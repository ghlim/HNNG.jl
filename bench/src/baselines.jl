# One method interface over hNNG and every baseline (spec §11).
export BenchMethod, method_name, build, query, build_time, canonical,
       HNNGIncremental, HNNGBulk, HNNGHandle, fraction_evaluated,
       BruteBLAS, BruteHandle, KDTreeNN, BallTreeNN, TreeHandle

"""
A benchmarked method. Interface:
- `method_name(m) -> String`: the CSV `method` value
- `build(m, X::Matrix{Float32}) -> handle`: X is d×n, columns are points, ids are column numbers
- `query(handle, q, k) -> (ids::Vector{Int}, dists::Vector{Float64})`: sorted by (distance, id)
- `build_time(m, X) -> (handle, seconds)`
"""
abstract type BenchMethod end

"True when (dists, ids) is already ordered by (distance, id): the common case, checked without allocating."
function is_canonical(ids::AbstractVector, dists::AbstractVector)
    @inbounds for j in 2:length(ids)
        dists[j-1] < dists[j] || (dists[j-1] == dists[j] && ids[j-1] <= ids[j]) || return false
    end
    return true
end

"""
Order by (distance, id) and convert to (Vector{Int}, Vector{Float64}). The trees are called
with `sortres = true` (the library's own in-place sort by distance), so `canonical` only fixes
the id order within equal distances and widens the distances; the sort here (which allocates)
runs only when `is_canonical` fails, e.g. on ties, so the common timed query path is not
charged for it.
"""
function canonical(ids::AbstractVector, dists::AbstractVector)
    if !is_canonical(ids, dists)
        p = sort(collect(eachindex(ids)); by = j -> (dists[j], ids[j]))
        ids, dists = ids[p], dists[p]
    end
    return convert(Vector{Int}, ids), convert(Vector{Float64}, dists)
end

"""
    build_time(m, X) -> (handle, seconds)

Untimed warm-up build on the first min(n, 2000) columns (compiles the build path; 200
columns left code paths of SimilaritySearch uncompiled), a full GC, then one timed full
build (spec assumption A11).

Compile guard: Julia's cumulative compile-time counter is read around the timed build only.
If more than max(0.05 s, 1% of the build time) went into compilation, a
"compilation inside timed build" warning is logged with the method, n, compile and build
seconds, so a polluted `build_s` is visible rather than silent.
"""
function build_time(m::BenchMethod, X::Matrix{Float32})
    build(m, X[:, 1:min(size(X, 2), 2000)])
    GC.gc()
    c0 = 0
    local h, seconds, c1
    Base.cumulative_compile_timing(true)
    try
        c0 = Base.cumulative_compile_time_ns()[1]
        t0 = time_ns()
        h = build(m, X)
        seconds = (time_ns() - t0) / 1e9
        c1 = Base.cumulative_compile_time_ns()[1]
    finally
        Base.cumulative_compile_timing(false)
    end
    compile_s = (c1 - c0) / 1e9
    if compile_s > max(0.05, 0.01 * seconds)
        @warn "compilation inside timed build" method = method_name(m) n = size(X, 2) compile_s = compile_s build_s = seconds
    end
    return h, seconds
end

# ---- hNNG ---------------------------------------------------------------------------
struct HNNGIncremental <: BenchMethod end
struct HNNGBulk <: BenchMethod end
struct HNNGHandle{I}
    idx::I
end
method_name(::HNNGIncremental) = "hNNG-incremental"
method_name(::HNNGBulk) = "hNNG-bulk"
new_hnng(d::Int) = HNNG.HNNGIndex(Euclidean(), d; T = Float32)

function build(::HNNGIncremental, X::Matrix{Float32})
    idx = new_hnng(size(X, 1))
    HNNG.add!(idx, X)                 # streaming inserts, one column at a time, in order
    return HNNGHandle(idx)
end
function build(::HNNGBulk, X::Matrix{Float32})
    idx = new_hnng(size(X, 1))
    HNNG.bulk_add!(idx, X)
    return HNNGHandle(idx)
end
query(h::HNNGHandle, q::AbstractVector{Float32}, k::Int) = HNNG.knn(h.idx, q, k)

"Mean over queries of (distance evaluations of the query) / (live elements)."
fraction_evaluated(h::HNNGHandle, qs, k::Int) =
    mean(HNNG.knn_counted(h.idx, q, k)[3] for q in qs) / length(h.idx)

# ---- brute force: Gram trick with gemv, exact Float64 re-rank -----------------------
struct BruteBLAS <: BenchMethod end
method_name(::BruteBLAS) = "brute-BLAS"
"Largest d for which the brute force scores on an n×d copy (gemv 'N'); above it the d×n 'T' gemv was faster (Fashion-MNIST, d = 784)."
const BRUTE_ROWMAJOR_MAXD = 128
"""
Brute-force handle. It reuses the mutable buffers `s`, `a` and `top` on every query, so it is
not thread-safe; the timing protocol runs one query at a time. `Xt` is the n×d copy
`permutedims(X)` when d <= $(BRUTE_ROWMAJOR_MAXD) (0×0 otherwise).
"""
struct BruteHandle
    X::Matrix{Float32}
    Xt::Matrix{Float32}     # n×d copy of X for d <= BRUTE_ROWMAJOR_MAXD, else 0×0
    n2::Vector{Float32}     # ‖x_i‖²
    maxn2::Float32
    s::Vector{Float32}      # Xᵀq
    a::Vector{Float32}      # ‖x_i‖² − 2 x_iᵀq  (= ‖x_i − q‖² − ‖q‖²)
    top::Vector{Float32}    # the k smallest scores, sorted ascending
end
function build(::BruteBLAS, X::Matrix{Float32})
    d, n = size(X)
    n2 = vec(sum(abs2, X; dims = 1))
    Xt = d <= BRUTE_ROWMAJOR_MAXD ? permutedims(X) : Matrix{Float32}(undef, 0, 0)
    return BruteHandle(X, Xt, n2, maximum(n2; init = 0f0), zeros(Float32, n), zeros(Float32, n), Float32[])
end

"""
    brute_tau(d, q, maxn2) -> τ

Candidate window of the brute force: τ = max(1e-4, 4(d+3)·eps(Float32))·(‖q‖² + max‖x‖²).

Bound. In the standard worst-case rounding model a Float32 score a_i = ‖x_i‖² − 2x_iᵀq is off by
at most γ_d·(‖x_i‖² + 2Σ_j|x_ij q_j|) + eps·|a_i| with γ_d = d·eps/(1 − d·eps), and both terms in
brackets are ≤ 2(‖q‖² + max‖x‖²). The k-th score and a candidate's score can both be off, so
the window must cover twice that error, 4(γ_d + eps)(‖q‖² + max‖x‖²); 4(d+3)·eps covers it with
a factor-2 margin for every d <= 2800 (Fashion-MNIST, d = 784, is inside the range). Beyond
d = 2800 the window is no longer a proven bound.
"""
brute_tau(d::Integer, q::AbstractVector{Float32}, maxn2::Float32) =
    max(1f-4, 4f0 * (d + 3) * eps(Float32)) * (sum(abs2, q) + maxn2)

"""
Scores every point with one BLAS gemv: on the n×d copy (gemv 'N') when d <= $(BRUTE_ROWMAJOR_MAXD),
on the d×n data (transposed gemv) otherwise. One pass writes the scores and keeps the k
smallest in a sorted k-slot buffer, whose last slot is the k-th score t. Candidates are all
points whose Float32 score lies within τ = `brute_tau` of t, a proven bound on the score's
rounding for d <= 2800; they are re-ranked with the Float64 kernel, so the answer is exact.
"""
function query(h::BruteHandle, q::AbstractVector{Float32}, k::Int)
    X = h.X
    d, n = size(X)
    kk = min(k, n)
    if size(h.Xt, 1) == n && n > 0
        mul!(h.s, h.Xt, q)                         # BLAS sgemv 'N' on the n×d copy
    else
        mul!(h.s, transpose(X), q)                 # BLAS sgemv 'T' on the d×n data
    end
    top = h.top
    resize!(top, kk)
    fill!(top, Inf32)
    @inbounds for i in 1:n
        a = h.n2[i] - 2f0 * h.s[i]
        h.a[i] = a
        if kk > 0 && a < top[kk]                   # insertion into the sorted k-slot buffer
            j = kk
            while j > 1 && top[j-1] > a
                top[j] = top[j-1]
                j -= 1
            end
            top[j] = a
        end
    end
    t = top[kk]
    τ = brute_tau(d, q, h.maxn2)
    cand = findall(<=(t + τ), h.a)
    ds = Float64[dist64(view(X, :, i), q) for i in cand]
    p = first(sort(collect(eachindex(cand)); by = j -> (ds[j], cand[j])), kk)
    return cand[p], ds[p]
end

# ---- NearestNeighbors.jl trees --------------------------------------------------------
Base.@kwdef struct KDTreeNN <: BenchMethod
    leafsize::Int = 25            # NearestNeighbors.jl default
end
Base.@kwdef struct BallTreeNN <: BenchMethod
    leafsize::Int = 25
end
method_name(::KDTreeNN) = "KD-tree"
method_name(::BallTreeNN) = "ball-tree"
"""
Handle over a NearestNeighbors.jl tree. Tree distances are computed in the data's element type
(Float32) and widened to Float64 by `canonical`; the recall rules recompute distances in
Float64 from the returned ids, so this does not affect the reported recall.
"""
struct TreeHandle{T}
    tree::T
    n::Int
end
build(m::KDTreeNN, X::Matrix{Float32}) =
    TreeHandle(NN.KDTree(X, Euclidean(); leafsize = m.leafsize), size(X, 2))
build(m::BallTreeNN, X::Matrix{Float32}) =
    TreeHandle(NN.BallTree(X, Euclidean(); leafsize = m.leafsize), size(X, 2))
function query(h::TreeHandle, q::AbstractVector{Float32}, k::Int)
    ids, ds = NN.knn(h.tree, q, min(k, h.n), true)
    return canonical(ids, ds)
end

export HNSWjl, HNSWHandle, build_hnsw, set_ef!, hnsw_insert!,
       SSSearchGraph, SSExhaustive, SSHandle, optional_methods

# ---- HNSW.jl ------------------------------------------------------------------------
"HNSW.jl with the spec's build parameters; `ef` is the search beam (`set_ef!` changes it)."
Base.@kwdef struct HNSWjl <: BenchMethod
    M::Int = 16
    efConstruction::Int = 200
    ef::Int = 10
end
method_name(::HNSWjl) = "HNSW"
struct HNSWHandle{G}
    g::G
end

"""
    build_hnsw(m, P; nbuild = size(P, 2)) -> HNSWHandle

Hands every column of P to HNSW.jl (so later inserts need no resize) and links columns
1:nbuild into the graph. Seeds the global RNG first: HNSW.jl draws levels from it.
"""
function build_hnsw(m::HNSWjl, P::Matrix{Float32}; nbuild::Int = size(P, 2))
    Random.seed!(SEED_HNSW)
    data = [P[:, i] for i in 1:size(P, 2)]
    # Effective configuration: standard HNSW with M = m.M (16): m.M links per upper layer,
    # 2m.M (32) on layer 0, m_L = 1/ln m.M. HNSW.jl 0.1.6 passes M and M0 through two swaps that
    # cancel: src/interface.jl:74 calls `LayeredGraph{T}(max_elements, M0, M, m_L)` against the
    # signature `(num_elements, M, M0, m_L)` at src/layered_graphs.jl:39, which forwards
    # `(linklist, M, M0, m_L)` positionally into the struct whose fields are declared
    # `(linklist, M0, M, m_L)` at src/layered_graphs.jl:18-22. So the keywords land in the
    # fields of the same name, and `max_connections` (src/layered_graphs.jl:94) caps layer 0 at
    # lgraph.M0. Passing M = 2m.M, M0 = m.M would invert them. The values are passed explicitly
    # (they equal HNSW.jl's defaults M0 = 2M, m_L = 1/log(M)); the tests assert the fields.
    g = HNSW.HierarchicalNSW(data; metric = Euclidean(), M = m.M, M0 = 2 * m.M,
                             m_L = 1 / log(m.M), efConstruction = m.efConstruction, ef = m.ef)
    HNSW.add_to_graph!(g, 1:nbuild)
    return HNSWHandle(g)
end
build(m::HNSWjl, X::Matrix{Float32}) = build_hnsw(m, X)
set_ef!(h::HNSWHandle, ef::Int) = HNSW.set_ef!(h.g, ef)

"""
HNSW.jl's online insert: link column i (already in the data vector) into the graph.
Throws if i is already linked: HNSW.jl itself would only warn and skip it.
"""
function hnsw_insert!(h::HNSWHandle, i::Int)
    h.g.added[i] && throw(ArgumentError("hnsw_insert!: id $i is already in the graph"))
    return HNSW.add_to_graph!(h.g, [i])
end

function query(h::HNSWHandle, q::AbstractVector{Float32}, k::Int)
    ids, ds = HNSW.knn_search(h.g, convert(Vector{Float32}, q), k)
    return canonical(ids, ds)
end

# ---- SimilaritySearch.jl (optional) -----------------------------------------------------
struct SSSearchGraph <: BenchMethod end
struct SSExhaustive <: BenchMethod end
method_name(::SSSearchGraph) = "SS-SearchGraph"
method_name(::SSExhaustive) = "SS-Exhaustive"
struct SSHandle{I,C}
    index::I
    ctx::C
end

"The SimilaritySearch methods, or none when the package did not load."
optional_methods() = HAS_SS ? BenchMethod[SSSearchGraph(), SSExhaustive()] : BenchMethod[]

# Calls checked against the installed SimilaritySearch 1.6.7: positional constructors
# `SearchGraph(dist, db)` and `ExhaustiveSearch(dist, db)`, SimilaritySearch's own metric
# `Dist.L2()` (Euclidean, not squared, like every other method), and one query at a time
# through `search(index, ctx, q, knnqueue(ctx, k))`, read with `IdDistView`. The default
# SearchGraphContext auto-tunes the beam (OptimizeParameters(MinRecall(0.97))) during index!.
# `neardup = 0f0` folds exact duplicate points into cluster members and a query then returns one
# representative per duplicate group, which under-counts recall on data with repeated points
# (Fashion-MNIST); `expand!(G, ctx, q, res)` after each search gives back the raw k nearest,
# members included. Since 1.6.6 expand! takes the build context: with it, members of a graph
# folded at distance zero take their representative's distance instead of being re-evaluated.
function build(::SSSearchGraph, X::Matrix{Float32})
    Random.seed!(SEED_SS)             # RandomHints and the parameter optimiser draw from the global RNG
    SS = SimilaritySearch
    g = SS.SearchGraph(SS.Dist.L2(), SS.MatrixDatabase(X))
    # reporters = []: silent build (searchgraph/context.jl:42 in 1.6.7: "Pass `reporters=[]` to
    # silence this context completely"); the auto-tuning callback is unaffected
    ctx = SS.SearchGraphContext(; neighborhood = SS.Neighborhood(; neardup = 0f0), reporters = [])
    SS.index!(g, ctx)
    return SSHandle(g, ctx)
end
function build(::SSExhaustive, X::Matrix{Float32})
    SS = SimilaritySearch             # no randomness: index! is a no-op, search scans every column
    return SSHandle(SS.ExhaustiveSearch(SS.Dist.L2(), SS.MatrixDatabase(X)), SS.GenericContext(; reporters = []))
end
function query(h::SSHandle, q::AbstractVector{Float32}, k::Int)
    SS = SimilaritySearch
    qv = convert(Vector{Float32}, q)
    res = SS.knnqueue(h.ctx, k)
    SS.search(h.index, h.ctx, qv, res)
    h.index isa SS.SearchGraph && SS.expand!(h.index, h.ctx, qv, res)   # raw neighbours, duplicates included
    ids = Int[]
    ds = Float64[]
    for p in SS.IdDistView(res)            # only filled slots are visited: no padding convention to decode
        push!(ids, Int(p.id))
        push!(ds, Float64(p.dist))
    end
    return canonical(ids, ds)
end
