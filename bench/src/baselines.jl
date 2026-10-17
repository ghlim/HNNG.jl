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
Order by (distance, id) and convert to (Vector{Int}, Vector{Float64}). NearestNeighbors.jl,
HNSW.jl and SimilaritySearch already return sorted results, so the sort (which allocates)
runs only when `is_canonical` fails, e.g. on ties; the timed query path of the baselines is
then not charged for it.
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

Untimed warm-up build on the first min(n, 200) columns (compiles the build path), a full
GC, then one timed full build (spec assumption A11).
"""
function build_time(m::BenchMethod, X::Matrix{Float32})
    build(m, X[:, 1:min(size(X, 2), 200)])
    GC.gc()
    t0 = time_ns()
    h = build(m, X)
    return h, (time_ns() - t0) / 1e9
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
struct BruteHandle
    X::Matrix{Float32}
    n2::Vector{Float32}     # ‖x_i‖²
    maxn2::Float32
    s::Vector{Float32}      # Xᵀq
    a::Vector{Float32}      # ‖x_i‖² − 2 x_iᵀq  (= ‖x_i − q‖² − ‖q‖²)
    buf::Vector{Float32}
end
function build(::BruteBLAS, X::Matrix{Float32})
    n = size(X, 2)
    n2 = vec(sum(abs2, X; dims = 1))
    return BruteHandle(X, n2, maximum(n2; init = 0f0), zeros(Float32, n), zeros(Float32, n), zeros(Float32, n))
end
"""
Scores every point with one BLAS gemv. Candidates are all points whose Float32 score lies
within τ = max(1e-4, 2·d·eps(Float32))·(‖q‖² + max‖x‖²) of the k-th score, a bound on the
score's rounding; candidates are re-ranked with the Float64 kernel, so the answer is exact.
"""
function query(h::BruteHandle, q::AbstractVector{Float32}, k::Int)
    X = h.X
    d, n = size(X)
    kk = min(k, n)
    mul!(h.s, transpose(X), q)                     # BLAS sgemv
    @inbounds for i in 1:n
        h.a[i] = h.n2[i] - 2f0 * h.s[i]
    end
    copyto!(h.buf, h.a)
    t = partialsort!(h.buf, kk)
    τ = max(1f-4, 2f0 * d * eps(Float32)) * (sum(abs2, q) + h.maxn2)
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
struct TreeHandle{T}
    tree::T
    n::Int
end
build(m::KDTreeNN, X::Matrix{Float32}) =
    TreeHandle(NN.KDTree(X, Euclidean(); leafsize = m.leafsize), size(X, 2))
build(m::BallTreeNN, X::Matrix{Float32}) =
    TreeHandle(NN.BallTree(X, Euclidean(); leafsize = m.leafsize), size(X, 2))
function query(h::TreeHandle, q::AbstractVector{Float32}, k::Int)
    ids, ds = NN.knn(h.tree, q, min(k, h.n))
    return canonical(ids, ds)
end
