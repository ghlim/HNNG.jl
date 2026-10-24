isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))
const KLDivergence = HNNG.Distances.KLDivergence

# The parent-distance filter (`HNNG._knn_pf!`) forced on and off through the internal
# `pf` argument, independent of the gate `HNNG._use_pf`: same ids, same distance bits,
# never more evaluations.

bitsof(v) = reinterpret(UInt64, collect(Float64, vec(v)))

"""`n` columns of `X`-like data with every fifth column repeated once or twice (exact duplicates)."""
function pf_with_dups(rng, X, n)
    cols = [X[:, j] for j in axes(X, 2)]
    for j in 1:5:size(X, 2), _ in 1:rand(rng, 1:2)
        length(cols) < n && push!(cols, copy(X[:, j]))
    end
    Y = reduce(hcat, cols)
    Y[:, randperm(rng, size(Y, 2))]
end

pf_data(rng, T, d, n, geom) =
    pf_with_dups(rng, geom === :blobs ? blobs(rng, T, d, n - n ÷ 5; nc=6) : rand(rng, T, d, n - n ÷ 5), n)

"""
Counters shared by the cases: `bad` mismatches, evaluation totals on/off, cases where
forced-on evaluated more, and the on/off totals of the clustered d >= 64 cases.
"""
mutable struct PFTally
    bad::Int
    on::Int
    off::Int
    worse::Int
    hi_on::Int
    hi_off::Int
end
PFTally() = PFTally(0, 0, 0, 0, 0, 0)

function pf_record!(t::PFTally, ok::Bool, eon::Int, eoff::Int, hi::Bool)
    ok || (t.bad += 1)
    t.on += eon
    t.off += eoff
    eon > eoff && (t.worse += 1)
    if hi
        t.hi_on += eon
        t.hi_off += eoff
    end
    ok && eon <= eoff
end

"""Forced-on vs forced-off `_knn!` on one prepared query (optionally seeded)."""
function pf_pair(idx, qp, k, seed=0)
    m1, m2 = HNNG.DistMemo(), HNNG.DistMemo()
    a1, d1 = HNNG._knn!(m1, idx, qp, k, seed, true)
    a2, d2 = HNNG._knn!(m2, idx, qp, k, seed, false)
    (a1 == a2 && bitsof(d1) == bitsof(d2)), m1.nevals, m2.nevals
end

"""Forced-on vs forced-off through the internal bodies of `knn_stored` (evaluations via `nevals`)."""
function pf_stored_pair(idx, id, k)
    e0 = idx.nevals
    r1 = HNNG._knn_stored(idx, id, k, true)
    e1 = idx.nevals - e0
    r2 = HNNG._knn_stored(idx, id, k, false)
    e2 = idx.nevals - e0 - e1
    (r1[1] == r2[1] && bitsof(r1[2]) == bitsof(r2[2])), e1, e2
end

"""Every check of one index: external queries, stored points, knn_stored, threaded knn_batch."""
function pf_check_index!(t::PFTally, rng, idx, X, ks, hi)
    T = eltype(X)
    d, n = size(X)
    live = findall(idx.alive)
    Q = hcat(X[:, rand(rng, 1:n, 4)] .+ T(0.05) .* randn(rng, T, d, 4),   # near the data
             X[:, rand(rng, live, 2)],                                     # stored points (distance 0)
             rand(rng, T, d, 2), 3 .* randn(rng, T, d, 1))                 # elsewhere
    allok = true
    for j in axes(Q, 2), k in ks
        ok, eon, eoff = pf_pair(idx, HNNG._prepare(idx, view(Q, :, j)), k)
        allok &= pf_record!(t, ok, eon, eoff, hi)
    end
    for id in live[1:max(1, length(live) ÷ 6):end], k in (1, 5, n + 3)
        ok, eon, eoff = pf_stored_pair(idx, id, k)
        allok &= pf_record!(t, ok, eon, eoff, hi)
        if k > 1                                   # the seeded engine itself, k + 1 as knn_stored runs it
            ok, eon, eoff = pf_pair(idx, HNNG._point(idx.store, id), k + 1, id)
            allok &= pf_record!(t, ok, eon, eoff, hi)
        end
    end
    for k in (1, 7)
        I1, D1 = HNNG._knn_batch(idx, Q, k, true, true)
        I2, D2 = HNNG._knn_batch(idx, Q, k, true, false)
        ok = I1 == I2 && bitsof(D1) == bitsof(D2)
        ok || (t.bad += 1)
        allok &= ok
    end
    # the public calls run the gated engine
    pf = HNNG._use_pf(idx)
    q = view(Q, :, 1)
    m = HNNG.DistMemo()
    a, b = HNNG._knn!(m, idx, HNNG._prepare(idx, q), 5, 0, pf)
    allok &= knn(idx, q, 5) == (a, HNNG._to_user.(Ref(idx), b))
    allok
end

@testset "parent-distance filter: forced on == forced off, bitwise, fewer evaluations" begin
    rng = StableRNG(6401)
    t = PFTally()
    dims = (3, 16, 64, 100)
    n = 160
    for M in (Euclidean(), SqEuclidean(), CosineDist()), T in (Float32, Float64), d in dims,
        geom in (:blobs, :uniform), build in (:add, :bulk)
        X = pf_data(rng, T, d, n, geom)
        idx = HNNGIndex(M, d; T=T)
        build === :add ? add!(idx, X) : bulk_add!(idx, X)
        @test verify(idx)
        hi = geom === :blobs && d >= 64
        ks = (1, 5, 20, n + 5)
        @test pf_check_index!(t, rng, idx, X, ks, hi)
        if build === :add && T === Float64       # after deletions (incl. duplicates and the root rep)
            for id in vcat(1:9:n, [top_rep(idx)])
                idx.alive[id] && delete!(idx, id)
            end
            @test verify(idx)
            @test pf_check_index!(t, rng, idx, X, ks, hi)
        end
    end
    # Float16: one small case per kernel path (accumulates in Float32)
    for M in (Euclidean(), CosineDist()), d in (5, 64)
        X = pf_data(rng, Float16, d, 80, :blobs)
        idx = HNNGIndex(M, d; T=Float16)
        add!(idx, X)
        @test verify(idx)
        @test pf_check_index!(t, rng, idx, X, (1, 5, 90), false)
    end
    @test t.bad == 0
    @test t.worse == 0
    @test t.on <= t.off
    @test t.hi_on < t.hi_off                       # the filter does skip on clustered d >= 64 data
    @info "pf_filter evaluations" on = t.on off = t.off hi_on = t.hi_on hi_off = t.hi_off
end

@testset "parent-distance filter: other stores and metrics, forced" begin
    rng = StableRNG(6402)
    t = PFTally()
    # ObjectStore under a Distances metric, UserMetric declared metric, and a non-metric
    # (τ stays +Inf there, so nothing is skipped)
    X = pf_data(rng, Float64, 6, 120, :blobs)
    objs = [X[:, j] for j in axes(X, 2)]
    o = HNNGIndex(Euclidean(), Vector{Float64})
    add!(o, objs)
    um = HNNGIndex(UserMetric((a, b) -> sqrt(sum(abs2, a .- b)); is_metric=true), 6; T=Float64)
    add!(um, X)
    nm = HNNGIndex(UserMetric((a, b) -> sum(abs2, a .- b)), 6; T=Float64)
    add!(nm, X)
    for idx in (o, um, nm)
        @test verify(idx)
        for j in 1:12, k in (1, 4, 130)
            q = idx === o ? objs[rand(rng, 1:120)] .+ 0.05 .* randn(rng, 6) : X[:, rand(rng, 1:120)] .+ 0.05 .* randn(rng, 6)
            ok, eon, eoff = pf_pair(idx, HNNG._prepare(idx, q), k)
            @test pf_record!(t, ok, eon, eoff, false)
        end
        for id in 1:15:120
            ok, eon, eoff = pf_stored_pair(idx, id, 6)
            @test pf_record!(t, ok, eon, eoff, false)
        end
    end
    @test t.bad == 0 && t.worse == 0
end

@testset "parent-distance filter: the gate" begin
    for M in (Euclidean(), SqEuclidean(), CosineDist()), T in (Float32, Float64)
        @test !HNNG._use_pf(HNNGIndex(M, 63; T=T))
        @test HNNG._use_pf(HNNGIndex(M, 64; T=T))
        @test HNNG._use_pf(HNNGIndex(M, 784; T=T))
    end
    @test HNNG._use_pf(HNNGIndex(Euclidean(), 64; T=Float16))
    @test HNNG.PF_MIN_DIM == 64
    @test !HNNG._use_pf(HNNGIndex(Euclidean(), Vector{Float64}))                      # ObjectStore
    @test !HNNG._use_pf(HNNGIndex(UserMetric((a, b) -> 0.0; is_metric=true), 64))     # UserMetric, VectorStore
    @test !HNNG._use_pf(HNNGIndex(UserMetric((a, b) -> 0.0; is_metric=true), Vector{Float64}))
    @test !HNNG._use_pf(HNNGIndex((a, b) -> 0.0, 128))                                 # bare callable: non-metric UserMetric
    @test !HNNG._use_pf(HNNGIndex(KLDivergence(), 128))                                # non-metric PreMetric
    idx = HNNGIndex(Euclidean(), 64)
    idx.is_metric = false                                                              # the is_metric clause itself
    @test !HNNG._use_pf(idx)
end
