isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))
using Test, HNNG, StableRNGs
using Distances: Cityblock, CorrDist

@testset "stores and index construction" begin
    idx = HNNGIndex(Euclidean(), 3)
    @test idx.store isa HNNG.VectorStore{Float32}
    @test length(idx) == 0 && nlevels(idx) == 0 && idx.nevals == 0
    @test is_metric(idx)
    idx64 = HNNGIndex(SqEuclidean(), 5; T=Float64)
    @test idx64.store isa HNNG.VectorStore{Float64}
    @test HNNG.dim(idx64.store) == 5
    s = HNNG.VectorStore{Float32}(2)
    for i in 1:40
        @test HNNG._append!(s, Float32[i, -i]) == i
    end
    @test length(s) == 40
    @test HNNG._point(s, 17) == Float32[17, -17]
    @test HNNG._point(s, 40) == Float32[40, -40]
    o = HNNGIndex((a, b) -> abs(a - b), Int)
    @test o.store isa HNNG.ObjectStore{Int}
    @test HNNG._append!(o.store, 7) == 1
    @test HNNG._point(o.store, 1) == 7
    @test_throws ArgumentError HNNGIndex(Euclidean(), 0)
    @test sprint(show, idx) == "HNNGIndex(Euclidean, 0 live, 0 levels)"

    # _ntotal counts stored ids (including tombstones); length(idx) is the nlive counter
    @test HNNG._ntotal(idx) == 0 && HNNG._ntotal(idx64) == 0
    @test length(o.store) == 1 && HNNG._ntotal(o) == 1
    HNNG._append!(idx.store, Float32[1, 2, 3])
    @test HNNG._ntotal(idx) == 1 && length(idx.store) == 1
    @test length(idx) == 0 && idx.nlive == 0
    idx.nlive = 3
    @test length(idx) == 3 && HNNG._ntotal(idx) == 1
    @test sprint(show, idx) == "HNNGIndex(Euclidean, 3 live, 0 levels)"
    @test length(HNNG.ObjectStore{String}()) == 0
    @test length(HNNG.ObjectStore{String}(["a", "b"])) == 2

    # Level(n) defaults
    lv = HNNG.Level(3)
    @test lv.outlink == [0, 0, 0]
    @test lv.outdist == [Inf, Inf, Inf]
    @test length(lv.inlinks) == 3 && all(isempty, lv.inlinks)
    @test length(lv.present) == 3 && !any(lv.present)
    @test lv.cluster_of == [0, 0, 0]
    @test isempty(lv.clusters) && isempty(lv.free)
    @test lv.npop == 0 && lv.nclusters == 0
    push!(lv.inlinks[1], 9)            # inlink vectors must not be aliased
    @test isempty(lv.inlinks[2]) && lv.inlinks[1] == [9]
    @test isempty(HNNG.Level(0).outlink)

    # _grow! keeps old entries, pads new ones with defaults, never shrinks
    lv.outlink[2] = 5; lv.outdist[2] = 1.5; lv.present[2] = true
    lv.cluster_of[2] = 4; lv.inlinks[2] = [1, 3]; lv.npop = 1
    @test HNNG._grow!(lv, 5) === lv
    @test lv.outlink == [0, 5, 0, 0, 0]
    @test lv.outdist == [Inf, 1.5, Inf, Inf, Inf]
    @test lv.present == Bool[0, 1, 0, 0, 0]
    @test lv.cluster_of == [0, 4, 0, 0, 0]
    @test length(lv.inlinks) == 5 && lv.inlinks[1] == [9] && lv.inlinks[2] == [1, 3]
    @test all(isempty, lv.inlinks[3:5])
    @test lv.npop == 1
    HNNG._grow!(lv, 2)
    @test length(lv.outlink) == 5 && length(lv.outdist) == 5 && length(lv.inlinks) == 5 &&
          length(lv.present) == 5 && length(lv.cluster_of) == 5

    # Cluster and the dead slot
    c = HNNG.Cluster(2, [2, 5], [0.0, 1.5], 3.0, [1.0, 2.0])
    @test c.rep == 2 && c.members == [2, 5] && c.repdist == [0.0, 1.5]
    @test c.radius == 3.0 && c.maxout == [1.0, 2.0]
    d1 = HNNG._dead_cluster(); d2 = HNNG._dead_cluster()
    @test d1 isa HNNG.Cluster && d1.rep == 0 && d1.radius == 0.0
    @test isempty(d1.members) && isempty(d1.repdist) && isempty(d1.maxout)
    @test d1.members !== d2.members && d1.maxout !== d2.maxout
    @test eltype(d1.members) == Int && eltype(d1.repdist) == Float64
end

@testset "prune guard" begin
    @test HNNGIndex(Euclidean(), 784).guard == 4 * 784 * eps(Float32)
    @test HNNGIndex(Euclidean(), 784; guard=nothing).guard == 4 * 784 * eps(Float32)
    @test HNNGIndex(Euclidean(), 2; T=Float64).guard == 1e-6
    @test HNNGIndex(Euclidean(), 3; guard=1e-3).guard == 1e-3
    @test HNNGIndex((a, b) -> abs(a - b), Float64).guard == 1e-6
    @test HNNGIndex((a, b) -> abs(a - b), Float64; guard=1e-3).guard == 1e-3
end

@testset "is_metric rules" begin
    @test is_metric(HNNGIndex(Euclidean(), 2))
    @test is_metric(HNNGIndex(SqEuclidean(), 2))
    @test is_metric(HNNGIndex(CosineDist(), 2))
    @test is_metric(HNNGIndex(Cityblock(), 2))
    @test !is_metric(HNNGIndex(CorrDist(), 2))
    @test !is_metric(HNNGIndex(SqEuclidean(), Vector{Float64}))
    @test is_metric(HNNGIndex(Euclidean(), Vector{Float64}))
    f(a, b) = abs(a - b)
    @test !is_metric(HNNGIndex(f, Float64))
    @test HNNGIndex(f, Float64).metric isa UserMetric
    @test is_metric(HNNGIndex(UserMetric(f; is_metric=true), Float64))
    @test !is_metric(HNNGIndex(UserMetric(f), Float64))
end

@testset "distance memo epochs" begin
    m = HNNG.DistMemo()
    HNNG._begin!(m, 10)
    @test length(m.seen) == 10 && m.epoch == 1
    @test m.nevals == 0 && isempty(m.front) && isempty(m.res)
    @test all(==(0), m.seen) && all(==(0), m.mark)
    m.seen[3] = m.epoch
    m.mark[4] = m.epoch
    push!(m.front, (1.0, 0, 1)); push!(m.res, (2.0, 3))
    @test HNNG._begin!(m, 12) === m
    @test length(m.val) == 12 && m.epoch == 2 && m.seen[3] != m.epoch
    @test m.mark[4] != m.epoch                     # stale mark from the previous pass
    @test length(m.mark) == 12 && m.seen[11] == 0 && m.mark[12] == 0   # grown slots are zero
    @test isempty(m.front) && isempty(m.res)       # buffers emptied by _begin!
    m.epoch = typemax(UInt32)
    m.seen[5] = typemax(UInt32)
    m.mark[7] = typemax(UInt32)
    push!(m.front, (1.0, 0, 1)); push!(m.res, (2.0, 3))
    HNNG._begin!(m, 12)
    @test m.epoch == 1 && m.seen[5] == 0
    @test m.mark[7] == 0 && all(==(0), m.seen) && all(==(0), m.mark)   # wrap clears both
    @test isempty(m.front) && isempty(m.res)
    HNNG._begin!(m, 5)                             # a smaller n never shrinks the buffers
    @test length(m.val) == 12 && length(m.seen) == 12 && m.epoch == 2
end

@testset "Euclidean kernel" begin
    rng = StableRNG(3)
    for T in (Float32, Float64), d in (1, 3, 33)
        a = randn(rng, T, d)
        b = randn(rng, T, d)
        k = HNNG._euclid(a, b)
        @test k isa Float64
        @test k == HNNG._euclid(b, a)
        @test isapprox(k, sqrt(sum(abs2, Float64.(a) .- Float64.(b))); rtol = T == Float32 ? 1e-5 : 1e-12)
        @test HNNG._euclid(a, a) == 0.0
    end
end

"""
Reference for `_euclid`, written out with plain scalar loops in the kernel's fixed order:
`W` lanes (8, or 4 for `Float64`) × 4 accumulators over blocks of `4W`, then blocks of
`W` into accumulator 1, lane-wise `(c1 + c2) + (c3 + c4)`, halving tree
`v[l] + v[l + W/2]`, then the `n mod W` tail left to right. `Float16` inputs are
widened to `Float32` first and accumulate in `Float32`.
"""
function ref_euclid(a::AbstractVector{T}, b::AbstractVector{T}) where {T}
    x, y = T === Float16 ? (Float32.(a), Float32.(b)) : (collect(a), collect(b))
    n = length(x)
    W = T === Float64 ? 4 : 8
    acc = zeros(eltype(x), W, 4)               # acc[l, c]: lane l of accumulator c
    i = 0
    while i + 4W <= n
        for c in 1:4, l in 1:W                 # lane l of c_j takes coordinate i + (j-1)W + l
            k = i + (c - 1) * W + l
            t = x[k] - y[k]
            acc[l, c] = acc[l, c] + t * t
        end
        i += 4W
    end
    while i + W <= n                           # leftover blocks of W go into c1
        for l in 1:W
            t = x[i+l] - y[i+l]
            acc[l, 1] = acc[l, 1] + t * t
        end
        i += W
    end
    v = [(acc[l, 1] + acc[l, 2]) + (acc[l, 3] + acc[l, 4]) for l in 1:W]
    while length(v) > 1                        # horizontal halving tree
        h = length(v) ÷ 2
        v = [v[l] + v[l+h] for l in 1:h]
    end
    s = v[1]
    for k in i+1:n                             # tail, left to right
        t = x[k] - y[k]
        s = s + t * t
    end
    sqrt(Float64(s))
end

"""Values with mixed magnitudes 2^-8 … 2^8, so different summation orders round differently."""
mixed(rng, ::Type{T}, dims...) where {T} = T.(randn(rng, dims...) .* exp2.(rand(rng, -8:8, dims...)))

@testset "kernel bits are call-site independent" begin
    rng = StableRNG(31)
    for T in (Float32, Float64), d in (1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 784)
        X = mixed(rng, T, d, 2)
        x, y = X[:, 1], X[:, 2]
        r = ref_euclid(x, y)
        # stride-2 views (getindex path) holding the same values
        Y = zeros(T, 2d, 2); Y[1:2:end, 1] = x; Y[1:2:end, 2] = y
        s1, s2 = view(Y, 1:2:2d, 1), view(Y, 1:2:2d, 2)
        # non-strided views (index vectors) and a unit-stride view at an offset (pointer path)
        u1, u2 = view(Y, collect(1:2:2d), 1), view(Y, collect(1:2:2d), 2)
        Z = zeros(T, d + 3, 2); Z[4:end, 1] = x; Z[4:end, 2] = y
        o1, o2 = view(Z, 4:d+3, 1), view(Z, 4:d+3, 2)
        k = HNNG._euclid(x, y)
        @test k === r
        @test k === HNNG._euclid(view(X, :, 1), view(X, :, 2)) === HNNG._euclid(y, x)
        @test k === HNNG._euclid(x, view(X, :, 2)) === HNNG._euclid(view(X, :, 1), y)
        @test k === HNNG._euclid(view(X, :, 2), view(X, :, 1)) === HNNG._euclid(reshape(X, :)[1:d], vec(X[:, 2:2]))
        @test k === HNNG._euclid(s1, s2) === HNNG._euclid(s2, s1) === HNNG._euclid(s1, y) === HNNG._euclid(x, s2)
        @test k === HNNG._euclid(u1, u2) === HNNG._euclid(u2, x) === HNNG._euclid(o1, o2) === HNNG._euclid(o1, s2)
    end
    for T in (Float32, Float64), d in (33, 784)            # random pairs against the reference
        ok = true
        for _ in 1:50
            x, y = mixed(rng, T, d), mixed(rng, T, d)
            ok &= HNNG._euclid(x, y) === ref_euclid(x, y) === HNNG._euclid(y, x)
        end
        @test ok
    end
    # the reference detects a reordering: a plain left-to-right sum differs somewhere
    for T in (Float32, Float64)
        differs = false
        for _ in 1:200
            x, y = mixed(rng, T, 784), mixed(rng, T, 784)
            seq = zero(T)
            for i in 1:784
                seq += (x[i] - y[i]) * (x[i] - y[i])
            end
            differs |= sqrt(Float64(seq)) !== ref_euclid(x, y)
        end
        @test differs
    end
    # Float16 runs the Float32 lanes on widened values (W = 8); the same bits at every call site
    H = mixed(rng, Float16, 37, 2)
    @test HNNG._euclid(H[:, 1], view(H, :, 2)) === ref_euclid(H[:, 1], H[:, 2]) === HNNG._euclid(H[:, 2], H[:, 1])
    Hs = zeros(Float16, 74, 2); Hs[1:2:end, :] = H
    @test HNNG._euclid(view(Hs, 1:2:74, 1), view(H, :, 2)) === ref_euclid(H[:, 1], H[:, 2])
    # any other float type takes the scalar path with the same lane map (W = 8)
    B = big.(mixed(rng, Float64, 37, 2))
    @test HNNG._euclid(B[:, 1], B[:, 2]) == ref_euclid(B[:, 1], B[:, 2])
    @test HNNG._euclid(Float32[], Float32[]) === 0.0 === HNNG._euclid(view(zeros(0, 1), :, 1), Float64[])
    @test_throws DimensionMismatch HNNG._euclid(zeros(3), zeros(4))
    @test_throws DimensionMismatch HNNG._euclid(zeros(Float32, 3), view(zeros(Float32, 8), 1:2:8))
end

@testset "Float16 data accumulates in Float32 (8-bit pixels)" begin
    rng = StableRNG(64)
    P = rand(rng, 0:255, 64, 300)                    # 8-bit pixels: Float16 holds them exactly
    Q = rand(rng, 0:255, 64, 25)
    for M in (Euclidean(), CosineDist())
        h = HNNGIndex(M, 64; T=Float16)
        add!(h, P)                                   # streaming
        @test verify(h)
        b = HNNGIndex(M, 64; T=Float16)
        bulk_add!(b, P)                              # bulk (direct path for Float16)
        @test verify(b)
        @test same_structure(structure(b), structure(h))
        f = HNNGIndex(M, 64; T=Float32)
        add!(f, P)
        @test verify(f)
        for j in axes(Q, 2)
            ids, ds = knn(h, Q[:, j], 5)
            @test all(isfinite, ds) && ds[1] > 0
            @test ids == knn(f, Q[:, j], 5)[1]
            @test knn(b, Q[:, j], 5) == (ids, ds)
            @test check_knn(h, Q[:, j], 5)
        end
    end
    # pixel distances are integers below 2^24, so Float16 storage loses nothing under Euclidean
    e16, e32 = HNNGIndex(Euclidean(), 64; T=Float16), HNNGIndex(Euclidean(), 64)
    @test HNNG.distance(e16, P[:, 1], P[:, 2]) === HNNG.distance(e32, P[:, 1], P[:, 2])
    # the cosine normalisation sums in Float32: a pixel vector keeps unit length
    c16 = HNNGIndex(CosineDist(), 64; T=Float16)
    @test isapprox(sum(abs2, Float64.(HNNG._prepare(c16, P[:, 1]))), 1.0; atol=1e-2)
end

@testset "units and input preparation" begin
    e = HNNGIndex(Euclidean(), 2; T=Float64)
    s = HNNGIndex(SqEuclidean(), 2; T=Float64)
    c = HNNGIndex(CosineDist(), 2; T=Float64)
    @test HNNG.distance(e, [0, 0], [3, 4]) == 5.0
    @test HNNG.distance(s, [0, 0], [3, 4]) ≈ 25.0
    @test HNNG.distance(c, [1, 0], [0, 2]) ≈ 1.0
    @test HNNG.distance(c, [1, 1], [2, 2]) ≈ 0.0 atol = 1e-12
    @test HNNG.distance(c, [1, 0], [-3, 0]) ≈ 2.0
    @test HNNG._prepare(c, [3.0, 4.0]) ≈ [0.6, 0.8]
    # Review Focus 4: the zero vector stays zero under cosine and is 0.5 from any unit vector
    z = HNNG._prepare(c, [0.0, 0.0])
    @test z == [0.0, 0.0] && all(isfinite, z)
    @test HNNG.distance(c, [0, 0], [5, 0]) ≈ 0.5
    f32 = HNNGIndex(Euclidean(), 3)
    v = HNNG._prepare(f32, [1, 2, 3])
    @test v isa Vector{Float32} && v == Float32[1, 2, 3]
    @test HNNG._prepare(f32, view([1.0 2 3; 4 5 6], 2, :)) == Float32[4, 5, 6]
    @test_throws "dimension mismatch" HNNG._prepare(f32, [1.0, 2.0])
    @test_throws "non-finite coordinate NaN" HNNG._prepare(f32, [1.0, NaN, 2.0])
    @test_throws "non-finite coordinate 1.0e300" HNNG._prepare(f32, [1.0, 1e300, 2.0])
    u = HNNGIndex(UserMetric((a, b) -> abs(a - b); is_metric=true), Int)
    @test HNNG.distance(u, 3, 10) === 7.0
end

@testset "element and counted distances" begin
    idx = HNNGIndex(Euclidean(), 2; T=Float64)
    for x in ([0.0, 0.0], [3.0, 4.0])
        HNNG._append!(idx.store, HNNG._prepare(idx, x))
        push!(idx.alive, true)
    end
    @test element(idx, 2) == [3.0, 4.0]
    @test HNNG._d(idx, 1, 2) == 5.0 && idx.nevals == 1
    m = HNNG._begin!(HNNG.DistMemo(), 2)
    q = [0.0, 4.0]
    @test HNNG._mdist(m, idx, q, 2) == 3.0
    @test HNNG._mdist(m, idx, q, 2) == 3.0
    @test m.nevals == 1
    @test_throws "unknown id 3" element(idx, 3)
    idx.alive[1] = false
    @test_throws "id 1 is deleted" element(idx, 1)
end
