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
