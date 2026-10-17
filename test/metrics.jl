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
    m.seen[3] = m.epoch
    HNNG._begin!(m, 12)
    @test length(m.val) == 12 && m.epoch == 2 && m.seen[3] != m.epoch
    m.epoch = typemax(UInt32)
    m.seen[5] = typemax(UInt32)
    HNNG._begin!(m, 12)
    @test m.epoch == 1 && m.seen[5] == 0
end
