isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))

@testset "save/load round trip: VectorStore" begin
    rng = StableRNG(17)
    for M in (Euclidean(), SqEuclidean(), CosineDist())
        idx = HNNGIndex(M, 4)
        add!(idx, blobs(rng, Float32, 4, 300))
        @test verify(idx)
        for id in (5, 77, 123)
            delete!(idx, id)
            @test verify(idx)
        end
        path = joinpath(mktempdir(), "idx.hnng")
        @test HNNG.save(path, idx) == path
        idx2 = HNNG.load(path)
        @test verify(idx2)
        @test idx2.metric == idx.metric && idx2.guard == idx.guard && length(idx2) == 297 && idx2.nevals == 0
        @test idx2.nlive == idx.nlive
        @test same_structure(structure(idx2), structure(idx))
        for _ in 1:20
            q = randn(rng, Float32, 4)
            @test knn(idx2, q, 7) == knn(idx, q, 7)
        end
        q = randn(rng, Float32, 4)
        @test HNNG.knn_levelsync(idx2, q, 7) == knn(idx, q, 7)
        @test add!(idx2, randn(rng, Float32, 4)) == 301
        @test verify(idx2)
    end
end

@testset "save/load round trip: ObjectStore needs metric=" begin
    rng = StableRNG(171)
    S = descriptor_sets(rng, 3, 60)
    idx = HNNGIndex(UserMetric(hausdorff; is_metric=true), Matrix{Float64})
    add!(idx, S)
    @test verify(idx)
    path = joinpath(mktempdir(), "obj.hnng")
    HNNG.save(path, idx)
    @test_throws ArgumentError HNNG.load(path)
    idx2 = HNNG.load(path; metric=UserMetric(hausdorff; is_metric=true))
    @test verify(idx2) && is_metric(idx2)
    for q in descriptor_sets(rng, 3, 5)
        @test knn(idx2, q, 4) == knn(idx, q, 4)
    end
end

@testset "load rejects other files" begin
    dir = mktempdir()
    path = joinpath(dir, "other.jls")
    open(io -> HNNG.serialize(io, (a = 1,)), path, "w")
    @test_throws "not an HNNG.jl index file" HNNG.load(path)
    idx = HNNGIndex(Euclidean(), 2)
    add!(idx, Float32[0 1; 0 1])
    @test verify(idx)
    p = open(HNNG.deserialize, HNNG.save(joinpath(dir, "good.hnng"), idx))
    newer = joinpath(dir, "newer.hnng")
    open(io -> HNNG.serialize(io, merge(p, (version = HNNG.FORMAT_VERSION + 1,))), newer, "w")
    @test_throws ArgumentError HNNG.load(newer)
    @test_throws "format version" HNNG.load(newer)
end
