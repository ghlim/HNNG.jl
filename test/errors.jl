isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))

@testset "errors (spec §9)" begin
    idx = HNNGIndex(Euclidean(), 3)
    add!(idx, Float32[1 2 3; 4 5 6; 7 8 9])
    @test verify(idx)
    @test_throws "dim must be at least 1, got 0" HNNGIndex(Euclidean(), 0)
    # dimension mismatch on add! and knn
    @test_throws "the input has length 2" add!(idx, [1.0, 2.0])
    @test_throws "the matrix has 2 rows" add!(idx, ones(2, 4))
    @test_throws "the input has length 4" knn(idx, [1.0, 2.0, 3.0, 4.0], 1)
    @test_throws ArgumentError knn_batch(idx, ones(2, 4), 1)
    # a non-finite coordinate; nothing is inserted
    @test_throws "non-finite coordinate Inf at position 2" add!(idx, [1.0, Inf, 0.0])
    @test_throws "non-finite coordinate NaN" add!(idx, [1.0 2.0; NaN 0.0; 0.0 0.0])
    @test length(idx) == 3 && HNNG._ntotal(idx) == 3 && verify(idx)
    @test_throws ArgumentError knn(idx, [NaN, 0.0, 0.0], 1)
    # k < 1
    @test_throws "k must be at least 1, got 0" knn(idx, [1.0, 2.0, 3.0], 0)
    @test_throws "got -1" knn_batch(idx, ones(3, 2), -1)
    @test_throws ArgumentError HNNG.knn_counted(idx, [1.0, 2.0, 3.0], 0)
    @test_throws ArgumentError HNNG.knn_levelsync(idx, [1.0, 2.0, 3.0], 0)
    # delete! of an unknown or already deleted id
    @test_throws "unknown id 4" delete!(idx, 4)
    @test_throws "unknown id 0" delete!(idx, 0)
    delete!(idx, 2)
    @test verify(idx)
    @test_throws "id 2 is deleted" delete!(idx, 2)
    # navigation on a dead id
    for f in (element, outlink, outlink_distance, inlinks, cluster, path_to_root)
        @test_throws "id 2 is deleted" f(idx, 2)
    end
    @test_throws "level 5 is out of range" outlink(idx, 1; level=5)
    @test_throws "stale or invalid cluster handle" members(idx, HNNG.ClusterRef(0, 99))
    # bulk_add! on a non-empty index or an ObjectStore
    @test_throws "needs an empty index" bulk_add!(idx, ones(3, 2))
    drained = HNNGIndex(Euclidean(), 1)
    add!(drained, [1.0])
    @test verify(drained)
    delete!(drained, 1)
    @test verify(drained)
    @test_throws "needs an empty index" bulk_add!(drained, ones(1, 2))
    obj = HNNGIndex(UserMetric((a, b) -> abs(a - b)), Float64)
    @test_throws "needs a VectorStore index" bulk_add!(obj, ones(1, 2))
    # load of an ObjectStore or UserMetric index without metric=
    dir = mktempdir()
    add!(obj, [1.0, 2.0])
    @test verify(obj)
    HNNG.save(joinpath(dir, "o"), obj)
    @test_throws "pass metric=" HNNG.load(joinpath(dir, "o"))
    um = HNNGIndex(UserMetric((a, b) -> sqrt(sum(abs2, a .- b)); is_metric=true), 2)
    add!(um, [0.0 1.0; 0.0 1.0])
    @test verify(um)
    HNNG.save(joinpath(dir, "u"), um)
    @test_throws "pass metric=" HNNG.load(joinpath(dir, "u"))
    @test verify(HNNG.load(joinpath(dir, "u"); metric=um.metric))
    # load of a file that is not an HNNG.jl index
    open(io -> HNNG.serialize(io, (a = 1,)), joinpath(dir, "x"), "w")
    @test_throws "$(joinpath(dir, "x")) is not an HNNG.jl index file" HNNG.load(joinpath(dir, "x"))
    # load of a file written with another index format version
    p = open(HNNG.deserialize, joinpath(dir, "u"))
    open(io -> HNNG.serialize(io, merge(p, (version = HNNG.FORMAT_VERSION + 1,))), joinpath(dir, "v"), "w")
    @test_throws "format version $(HNNG.FORMAT_VERSION + 1)" HNNG.load(joinpath(dir, "v"); metric=um.metric)
end
