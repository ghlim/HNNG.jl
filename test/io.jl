isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))
using Distances: Cityblock, Minkowski

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
    good = HNNG.save(joinpath(dir, "good.hnng"), idx)
    header, payload = open(io -> (HNNG.deserialize(io), HNNG.deserialize(io)), good)
    @test header == (HNNG.FORMAT, HNNG.FORMAT_VERSION)
    newer = joinpath(dir, "newer.hnng")
    open(newer, "w") do io
        HNNG.serialize(io, (HNNG.FORMAT, HNNG.FORMAT_VERSION + 1))
        HNNG.serialize(io, payload)
    end
    @test_throws ArgumentError HNNG.load(newer)
    @test_throws "format version" HNNG.load(newer)
end

@testset "load: empty, truncated and garbage files are ArgumentErrors (I-4)" begin
    rng = StableRNG(140)
    dir = mktempdir()
    idx = HNNGIndex(Euclidean(), 3)
    add!(idx, gauss(rng, Float32, 3, 50))
    @test verify(idx)
    good = HNNG.save(joinpath(dir, "good.hnng"), idx)
    bytes = read(good)
    files = (empty = UInt8[], truncated = bytes[1:end÷2], garbage = rand(rng, UInt8, 4096),
             text = Vector{UInt8}("hello world, not an index\n"))
    for (name, b) in pairs(files)
        p = joinpath(dir, String(name))
        write(p, b)
        @test_throws ArgumentError HNNG.load(p)
        @test_throws "$p is not an HNNG.jl index file" HNNG.load(p)
    end
    @test_throws "or is damaged: EOFError" HNNG.load(joinpath(dir, "empty"))
    @test_throws "or is damaged" HNNG.load(joinpath(dir, "truncated"))
    # a header with another version is rejected before its payload is read
    newer = joinpath(dir, "newer")
    open(newer, "w") do io
        HNNG.serialize(io, (HNNG.FORMAT, HNNG.FORMAT_VERSION + 1))
        write(io, "a payload in a layout this version cannot read")
    end
    @test_throws "has HNNG.jl index format version $(HNNG.FORMAT_VERSION + 1)" HNNG.load(newer)
    @test verify(HNNG.load(good))
end

@testset "load: metric= must have the saved metric's type (M-4)" begin
    rng = StableRNG(141)
    dir = mktempdir()
    idx = HNNGIndex(Euclidean(), 4; T=Float64)
    add!(idx, gauss(rng, Float64, 4, 100))
    @test verify(idx)
    p = HNNG.save(joinpath(dir, "e.hnng"), idx)
    for m in (SqEuclidean(), CosineDist(), Cityblock(), UserMetric((a, b) -> 0.0))
        @test_throws ArgumentError HNNG.load(p; metric=m)
        @test_throws "was saved with Euclidean" HNNG.load(p; metric=m)
    end
    @test verify(HNNG.load(p; metric=Euclidean()))
    mk = HNNGIndex(Minkowski(3.0), 2; T=Float64)            # same type, other parameter: allowed
    add!(mk, gauss(rng, Float64, 2, 20))
    @test verify(mk)
    q = HNNG.save(joinpath(dir, "m.hnng"), mk)
    @test HNNG.load(q; metric=Minkowski(4.0)).metric == Minkowski(4.0)
end

@testset "save writes only the live columns (M-7)" begin
    rng = StableRNG(142)
    dir = mktempdir()
    idx = HNNGIndex(Euclidean(), 64; T=Float64)
    add!(idx, gauss(rng, Float64, 64, 17))
    @test verify(idx)
    @test size(idx.store.data, 2) == 32                    # capacity doubled past 16
    p1 = HNNG.save(joinpath(dir, "a.hnng"), idx)
    p2 = HNNG.save(joinpath(dir, "b.hnng"), idx)
    @test filesize(p1) < 32 * 64 * 8                       # the 15 spare columns are not written
    @test read(p1) == read(p2)                             # no uninitialised bytes: reproducible
    idx2 = HNNG.load(p1)
    @test size(idx2.store.data, 2) == 17 && length(idx2) == 17
    @test verify(idx2)
    @test same_structure(structure(idx2), structure(idx))
    @test add!(idx2, randn(rng, 64)) == 18                 # the store still grows
    @test verify(idx2)
end
