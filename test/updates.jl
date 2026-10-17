isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))

@testset "delete!: four points on a line" begin
    idx = HNNGIndex(Euclidean(), 1; T=Float64)
    stream!(idx, reshape([0.0, 1.0, 10.0, 11.0], 1, :))
    @test delete!(idx, 3) === idx
    @test idx.last_top == 1
    @test verify(idx)
    @test length(idx) == 3 && nlevels(idx) == 1
    @test idx.levels[1].outlink[4] == 2
    @test knn(idx, [10.0], 4) == ([4, 2, 1], [1.0, 9.0, 10.0])
    @test_throws ArgumentError delete!(idx, 3)
    @test_throws ArgumentError delete!(idx, 0)
    @test_throws ArgumentError delete!(idx, 5)
    @test verify(idx) && length(idx) == 3
end

@testset "delete!: drain to empty and refill" begin
    rng = StableRNG(10)
    X = gauss(rng, Float64, 2, 30)
    idx = HNNGIndex(Euclidean(), 2; T=Float64)
    stream!(idx, X)
    oldtop = -1
    for id in randperm(rng, 30)
        oldtop = nlevels(idx) - 1
        delete!(idx, id)
        @test verify(idx)
        q = randn(rng, 2)
        @test check_knn(idx, q, 3)
        @test HNNG.knn_levelsync(idx, q, 3) == knn(idx, q, 3)
    end
    # one live element means one level, so the old top is 0
    @test idx.last_top == oldtop
    @test length(idx) == 0 && nlevels(idx) == 0
    @test knn(idx, [0.0, 0.0], 1) == (Int[], Float64[])
    @test stream!(idx, X) == 31:60
    @test check_knn(idx, randn(rng, 2), 5)
    idx2 = HNNGIndex(Euclidean(), 1; T=Float64)
    stream!(idx2, reshape([0.0, 5.0], 1, :))
    @test nlevels(idx2) == 1
    delete!(idx2, 1)
    @test verify(idx2) && nlevels(idx2) == 1 && idx2.last_top == 0
    delete!(idx2, 2)
    @test verify(idx2)
    @test nlevels(idx2) == 0 && idx2.last_top == 0
end

@testset "delete!: removing the root representative again and again" begin
    rng = StableRNG(101)
    idx = HNNGIndex(Euclidean(), 3)
    stream!(idx, blobs(rng, Float32, 3, 150))
    while length(idx) > 1
        delete!(idx, top_rep(idx))
        @test verify(idx)
        q = 4 .* randn(rng, Float32, 3)
        @test check_knn(idx, q, 5)
        @test HNNG.knn_levelsync(idx, q, 5) == knn(idx, q, 5)
    end
    @test nlevels(idx) == 1
end

@testset "random interleaved inserts and deletes stay exact" begin
    rng = StableRNG(13)
    for (M, T, d) in ((Euclidean(), Float64, 2), (SqEuclidean(), Float32, 5), (CosineDist(), Float64, 3))
        idx = HNNGIndex(M, d; T=T)
        live = Int[]
        for _ in 1:300
            if isempty(live) || rand(rng) < 0.6
                push!(live, add!(idx, randn(rng, T, d)))
            else
                delete!(idx, popat!(live, rand(rng, 1:length(live))))
            end
            @test verify(idx)
            @test length(idx) == length(live)
            @test check_knn(idx, randn(rng, T, d), 4)
        end
    end
end

@testset "deleting copies of duplicated points" begin      # Review Focus 1
    rng = StableRNG(131)
    base = gauss(rng, Float32, 2, 20)
    idx = HNNGIndex(Euclidean(), 2)
    stream!(idx, hcat(base, base, base))
    for j in 1:20
        delete!(idx, j + 20 * rand(rng, 0:2))
        @test verify(idx)
    end
    for j in 1:20
        ids, ds = knn(idx, base[:, j], 2)
        @test ds == [0.0, 0.0] && all(i -> mod1(i, 20) == j, ids) && issorted(ids)
        @test check_knn(idx, base[:, j], 5)
    end
end

@testset "delete everything and re-insert gives the same structure" begin
    rng = StableRNG(132)
    X = blobs(rng, Float64, 3, 90)
    idx = HNNGIndex(Euclidean(), 3; T=Float64)
    stream!(idx, X)
    ref = structure(idx)
    for id in randperm(rng, 90)
        delete!(idx, id); verify(idx)
    end
    @test nlevels(idx) == 0
    stream!(idx, X)
    @test same_structure(structure(idx, vcat(1:90, 1:90)), ref)
end
