isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))

@testset "knn equals the brute-force oracle (Euclidean family)" begin
    rng = StableRNG(8)
    for M in (Euclidean(), SqEuclidean(), CosineDist()), T in (Float32, Float64),
        d in (2, 8, 32), n in (1, 2, 50, 2000)
        idx = HNNGIndex(M, d; T=T)
        add!(idx, gauss(rng, T, d, n))
        @test verify(idx)
        for _ in 1:5, k in (1, 10, n + 5)
            @test check_knn(idx, randn(rng, T, d), k)
        end
    end
end

@testset "knn basics" begin
    idx = HNNGIndex(Euclidean(), 2; T=Float64)
    @test knn(idx, [0.0, 0.0], 3) == (Int[], Float64[])
    add!(idx, [1.0 2.0 4.0; 0.0 0.0 0.0])
    @test verify(idx)
    @test knn(idx, [0.0, 0.0], 2) == ([1, 2], [1.0, 2.0])
    @test knn(idx, [0, 0], 10) == ([1, 2, 3], [1.0, 2.0, 4.0])          # Int query, k > n
    s = HNNGIndex(SqEuclidean(), 2; T=Float64)
    add!(s, [1.0 2.0; 0.0 0.0])
    @test verify(s)
    @test knn(s, [0.0, 0.0], 2)[2] ≈ [1.0, 4.0]
    f = HNNGIndex(Euclidean(), 3)                    # Review Focus 3: default Float32 index
    add!(f, [1.0 4.0 0.5; 2.0 5.0 0.25; 3.0 6.0 2.0])
    @test verify(f)
    @test knn(f, [1, 2, 3], 1) == ([1], [0.0]) && check_knn(f, [4, 5, 6], 3)   # Int queries
end

@testset "nevals counts the serial knn path" begin
    rng = StableRNG(81)
    idx = HNNGIndex(Euclidean(), 4)
    add!(idx, gauss(rng, Float32, 4, 300))
    @test verify(idx)
    before = idx.nevals
    @test before > 0
    ids, ds, n1 = HNNG.knn_counted(idx, randn(rng, Float32, 4), 5)
    @test 1 <= n1 <= 300                    # memoised: every element at most once
    @test idx.nevals == before + n1
    @test length(ids) == 5 && issorted(ds)
    knn(idx, randn(rng, Float32, 4), 5)
    @test idx.nevals > before + n1
end

@testset "knn prunes on clustered low-dimensional data" begin
    rng = StableRNG(83)
    idx = HNNGIndex(Euclidean(), 2; T=Float64)
    X = blobs(rng, Float64, 2, 2020; nc=20)          # queries from the same blob centres
    add!(idx, X[:, 1:2000])
    @test verify(idx)
    ne = [HNNG.knn_counted(idx, X[:, j], 10)[3] for j in 2001:2020]
    # a full scan is 2000; pruning on 20 blobs in 2-D is far below 400
    @test maximum(ne) < 400
end

@testset "exact duplicates: ties go to the smallest id" begin    # Review Focus 1
    rng = StableRNG(82)
    base = gauss(rng, Float64, 3, 25)
    idx = HNNGIndex(Euclidean(), 3; T=Float64)
    stream!(idx, hcat(base, base, base))
    for j in 1:25
        ids, ds = knn(idx, base[:, j], 3)
        @test ids == [j, j + 25, j + 50] && ds == [0.0, 0.0, 0.0]
        @test check_knn(idx, base[:, j] .+ 0.01, 7)
    end
end

@testset "integer lattice: massive distance ties" begin           # Review Focus 2
    pts = reduce(hcat, [Float32[i, j] for i in 0:7 for j in 0:7])
    idx = HNNGIndex(Euclidean(), 2)
    stream!(idx, pts)
    for q in ([0.5, 0.5], [3, 3], [7.5, -1], [3.5, 3.5]), k in (1, 4, 9, 64)
        @test check_knn(idx, q, k)
    end
end

@testset "cosine: zero vectors are stored and queried" begin       # Review Focus 4
    idx = HNNGIndex(CosineDist(), 3; T=Float64)
    stream!(idx, [0.0 1.0 0.0 2.0; 0.0 0.0 1.0 2.0; 0.0 0.0 0.0 0.0])
    ids, ds = knn(idx, [0.0, 0.0, 0.0], 4)
    @test ids[1] == 1 && sort(ids) == [1, 2, 3, 4]
    @test ds[1] == 0.0 && all(isapprox.(ds[2:end], 0.5; atol=1e-12))
    @test check_knn(idx, [0.0, 0.0, 0.0], 4)
    @test check_knn(idx, [1.0, 1.0, 0.0], 4)
end
