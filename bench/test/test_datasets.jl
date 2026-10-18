include(joinpath(@__DIR__, "setup.jl"))

using DataFrames
struct AtomicBoom end
Base.print(io::IO, ::AtomicBoom) = error("boom: the CSV write failed half-way")

@testset "datasets" begin
    @testset "results directories and CSV schemas" begin
        @test results_dir(true) != results_dir(false)
        @test endswith(results_dir(true), "results-smoke")
        @test endswith(results_dir(false), "results")
        @test withenv(() -> results_dir(true), "BENCH_RESULTS" => "/tmp/elsewhere") == "/tmp/elsewhere"
        @test SCHEMA_EXP1 == ["dataset", "n", "d", "nq", "k", "method", "param", "build_s", "qps",
                              "recall_ann", "recall_id", "frac_evaluated", "nlevels", "nclusters"]
        @test SCHEMA_EXP2 == ["geometry", "n", "d", "nq", "k", "method", "build_s", "qps",
                              "recall_ann", "frac_evaluated", "nlevels"]
        @test SCHEMA_EXP3 == ["config", "n", "d", "run", "method", "op", "seq", "latency_s",
                              "nevals", "rewired", "top_level", "note"]
    end

    @testset "write_csv_atomic: .tmp then mv, the old file survives a failed write" begin
        mktempdir() do dir
            p = joinpath(dir, "exp9.csv")
            @test write_csv_atomic(p, DataFrame(a = [1, 2])) == p
            @test read(p, String) == "a\n1\n2\n" && !isfile(p * ".tmp")
            write_csv_atomic(p, DataFrame(a = [3]))                       # replaces an existing file
            @test read(p, String) == "a\n3\n"
            @test_throws ErrorException write_csv_atomic(p, DataFrame(a = [AtomicBoom()]))
            @test read(p, String) == "a\n3\n" && !isfile(p * ".tmp")    # untouched, no stray .tmp
            @test readdir(dir) == ["exp9.csv"]
        end
    end

    @testset "gaussian: seeded, d×n Float32, prefix-stable" begin
        A = gaussian(50, 4, SEED_GAUSS_TRAIN + 4)
        @test size(A) == (4, 50) && eltype(A) == Float32
        @test A == gaussian(50, 4, SEED_GAUSS_TRAIN + 4)
        @test A != gaussian(50, 4, SEED_GAUSS_QUERY + 4)
        @test gaussian(80, 4, SEED_GAUSS_TRAIN + 4)[:, 1:50] == A   # Exp.3 relies on this
    end

    @testset "blobs: 100 centres in [-10,10]^d, unit spread, round-robin" begin
        C = blob_centres(3, SEED_BLOBS_CENTRES + 3)
        @test size(C) == (3, 100) && all(-10 .<= C .<= 10)
        X = blobs(1000, C, SEED_BLOBS_TRAIN + 3)
        @test size(X) == (3, 1000) && eltype(X) == Float32
        r = [dist64(view(X, :, i), view(C, :, (i - 1) % 100 + 1)) for i in 1:1000]
        @test 1.2 < sum(r) / 1000 < 2.0          # mean of a chi_3 variable is 1.596
        Xs, Qs = synthetic("blobs", 200, 10, 3)
        @test size(Xs) == (3, 200) && size(Qs) == (3, 10)
        Xg, Qg = synthetic("gaussian", 200, 10, 3)
        @test Xg == gaussian(200, 3, SEED_GAUSS_TRAIN + 3)
        @test_throws ArgumentError synthetic("uniform", 10, 2, 3)
    end

    @testset "ground truth: Float64, ties by smaller id" begin
        X = Float32[0 2 5 1]                     # 1×4: points at 0, 2, 5, 1
        q = reshape(Float32[1], 1, 1)            # distances 1, 1, 4, 0
        nb, gd = ground_truth(X, q, 3)
        @test nb[:, 1] == [4, 1, 2]
        @test gd[:, 1] == [0.0, 1.0, 1.0]
        nb2, _ = ground_truth(X, q, 10)          # k larger than n
        @test size(nb2) == (4, 1)
    end

    @testset "smoke_subset recomputes ground truth" begin
        ds = Dataset("toy", gaussian(40, 3, 1), gaussian(5, 3, 2), zeros(Int, 1, 5), zeros(1, 5))
        s = smoke_subset(ds, 30, 4)
        @test size(s.train) == (3, 30) && size(s.test) == (3, 4)
        nb, gd = ground_truth(s.train, s.test, 10)
        @test s.neighbors == nb && s.distances == gd
    end

    @testset "HDF5 loader: ann-benchmarks layout, 1-based ids" begin
        p = joinpath(data_dir(), "random-xs-20-euclidean.hdf5")
        if isfile(p)
            ds = load_hdf5(p)
            @test ds.name == "random-xs-20-euclidean"
            @test size(ds.train) == (20, 9000) && eltype(ds.train) == Float32
            @test size(ds.test) == (20, 1000)
            @test size(ds.neighbors) == (100, 1000) && size(ds.distances) == (100, 1000)
            @test ds.neighbors[1, 1] == 3619                 # h5py shows 3618, 0-based
            @test ds.distances[1, 1] ≈ 4.43531921 atol = 1e-6
            @test dist64(view(ds.train, :, 3619), view(ds.test, :, 1)) ≈ 4.43531921 rtol = 1e-5
        else
            @warn "random-xs HDF5 not found at $p; loader test skipped"
            @test_skip isfile(p)
        end
    end
end
