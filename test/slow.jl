isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))
using HDF5

const SLOW_DATA = "/home/coral/working/benchmarks/data"

@testset "slow: random-xs-20-euclidean, all 9000 train points" begin
    path = joinpath(SLOW_DATA, "random-xs-20-euclidean.hdf5")
    if isfile(path)
        train, test, gt = h5open(path) do f
            read(f["train"]), read(f["test"]), read(f["distances"])
        end
        idx = HNNGIndex(Euclidean(), 20; T=Float64)
        add!(idx, train)
        @test verify(idx)
        for j in 1:200
            _, ds = knn(idx, test[:, j], 10)
            @test all(isapprox.(ds, gt[1:10, j]; rtol=1e-9))
        end
        B = HNNGIndex(Euclidean(), 20; T=Float64)
        bulk_add!(B, train)
        @test verify(B)
        @test same_structure(structure(B), structure(idx))
    else
        @test_skip isfile(path)
    end
end

@testset "slow: fashion-mnist-784, 5000-point Float32 subset" begin
    path = joinpath(SLOW_DATA, "fashion-mnist-784-euclidean.hdf5")
    if isfile(path)
        train, test = h5open(path) do f
            f["train"][:, 1:5000], f["test"][:, 1:50]
        end
        idx = HNNGIndex(Euclidean(), 784)
        add!(idx, train)
        @test verify(idx)
        for j in 1:50
            @test check_knn(idx, test[:, j], 10)
        end
        B = HNNGIndex(Euclidean(), 784)
        bulk_add!(B, train)
        @test verify(B)
        @test same_structure(structure(B), structure(idx))
    else
        @test_skip isfile(path)
    end
end

@testset "slow: concurrent knn and add! from 4 threads (subprocess, julia -t 4)" begin
    script = """
    include($(repr(joinpath(@__DIR__, "oracle.jl"))))
    rng = StableRNG(2026)
    idx = HNNGIndex(Euclidean(), 8)
    add!(idx, randn(rng, Float32, 8, 3000))
    verify(idx)
    nq = 4000
    Q = randn(rng, Float32, 8, nq)
    ref = [brute_knn(idx, Q[:, j], 10) for j in 1:nq]
    out = Vector{Any}(undef, nq)
    Threads.@threads for j in 1:nq
        out[j] = try knn(idx, Q[:, j], 10) catch e; e end
    end
    nwrong = count(j -> out[j] != ref[j], 1:nq)
    bi, bd = knn_batch(idx, Q, 10; threads=true)
    nbatch = count(j -> (bi[:, j], bd[:, j]) != ref[j], 1:nq)
    println("KNN ", Threads.nthreads(), " ", nwrong, " ", nbatch)
    flush(stdout)
    # concurrent add! is serialized: every column inserted once, the index stays valid
    X = randn(rng, Float32, 8, 300)
    par = HNNGIndex(Euclidean(), 8)
    ids = zeros(Int, 300)
    Threads.@threads for j in 1:300
        ids[j] = add!(par, X[:, j])
    end
    ok = sort(ids) == 1:300 && all(j -> element(par, ids[j]) == X[:, j], 1:300) && verify(par)
    println("ADD ", ok)
    """
    cmd = `$(Base.julia_cmd()) -t 4 --project=$(Base.active_project()) --startup-file=no -e $script`
    out = read(pipeline(ignorestatus(cmd); stderr=devnull), String)
    m = match(r"KNN (\d+) (\d+) (\d+)", out)
    @test m !== nothing
    if m !== nothing
        @test m[1] == "4"
        @test m[2] == "0"                 # Threads.@threads knn: every result equals brute force
        @test m[3] == "0"                 # knn_batch(threads=true)
    end
    @test occursin("ADD true", out)       # Threads.@threads add!
end
