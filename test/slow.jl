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
