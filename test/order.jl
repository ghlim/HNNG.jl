isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))

@testset "five insertion orders give the same level-0 graph and clusters" begin
    rng = StableRNG(12)
    X = gauss(rng, Float64, 3, 200)
    ref = nothing
    for t in 1:5
        p = randperm(rng, 200)
        idx = HNNGIndex(Euclidean(), 3; T=Float64)
        stream!(idx, X[:, p])
        s0 = structure(idx, p)[1]                      # id j holds point p[j]
        g = ([(o[1], o[2]) for o in s0.outs], sort!([c[1] for c in s0.clusters]))
        if t == 1
            ref = g
        else
            @test g == ref
        end
    end
end

@testset "the hierarchy depends only on the live (id, point) set" begin
    rng = StableRNG(121)
    n = 150
    X = blobs(rng, Float64, 2, n)
    A = HNNGIndex(Euclidean(), 2; T=Float64)
    stream!(A, X)
    B = HNNGIndex(Euclidean(), 2; T=Float64)
    key = Int[]                    # key[id] = point index, 0 for noise
    noise = Int[]
    for j in 1:n
        add!(B, X[:, j]); push!(key, j); verify(B)
        if rand(rng) < 0.5
            push!(noise, add!(B, 20 .* rand(rng, 2) .- 10)); push!(key, 0); verify(B)
        end
        if !isempty(noise) && rand(rng) < 0.4
            delete!(B, popat!(noise, rand(rng, 1:length(noise)))); verify(B)
        end
    end
    for id in noise
        delete!(B, id); verify(B)
    end
    @test same_structure(structure(B, key), structure(A))
end

@testset "delete and re-insert equals a fresh build in the same id order" begin
    rng = StableRNG(122)
    n = 120
    X = gauss(rng, Float64, 4, n)
    S = sort!(randperm(rng, n)[1:40])
    C = HNNGIndex(Euclidean(), 4; T=Float64)
    stream!(C, X)
    for i in S
        delete!(C, i); verify(C)
    end
    for i in S
        add!(C, X[:, i]); verify(C)
    end
    rest = setdiff(1:n, S)
    D = HNNGIndex(Euclidean(), 4; T=Float64)
    stream!(D, X[:, vcat(rest, S)])
    @test same_structure(structure(C, vcat(collect(1:n), S)), structure(D, vcat(rest, S)))
end
