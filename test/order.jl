isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))
using Distances: Cityblock

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

@testset "bulk_add! equals streaming add! in the same column order" begin
    rng = StableRNG(19)
    for (M, T, d, n, gen) in ((Euclidean(), Float64, 2, 300, blobs), (SqEuclidean(), Float32, 6, 250, gauss),
                              (CosineDist(), Float32, 5, 200, gauss), (Cityblock(), Float64, 3, 150, blobs))
        X = gen(rng, T, d, n)
        S = HNNGIndex(M, d; T=T)
        add!(S, X)
        @test verify(S)
        B = HNNGIndex(M, d; T=T)
        @test bulk_add!(B, X) == 1:n
        @test verify(B)
        @test same_structure(structure(B), structure(S))
        q = randn(rng, T, d)
        @test knn(B, q, 5) == knn(S, q, 5)
        @test B.nevals > 0
        @test_throws ArgumentError bulk_add!(S, X)           # S has assigned ids
    end
    base = gauss(StableRNG(191), Float64, 3, 40)
    S = HNNGIndex(Euclidean(), 3; T=Float64)
    add!(S, hcat(base, base))                       # exact duplicates
    @test verify(S)
    B = HNNGIndex(Euclidean(), 3; T=Float64)
    bulk_add!(B, hcat(base, base))
    @test verify(B)
    @test same_structure(structure(B), structure(S))
    for p in (randperm(StableRNG(193), 80), randperm(StableRNG(194), 80))
        Xp = hcat(base, base)[:, p]
        S2 = HNNGIndex(Euclidean(), 3; T=Float64); add!(S2, Xp); @test verify(S2)
        B2 = HNNGIndex(Euclidean(), 3; T=Float64); bulk_add!(B2, Xp); @test verify(B2)
        @test same_structure(structure(B2), structure(S2))
    end
    # triplicates: every point has two partners at distance 0, so outlinks tie canonically
    base = gauss(StableRNG(195), Float64, 3, 30)
    X3 = hcat(base, base, base)
    for (k, p) in enumerate((collect(1:90), randperm(StableRNG(196), 90), randperm(StableRNG(197), 90)))
        Xp = X3[:, p]
        S3 = HNNGIndex(Euclidean(), 3; T=Float64); add!(S3, Xp); @test verify(S3)
        B3 = HNNGIndex(Euclidean(), 3; T=Float64); bulk_add!(B3, Xp); @test verify(B3)
        @test same_structure(structure(B3), structure(S3))
        if k == 1                                   # plain order: smallest other copy wins the tie
            for i in 1:90
                q = mod1(i, 30)
                @test outlink(B3, i) == minimum(c for c in (q, q + 30, q + 60) if c != i)
            end
        end
    end
    S3 = HNNGIndex(Euclidean(), 3); add!(S3, X3); @test verify(S3)
    B3 = HNNGIndex(Euclidean(), 3); bulk_add!(B3, X3); @test verify(B3)
    @test same_structure(structure(B3), structure(S3))
    @test bulk_add!(HNNGIndex(Euclidean(), 3), zeros(Float32, 3, 0)) == 1:0
    one = HNNGIndex(Euclidean(), 3)
    @test bulk_add!(one, ones(Float32, 3, 1)) == 1:1 && verify(one) && nlevels(one) == 1
    # errors (spec §9, Spec assumption 8) and Task 18's stats contract
    @test_throws ArgumentError bulk_add!(HNNGIndex(UserMetric(hausdorff; is_metric=true), Matrix{Float64}), zeros(1, 1))
    @test_throws ArgumentError bulk_add!(HNNGIndex(Euclidean(), 3), zeros(Float32, 4, 5))
    F = HNNGIndex(Euclidean(), 3)
    @test bulk_add!(F, gauss(StableRNG(195), Float32, 3, 12)) == 1:12
    @test verify(F)
    @test HNNG.last_update_stats(F) == (rewired=0, top_level_touched=0)
    for id in 1:12
        delete!(F, id)
        @test verify(F)
    end
    @test length(F) == 0
    @test_throws ArgumentError bulk_add!(F, ones(Float32, 3, 1))   # drained, not fresh
end

@testset "bulk_add! then streaming updates" begin
    rng = StableRNG(192)
    B = HNNGIndex(Euclidean(), 4)
    bulk_add!(B, gauss(rng, Float32, 4, 400))
    @test verify(B)
    for _ in 1:30
        add!(B, randn(rng, Float32, 4))
        @test verify(B)
    end
    for id in 1:15
        delete!(B, id)
        @test verify(B)
    end
    @test check_knn(B, randn(rng, Float32, 4), 10)
end

@testset "BLAS all-pairs equals the direct all-pairs kernel" begin
    rng = StableRNG(20)
    for (T, d, n) in ((Float32, 2, 700), (Float32, 20, 1100), (Float64, 7, 600), (Float32, 100, 300))
        idx = HNNGIndex(Euclidean(), d; T=T)
        X = gauss(rng, T, d, n)
        X[:, 2:2:40] .= X[:, 1:2:39]                # exact duplicates
        X .+= T(1000)                               # norms far above the distances
        for j in axes(X, 2)
            HNNG._append!(idx.store, HNNG._prepare(idx, view(X, :, j)))
            push!(idx.alive, true)
        end
        pop = collect(1:n)
        @test HNNG._allpairs_nn_blas(idx, pop) == HNNG._allpairs_nn_direct(idx, pop)
        sub = sort!(randperm(rng, n)[1:150])
        @test HNNG._allpairs_nn_blas(idx, sub) == HNNG._allpairs_nn_direct(idx, sub)
    end
    # 8-bit pixels at d = 784, with one-ulp near duplicates and exact duplicates
    d, n = 784, 400
    X = Float32.(rand(rng, 0:255, d, n))
    X[1, 1] = 0f0                                   # one near duplicate is a subnormal step
    near = Matrix{Float32}(undef, d, 20)
    for c in 1:20
        src = c == 1 ? 1 : rand(rng, 1:n)
        near[:, c] .= X[:, src]
        k = c == 1 ? 1 : rand(rng, 1:d)
        near[k, c] = nextfloat(near[k, c])
    end
    X = hcat(X, near, X[:, rand(rng, 1:n, 10)])
    idx = HNNGIndex(Euclidean(), d; T=Float32)
    for j in axes(X, 2)
        HNNG._append!(idx.store, HNNG._prepare(idx, view(X, :, j)))
        push!(idx.alive, true)
    end
    pop = collect(1:size(X, 2))
    @test HNNG._allpairs_nn_blas(idx, pop) == HNNG._allpairs_nn_direct(idx, pop)
    sub = sort!(randperm(rng, size(X, 2))[1:150])
    @test HNNG._allpairs_nn_blas(idx, sub) == HNNG._allpairs_nn_direct(idx, sub)
    # squares that underflow (kernel distances 0) and sums that overflow
    for (T, d, n, s) in ((Float32, 8, 60, 1f-25), (Float32, 4, 50, 1f19), (Float64, 4, 50, 1e154))
        idx = HNNGIndex(Euclidean(), d; T=T)
        X = gauss(rng, T, d, n) .* s
        for j in axes(X, 2)
            HNNG._append!(idx.store, HNNG._prepare(idx, view(X, :, j)))
            push!(idx.alive, true)
        end
        pop = collect(1:n)
        @test HNNG._allpairs_nn_blas(idx, pop) == HNNG._allpairs_nn_direct(idx, pop)
    end
end
