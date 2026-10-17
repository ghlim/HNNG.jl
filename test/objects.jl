isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))

@testset "ObjectStore: Hausdorff on variable-size descriptor sets (pruned, exact)" begin
    rng = StableRNG(11)
    S = descriptor_sets(rng, 3, 80)
    idx = HNNGIndex(UserMetric(hausdorff; is_metric=true), Matrix{Float64})
    @test is_metric(idx)
    for s in S
        add!(idx, s)
        @test verify(idx)
    end
    for q in descriptor_sets(rng, 3, 10), k in (1, 5, 100)
        @test check_knn(idx, q, k)
        @test HNNG.knn_levelsync(idx, q, k) == knn(idx, q, k)
    end
    delete!(idx, 7)
    @test verify(idx)
    delete!(idx, top_rep(idx))
    @test verify(idx)
    @test check_knn(idx, S[7], 3)
end

@testset "ObjectStore: 2015 ROS view distance (not a metric: unpruned, exact)" begin
    rng = StableRNG(111)
    S = descriptor_sets(rng, 4, 70)
    idx = HNNGIndex(ros_view_distance, Matrix{Float64})   # bare function -> UserMetric(is_metric=false)
    @test !is_metric(idx)
    for s in S
        add!(idx, s)
        @test verify(idx)
    end
    for q in descriptor_sets(rng, 4, 10), k in (1, 5)
        @test check_knn(idx, q, k)
        _, _, ne = HNNG.knn_counted(idx, q, k)
        @test ne == length(idx)                        # unpruned: every live element once
    end
end

@testset "ObjectStore: integer-valued Hamming distance on strings" begin   # Review Focus 5
    rng = StableRNG(112)
    hamming(a, b) = count(i -> a[i] != b[i], eachindex(a))
    alphabet = ['a', 'c', 'g', 't']
    words = [String(rand(rng, alphabet, 6)) for _ in 1:90]
    idx = HNNGIndex(UserMetric(hamming; is_metric=true), String)
    for w in words
        add!(idx, w)
        @test verify(idx)
    end
    ids, ds = knn(idx, words[5], 4)
    @test ds isa Vector{Float64} && ds[1] == 0.0 && ids[1] == findfirst(==(words[5]), words)
    for _ in 1:15
        @test check_knn(idx, String(rand(rng, alphabet, 6)), 6)
    end
end
