include(joinpath(@__DIR__, "setup.jl"))

@testset "bench" begin
    include("test_datasets.jl")

    @testset "recall rules (hand-made tie case)" begin
        # truth: ids 10, 20, 30, 40 at distances 1, 2, 2, 3; ids 20 and 30 tie for 2nd place
        tid = [10, 20, 30, 40]
        tdist = [1.0, 2.0, 2.0, 3.0]
        k = 2
        # a correct answer that took the other member of the tie
        @test recall_ann([1.0, 2.0], tdist, k) == 1.0
        @test recall_id([10, 30], tid, k) == 0.5
        # the additive 1e-3 rule: inside counts, outside does not
        @test recall_ann([1.0, 2.0009], tdist, k) == 1.0
        @test recall_ann([1.0, 2.0011], tdist, k) == 0.5
        # fewer than k results
        @test recall_ann([1.0], tdist, k) == 0.5
        @test recall_id([10], tid, k) == 0.5
        # duplicates count once; results beyond k are ignored
        @test recall_id([10, 10], tid, k) == 0.5
        @test recall_ann([1.0, 9.0, 1.5], tdist, k) == 0.5
        @test recall_id([10, 40, 20], tid, k) == 0.5
        @test_throws ArgumentError recall_ann([1.0], tdist, 0)
        @test_throws ArgumentError recall_id([1], tid, 0)
        # mean over queries, distances recomputed in Float64 from the returned ids
        X = Float32[0 1 2 3]
        Q = reshape(Float32[0.4, 2.6], 1, 2)
        nb, gd = ground_truth(X, Q, 2)                       # [1, 2] and [4, 3]
        res = [([1, 2], [0.0, 0.0]), ([4, 2], [0.0, 0.0])]   # reported distances are ignored
        ra, ri = mean_recalls(X, Q, nb, gd, res, 2)
        @test ra == 0.75 && ri == 0.75
        # a duplicated returned id is credited once by both rules, even with the right distance
        resdup = [([1, 1], [0.0, 0.0]), ([4, 3], [0.0, 0.0])]
        ra2, ri2 = mean_recalls(X, Q, nb, gd, resdup, 2)
        @test ra2 == 0.75 && ri2 == 0.75                     # query 1: 0.5, query 2: 1.0
    end

    @testset "distance_match: the multiset check the additive rule cannot make" begin
        tdist = [1.0, 2.0, 2.0, 3.0]
        @test distance_match([1.0, 2.0], tdist, 2) == 1.0
        @test distance_match([2.0, 1.0], tdist, 2) == 1.0            # order of the found distances does not matter
        @test distance_match([1.0, 2.0009], tdist, 2) == 0.5         # recall_ann calls this 1.0
        @test distance_match([1.0], tdist, 2) == 0.5                 # fewer than k results
        @test distance_match([1.0, 1.0], tdist, 2) == 0.5            # a repeated distance is not a second neighbour
        @test distance_match([1.0, 2.0, 9.0], tdist, 2) == 1.0       # results beyond k are ignored
        @test_throws ArgumentError distance_match([1.0], tdist, 0)
        X = Float32[0 1 2 3]
        Q = reshape(Float32[0.4, 2.6], 1, 2)
        nb, gd = ground_truth(X, Q, 2)
        @test min_match(X, Q, gd, [([1, 2], [0.0, 0.0]), ([4, 3], [0.0, 0.0])], 2) == 1.0
        @test min_match(X, Q, gd, [([1, 2], [0.0, 0.0]), ([4, 2], [0.0, 0.0])], 2) == 0.5
    end
end
