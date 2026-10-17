isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))

# Points 0, 1, 10, 11: level 0 = {1,2} (rep 1) and {3,4} (rep 3); level 1 = {1,3} (rep 1, radius 11).
function line4()
    idx = HNNGIndex(Euclidean(), 1; T=Float64)
    stream!(idx, reshape([0.0, 1.0, 10.0, 11.0], 1, :))
    idx
end

@testset "navigation on four points on a line" begin
    idx = line4()
    n0 = idx.nevals
    @test outlink(idx, 3) == 4 && outlink_distance(idx, 3) == 1.0
    @test inlinks(idx, 2) == [1]
    c1 = cluster(idx, 2)
    @test members(idx, c1) == [1, 2] && representative(idx, c1) == 1 && HNNG.radius(idx, c1) == 1.0
    r = root(idx)
    @test r.level == 1 && members(idx, r) == [1, 3]
    @test HNNG.radius(idx, r) == 11.0
    @test parent(idx, cluster(idx, 4)) == r && parent(idx, r) === nothing
    @test children(idx, r) == [cluster(idx, 1), cluster(idx, 3)]
    @test children(idx, c1) == HNNG.ClusterRef[]
    @test path_to_root(idx, 4) == [cluster(idx, 4), r]
    @test outlink(idx, 1; level=1) == 3 && outlink_distance(idx, 1; level=1) == 10.0
    @test outlink(idx, 2; level=1) == 0 && cluster(idx, 2; level=1) === nothing
    @test inlinks(idx, 2; level=1) == Int[] && outlink_distance(idx, 2; level=1) == Inf
    @test_throws ArgumentError outlink(idx, 1; level=2)
    @test_throws ArgumentError members(idx, HNNG.ClusterRef(0, 99))
    @test idx.nevals == n0                                  # no distance evaluated
    @test root(HNNGIndex(Euclidean(), 2)) === nothing
end

@testset "navigation reports the metric's units" begin
    s = HNNGIndex(SqEuclidean(), 1; T=Float64)
    stream!(s, reshape([0.0, 1.0, 10.0, 11.0], 1, :))
    @test outlink_distance(s, 1; level=1) == 100.0
    @test HNNG.radius(s, root(s)) == 121.0
end

@testset "outlink chains end at a mutual pair; paths end at the root" begin
    rng = StableRNG(14)
    idx = HNNGIndex(Euclidean(), 2; T=Float64)
    stream!(idx, blobs(rng, Float64, 2, 200))
    for s in 1:200
        x, seen = s, Set{Int}()
        while !(x in seen)
            push!(seen, x)
            x = outlink(idx, x)
        end
        @test outlink(idx, outlink(idx, x)) == x
        p = path_to_root(idx, s)
        @test p[end] == root(idx) && length(p) == nlevels(idx)
        @test all(i -> representative(idx, p[i]) in members(idx, p[i+1]), 1:length(p)-1)
    end
end

@testset "leaves and newick" begin
    idx = line4()
    @test leaves(idx, root(idx)) == [1, 2, 3, 4]
    @test leaves(idx, cluster(idx, 4)) == [3, 4]
    @test newick(idx) == "((1:0.0,2:1.0):1.0,(3:0.0,4:1.0):1.0):11.0;"
    @test newick(idx; labels=["a", "b", "c d", "e'f"]) ==
          "((a:0.0,b:1.0):1.0,('c d':0.0,'e''f':1.0):1.0):11.0;"
    @test newick(HNNGIndex(Euclidean(), 1)) == ";"
    rng = StableRNG(15)
    big = HNNGIndex(Euclidean(), 3)
    stream!(big, blobs(rng, Float32, 3, 120))
    @test leaves(big, root(big)) == 1:120
    @test count(==(','), newick(big)) == 119
    n0 = big.nevals
    newick(big)
    leaves(big, root(big))
    @test big.nevals == n0
end
