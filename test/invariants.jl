isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))

# Hand-built level 0 over the points 0, 1, 5 on a line: 1 <-> 2 is the mutual pair,
# 3 -> 2; one cluster with representative 2 (two inlinks).
function three_point_index()
    idx = HNNGIndex(Euclidean(), 1; T=Float64)
    for x in (0.0, 1.0, 5.0)
        HNNG._append!(idx.store, [x])
        push!(idx.alive, true)
    end
    idx.nlive = 3
    lv = HNNG.Level(3)
    lv.present .= true
    lv.npop = 3
    lv.outlink .= [2, 1, 2]
    lv.outdist .= [1.0, 1.0, 4.0]
    lv.inlinks .= [[2], [1, 3], Int[]]
    push!(lv.clusters, HNNG.Cluster(2, [1, 2, 3], [1.0, 0.0, 4.0], 4.0, [4.0]))
    lv.cluster_of .= 1
    lv.nclusters = 1
    push!(idx.levels, lv)
    idx
end

# Hand-built level 0 over 0, 1, 10, 11: two clusters, so level 0 cannot be the top.
function two_cluster_level0()
    idx = HNNGIndex(Euclidean(), 1; T=Float64)
    for x in (0.0, 1.0, 10.0, 11.0)
        HNNG._append!(idx.store, [x])
        push!(idx.alive, true)
    end
    idx.nlive = 4
    lv = HNNG.Level(4)
    lv.present .= true
    lv.npop = 4
    lv.outlink .= [2, 1, 4, 3]
    lv.outdist .= 1.0
    lv.inlinks .= [[2], [1], [4], [3]]
    push!(lv.clusters, HNNG.Cluster(1, [1, 2], [0.0, 1.0], 1.0, [1.0]))
    push!(lv.clusters, HNNG.Cluster(3, [3, 4], [0.0, 1.0], 1.0, [1.0]))
    lv.cluster_of .= [1, 1, 2, 2]
    lv.nclusters = 2
    push!(idx.levels, lv)
    idx
end

@testset "verify accepts consistent indexes" begin
    @test verify(HNNGIndex(Euclidean(), 2))
    @test verify(three_point_index())
    @test HNNG._verify_level(two_cluster_level0(), 0)
end

@testset "verify rejects each broken invariant" begin
    function broken(f)
        idx = three_point_index()
        f(idx)
        idx
    end
    @test_throws "population differs from the live set" verify(broken(i -> i.alive[3] = false))
    @test_throws "outlink of 1 is 3 but its nearest neighbour is 2" verify(broken(function (i)
        lv = i.levels[1]
        lv.outlink[1] = 3; lv.inlinks[3] = [1]; lv.inlinks[2] = [3]
    end))
    @test_throws "outdist of 1" verify(broken(i -> i.levels[1].outdist[1] = 0.5))
    @test_throws "inlinks of 1 are not sorted and unique" verify(broken(i -> i.levels[1].inlinks[1] = [2, 2]))
    @test_throws "1 is missing from the inlinks of 2" verify(broken(i -> i.levels[1].inlinks[2] = [3]))
    @test_throws "is not the connected component" verify(broken(function (i)
        lv = i.levels[1]
        lv.clusters[1] = HNNG.Cluster(2, [1, 2], [1.0, 0.0], 1.0, [1.0])
        push!(lv.clusters, HNNG.Cluster(3, [3], [0.0], 0.0, [4.0]))
        lv.cluster_of[3] = 2
        lv.nclusters = 2
    end))
    @test_throws "has representative 1, the election gives 2" verify(broken(i -> i.levels[1].clusters[1].rep = 1))
    @test_throws "repdist of 3" verify(broken(i -> i.levels[1].clusters[1].repdist[3] = 4.5))
    @test_throws "radius is 3.0, expected 4.0" verify(broken(i -> i.levels[1].clusters[1].radius = 3.0))
    @test_throws "maxout[0]" verify(broken(i -> i.levels[1].clusters[1].maxout = [1.0]))
    @test_throws "live counter is 2 but level 0 holds 3 present elements" verify(broken(i -> i.nlive = 2))
    @test_throws "top level 0 has 2 clusters" verify(two_cluster_level0())
    nolevels = HNNGIndex(Euclidean(), 1)
    HNNG._append!(nolevels.store, Float32[1])
    push!(nolevels.alive, true)
    @test_throws "live elements but no levels" verify(nolevels)
end
