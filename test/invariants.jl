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

# Two levels over 0, 1, 10, 11. Level 0: mutual pairs 1<->2 and 3<->4, representatives 1 and 3
# (one inlink each, equal sums, smallest id). Level 1: the representatives 1 and 3 as a mutual
# pair (distance 10), one cluster with representative 1 (equal inlinks and sums, smallest id).
# radius = max(0 + 1, 10 + 1) = 11; maxout = [max level-0 outdist = 1, max level-1 outdist = 10].
function two_level_index()
    idx = two_cluster_level0()
    lv = HNNG.Level(4)
    lv.present .= [true, false, true, false]
    lv.npop = 2
    lv.outlink .= [3, 0, 1, 0]
    lv.outdist .= [10.0, Inf, 10.0, Inf]
    lv.inlinks .= [[3], Int[], [1], Int[]]
    push!(lv.clusters, HNNG.Cluster(1, [1, 3], [0.0, 10.0], 11.0, [1.0, 10.0]))
    lv.cluster_of .= [1, 0, 1, 0]
    lv.nclusters = 1
    push!(idx.levels, lv)
    idx
end

@testset "verify accepts consistent indexes" begin
    @test verify(HNNGIndex(Euclidean(), 2))
    @test verify(three_point_index())
    @test HNNG._verify_level(two_cluster_level0(), 0)
    @test verify(two_level_index())
    # most inlinks wins, even over the smaller id: 3 has two inlinks, 1 has one
    lv = HNNG.Level(3)
    lv.inlinks .= [[3], Int[], [1, 2]]
    lv.outdist .= 1.0
    @test HNNG._elect(lv, [1, 2, 3]) == 3
    # equal inlinks: the smaller inlinker-distance sum wins over the smaller id (2: sum 2, 1: sum 5)
    lv = HNNG.Level(4)
    lv.inlinks .= [[3], [4], Int[], Int[]]
    lv.outdist .= [1.0, 1.0, 5.0, 2.0]
    @test HNNG._elect(lv, [1, 2, 3, 4]) == 2
    # equal inlinks and equal sums: the smaller id wins
    lv.outdist[3] = 2.0
    @test HNNG._elect(lv, [1, 2, 3, 4]) == 1
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

    # multi-level bounds and populations
    function broken2(f)
        idx = two_level_index()
        f(idx)
        idx
    end
    @test_throws "radius is 10.0, expected 11.0" verify(broken2(i -> i.levels[2].clusters[1].radius = 10.0))
    @test_throws "maxout[0] is 2.0, expected 1.0" verify(broken2(i -> i.levels[2].clusters[1].maxout = [2.0, 10.0]))
    @test_throws "maxout[1] is 9.0, expected 10.0" verify(broken2(i -> i.levels[2].clusters[1].maxout = [1.0, 9.0]))
    @test_throws "level 1: population differs from the representatives of level 0" verify(broken2(i -> i.levels[2].present[2] = true))
    @test_throws "level 1: population differs from the representatives of level 0" verify(broken2(i -> i.levels[2].present[3] = false))
    @test_throws "level 0 has one cluster but is not the top level" verify(broken(function (i)
        lv = HNNG.Level(3)
        lv.present[2] = true
        lv.npop = 1
        push!(lv.clusters, HNNG.Cluster(2, [2], [0.0], 4.0, [4.0, Inf]))
        lv.cluster_of[2] = 1
        lv.nclusters = 1
        push!(i.levels, lv)
    end))

    # classic invariants
    @test_throws "element 3 is in two clusters" verify(broken(function (i)
        lv = i.levels[1]
        push!(lv.clusters, HNNG.Cluster(3, [3], [0.0], 0.0, [4.0]))
        lv.nclusters = 2
    end))
    @test_throws "cluster_of[3] is 2, expected 1" verify(broken(i -> i.levels[1].cluster_of[3] = 2))
    @test_throws "3 is an inlink of 1 but its outlink is 2" verify(broken(i -> i.levels[1].inlinks[1] = [2, 3]))

    # free list
    @test_throws "free slot 2 is listed twice" verify(broken(function (i)
        lv = i.levels[1]
        push!(lv.clusters, HNNG._dead_cluster())
        lv.free = [2, 2]
    end))
    @test_throws "free slot 5 is out of range" verify(broken(function (i)
        lv = i.levels[1]
        push!(lv.clusters, HNNG._dead_cluster())
        lv.free = [2, 5]
    end))
    @test_throws "free slot 1 holds a live cluster" verify(broken(i -> i.levels[1].free = [1]))
end

@testset "level 0: streaming inserts keep the exact 1-NN graph" begin
    rng = StableRNG(5)
    for (T, d, n) in ((Float64, 2, 120), (Float32, 8, 100))
        idx = HNNGIndex(Euclidean(), d; T=T)
        X = gauss(rng, T, d, n)
        for j in 1:n
            @test add!(idx, view(X, :, j)) == j
            @test HNNG._verify_level(idx, 0)
        end
        @test length(idx) == n
        @test idx.levels[1].npop == n
    end
end

@testset "level 0: four points on a line" begin
    idx = HNNGIndex(Euclidean(), 1; T=Float64)
    for x in (0.0, 1.0, 10.0, 11.0)
        add!(idx, [x])
        @test HNNG._verify_level(idx, 0)
    end
    lv = idx.levels[1]
    @test lv.outlink == [2, 1, 4, 3]
    @test lv.inlinks == [[2], [1], [4], [3]]
    @test sort([c.rep for c in lv.clusters if c.rep != 0]) == [1, 3]
end

@testset "add! accepts any real input on a Float32 index" begin   # Review Focus 3
    idx = HNNGIndex(Euclidean(), 3)
    @test add!(idx, [1, 2, 3]) == 1
    @test add!(idx, [0.5, 0.25, 2.0]) == 2
    M = [1.0 4.0; 2.0 5.0; 3.5 6.0]
    @test add!(idx, M) == 3:4
    @test add!(idx, view(permutedims(M), 2, :)) == 5
    @test add!(idx, Float16[1, 1, 1]) == 6
    @test HNNG._verify_level(idx, 0)
    @test element(idx, 1) == Float32[1, 2, 3]
    @test element(idx, 5) == Float32[4, 5, 6]
    @test eltype(element(idx, 2)) == Float32
end

@testset "add! on an ObjectStore" begin
    idx = HNNGIndex(UserMetric((a, b) -> abs(a - b); is_metric=true), Float64)
    @test add!(idx, 3.0) == 1
    @test add!(idx, [1.0, 7.5, 2.0]) == 2:4
    @test HNNG._verify_level(idx, 0)
    @test element(idx, 3) == 7.5
end
