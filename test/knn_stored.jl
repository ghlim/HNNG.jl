isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))

"""Exhaustive k-NN of the stored element `id` over the live elements other than `id`: (distance, id) order, user units."""
function brute_stored(idx, id, k)
    p = HNNG._point(idx.store, id)
    cand = Tuple{Float64,Int}[]
    for j in 1:HNNG._ntotal(idx)
        (idx.alive[j] && j != id) || continue
        push!(cand, (HNNG._rawdist(idx, p, HNNG._point(idx.store, j)), j))
    end
    sort!(cand)
    top = cand[1:min(k, length(cand))]
    [c[2] for c in top], [HNNG._to_user(idx, c[1]) for c in top]
end

check_stored(idx, id, k) = knn_stored(idx, id, k) == brute_stored(idx, id, k)

"""`n` columns with every fifth point duplicated 2 or 3 times (exact duplicates, distance 0)."""
function with_duplicates(rng, T, d, n)
    base = gauss(rng, T, d, n - n ÷ 4)
    cols = [base[:, j] for j in axes(base, 2)]
    for j in 1:5:size(base, 2), _ in 1:rand(rng, 1:2)
        length(cols) < n && push!(cols, copy(base[:, j]))
    end
    X = reduce(hcat, cols)
    X[:, randperm(rng, size(X, 2))]
end

@testset "knn_stored equals the brute-force oracle (duplicates, many k)" begin
    rng = StableRNG(701)
    for d in (2, 8), (M, T) in ((Euclidean(), Float32), (Euclidean(), Float64))
        X = with_duplicates(rng, T, d, 400)
        n = size(X, 2)
        idx = HNNGIndex(M, d; T=T)
        add!(idx, X)
        @test verify(idx)
        dup = 0
        for id in vcat(1:5:n, [1, n]), k in (1, 2, 5, n - 1, n + 3)
            @test check_stored(idx, id, k)
        end
        for id in 1:n
            ids, ds = knn_stored(idx, id, 4)
            dup += count(iszero, ds) > 1
            @test id ∉ ids
        end
        @test dup > 0                      # some ids really have several elements at distance 0
    end
end

@testset "knn_stored: all-identical points are ordinary neighbours" begin
    idx = HNNGIndex(Euclidean(), 3; T=Float64)
    add!(idx, repeat([1.0, 2.0, 3.0], 1, 9))
    @test verify(idx)
    for id in 1:9, k in (1, 3, 8, 20)
        @test check_stored(idx, id, k)
    end
    @test knn_stored(idx, 5, 3) == ([1, 2, 3], [0.0, 0.0, 0.0])
    @test knn_stored(idx, 1, 3) == ([2, 3, 4], [0.0, 0.0, 0.0])
end

@testset "knn_stored after delete!" begin
    rng = StableRNG(702)
    X = with_duplicates(rng, Float64, 4, 300)
    n = size(X, 2)
    idx = HNNGIndex(Euclidean(), 4; T=Float64)
    add!(idx, X)
    gone = collect(1:7:n)
    for g in gone
        delete!(idx, g)
    end
    @test verify(idx)
    live = [i for i in 1:n if i ∉ gone]
    for id in live[1:5:end], k in (1, 2, 5, n - 1)
        @test check_stored(idx, id, k)
        @test isdisjoint(knn_stored(idx, id, k)[1], gone)
    end
    @test_throws ArgumentError knn_stored(idx, gone[1], 3)
    @test_throws ArgumentError knn_stored(idx, n + 1, 3)
    @test_throws ArgumentError knn_stored(idx, 0, 3)
    @test_throws ArgumentError knn_stored(idx, live[1], 0)
    @test_throws ArgumentError knn_stored(idx, live[1], -2)
    # delete down to one element: nothing left to return
    for i in live[2:end]
        delete!(idx, i)
    end
    @test verify(idx)
    @test knn_stored(idx, live[1], 1) == (Int[], Float64[])
    @test knn_stored(idx, live[1], 4) == (Int[], Float64[])
end

@testset "knn_stored: metrics and units match knn" begin
    rng = StableRNG(703)
    for M in (Euclidean(), SqEuclidean(), CosineDist()), T in (Float32, Float64)
        X = with_duplicates(rng, T, 6, 250)
        n = size(X, 2)
        idx = HNNGIndex(M, 6; T=T)
        add!(idx, X)
        @test verify(idx)
        for id in 1:11:n, k in (1, 3, 10, n - 1)
            @test check_stored(idx, id, k)
        end
        # the same answer as knn on the stored point (Euclidean family: query normalised the same way)
        id = 17
        ids, ds = knn(idx, X[:, id], 6)
        sids, sds = knn_stored(idx, id, 5)
        keep = ids .!= id
        @test sids == ids[keep][1:5] && sds ≈ ds[keep][1:5]
    end
end

@testset "knn_stored: ObjectStore, metric and non-metric UserMetric" begin
    rng = StableRNG(704)
    S = descriptor_sets(rng, 3, 40)
    push!(S, copy(S[3]), copy(S[3]), copy(S[10]))
    for (f, met) in ((hausdorff, true), (ros_view_distance, false))
        idx = HNNGIndex(UserMetric(f; is_metric=met), Matrix{Float64})
        for s in S
            add!(idx, s)
        end
        @test verify(idx) && is_metric(idx) == met
        for id in (1, 3, 10, 20, 41, 42, 43), k in (1, 2, 5, 100)
            @test check_stored(idx, id, k)
        end
        _, _, ne = HNNG.knn_counted(idx, S[3], 6)
        met || @test ne == length(idx)             # the unseeded path is the unpruned scan
    end
    hamming(a, b) = count(i -> a[i] != b[i], eachindex(a))
    words = [String(rand(rng, ['a', 'c', 'g', 't'], 5)) for _ in 1:60]
    idx = HNNGIndex(UserMetric(hamming; is_metric=true), String)
    add!(idx, words)
    for id in 1:7:60, k in (1, 4, 59)
        @test check_stored(idx, id, k)
    end
end

@testset "knn_stored: one element, errors" begin
    idx = HNNGIndex(Euclidean(), 2)
    @test_throws ArgumentError knn_stored(idx, 1, 1)
    add!(idx, Float32[1, 2])
    @test knn_stored(idx, 1, 1) == (Int[], Float64[])
    @test knn_stored(idx, 1, 5) == (Int[], Float64[])
    @test_throws ArgumentError knn_stored(idx, 1, 0)
    @test_throws ArgumentError knn_stored(idx, 2, 1)
end

@testset "knn_stored: k = 1 is free and agrees with k = 5" begin
    rng = StableRNG(705)
    X = with_duplicates(rng, Float32, 5, 300)
    idx = HNNGIndex(Euclidean(), 5)
    add!(idx, X)
    @test verify(idx)
    for id in 1:size(X, 2)
        reset_evals!(idx)
        ids1, d1 = knn_stored(idx, id, 1)
        @test idx.nevals == 0
        @test (ids1, d1) == (outlink(idx, id) == 0 ? Int[] : [outlink(idx, id)],
                             outlink(idx, id) == 0 ? Float64[] : [outlink_distance(idx, id)])
        ids5, d5 = knn_stored(idx, id, 5)
        @test ids1 == ids5[1:1] && d1 == d5[1:1]
    end
    reset_evals!(idx)
    knn_stored(idx, 3, 5)
    @test idx.nevals > 0
end

@testset "knn_stored never evaluates more than knn on the stored point with k + 1" begin
    rng = StableRNG(706)
    for d in (2, 8), M in (Euclidean(), SqEuclidean())
        idx = HNNGIndex(M, d; T=Float64)
        X = with_duplicates(rng, Float64, d, 400)
        add!(idx, X)
        tot_used = tot_ne = 0
        for id in 1:9:size(X, 2), k in (2, 5, 20)
            q = copy(element(idx, id))
            before = idx.nevals
            knn_stored(idx, id, k)
            used = idx.nevals - before
            _, _, ne = HNNG.knn_counted(idx, q, k + 1)
            @test used <= ne
            tot_used += used
            tot_ne += ne
        end
        @test tot_used < tot_ne                # the seeds do save evaluations
    end
end

@testset "knn_stored holds the lock and counts into nevals" begin
    rng = StableRNG(707)
    idx = HNNGIndex(Euclidean(), 4)
    add!(idx, gauss(rng, Float32, 4, 200))
    reset_evals!(idx)
    knn_stored(idx, 9, 5)
    @test idx.nevals > 0 && idx.memo.nevals == 0
    @test !islocked(idx.lock)
end
