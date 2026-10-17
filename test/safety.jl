isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))

# Public calls that take the per-index lock (spec §4, §6.5), each as a closure.
function locked_calls(idx, q, path)
    Q = reshape(q, :, 1)
    [() -> knn(idx, q, 3), () -> HNNG.knn_counted(idx, q, 3), () -> HNNG.knn_levelsync(idx, q, 3),
     () -> knn_batch(idx, Q, 3), () -> knn_batch(idx, Q, 3; threads=true),
     () -> delete!(idx, add!(idx, q)), () -> verify(idx), () -> HNNG.save(path, idx),
     () -> stats(idx), () -> reset_evals!(idx)]
end

@testset "per-index lock: every public entry takes it, and it is reentrant" begin
    rng = StableRNG(110)
    idx = HNNGIndex(Euclidean(), 3)
    @test idx.lock isa ReentrantLock
    add!(idx, gauss(rng, Float32, 3, 200))
    @test verify(idx)
    q = randn(rng, Float32, 3)
    r = knn(idx, q, 3)
    path = joinpath(mktempdir(), "lock.hnng")
    # another task blocks on each call while this task holds the lock
    for f in locked_calls(idx, q, path)
        lock(idx.lock)
        t = @async f()
        for _ in 1:20
            yield()
        end
        @test !istaskdone(t)
        unlock(idx.lock)
        fetch(t)
        @test verify(idx)
    end
    e = HNNGIndex(Euclidean(), 3)
    lock(e.lock)
    t = @async bulk_add!(e, ones(3, 2))
    for _ in 1:20
        yield()
    end
    @test !istaskdone(t)
    unlock(e.lock)
    @test fetch(t) == 1:2 && verify(e)
    # reentrant: a caller that holds the lock may call every public function
    lock(idx.lock) do
        for f in locked_calls(idx, q, path)
            f()
        end
        @test knn(idx, q, 3) == r
        @test verify(idx)
    end
    @test !islocked(idx.lock)
    @test_throws ArgumentError knn(idx, q, 0)              # a throwing call releases the lock
    @test_throws ArgumentError add!(idx, [1.0, 2.0])
    @test !islocked(idx.lock)
    @test fetch(Threads.@spawn knn(idx, q, 3)) == r
end

using Distances: CorrDist

@testset "a NaN distance throws ArgumentError and rolls back (I-2)" begin
    rng = StableRNG(120)
    X = randn(rng, 4, 30)
    X[:, 7] .= 1.0                                     # constant column: CorrDist is NaN against everything
    idx = HNNGIndex(CorrDist(), 4; T=Float64)
    @test_throws ArgumentError add!(idx, X)            # columns 1:6 stay, column 7 is rolled back
    @test length(idx) == 6 && HNNG._ntotal(idx) == 6 && !idx.invalid
    @test verify(idx)
    @test add!(idx, X[:, 8]) == 7                      # the rolled-back column got no id
    @test verify(idx)
    err = try add!(idx, X[:, 7]); nothing catch e; e end
    @test err isa ArgumentError && occursin("is NaN", err.msg) && occursin("ids 8 and", err.msg)
    @test length(idx) == 7 && verify(idx)
    # a query whose distance is NaN
    @test_throws ArgumentError knn(idx, ones(4), 3)
    @test_throws "the query and id" knn(idx, ones(4), 3)
    @test_throws ArgumentError HNNG.knn_counted(idx, ones(4), 3)
    @test_throws ArgumentError HNNG.knn_levelsync(idx, ones(4), 3)
    @test_throws ArgumentError knn_batch(idx, ones(4, 2), 3)
    @test_throws ArgumentError knn_batch(idx, ones(4, 2), 3; threads=true)
    q = randn(rng, 4)
    @test check_knn(idx, q, 5) && verify(idx)         # the index is untouched
    # bulk_add! resets the index to the empty state
    b = HNNGIndex(CorrDist(), 4; T=Float64)
    @test_throws ArgumentError bulk_add!(b, X)
    @test length(b) == 0 && HNNG._ntotal(b) == 0 && nlevels(b) == 0 && !b.invalid
    @test verify(b)
    @test bulk_add!(b, X[:, 1:6]) == 1:6
    @test verify(b)
    # a UserMetric that returns NaN
    f(a, b) = (a == 3.0 || b == 3.0) ? NaN : abs(a - b)
    o = HNNGIndex(UserMetric(f; is_metric=true), Float64)
    @test add!(o, [1.0, 2.0]) == 1:2
    @test verify(o)
    @test_throws ArgumentError add!(o, 3.0)
    @test length(o) == 2 && HNNG._ntotal(o) == 2 && verify(o)
    @test_throws "the query and id" knn(o, 3.0, 1)
    @test add!(o, [10.0, 11.0]) == 3:4
    @test verify(o)
    @test knn(o, 2.9, 2) == ([2, 1], [0.8999999999999999, 1.9])
end

"""A `Float64` metric `|a - b|` that counts its calls and throws on call `at[]` (0: never)."""
function tripwire(; err=() -> error("tripwire"))
    calls, at = Ref(0), Ref(0)
    f(a, b) = (calls[] += 1; calls[] == at[] && throw(err()); abs(a - b))
    f, calls, at
end

const INVALID = "index is invalid after an exception in add!/delete!/bulk_add!"

@testset "an exception in add! before any structural write rolls back (I-3)" begin
    rng = StableRNG(130)
    S = descriptor_sets(rng, 3, 30)
    idx = HNNGIndex(UserMetric(hausdorff; is_metric=true), Matrix{Float64})
    add!(idx, S)
    @test verify(idx)
    q = randn(rng, 3, 4)
    r = knn(idx, q, 5)
    @test_throws DimensionMismatch add!(idx, randn(rng, 4, 3))   # the metric throws on the bad object
    @test length(idx) == 30 && HNNG._ntotal(idx) == 30 && !idx.invalid
    @test verify(idx)
    @test knn(idx, q, 5) == r
    @test add!(idx, S[1] .+ 0.01) == 31                         # the next add! gets the next id
    @test verify(idx)
    delete!(idx, 31)
    @test verify(idx)
    # a vector add! keeps the objects before the failing one
    @test_throws DimensionMismatch add!(idx, [S[2] .+ 0.01, randn(rng, 4, 2), S[3] .+ 0.01])
    @test length(idx) == 31 && HNNG._ntotal(idx) == 32 && verify(idx)
    @test check_knn(idx, q, 7)
    # InterruptException (Ctrl-C) is handled the same way
    f, calls, at = tripwire(err=() -> InterruptException())
    v = HNNGIndex(UserMetric(f; is_metric=true), Float64)
    add!(v, collect(1.0:20.0))
    @test verify(v)
    at[] = calls[] + 1
    @test_throws InterruptException add!(v, 7.5)
    @test length(v) == 20 && HNNG._ntotal(v) == 20 && !v.invalid
    @test verify(v)
    @test add!(v, 7.5) == 21
    @test verify(v)
    # the first element: a level 0 is created; rolled back to no levels
    g, gcalls, gat = tripwire()
    w = HNNGIndex(UserMetric(g; is_metric=true), Float64)
    gat[] = 1
    @test_throws "tripwire" add!(w, [1.0, 2.0])                 # element 1 needs no distance
    @test length(w) == 1 && nlevels(w) == 1 && verify(w)
end

@testset "an exception after a structural write invalidates the index (I-3)" begin
    rng = StableRNG(131)
    X = 10 .* randn(rng, 60)
    x = 0.123
    f, calls, at = tripwire()
    build() = (i = HNNGIndex(UserMetric(f; is_metric=true), Float64); add!(i, X); verify(i); i)
    # count the evaluations of the level-0 search for x and of the whole insertion
    dry = build()
    memo0 = dry.memo.nevals
    HNNG._nn_search(dry, x, 0, 0, 0, Int[], true)
    esearch = dry.memo.nevals - memo0
    HNNG._flush!(dry)
    n0 = dry.nevals
    add!(dry, x)
    @test verify(dry)
    etotal = dry.nevals - n0
    @test 0 < esearch < etotal
    # a throw on the last evaluation lands after outlinks were written
    idx = build()
    at[] = calls[] + etotal
    @test_throws "tripwire" add!(idx, x)
    at[] = 0
    @test idx.invalid
    path = joinpath(mktempdir(), "bad.hnng")
    for g in (() -> knn(idx, 0.0, 1), () -> add!(idx, 1.0), () -> add!(idx, [1.0, 2.0]),
              () -> delete!(idx, 1), () -> verify(idx), () -> HNNG.save(path, idx),
              () -> knn_batch(idx, [0.0], 1), () -> knn_batch(idx, [0.0], 1; threads=true),
              () -> HNNG.knn_counted(idx, 0.0, 1), () -> HNNG.knn_levelsync(idx, 0.0, 1),
              () -> element(idx, 1), () -> outlink(idx, 1), () -> inlinks(idx, 1),
              () -> cluster(idx, 1), () -> members(idx, HNNG.ClusterRef(0, 1)),
              () -> root(idx), () -> newick(idx), () -> path_to_root(idx, 1))
        @test_throws ArgumentError g()
        @test_throws INVALID g()
    end
    @test !isfile(path)
    @test length(idx) == 61 && nlevels(idx) >= 1 && is_metric(idx)   # these still answer
    @test stats(idx).n_live == 61
    # the same throw on the first evaluation lands in the search: rolled back
    idx = build()
    at[] = calls[] + 1
    @test_throws "tripwire" add!(idx, x)
    at[] = 0
    @test !idx.invalid && length(idx) == 60
    @test verify(idx)
    @test add!(idx, x) == 61
    @test verify(idx)
end

@testset "an exception in delete!: rolled back in the level-0 re-search, else invalid (I-3)" begin
    rng = StableRNG(132)
    X = 10 .* randn(rng, 60)
    f, calls, at = tripwire()
    build() = (i = HNNGIndex(UserMetric(f; is_metric=true), Float64); add!(i, X); verify(i); i)
    # an id with inlinkers re-searches its inlinkers before any write: rolled back
    idx = build()
    id = findfirst(i -> !isempty(inlinks(idx, i)), 1:60)
    q = 1.5
    r = knn(idx, q, 5)
    at[] = calls[] + 1
    @test_throws "tripwire" delete!(idx, id)
    at[] = 0
    @test !idx.invalid && length(idx) == 60 && idx.alive[id]
    @test verify(idx)
    @test knn(idx, q, 5) == r
    delete!(idx, id)
    @test verify(idx) && length(idx) == 59
    # find an id whose delete! evaluates distances after the level-0 re-search
    victim, etotal = 0, 0
    for i in 1:60
        a = build()
        m0 = a.memo.nevals
        HNNG._relink_search(a, i, 0, Int[])
        es = a.memo.nevals - m0
        HNNG._flush!(a)
        n0 = a.nevals
        delete!(a, i)
        et = a.nevals - n0
        if et > es
            victim, etotal = i, et
            break
        end
    end
    @test victim != 0
    idx = build()
    at[] = calls[] + etotal
    @test_throws "tripwire" delete!(idx, victim)
    at[] = 0
    @test idx.invalid
    @test_throws INVALID verify(idx)
    @test_throws INVALID knn(idx, q, 1)
end
