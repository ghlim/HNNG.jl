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
