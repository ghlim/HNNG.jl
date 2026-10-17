include(joinpath(@__DIR__, "setup.jl"))

@testset "baselines (exact methods)" begin
    exact = (HNNGIncremental(), HNNGBulk(), BruteBLAS(), KDTreeNN(), BallTreeNN())
    @test [method_name(m) for m in exact] ==
          ["hNNG-incremental", "hNNG-bulk", "brute-BLAS", "KD-tree", "ball-tree"]

    @testset "exact on $geometry, d = $d" for (geometry, d) in (("gaussian", 2), ("gaussian", 8), ("blobs", 8))
        X, Q = synthetic(geometry, 300, 20, d)
        nb, gd = ground_truth(X, Q, 10)
        for m in exact
            h, t = build_time(m, X)
            @test t > 0
            res = [query(h, q, 10) for q in columns(Q)]
            @test all(length(r[1]) == 10 for r in res)
            @test all(r[1] isa Vector{Int} && r[2] isa Vector{Float64} for r in res)
            @test all(issorted(collect(zip(r[2], r[1]))) for r in res)
            ra, _ = mean_recalls(X, Q, nb, gd, res, 10)
            @test ra == 1.0
            @test min_match(X, Q, gd, res, 10) == 1.0       # all ten distances, not only the additive rule
        end
    end

    @testset "canonical sorts by (distance, id) only when needed" begin
        @test canonical([5, 3, 9], [1.0, 1.0, 0.5]) == ([9, 3, 5], [0.5, 1.0, 1.0])
        @test canonical([9, 3], [1.0, 1.0]) == ([3, 9], [1.0, 1.0])        # tie in distance: smaller id first
        ids = [3, 5, 9]
        cid, cds = canonical(ids, Float32[0.5, 1.0, 1.0])                  # already sorted: no reordering,
        @test cid === ids && cds == [0.5, 1.0, 1.0] && cds isa Vector{Float64}   # ids passed through, dists widened
        @test canonical(Int[], Float64[]) == (Int[], Float64[])
    end

    @testset "tree results arrive sorted: canonical has nothing to sort" begin
        X, Q = synthetic("gaussian", 300, 20, 8)
        for m in (KDTreeNN(), BallTreeNN())
            h = build(m, X)
            for q in columns(Q)
                _, ds = BenchCore.NN.knn(h.tree, q, 10, true)
                @test issorted(ds)
            end
        end
    end

    @testset "hNNG fraction evaluated" begin
        X, Q = synthetic("blobs", 300, 20, 2)
        h = build(HNNGBulk(), X)
        ρ = fraction_evaluated(h, columns(Q), 10)
        @test 0 < ρ <= 1
        @test length(h.idx) == 300
    end

    @testset "brute force stays exact under Float32 cancellation" begin
        # common offset 1000: ‖x‖² ≈ 8e6 while neighbour distances² are O(10)
        X = gaussian(500, 8, 11) .+ 1000f0
        Q = gaussian(20, 8, 12) .+ 1000f0
        nb, gd = ground_truth(X, Q, 10)
        h = build(BruteBLAS(), X)
        for (j, q) in enumerate(columns(Q))
            ids, ds = query(h, q, 10)
            @test ids == nb[:, j]
            @test ds == gd[:, j]
        end
    end

    @testset "k larger than n returns every point" begin
        X, Q = synthetic("gaussian", 7, 1, 3)
        for m in exact
            ids, _ = query(build(m, X), columns(Q)[1], 10)
            @test sort(ids) == 1:7
        end
    end
end

# The compile guard in build_time. These run in a child julia: in the test process the
# build paths are already compiled by the tests above, so "no warning" would be vacuous.
@testset "build_time compile guard (fresh process)" begin
    core = joinpath(BENCH_DIR, "src", "BenchCore.jl")
    function run_child(body::String)
        script = "include(raw\"$core\"); using .BenchCore; $body"
        cmd = `$(Base.julia_cmd()) -t 1 --project=$(BENCH_DIR) -e $script`
        err = IOBuffer()
        proc = run(pipeline(ignorestatus(cmd); stdout = devnull, stderr = err))
        return success(proc), String(take!(err))
    end
    # smoke-size build in a fresh process: the first call to build_time for each method
    ok, msg = run_child("""
        X, _ = synthetic("gaussian", 300, 20, 20)
        ms = BenchMethod[HNNGIncremental()]
        HAS_SS && push!(ms, SSSearchGraph())
        for m in ms
            build_time(m, X)
        end
        """)
    @test ok
    @test !occursin("compilation inside timed build", msg)
    # control: a method whose timed build (second call) compiles new code does trigger the warning
    ok, msg = run_child("""
        struct Compiles <: BenchMethod; calls::Ref{Int}; end
        BenchCore.method_name(::Compiles) = "compiles"
        function BenchCore.build(m::Compiles, X::Matrix{Float32})
            m.calls[] += 1
            m.calls[] == 2 && include_string(Main, "using LinearAlgebra; ctl_h(n) = sum(abs2, svd(randn(n, n)).S) + sum(sort(rand(Int8, n)) .+ 1); ctl_h(20)")
            return nothing
        end
        build_time(Compiles(Ref(0)), zeros(Float32, 2, 10))
        """)
    @test ok
    @test occursin("compilation inside timed build", msg)
end
