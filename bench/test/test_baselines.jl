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
        # concretely inferred (no boxed captured variable): the tree and HNSW query paths call it per query
        @test only(Base.return_types(canonical, (Vector{Int}, Vector{Float32}))) == Tuple{Vector{Int},Vector{Float64}}
        @test only(Base.return_types(canonical, (Vector{Int32}, Vector{Float32}))) == Tuple{Vector{Int},Vector{Float64}}
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

# The brute force before the final-review fix (39f7c17), kept verbatim as the reference: d×n
# transposed gemv, copy + partialsort! for the k-th score, τ = max(1e-4, 2d·eps)·(‖q‖² + max‖x‖²),
# Float64 re-rank. The fixed BruteBLAS must return identical (ids, dists) on every query.
function brute_reference(X::Matrix{Float32}, q::AbstractVector{Float32}, k::Int)
    d, n = size(X)
    kk = min(k, n)
    n2 = vec(sum(abs2, X; dims = 1))
    a = n2 .- 2f0 .* (transpose(X) * q)
    t = partialsort!(copy(a), kk)
    τ = max(1f-4, 2f0 * d * eps(Float32)) * (sum(abs2, q) + maximum(n2; init = 0f0))
    cand = findall(<=(t + τ), a)
    ds = Float64[dist64(view(X, :, i), q) for i in cand]
    p = first(sort(collect(eachindex(cand)); by = j -> (ds[j], cand[j])), kk)
    return cand[p], ds[p]
end

@testset "brute force: n×d layout up to d = 128, single-pass top-k, same answers as before" begin
    function same_as_reference(X, Q; ks = (1, 10))
        h = build(BruteBLAS(), X)
        return all(query(h, q, k) == brute_reference(X, q, k) for q in columns(Q), k in ks)
    end
    @testset "$geometry, n = $n, d = $d" for (geometry, n, d) in
            (("gaussian", 300, 2), ("gaussian", 300, 8), ("blobs", 300, 8),     # the exact-method test sets
             ("gaussian", 2000, 2), ("gaussian", 2000, 32), ("gaussian", 2000, 128), ("gaussian", 2000, 200),
             ("blobs", 2000, 16))
        X, Q = synthetic(geometry, n, 50, d)
        @test same_as_reference(X, Q)
        h = build(BruteBLAS(), X)
        if d <= 128
            @test h.Xt == permutedims(X)                    # gemv 'N' on the n×d copy
        else
            @test isempty(h.Xt)                             # d > 128 keeps the d×n transposed gemv
        end
    end
    @testset "offset data (Float32 cancellation) and k ≥ n" begin
        X = gaussian(500, 8, 11) .+ 1000f0
        Q = gaussian(20, 8, 12) .+ 1000f0
        @test same_as_reference(X, Q; ks = (1, 10, 499, 500, 600))
    end
    rx = joinpath(data_dir(), "random-xs-20-euclidean.hdf5")
    if isfile(rx)
        ds = load_hdf5(rx)
        @test same_as_reference(ds.train, ds.test[:, 1:200])
    else
        @info "random-xs-20 not found under $(data_dir()); its brute-force identity check is skipped"
    end
    @testset "τ window: 4(d+3)·eps(Float32), floor 1e-4" begin
        q = Float32[1, 2, 3]
        @test BenchCore.brute_tau(20, q, 5f0) == max(1f-4, 4f0 * 23 * eps(Float32)) * (14f0 + 5f0)
        @test BenchCore.brute_tau(20, q, 5f0) == 1f-4 * 19f0          # floor at small d
        @test BenchCore.brute_tau(784, q, 5f0) == 4f0 * 787 * eps(Float32) * 19f0
    end
end
