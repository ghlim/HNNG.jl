include(joinpath(@__DIR__, "setup.jl"))
using LinearAlgebra

# Run `check_protocol()` in a child julia started with `flags`, with JULIA_NUM_THREADS
# removed from its environment; return (succeeded, stderr text).
function run_protocol_child(flags::Vector{String})
    core = joinpath(BENCH_DIR, "src", "BenchCore.jl")
    script = "include(raw\"$core\"); using .BenchCore; check_protocol()"
    env = copy(ENV)
    delete!(env, "JULIA_NUM_THREADS")
    cmd = setenv(`$(Base.julia_cmd()) $flags --project=$(BENCH_DIR) -e $script`, env)
    err = IOBuffer()
    proc = run(pipeline(ignorestatus(cmd); stdout = devnull, stderr = err))
    return success(proc), String(take!(err))
end

@testset "timing protocol" begin
    @testset "check_protocol accepts -t 1 and pins BLAS to one thread" begin
        @test Threads.maxthreadid() == 1
        BLAS.set_num_threads(2)
        @test check_protocol() === nothing
        @test BLAS.get_num_threads() == 1
    end

    @testset "check_protocol refuses more than one thread" begin
        ok, msg = run_protocol_child(["-t", "2"])
        @test !ok
        # the refusal message, so a child that merely failed to load BenchCore does not pass
        @test occursin("needs `julia -t 1`", msg)
    end

    @testset "check_protocol refuses plain julia (default + interactive thread)" begin
        ok, msg = run_protocol_child(String[])
        @test !ok
        @test occursin("needs `julia -t 1`", msg)
    end

    @testset "best_of keeps the fastest of three passes" begin
        durations = [0.06, 0.02, 0.04]
        calls = Ref(0)
        spans = Float64[]                 # the time each pass really took
        out, best = best_of() do
            calls[] += 1
            t0 = time_ns()
            sleep(durations[calls[]])
            push!(spans, (time_ns() - t0) / 1e9)
            calls[]
        end
        @test calls[] == 3 && out == 3
        # the fastest pass is the 0.02 s one; no tight upper bound (a loaded CI machine oversleeps)
        @test 0.019 <= best <= 0.2
        @test best <= minimum(spans[[1, 3]])
    end

    @testset "time_queries: one warm-up call, then three timed passes" begin
        calls = Ref(0)
        f = q -> (calls[] += 1; ([round(Int, q[1])], [0.0]))
        qs = [Float32[i] for i in 1:5]
        results, best = time_queries(f, qs)
        @test calls[] == 1 + 3 * 5
        @test [r[1][1] for r in results] == 1:5
        @test best >= 0
    end

    @testset "timed_queries: function barrier, so query(h, q, k) dispatches statically" begin
        X, Q = synthetic("gaussian", 200, 10, 4)
        qs = columns(Q)
        hs = Any[build(BruteBLAS(), X), build(KDTreeNN(), X)]   # Any, as build_time's handle is in exp1/exp2
        for h in hs
            results, best = timed_queries(h, qs, 5)
            @test results == [query(h, q, 5) for q in qs] && best > 0
            # inside the barrier the results vector is concretely typed (a dynamic closure gives Vector{Any})
            rt = only(Base.return_types(timed_queries, (typeof(h), typeof(qs), Int)))
            @test rt <: Tuple{Union{Nothing,Vector{Tuple{Vector{Int},Vector{Float64}}}},Float64}
        end
    end

    @testset "parse_cpu_list: the kernel's cpulist format" begin
        @test BenchCore.parse_cpu_list("2") == [2]
        @test BenchCore.parse_cpu_list("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
        @test BenchCore.parse_cpu_list("4-7") == 4:7
        @test BenchCore.parse_cpu_list(nothing) == Int[]
    end

    @testset "env.txt records the run environment" begin
        check_protocol()
        mktempdir() do dir
            p = write_env(joinpath(dir, "env.txt"))
            txt = read(p, String)
            for key in ("julia_version=", "manifest_sha256=", "cpu_model=", "governor=",
                        "cpus_allowed=", "threads=1", "interactive_threads=0", "blas_threads=1",
                        "hnng_commit=", "date=",
                        "dirty=", "smoke=", "seeds=", "pkg_versions=", "kernel=",
                        "cpu_sibling=", "no_turbo=", "epp=", "blas_config=")
                @test occursin(key, txt)
            end
            # one key=value per line; the BLAS configuration is flattened onto its line
            @test all(l -> occursin('=', l), split(chomp(txt), '\n'))
            @test match(r"^blas_config=.*\S"m, txt) !== nothing
            @test match(r"^no_turbo=(0|1|unknown)$"m, txt) !== nothing
            # the hyperthread siblings of every allowed CPU, as cpu:siblings
            allowed = BenchCore.parse_cpu_list(BenchCore.first_line_value("/proc/self/status", "Cpus_allowed_list"))
            sib = match(r"^cpu_sibling=(.*)$"m, txt)[1]
            @test sib == "unknown" || [parse(Int, split(e, ':')[1]) for e in split(sib, ';')] == allowed
            @test occursin("julia_version=$(VERSION)", txt)
            @test match(r"manifest_sha256=[0-9a-f]{64}", txt) !== nothing
            @test occursin("smoke=true", txt)                       # the test preamble sets BENCH_SMOKE=1
            @test occursin("seeds=GAUSS_TRAIN=1000,", txt) && occursin(",HNSW=9000,SS=9500", txt)
            @test match(r"^dirty=(true|false|unknown)$"m, txt) !== nothing
            @test occursin("pkg_versions=", txt) && occursin("NearestNeighbors=", txt)
            @test match(r"^kernel=\S+"m, txt) !== nothing
        end
    end
end
