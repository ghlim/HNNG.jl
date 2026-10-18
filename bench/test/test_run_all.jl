include(joinpath(@__DIR__, "setup.jl"))
isdefined(Main, :run_all) || include(joinpath(BENCH_DIR, "run_all.jl"))

# A fake experiment that records its call and writes its CSV, to observe order and skipping.
fake_experiment(name, file, calls) = (name = name, file = file,
    run = function (args...; kw...)
        push!(calls, name)
        write(joinpath(kw[:outdir], file), "x\n")
        return nothing
    end)

@testset "run_experiments: order, skip-existing, BENCH_ONLY, BENCH_FORCE" begin
    @test [e.name for e in EXPERIMENTS] == [:exp2, :exp3, :exp1, :exp1b]       # Fashion-MNIST last
    @test [e.file for e in EXPERIMENTS] == ["exp2.csv", "exp3.csv", "exp1.csv", "exp1b.csv"]
    mktempdir() do out
        calls = Symbol[]
        fakes = [fake_experiment(:exp2, "exp2.csv", calls), fake_experiment(:exp3, "exp3.csv", calls),
                 fake_experiment(:exp1, "exp1.csv", calls), fake_experiment(:exp1b, "exp1b.csv", calls)]
        go(; smoke = false, force = false, select = nothing) =
            run_experiments(; smoke, outdir = out, force, select, experiments = fakes)
        @test go() == [:exp2, :exp3, :exp1, :exp1b] && calls == [:exp2, :exp3, :exp1, :exp1b]
        empty!(calls)
        @test isempty(go()) && isempty(calls)                       # every CSV exists: nothing is called
        rm(joinpath(out, "exp3.csv"))                               # a crashed experiment left no CSV
        @test go() == [:exp3] && calls == [:exp3]
        empty!(calls)
        @test go(; force = true) == [:exp2, :exp3, :exp1, :exp1b]   # BENCH_FORCE=1 reruns everything
        empty!(calls)
        @test isempty(go(; select = :exp1))                         # BENCH_ONLY alone still skips a finished CSV
        @test go(; select = :exp1, force = true) == [:exp1] && calls == [:exp1]
        empty!(calls)
        @test go(; smoke = true) == [:exp2, :exp3, :exp1, :exp1b]   # smoke runs always rerun
    end
    withenv("BENCH_FORCE" => "1", "BENCH_ONLY" => "exp1b") do
        @test bench_force() && bench_only() == :exp1b
    end
    withenv("BENCH_FORCE" => nothing, "BENCH_ONLY" => nothing) do
        @test !bench_force() && bench_only() === nothing
    end
    withenv("BENCH_ONLY" => "exp9") do
        @test_throws ErrorException bench_only()
    end
end

@testset "runs.txt: a start and an end line per experiment that runs" begin
    mktempdir() do out
        calls = Symbol[]
        fakes = [fake_experiment(:exp2, "exp2.csv", calls), fake_experiment(:exp3, "exp3.csv", calls)]
        run_experiments(; smoke = false, outdir = out, force = false, select = nothing, experiments = fakes)
        run_experiments(; smoke = false, outdir = out, force = false, select = nothing, experiments = fakes)  # all skipped
        lines = readlines(joinpath(out, "runs.txt"))
        @test [join(split(l)[1:2], " ") for l in lines] == ["start exp2", "end exp2", "start exp3", "end exp3"]
        @test all(l -> match(r"^(start|end) exp\d \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$", l) !== nothing, lines)
        # a failing experiment leaves its start line and no end line
        boom = [(name = :exp1, file = "exp1.csv", run = (args...; kw...) -> error("crash"))]
        @test_throws ErrorException run_experiments(; smoke = false, outdir = out, force = false,
                                                    select = nothing, experiments = boom)
        @test startswith(readlines(joinpath(out, "runs.txt"))[end], "start exp1 ")
    end
end

@testset "record_env: env.txt on a fresh run, env-resume-<time>.txt on a resume, commit guard" begin
    head = BenchCore.git_commit()
    envkey(p) = match(r"^hnng_commit=(.*)$"m, read(p, String))[1]
    resumes(out) = filter(f -> match(r"^env-resume-\d{8}T\d{6}\.txt$", f) !== nothing, readdir(out))
    mktempdir() do out                                         # nothing there: fresh
        @test record_env(out; smoke = false, force = false, select = nothing) == joinpath(out, "env.txt")
        @test envkey(joinpath(out, "env.txt")) == head && isempty(resumes(out))
    end
    mktempdir() do out                                         # a CSV and env.txt from this commit: resume
        write(joinpath(out, "exp2.csv"), "x\n")
        write(joinpath(out, "env.txt"), "hnng_commit=$head\nfirst=1\n")
        p = record_env(out; smoke = false, force = false, select = nothing)
        @test p == joinpath(out, only(resumes(out)))
        @test read(joinpath(out, "env.txt"), String) == "hnng_commit=$head\nfirst=1\n"   # untouched
        @test envkey(p) == head
    end
    mktempdir() do out                                         # resume across a commit change: refused
        write(joinpath(out, "exp2.csv"), "x\n")
        write(joinpath(out, "env.txt"), "hnng_commit=0123456789abcdef\n")
        err = try
            record_env(out; smoke = false, force = false, select = nothing); nothing
        catch e
            e
        end
        @test err isa ErrorException && occursin("BENCH_FORCE=1", err.msg) && occursin("0123456789abcdef", err.msg)
        @test isempty(resumes(out))
        # BENCH_FORCE=1 starts afresh and rewrites env.txt; smoke always reruns everything
        @test record_env(out; smoke = false, force = true, select = nothing) == joinpath(out, "env.txt")
        @test envkey(joinpath(out, "env.txt")) == head
        write(joinpath(out, "env.txt"), "hnng_commit=0123456789abcdef\n")
        @test record_env(out; smoke = true, force = false, select = nothing) == joinpath(out, "env.txt")
        # BENCH_FORCE=1 with BENCH_ONLY reruns one experiment: the others' env.txt is kept
        write(joinpath(out, "env.txt"), "hnng_commit=0123456789abcdef\n")
        p = record_env(out; smoke = false, force = true, select = :exp1)
        @test p == joinpath(out, only(resumes(out)))
        @test read(joinpath(out, "env.txt"), String) == "hnng_commit=0123456789abcdef\n"
    end
end

@testset "run_all: tables and plots only when all four CSVs exist" begin
    mktempdir() do out
        run_all(; outdir = out, force = true, select = :exp2)
        @test isfile(joinpath(out, "exp2.csv")) && isfile(joinpath(out, "env.txt"))
        @test [split(l)[1] for l in readlines(joinpath(out, "runs.txt"))] == ["start", "end"]
        @test !isfile(joinpath(out, "exp1.csv"))
        @test !isdir(joinpath(out, "tex")) && !isdir(joinpath(out, "fig"))
    end
end

@testset "run_all smoke: every output file" begin
    mktempdir() do out
        @test run_all(; outdir = out) == out
        for f in ("exp1.csv", "exp1b.csv", "exp2.csv", "exp3.csv", "env.txt",
                  "tex/tab_annbench.tex", "tex/tab_fashion.tex", "tex/tab_ops.tex", "tex/macros.tex",
                  "fig/recall_qps.pdf", "fig/dist_eval_vs_dim.pdf", "fig/qps_vs_dim.pdf", "fig/ops_comparison.pdf")
            @test isfile(joinpath(out, f))
        end
        ops = readlines(joinpath(out, "tex", "tab_ops.tex"))
        for row in ("200 (\$2\$) & ", "300 (\$2\$) & ", "300 (\$8\$) & ")
            @test count(l -> startswith(l, row), ops) == 1
        end
        annbench = read(joinpath(out, "tex", "tab_annbench.tex"), String)
        @test occursin("hNNG (incremental build) & 1.0000 & ", annbench)
        @test occursin("HNSW (sweet spot) & ", annbench) == HAS_HNSW
        macros = read(joinpath(out, "tex", "macros.tex"), String)
        @test occursin("\\newcommand{\\rhoDtwo}{", macros) && occursin("\\newcommand{\\rhoDsixteen}{n/a}", macros)
        @test occursin("threads=1", read(joinpath(out, "env.txt"), String))
    end
end
