# Every experiment, then the tables and the figures (spec §11).
# Pinned full run: bench/run.sh. Direct: julia -t 1 --project=bench bench/run_all.jl
isdefined(Main, :BenchCore) || include(joinpath(@__DIR__, "src", "BenchCore.jl"))
using .BenchCore
for (file, entry) in (("exp1.jl", :run_exp1), ("exp2.jl", :run_exp2), ("exp3.jl", :run_exp3),
                      ("tables.jl", :write_tables), ("plots.jl", :write_plots))
    isdefined(Main, entry) || include(joinpath(@__DIR__, file))
end

# Run order: exp2 and exp3 first (exp3 at n = 32000, d = 128 has never run at full size before
# the real run), then exp1 (random-xs), and the multi-hour Fashion-MNIST experiment last, so a
# late failure costs the least and the finished CSVs survive it.
const EXPERIMENTS = [
    (name = :exp2,  file = "exp2.csv",  run = run_exp2),
    (name = :exp3,  file = "exp3.csv",  run = run_exp3),
    (name = :exp1,  file = "exp1.csv",  run = run_exp1),
    (name = :exp1b, file = "exp1b.csv", run = run_exp1b),
]
const RESULT_CSVS = ["exp1.csv", "exp1b.csv", "exp2.csv", "exp3.csv"]

"BENCH_FORCE=1: rerun experiments whose CSV already exists."
bench_force() = get(ENV, "BENCH_FORCE", "0") == "1"

"BENCH_ONLY=exp1|exp1b|exp2|exp3: run only that experiment (`nothing` when unset)."
function bench_only()
    s = strip(get(ENV, "BENCH_ONLY", ""))
    isempty(s) && return nothing
    Symbol(s) in (e.name for e in EXPERIMENTS) ||
        error("BENCH_ONLY must be one of exp1, exp1b, exp2, exp3; got $(repr(s))")
    return Symbol(s)
end

"""
    run_experiments(; smoke, outdir, force, select, experiments) -> Vector{Symbol}

Runs `experiments` in order and returns the names of those that ran. An experiment whose CSV
is already in `outdir` is skipped unless `force`; smoke runs always rerun, since their
directory is scratch. `select` (a name) restricts the run to that experiment.
"""
function run_experiments(; smoke::Bool = is_smoke(), outdir::AbstractString = results_dir(smoke),
                         force::Bool = bench_force(), select::Union{Nothing,Symbol} = bench_only(),
                         experiments = EXPERIMENTS)
    ran = Symbol[]
    for e in experiments
        select === nothing || e.name == select || continue
        if !(force || smoke) && isfile(joinpath(outdir, e.file))
            @info "$(e.name): $(e.file) exists, skipped (BENCH_FORCE=1 reruns it)"
            continue
        end
        @info "running $(e.name)"
        e.run(; smoke, outdir)
        push!(ran, e.name)
    end
    return ran
end

function run_all(; smoke::Bool = is_smoke(), outdir::AbstractString = results_dir(smoke),
                 force::Bool = bench_force(), select::Union{Nothing,Symbol} = bench_only())
    check_protocol()
    mkpath(outdir)
    write_env(joinpath(outdir, "env.txt"))
    @info "benchmark results go to $outdir" smoke force select
    run_experiments(; smoke, outdir, force, select)
    absent = [f for f in RESULT_CSVS if !isfile(joinpath(outdir, f))]
    if isempty(absent)
        write_tables(outdir)
        write_plots(outdir)
    else
        @warn "tables and figures not written: missing $(join(absent, ", ")); rerun to continue" outdir
    end
    return outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
