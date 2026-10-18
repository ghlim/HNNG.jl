# Exp.1 (random-xs-20-euclidean) and Exp.1b (fashion-mnist-784-euclidean), spec §11.
# Run: julia -t 1 --project=bench bench/exp1.jl        (BENCH_SMOKE=1 for the tiny version; runs both)
isdefined(Main, :BenchCore) || include(joinpath(@__DIR__, "src", "BenchCore.jl"))
using .BenchCore
using CSV, DataFrames
import HNNG

const EXP1_EFS = [10, 20, 40, 80, 160, 320, 640]
const EXP1_EFS_SMOKE = [10, 40]
const EXP1_K = 10

function exp1_dataset(name::AbstractString, smoke::Bool)
    ds = load_hdf5(joinpath(data_dir(), name * ".hdf5"))
    return smoke ? smoke_subset(ds, 300, 20) : ds
end

"One row per method on `ds`; HNSW.jl gets one row per `ef`."
function bench_dataset(ds::Dataset; k::Int = EXP1_K, efs = EXP1_EFS)
    df = empty_exp1()
    qs = columns(ds.test)
    base = (dataset = ds.name, n = size(ds.train, 2), d = size(ds.train, 1), nq = length(qs), k = k)
    function row!(name, param, build_s, results, seconds; frac = missing, nlev = missing, ncl = missing)
        ra, ri = mean_recalls(ds, results, k)
        push!(df, (; base..., method = name, param = param, build_s = build_s,
                   qps = length(qs) / seconds, recall_ann = ra, recall_id = ri,
                   frac_evaluated = frac, nlevels = nlev, nclusters = ncl))
    end
    for m in BenchMethod[HNNGIncremental(), HNNGBulk(), BruteBLAS(), KDTreeNN(), BallTreeNN()]
        h, bt = build_time(m, ds.train)
        results, seconds = timed_queries(h, qs, k)
        mm = min_match(ds, results, k)         # exactness beyond the additive recall rule (A25)
        @info "Exp.1 exactness dataset=$(ds.name) method=$(method_name(m)) min_match=$mm"
        mm == 1.0 || error("Exp.1: $(method_name(m)) is not exact on $(ds.name) (min_match $mm)")
        if h isa HNNGHandle
            s = HNNG.stats(h.idx)
            row!(method_name(m), "", bt, results, seconds;
                 frac = fraction_evaluated(h, qs, k), nlev = s.nlevels, ncl = join(s.nclusters, ";"))
        else
            row!(method_name(m), "", bt, results, seconds)
        end
        @info "Exp.1" ds.name method_name(m) build_s = bt
    end
    if HAS_HNSW
        hh, hbt = build_time(HNSWjl(), ds.train)
        for ef in efs
            set_ef!(hh, ef)
            results, seconds = timed_queries(hh, qs, k)
            row!("HNSW", "ef=$ef", hbt, results, seconds)
        end
    end
    for m in optional_methods()
        try
            hs, sbt = build_time(m, ds.train)
            results, seconds = timed_queries(hs, qs, k)
            row!(method_name(m), "", sbt, results, seconds)
        catch err
            @warn "$(method_name(m)) row skipped on $(ds.name)" exception = err
        end
    end
    return df
end

function run_exp1_file(file::AbstractString, name::AbstractString, smoke::Bool, outdir::AbstractString)
    check_protocol()
    mkpath(outdir)
    df = bench_dataset(exp1_dataset(name, smoke); efs = smoke ? EXP1_EFS_SMOKE : EXP1_EFS)
    write_csv_atomic(joinpath(outdir, file), df)
    return nothing
end

"Exp.1: random-xs-20-euclidean -> exp1.csv."
run_exp1(; smoke::Bool = is_smoke(), outdir::AbstractString = results_dir(smoke)) =
    run_exp1_file("exp1.csv", "random-xs-20-euclidean", smoke, outdir)

"Exp.1b: fashion-mnist-784-euclidean -> exp1b.csv (hours in a full run)."
run_exp1b(; smoke::Bool = is_smoke(), outdir::AbstractString = results_dir(smoke)) =
    run_exp1_file("exp1b.csv", "fashion-mnist-784-euclidean", smoke, outdir)

if abspath(PROGRAM_FILE) == @__FILE__
    run_exp1()
    run_exp1b()
end
