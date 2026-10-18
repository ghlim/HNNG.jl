# Exp.2: exact-search scaling with dimension on Gaussian and blob data, spec §11.
# Run: julia -t 1 --project=bench bench/exp2.jl        (BENCH_SMOKE=1 for the tiny version)
isdefined(Main, :BenchCore) || include(joinpath(@__DIR__, "src", "BenchCore.jl"))
using .BenchCore
using CSV, DataFrames
import HNNG

const EXP2_DIMS = [2, 4, 8, 16, 32, 64, 128]
const EXP2_K = 10

exp2_settings(smoke::Bool) = smoke ?
    (dims = [2, 8], nq = 20, n = Dict("gaussian" => 300, "blobs" => 300)) :
    (dims = EXP2_DIMS, nq = 1000, n = Dict("gaussian" => 5000, "blobs" => 10000))

function run_exp2(; smoke::Bool = is_smoke(), outdir::AbstractString = results_dir(smoke))
    check_protocol()
    s = exp2_settings(smoke)
    df = empty_exp2()
    for geometry in ("gaussian", "blobs"), d in s.dims
        n = s.n[geometry]
        X, Q = synthetic(geometry, n, s.nq, d)
        nb, gd = ground_truth(X, Q, EXP2_K)
        qs = columns(Q)
        for m in BenchMethod[HNNGBulk(), KDTreeNN(), BallTreeNN(), BruteBLAS()]
            h, bt = build_time(m, X)
            results, seconds = timed_queries(h, qs, EXP2_K)
            ra, _ = mean_recalls(X, Q, nb, gd, results, EXP2_K)
            ra == 1.0 || error("Exp.2: $(method_name(m)) is not exact on $geometry d=$d (recall $ra)")
            # at low d the additive recall rule cannot see a (k+1)-th-neighbour answer; compare the distances (A25)
            mm = min_match(X, Q, gd, results, EXP2_K)
            @info "Exp.2 exactness geometry=$geometry d=$d method=$(method_name(m)) min_match=$mm"
            mm == 1.0 || error("Exp.2: $(method_name(m)) returns a wrong neighbour on $geometry d=$d (min_match $mm)")
            frac, nlev = h isa HNNGHandle ?
                (fraction_evaluated(h, qs, EXP2_K), HNNG.nlevels(h.idx)) : (missing, missing)
            push!(df, (geometry = geometry, n = n, d = d, nq = s.nq, k = EXP2_K,
                       method = method_name(m), build_s = bt, qps = s.nq / seconds,
                       recall_ann = ra, frac_evaluated = frac, nlevels = nlev))
        end
        @info "Exp.2" geometry d
    end
    mkpath(outdir)
    write_csv_atomic(joinpath(outdir, "exp2.csv"), df)
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_exp2()
end
