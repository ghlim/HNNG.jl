# Exp.3: incremental insert/delete latency against full rebuilds, spec §11.
# Run: julia -t 1 --project=bench bench/exp3.jl        (BENCH_SMOKE=1 for the tiny version)
isdefined(Main, :BenchCore) || include(joinpath(@__DIR__, "src", "BenchCore.jl"))
using .BenchCore
using CSV, DataFrames, StableRNGs
using Distances: Euclidean
import HNNG

const EXP3_NS = [8000, 16000, 32000]                     # TKDE grid
const EXP3_DS = [16, 64, 128]
const EXP3_TABLE = [(8000, 16), (32000, 16), (32000, 128)]  # tab_ops rows
const EXP3_K = 10

struct Exp3Config
    config::String   # "table" (also a tab_ops row) or "grid"; the figure uses both
    n::Int
    d::Int
    nops::Int        # inserts, then the same number of deletes
    runs::Int
    ncheck::Int      # brute-force-checked queries after every operation
end

function exp3_configs(smoke::Bool)
    ns, ds, table = smoke ? ([200, 300], [2, 8], [(200, 2), (300, 2), (300, 8)]) :
                            (EXP3_NS, EXP3_DS, EXP3_TABLE)
    nops, runs, ncheck = smoke ? (20, 3, 5) : (200, 3, 20)
    return [Exp3Config((n, d) in table ? "table" : "grid", n, d, nops, runs, ncheck) for d in ds for n in ns]
end

new_index3(d::Int) = HNNG.HNNGIndex(Euclidean(), d; T = Float32)
seconds_since(t0) = (time_ns() - t0) / 1e9

"""
    check_exact!(idx, P, alive, Qc, qi, ncheck; k = 10) -> qi

Answers `ncheck` queries (cycling through the columns of `Qc` after column `qi`) with
`HNNG.knn` and compares them with Float64 brute force over the live columns of `P`.
Throws on any difference beyond rtol 1e-5 or on a dead id. Not timed.
"""
function check_exact!(idx, P::Matrix{Float32}, alive::BitVector, Qc::Matrix{Float32}, qi::Int,
                      ncheck::Int; k::Int = EXP3_K)
    live = findall(alive)
    dist = Vector{Float64}(undef, length(live))
    kk = min(k, length(live))
    for _ in 1:ncheck
        qi = qi % size(Qc, 2) + 1
        q = Qc[:, qi]
        ids, ds = HNNG.knn(idx, q, k)
        for (t, i) in enumerate(live)
            dist[t] = dist64(view(P, :, i), q)
        end
        truth = collect(partialsort!(dist, 1:kk))
        ok = length(ds) == kk && all(isapprox.(ds, truth; rtol = 1e-5)) && all(alive[i] for i in ids)
        ok || error("Exp.3 exactness violated on query $qi: got $ds (ids $ids), expected $truth")
    end
    return qi
end

"Compile every timed path on a tiny index so that run 1 does not time compilation."
function exp3_warmup(d::Int)
    P = gaussian(60, d, SEED_EXP3_POINTS + d)
    idx = new_index3(d)
    HNNG.add!(idx, P[:, 1:50])
    for j in 51:60
        HNNG.add!(idx, P[:, j])
        HNNG.last_update_stats(idx)
    end
    for id in 1:10
        HNNG.delete!(idx, id)
    end
    HNNG.bulk_add!(new_index3(d), P)
    HNNG.knn(idx, P[:, 1], EXP3_K)
    HAS_HNSW && build_hnsw(HNSWjl(), P; nbuild = 50)
    return nothing
end

function exp3_hnng!(df::DataFrame, c::Exp3Config, P::Matrix{Float32}, Qc::Matrix{Float32}, run::Int)
    n, d = c.n, c.d
    base = (config = c.config, n = n, d = d, run = run, method = "hNNG")
    X = P[:, 1:n]
    GC.gc()
    idx = new_index3(d)
    t0 = time_ns()
    HNNG.add!(idx, X)
    t_stream = seconds_since(t0)
    push!(df, (; base..., op = "rebuild", seq = 1, latency_s = t_stream, nevals = HNNG.stats(idx).nevals,
               rewired = missing, top_level = missing, note = "streaming add! of n points"))
    bulk = new_index3(d)
    GC.gc()
    t0 = time_ns()
    HNNG.bulk_add!(bulk, X)
    t_bulk = seconds_since(t0)
    push!(df, (; base..., op = "rebuild-bulk", seq = 1, latency_s = t_bulk, nevals = HNNG.stats(bulk).nevals,
               rewired = missing, top_level = missing,
               note = "bulk_add! of n points; nevals = Gram pairs + re-ranks (not comparable with streaming nevals)"))
    bulk = nothing
    alive = falses(size(P, 2))
    alive[1:n] .= true
    qi = 0
    for j in 1:c.nops
        x = P[:, n + j]
        HNNG.reset_evals!(idx)
        GC.gc(false)        # outside the timed region: garbage from the untimed check is not collected inside the insert (A26)
        t0 = time_ns()
        id = HNNG.add!(idx, x)
        t = seconds_since(t0)
        id == n + j || error("HNNG gave id $id to column $(n + j)")
        rew, top = HNNG.last_update_stats(idx)
        push!(df, (; base..., op = "insert", seq = j, latency_s = t, nevals = HNNG.stats(idx).nevals,
                   rewired = rew, top_level = top, note = ""))
        alive[id] = true
        qi = check_exact!(idx, P, alive, Qc, qi, c.ncheck)
    end
    rng = StableRNG(SEED_EXP3_DELETE + d)       # same deletes in every run
    for j in 1:c.nops
        live = findall(alive)
        id = live[draw_index(rng, length(live))]
        HNNG.reset_evals!(idx)
        GC.gc(false)        # as for the insert loop (A26)
        t0 = time_ns()
        HNNG.delete!(idx, id)
        t = seconds_since(t0)
        rew, top = HNNG.last_update_stats(idx)
        alive[id] = false
        push!(df, (; base..., op = "delete", seq = j, latency_s = t, nevals = HNNG.stats(idx).nevals,
                   rewired = rew, top_level = top, note = ""))
        qi = check_exact!(idx, P, alive, Qc, qi, c.ncheck)
    end
    length(idx) == n || error("live count $(length(idx)) != $n after $(c.nops) inserts and deletes")
    return df
end

function exp3_baselines!(df::DataFrame, c::Exp3Config, P::Matrix{Float32}, run::Int)
    n, d = c.n, c.d
    base = (config = c.config, n = n, d = d, run = run)
    X = P[:, 1:n]
    for m in BenchMethod[KDTreeNN(), BallTreeNN()]
        _, t = build_time(m, X)
        push!(df, (; base..., method = method_name(m), op = "rebuild", seq = 1, latency_s = t,
                   nevals = missing, rewired = missing, top_level = missing, note = "full rebuild"))
    end
    push!(df, (; base..., method = "brute-BLAS", op = "update", seq = 1, latency_s = 0.0,
               nevals = missing, rewired = missing, top_level = missing, note = "no update cost"))
    HAS_HNSW || return df
    h = build_hnsw(HNSWjl(), P; nbuild = n)
    for j in 1:c.nops
        GC.gc(false)        # same treatment as the hNNG loops (A26)
        t0 = time_ns()
        hnsw_insert!(h, n + j)
        push!(df, (; base..., method = "HNSW", op = "insert", seq = j, latency_s = seconds_since(t0),
                   nevals = missing, rewired = missing, top_level = missing, note = ""))
    end
    push!(df, (; base..., method = "HNSW", op = "delete", seq = 1, latency_s = missing,
               nevals = missing, rewired = missing, top_level = missing, note = "mark only"))
    return df
end

function run_exp3(; smoke::Bool = is_smoke(), outdir::AbstractString = results_dir(smoke))
    check_protocol()
    df = empty_exp3()
    for c in exp3_configs(smoke)
        P = gaussian(c.n + c.nops, c.d, SEED_EXP3_POINTS + c.d)
        Qc = gaussian(2 * c.nops * c.ncheck, c.d, SEED_EXP3_CHECK + c.d)
        exp3_warmup(c.d)
        for run in 1:c.runs
            exp3_hnng!(df, c, P, Qc, run)
            exp3_baselines!(df, c, P, run)
        end
        @info "Exp.3" c.config c.n c.d
    end
    mkpath(outdir)
    CSV.write(joinpath(outdir, "exp3.csv"), df)
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_exp3()
end
