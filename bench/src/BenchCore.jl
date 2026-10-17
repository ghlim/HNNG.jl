# Shared code for the HNNG.jl benchmark harness (spec §11).
module BenchCore

using LinearAlgebra, Statistics, Printf, Dates, Random
using HDF5, StableRNGs, CSV, DataFrames, SHA
using Distances: Euclidean
import HNNG
import Pkg
import NearestNeighbors as NN

# HNSW.jl and SimilaritySearch are optional: their rows are skipped when they do not load
# (spec §13: SearchGraph then takes HNSW's table rows). `import`, not `using`, keeps their
# exported names out of this module.
try
    import HNSW
catch err
    @warn "HNSW.jl did not load; HNSW rows will be skipped" exception = err
end
try
    import SimilaritySearch
catch err
    @warn "SimilaritySearch did not load; SS-* rows will be skipped" exception = err
end
const HAS_HNSW = isdefined(@__MODULE__, :HNSW)
const HAS_SS = isdefined(@__MODULE__, :SimilaritySearch)

export BENCH_DIR, HAS_HNSW, HAS_SS, is_smoke, results_dir,
       SCHEMA_EXP1, SCHEMA_EXP2, SCHEMA_EXP3, empty_exp1, empty_exp2, empty_exp3

const BENCH_DIR = dirname(@__DIR__)   # .../HNNG.jl/bench, no trailing slash

"BENCH_SMOKE=1 shrinks every experiment to tiny sizes."
is_smoke() = get(ENV, "BENCH_SMOKE", "0") == "1"

"Where CSV, tex, fig and env.txt land. Smoke runs never write into bench/results."
results_dir(smoke::Bool = is_smoke()) =
    get(ENV, "BENCH_RESULTS", joinpath(BENCH_DIR, smoke ? "results-smoke" : "results"))

# Fixed CSV schemas: the column order here is the column order of the files.
empty_exp1() = DataFrame(dataset = String[], n = Int[], d = Int[], nq = Int[], k = Int[],
    method = String[], param = String[], build_s = Float64[], qps = Float64[],
    recall_ann = Float64[], recall_id = Float64[],
    frac_evaluated = Union{Missing,Float64}[], nlevels = Union{Missing,Int}[],
    nclusters = Union{Missing,String}[])
empty_exp2() = DataFrame(geometry = String[], n = Int[], d = Int[], nq = Int[], k = Int[],
    method = String[], build_s = Float64[], qps = Float64[], recall_ann = Float64[],
    frac_evaluated = Union{Missing,Float64}[], nlevels = Union{Missing,Int}[])
empty_exp3() = DataFrame(config = String[], n = Int[], d = Int[], run = Int[],
    method = String[], op = String[], seq = Int[], latency_s = Union{Missing,Float64}[],
    nevals = Union{Missing,Int}[], rewired = Union{Missing,Int}[],
    top_level = Union{Missing,Int}[], note = String[])
const SCHEMA_EXP1 = names(empty_exp1())
const SCHEMA_EXP2 = names(empty_exp2())
const SCHEMA_EXP3 = names(empty_exp3())

export op_latency

"""
    op_latency(exp3, config, method, op) -> DataFrame(n, d, latency_s)

Per (n, d) of `config` (`nothing` = every configuration): the median over runs of each
run's mean latency of `op` by `method` (spec assumption A9). Rows with a missing latency
are ignored; rows sorted by (n, d).
"""
function op_latency(exp3::AbstractDataFrame, config::Union{Nothing,AbstractString},
                    method::AbstractString, op::AbstractString)
    out = DataFrame(n = Int[], d = Int[], latency_s = Float64[])
    inconfig = config === nothing ? trues(nrow(exp3)) : (exp3.config .== config)
    sel = exp3[inconfig .& (exp3.method .== method) .& (exp3.op .== op), :]
    nrow(sel) == 0 && return out
    for g in groupby(sel, [:n, :d]; sort = true)
        per_run = Float64[]
        for r in groupby(g, :run; sort = true)
            vals = collect(skipmissing(r.latency_s))
            isempty(vals) || push!(per_run, mean(vals))
        end
        isempty(per_run) || push!(out, (n = g.n[1], d = g.d[1], latency_s = median(per_run)))
    end
    return out
end

include("datasets.jl")
include("recall.jl")
include("timing.jl")
include("baselines.jl")

end # module
