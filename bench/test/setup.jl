# Shared preamble: every bench test file starts with include("setup.jl").
using Test
ENV["BENCH_SMOKE"] = "1"   # tests never run the full-size experiments
for k in ("BENCH_ONLY", "BENCH_FORCE", "BENCH_RESULTS")   # a stray export must not change what the tests do
    delete!(ENV, k)
end
isdefined(Main, :BenchCore) || include(joinpath(@__DIR__, "..", "src", "BenchCore.jl"))
using .BenchCore
