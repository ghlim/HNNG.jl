include(joinpath(@__DIR__, "setup.jl"))
using CSV, DataFrames
isdefined(Main, :run_exp2) || include(joinpath(BENCH_DIR, "exp2.jl"))

@testset "Exp.2 smoke" begin
    mktempdir() do out
        run_exp2(; outdir = out)
        df = CSV.read(joinpath(out, "exp2.csv"), DataFrame)
        @test names(df) == SCHEMA_EXP2
        @test sort(unique(string.(df.geometry))) == ["blobs", "gaussian"]
        @test sort(unique(df.d)) == [2, 8]
        @test nrow(df) == 2 * 2 * 4
        @test sort(unique(string.(df.method))) == ["KD-tree", "ball-tree", "brute-BLAS", "hNNG-bulk"]
        @test all(df.recall_ann .== 1.0) && all(df.qps .> 0) && all(df.n .== 300) && all(df.nq .== 20)
        h = df[df.method .== "hNNG-bulk", :]
        @test nrow(h) == 4 && all(0 .< h.frac_evaluated .<= 1) && all(h.nlevels .>= 1)
        @test all(ismissing, df[df.method .!= "hNNG-bulk", :frac_evaluated])
    end
end
