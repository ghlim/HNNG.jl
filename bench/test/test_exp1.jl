include(joinpath(@__DIR__, "setup.jl"))
using CSV, DataFrames
isdefined(Main, :run_exp1) || include(joinpath(BENCH_DIR, "exp1.jl"))

@testset "Exp.1 / Exp.1b smoke" begin
    mktempdir() do out
        run_exp1(; outdir = out)
        @test !isfile(joinpath(out, "exp1b.csv"))       # run_exp1 is random-xs only
        run_exp1b(; outdir = out)
        core = ["hNNG-incremental", "hNNG-bulk", "brute-BLAS", "KD-tree", "ball-tree"]
        for (file, name) in (("exp1.csv", "random-xs-20-euclidean"), ("exp1b.csv", "fashion-mnist-784-euclidean"))
            df = CSV.read(joinpath(out, file), DataFrame)
            @test names(df) == SCHEMA_EXP1
            @test all(df.dataset .== name)
            @test all(df.n .== 300) && all(df.nq .== 20) && all(df.k .== 10)
            @test issubset(core, df.method)
            @test count(==("HNSW"), df.method) == (HAS_HNSW ? length(EXP1_EFS_SMOKE) : 0)
            HAS_HNSW && @test sort(string.(df[df.method .== "HNSW", :param])) == ["ef=10", "ef=40"]
            @test all(df.qps .> 0) && all(df.build_s .> 0)
            exact = df[in.(df.method, Ref(core)), :]
            @test all(exact.recall_ann .== 1.0)
            h = df[startswith.(df.method, "hNNG"), :]
            @test all(0 .< h.frac_evaluated .<= 1) && all(h.nlevels .>= 1)
            @test all(occursin.(r"^\d+(;\d+)*$", string.(h.nclusters)))
        end
    end
end
