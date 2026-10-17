include(joinpath(@__DIR__, "setup.jl"))
using CSV, DataFrames
using Distances: Euclidean
import HNNG
isdefined(Main, :run_exp3) || include(joinpath(BENCH_DIR, "exp3.jl"))

@testset "Exp.3" begin
    @testset "op_latency: median over runs of each run's mean" begin
        df = empty_exp3()
        for (run, lats) in enumerate(([1.0, 3.0], [5.0, 7.0], [0.5, 1.5]))   # run means 2, 6, 1
            for (seq, t) in enumerate(lats)
                push!(df, (config = "table", n = 100, d = 4, run = run, method = "hNNG", op = "insert",
                           seq = seq, latency_s = t, nevals = 1, rewired = 0, top_level = 0, note = ""))
            end
        end
        push!(df, (config = "table", n = 100, d = 4, run = 1, method = "HNSW", op = "delete", seq = 1,
                   latency_s = missing, nevals = missing, rewired = missing, top_level = missing, note = "mark only"))
        s = op_latency(df, "table", "hNNG", "insert")
        @test nrow(s) == 1 && s.n[1] == 100 && s.d[1] == 4 && s.latency_s[1] == 2.0
        @test nrow(op_latency(df, "table", "HNSW", "delete")) == 0
        @test nrow(op_latency(df, "grid", "hNNG", "insert")) == 0
        @test nrow(op_latency(df, nothing, "hNNG", "insert")) == 1
    end

    @testset "configurations: the TKDE grid, table rows at its corners" begin
        full = exp3_configs(false)
        @test length(full) == 9
        @test sort([(c.n, c.d) for c in full]) == sort([(n, d) for n in (8000, 16000, 32000) for d in (16, 64, 128)])
        @test sort([(c.n, c.d) for c in full if c.config == "table"]) == [(8000, 16), (32000, 16), (32000, 128)]
        @test all(c -> (c.nops, c.runs, c.ncheck) == (200, 3, 20), full)
        smoke = exp3_configs(true)
        @test length(smoke) == 4
        @test sort([(c.n, c.d) for c in smoke if c.config == "table"]) == [(200, 2), (300, 2), (300, 8)]
    end

    @testset "exactness check catches a wrong answer" begin
        P = gaussian(40, 2, 1)
        idx = HNNG.HNNGIndex(Euclidean(), 2; T = Float32)
        HNNG.add!(idx, P[:, 1:30])
        alive = falses(40)
        alive[1:31] .= true          # claims column 31 is live, but it was never inserted
        Qc = P[:, 31:31]             # a query on column 31: its true nearest distance is 0
        @test_throws ErrorException check_exact!(idx, P, alive, Qc, 0, 1)
        alive[31] = false
        @test check_exact!(idx, P, alive, Qc, 0, 1) == 1
    end

    @testset "smoke run" begin
        mktempdir() do out
            run_exp3(; outdir = out)
            df = CSV.read(joinpath(out, "exp3.csv"), DataFrame)
            @test names(df) == SCHEMA_EXP3
            h = df[df.method .== "hNNG", :]
            for c in exp3_configs(true), run in 1:c.runs
                g = h[(h.config .== c.config) .& (h.n .== c.n) .& (h.d .== c.d) .& (h.run .== run), :]
                @test count(==("insert"), g.op) == c.nops
                @test count(==("delete"), g.op) == c.nops
                @test count(==("rebuild"), g.op) == 1 && count(==("rebuild-bulk"), g.op) == 1
            end
            ins = h[h.op .== "insert", :]
            del = h[h.op .== "delete", :]
            @test all(ins.latency_s .> 0) && all(ins.nevals .> 0) && all(del.nevals .>= 0)
            @test all(vcat(ins.rewired, del.rewired) .>= 0) && all(vcat(ins.top_level, del.top_level) .>= 0)
            hd = df[(df.method .== "HNSW") .& (df.op .== "delete"), :]
            @test nrow(hd) == (HAS_HNSW ? 12 : 0)
            @test all(hd.note .== "mark only") && all(ismissing, hd.latency_s)
            @test nrow(op_latency(df, "table", "hNNG", "insert")) == 3
            @test nrow(op_latency(df, "grid", "hNNG", "insert")) == 1
            @test nrow(op_latency(df, nothing, "hNNG", "rebuild")) == 4
            @test nrow(op_latency(df, nothing, "KD-tree", "rebuild")) == 4
            @test nrow(op_latency(df, nothing, "HNSW", "insert")) == (HAS_HNSW ? 4 : 0)
        end
    end
end
