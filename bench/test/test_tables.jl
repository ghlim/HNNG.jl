include(joinpath(@__DIR__, "setup.jl"))
using DataFrames
isdefined(Main, :write_tables) || include(joinpath(BENCH_DIR, "tables.jl"))

bs(s) = s * "\\\\"   # append the LaTeX row end \\ (a raw string cannot end in a backslash)

@testset "tables" begin
    @testset "formatters" begin
        @test fmt_int(51679.4) == "51{,}679"
        @test fmt_int(635) == "635"
        @test fmt_int(1000) == "1{,}000"
        @test fmt_int(1234567) == "1{,}234{,}567"
        @test fmt_qps(9171.2) == "9{,}171"
        @test fmt_qps(59.74) == "59.7"
        @test fmt_sig3(552.4) == "552"
        @test fmt_sig3(43.61) == "43.6"
        @test fmt_sig3(0.0712) == "0.0712"
        @test fmt_sig3(0.46) == "0.460"
        @test fmt_sig3(1234.0) == "1{,}234"
        @test fmt_ms(0.0011) == raw"1.10\,ms"
        @test fmt_s(30.8) == raw"30.8\,s"
    end

    @testset "method table: TKDE layout, spec row set and order" begin
        df = DataFrame(
            method = ["HNSW", "ball-tree", "hNNG-bulk", "brute-BLAS", "KD-tree", "hNNG-incremental", "HNSW", "HNSW"],
            recall_ann = [0.82, 1.0, 1.0, 1.0, 1.0, 1.0, 0.995, 1.0],
            qps = [92000.0, 15000.0, 49256.0, 9171.4, 20000.0, 51679.2, 60000.0, 54000.0],
            build_s = [1.234, 0.0123, 0.0712, 0.00213, 0.0105, 0.583, 1.234, 1.234])
        expected = join([
            raw"\begin{tabular}{lccc}", raw"\hline", bs("Method & Recall & QPS & Build (s)"), raw"\hline",
            bs("hNNG (incremental build) & 1.000 & 51{,}679 & 0.583"),
            bs("hNNG (bulk build) & 1.000 & 49{,}256 & 0.0712"),
            bs("brute force (BLAS) & 1.000 & 9{,}171 & 0.00213"),
            bs("KD-tree & 1.000 & 20{,}000 & 0.0105"),
            bs("ball tree & 1.000 & 15{,}000 & 0.0123"),
            bs("HNSW (max recall attained) & 1.000 & 54{,}000 & 1.23"),
            bs("HNSW (sweet spot) & 0.995 & 60{,}000 & 1.23"),
            raw"\hline", raw"\end{tabular}"], "\n") * "\n"
        @test method_table_tex(df; recall_digits = 3) == expected
    end

    @testset "sweet spot falls back to max recall when no ef reaches 0.99" begin
        df = DataFrame(method = ["HNSW", "HNSW"], recall_ann = [0.5, 0.8], qps = [100.0, 50.0], build_s = [1.0, 1.0])
        p = hnsw_picks(df)
        @test p.max.recall_ann == 0.8 && p.sweet.recall_ann == 0.8
        @test occursin(bs("HNSW (sweet spot) & 0.80000 & 50.0 & 1.00"), method_table_tex(df; recall_digits = 5))
    end

    @testset "SearchGraph takes the HNSW place when HNSW is absent" begin
        df = DataFrame(method = ["hNNG-bulk", "SS-SearchGraph"], recall_ann = [1.0, 0.97],
                       qps = [10.0, 2000.0], build_s = [0.5, 2.0])
        tex = method_table_tex(df; recall_digits = 3)
        @test occursin(bs("SearchGraph (SimilaritySearch.jl) & 0.970 & 2{,}000 & 2.00"), tex)
        @test !occursin("HNSW", tex)
        @test hnsw_picks(df) === nothing
    end

    @testset "tab_ops: row format and median over runs" begin
        @test ops_row(8000, 16, 0.0011, 0.0016, 0.46) ==
              bs(raw"8{,}000 ($16$) & 1.10\,ms & 1.60\,ms & 0.460\,s \;\;($\sim$418$\times$)")
        df = empty_exp3()
        for run in 1:3, (op, t) in (("insert", [0.0010, 0.0011, 0.0012][run]), ("delete", 0.0016),
                                    ("rebuild", 0.46), ("rebuild-bulk", 0.05))
            push!(df, (config = "table", n = 8000, d = 16, run = run, method = "hNNG", op = op, seq = 1,
                       latency_s = t, nevals = 1, rewired = 2, top_level = 1, note = ""))
        end
        lines = split(chomp(ops_tex(df)), "\n")
        @test lines[1] == raw"\begin{tabular}{lccc}"
        @test lines[3] == bs(raw"$n$ ($d$) & insert/op & delete/op & rebuild/op \;(speedup)")
        @test lines[5] == ops_row(8000, 16, 0.0011, 0.0016, 0.46)
        @test lines[end] == raw"\end{tabular}"
    end

    @testset "macros.tex" begin
        exp1 = DataFrame(method = ["hNNG-incremental", "hNNG-bulk", "brute-BLAS", "HNSW", "HNSW"],
                         recall_ann = [1.0, 1.0, 1.0, 0.82, 1.0],
                         qps = [51679.0, 49256.0, 9171.0, 92000.0, 54000.0],
                         build_s = [0.58, 0.07, 0.002, 1.0, 1.0])
        exp1b = DataFrame(method = ["hNNG-incremental", "hNNG-bulk", "brute-BLAS", "HNSW", "HNSW"],
                          recall_ann = [1.0, 1.0, 1.0, 0.99996, 0.994],
                          qps = [70.0, 66.0, 59.7, 635.0, 5245.0],
                          build_s = [552.0, 43.6, 0.1, 6.3, 6.3])
        exp2 = DataFrame(geometry = ["gaussian", "gaussian"], method = ["hNNG-bulk", "hNNG-bulk"],
                         d = [2, 8], frac_evaluated = [0.024, 0.642])
        exp3 = empty_exp3()
        for (op, t, rw) in (("insert", 0.00586, 3), ("delete", 0.00859, 5),
                            ("rebuild", 30.8, missing), ("rebuild-bulk", 2.3, missing))
            push!(exp3, (config = "table", n = 32000, d = 128, run = 1, method = "hNNG", op = op, seq = 1,
                         latency_s = t, nevals = 1, rewired = rw, top_level = 1, note = ""))
        end
        tex = macros_tex(exp1, exp1b, exp2, exp3)
        has(name, val) = occursin("\\newcommand{\\$name}{$val}\n", tex)
        @test has("rhoDtwo", "0.024") && has("rhoDeight", "0.642")
        @test has("rhoDsixteen", "n/a") && has("rhoDthirtytwo", "n/a")
        @test has("pruneDtwo", "97.6")
        @test has("hnngQpsRandomXs", "51{,}679") && has("hnngBulkQpsRandomXs", "49{,}256")
        @test has("bruteQpsRandomXs", "9{,}171") && has("hnngOverBruteRandomXs", "5.6")
        @test has("hnswPeakQpsRandomXs", "92{,}000") && has("hnswPeakRecallRandomXs", "0.82")
        @test has("hnswExactQpsRandomXs", "54{,}000")
        @test has("hnngQpsFashion", "70.0") && has("hnngBulkQpsFashion", "66.0") && has("bruteQpsFashion", "59.7")
        @test has("hnswMaxRecallFashion", "0.99996") && has("hnswMaxRecallQpsFashion", "635")
        @test has("hnswSweetRecallFashion", "0.994") && has("hnswSweetQpsFashion", "5{,}245")
        @test has("hnngIncBuildFashion", "552") && has("hnngBulkBuildFashion", "43.6")
        @test has("bulkBuildSpeedupFashion", "12.7")
        @test has("rewireMean", "4.00") && has("rewireMax", "5")
        @test has("insertMsLargest", "5.86") && has("deleteMsLargest", "8.59")
        @test has("rebuildSLargest", "30.8") && has("speedupLargest", "5{,}256")
        @test has("speedupBulkLargest", "392") && has("bulkRebuildSLargest", "2.30")
        @test count("\\newcommand", tex) == 30
        for name in ("rhoDtwo", "rhoDeight", "rhoDsixteen", "rhoDthirtytwo", "rewireMean", "rewireMax",
                     "hnngQpsRandomXs", "bruteQpsRandomXs", "hnngBulkBuildFashion", "hnngIncBuildFashion",
                     "insertMsLargest", "rebuildSLargest", "bulkRebuildSLargest")   # used by the manuscript
            @test occursin("\\newcommand{\\$name}{", tex)
        end
    end
end
