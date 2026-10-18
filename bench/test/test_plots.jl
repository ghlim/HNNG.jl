include(joinpath(@__DIR__, "setup.jl"))
using DataFrames
isdefined(Main, :write_plots) || include(joinpath(BENCH_DIR, "plots.jl"))

ispdf(p) = isfile(p) && filesize(p) > 1000 && open(io -> String(read(io, 4)), p) == "%PDF"

@testset "plots" begin
    exp1 = DataFrame(method = ["hNNG-incremental", "hNNG-bulk", "brute-BLAS", "KD-tree", "ball-tree", "HNSW", "HNSW"],
                     recall_ann = [1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.99],
                     qps = [5e4, 4.9e4, 9e3, 2e4, 1.5e4, 9e4, 5e4])
    methods2 = ["hNNG-bulk", "KD-tree", "ball-tree", "brute-BLAS"]
    exp2 = DataFrame(geometry = repeat(["gaussian", "blobs"]; inner = 8),
                     n = repeat([5000, 10000]; inner = 8),
                     d = repeat([2, 2, 2, 2, 8, 8, 8, 8], 2),
                     method = repeat(methods2, 4),
                     qps = collect(range(10.0, 1e4; length = 16)),
                     frac_evaluated = [m == "hNNG-bulk" ? 0.3 : missing for m in repeat(methods2, 4)])
    exp3 = empty_exp3()
    series = (("hNNG", "insert", 1e-3), ("hNNG", "delete", 2e-3), ("hNNG", "rebuild", 0.5),
              ("hNNG", "rebuild-bulk", 0.1), ("HNSW", "insert", 5e-4), ("KD-tree", "rebuild", 0.01))
    for n in (8000, 16000, 32000), d in (16, 64, 128), run in 1:3, (meth, op, t) in series
        cfg = (n, d) in ((8000, 16), (32000, 16), (32000, 128)) ? "table" : "grid"
        push!(exp3, (config = cfg, n = n, d = d, run = run, method = meth, op = op, seq = 1,
                     latency_s = t * (n / 8000) * (d / 16), nevals = missing, rewired = missing,
                     top_level = missing, note = ""))
    end
    @test n_label(8000) == "8k" && n_label(32000) == "32k" && n_label(300) == "300"
    @test speedup_label(0.46, 0.0011) == "rebuild/insert ≈ 418×"
    @test FIG_SINGLE == (215, 165) && FIG_RECALL == (215, 197) && FIG_DOUBLE == (450, 150) && FIG_FONT == 8
    @testset "recall_qps: exact methods as lines, HNSW frontier, no SimilaritySearch, 1-3-10 ticks" begin
        @test qps_label(300) == "300" && qps_label(3000.0) == "3k" && qps_label(1e5) == "100k" && qps_label(3e6) == "3M"
        # the real random-xs range (3,594 to 124,829 QPS): 3k ... 100k inside the padded limits
        @test qps_axis(3594.5, 124829.4) ==
              ((3594.5 / 1.25, 124829.4 * 1.25), ([3e3, 1e4, 3e4, 1e5], ["3k", "10k", "30k", "100k"]))
        # a narrow range with no 1-3-10 step inside the padding widens to the nearest steps
        @test qps_axis(4e4, 6e4) == ((3e4, 1e5), ([3e4, 1e5], ["30k", "100k"]))
        @test qps_axis(1e4, 1e4) == ((3e3, 3e4), ([3e3, 1e4, 3e4], ["3k", "10k", "30k"]))
        @test_throws ArgumentError qps_axis(0, 10)
        withss = vcat(exp1, DataFrame(method = ["SS-SearchGraph", "SS-Exhaustive"], recall_ann = [0.99, 1.0],
                                      qps = [1e5, 1.4e4]))
        fig, ax = recall_qps_figure(withss)
        plots = ax.scene.plots
        hl = filter(p -> p isa HLines, plots)
        @test [p.label[] for p in hl] == ["hNNG (incremental)", "hNNG (bulk)", "brute force", "KD-tree", "ball tree"]
        @test [p.linestyle[] for p in hl] == [:solid, :dash, :dot, :dashdot, :dashdotdot]
        @test only(p.label[] for p in plots if p isa Scatter || p isa ScatterLines) == "HNSW (ef sweep)"
        @test !any(p -> occursin(r"SearchGraph|SS|Similarity", string(p.label[])), plots)
        @test ax.yticks[] == qps_axis(9e3, 9e4)[2]
        @test fig.scene.viewport[].widths == [FIG_RECALL...]
    end
    @testset "qps_vs_dim panel titles use n_label" begin
        _, panels = qps_dim_figure(exp2)
        @test [ax.title[] for ax in panels] == ["Gaussian, n = 5k", "100 blobs, n = 10k"]
    end
    # the page size in pt, as poppler's pdfinfo reports it: 1 Makie unit = 1 pt (pt_per_unit = 1)
    page_pts(p) = (m = match(r"Page size:\s+([\d.]+) x ([\d.]+)", read(`pdfinfo $p`, String));
                   (parse(Float64, m[1]), parse(Float64, m[2])))
    mktempdir() do dir
        made = ("recall_qps" => plot_recall_qps(exp1, joinpath(dir, "recall_qps.pdf")),
                "dist_eval_vs_dim" => plot_dist_eval(exp2, joinpath(dir, "dist_eval_vs_dim.pdf")),
                "qps_vs_dim" => plot_qps_dim(exp2, joinpath(dir, "qps_vs_dim.pdf")),
                "ops_comparison" => plot_ops(exp3, joinpath(dir, "ops_comparison.pdf")))
        for (_, p) in made
            @test ispdf(p)
        end
        if Sys.which("pdfinfo") === nothing
            @info "pdfinfo not found; PDF page sizes not checked"
        else
            for (name, p) in made
                want = name == "ops_comparison" ? FIG_DOUBLE : name == "recall_qps" ? FIG_RECALL : FIG_SINGLE
                w, h = page_pts(p)
                @test abs(w - want[1]) <= 1 && abs(h - want[2]) <= 1
            end
        end
    end
end
