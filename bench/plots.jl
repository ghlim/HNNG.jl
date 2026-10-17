# The four benchmark figures as PDFs (CairoMakie), spec §11.
# Run: julia -t 1 --project=bench bench/plots.jl       (reads results_dir())
isdefined(Main, :BenchCore) || include(joinpath(@__DIR__, "src", "BenchCore.jl"))
using .BenchCore
using CairoMakie, CSV, DataFrames

# Fixed order, one colour + marker per method, identical in every figure.
const METHOD_ORDER = ["hNNG-incremental", "hNNG-bulk", "brute-BLAS", "KD-tree", "ball-tree",
                      "HNSW", "SS-SearchGraph", "SS-Exhaustive"]
const METHOD_STYLE = Dict(
    "hNNG-incremental" => (color = "#2a78d6", marker = :circle,    label = "hNNG (incremental build)"),
    "hNNG-bulk"        => (color = "#eb6834", marker = :rect,      label = "hNNG (bulk build)"),
    "brute-BLAS"       => (color = "#1baf7a", marker = :diamond,   label = "brute force (BLAS)"),
    "KD-tree"          => (color = "#eda100", marker = :utriangle, label = "KD-tree"),
    "ball-tree"        => (color = "#e87ba4", marker = :dtriangle, label = "ball tree"),
    "HNSW"             => (color = "#008300", marker = :xcross,    label = "HNSW (ef sweep)"),
    "SS-SearchGraph"   => (color = "#6250d6", marker = :star5,     label = "SearchGraph"),
    "SS-Exhaustive"    => (color = "#e34948", marker = :hexagon,   label = "SimilaritySearch exhaustive"),
)
const GEOMETRY_STYLE = Dict(
    "gaussian" => (color = "#2a78d6", marker = :circle, label = "Gaussian"),
    "blobs"    => (color = "#eb6834", marker = :rect,   label = "100 blobs"),
)
# ops_comparison lines (hNNG only), fixed order: (op, label, colour, marker, line style).
# The three operations are the entities here, so they take categorical slots 1-3.
const OPS_LINES = [
    ("rebuild", "full rebuild per op", "#2a78d6", :rect,    :dot),
    ("insert",  "incremental insert",  "#eb6834", :circle,  :solid),
    ("delete",  "incremental delete",  "#1baf7a", :diamond, :dash),
]

# Final print sizes in pt: with `pt_per_unit = 1` one Makie unit is one PDF point, so the text
# is 8 pt on the page. sn-jnl `iicol`: column 215 pt (76 mm), double column 453 pt (160 mm).
const FIG_SINGLE = (215, 165)    # one column
const FIG_DOUBLE = (450, 150)    # ops_comparison: figure* across both columns
const FIG_FONT = 8
const FIG_PAD = 4                # figure padding (pt); the Makie default of 16 wastes a print column
const MS = 5                     # marker size (pt)
const LW = 1                     # line width (pt)
# Compact legend for a 215 pt column.
const LEGEND_STYLE = (framevisible = false, labelsize = 7, patchsize = (14, 8), padding = (2, 2, 2, 2),
                rowgap = 0, colgap = 6)

function save_pdf(fig, path::AbstractString)
    mkpath(dirname(path))
    save(path, fig; pt_per_unit = 1)
    return path
end
"Ticks at the dimensions `ds`, optionally only every `step`-th one (a half-column panel fits four labels)."
dim_ticks(ds; step::Int = 1) = (ds[1:step:end], string.(ds[1:step:end]))

"Recall vs QPS on random-xs: HNSW traces a frontier, exact methods are points at recall 1."
function plot_recall_qps(exp1::AbstractDataFrame, path::AbstractString)
    fig = Figure(size = FIG_SINGLE, fontsize = FIG_FONT, figure_padding = FIG_PAD)
    ax = Axis(fig[1, 1]; xlabel = "recall (ann-benchmarks rule, k = 10)", ylabel = "queries per second",
              yscale = log10)
    for m in METHOD_ORDER
        r = sort(exp1[exp1.method .== m, :], :recall_ann)
        nrow(r) == 0 && continue
        st = METHOD_STYLE[m]
        x, y = Float64.(r.recall_ann), Float64.(r.qps)
        if nrow(r) > 1
            scatterlines!(ax, x, y; color = st.color, marker = st.marker, markersize = MS, linewidth = LW, label = st.label)
        else
            scatter!(ax, x, y; color = st.color, marker = st.marker, markersize = MS + 2, label = st.label)
        end
    end
    axislegend(ax; position = :lb, LEGEND_STYLE...)
    return save_pdf(fig, path)
end

"ρ(d): fraction of points hNNG evaluates per query, both Exp.2 geometries."
function plot_dist_eval(exp2::AbstractDataFrame, path::AbstractString)
    ds = sort(unique(Int.(exp2.d)))
    fig = Figure(size = FIG_SINGLE, fontsize = FIG_FONT, figure_padding = FIG_PAD)
    ax = Axis(fig[1, 1]; xlabel = "dimension d", ylabel = "fraction of points evaluated ρ",
              xscale = log2, xticks = dim_ticks(ds), limits = (nothing, (0, 1.05)))
    for g in ("gaussian", "blobs")
        r = sort(exp2[(exp2.geometry .== g) .& (exp2.method .== "hNNG-bulk"), :], :d)
        nrow(r) == 0 && continue
        st = GEOMETRY_STYLE[g]
        scatterlines!(ax, Int.(r.d), Float64.(r.frac_evaluated); color = st.color, marker = st.marker,
                      markersize = MS, linewidth = LW, label = "$(st.label), n = $(r.n[1])")
    end
    axislegend(ax; position = :rb, LEGEND_STYLE...)
    return save_pdf(fig, path)
end

"QPS vs d for the exact methods; one panel per geometry (half a column each), shared legend below."
function plot_qps_dim(exp2::AbstractDataFrame, path::AbstractString)
    ds = sort(unique(Int.(exp2.d)))
    fig = Figure(size = FIG_SINGLE, fontsize = FIG_FONT, figure_padding = FIG_PAD)
    panels = Axis[]
    for g in ("gaussian", "blobs")
        sub = exp2[exp2.geometry .== g, :]
        nrow(sub) == 0 && continue
        ax = Axis(fig[1, length(panels) + 1]; title = "$(GEOMETRY_STYLE[g].label), n = $(sub.n[1])",
                  xlabel = "dimension d", ylabel = isempty(panels) ? "queries per second" : "",
                  xscale = log2, yscale = log10, xticks = dim_ticks(ds; step = 2))
        push!(panels, ax)
        for m in METHOD_ORDER
            r = sort(sub[sub.method .== m, :], :d)
            nrow(r) == 0 && continue
            st = METHOD_STYLE[m]
            scatterlines!(ax, Int.(r.d), Float64.(r.qps); color = st.color, marker = st.marker,
                          markersize = MS, linewidth = LW, label = st.label)
        end
    end
    isempty(panels) && error("exp2 has no rows to plot")
    length(panels) > 1 && linkyaxes!(panels...)
    Legend(fig[2, 1:length(panels)], panels[1]; orientation = :horizontal, nbanks = 2, LEGEND_STYLE...)
    rowgap!(fig.layout, 1, 4)
    return save_pdf(fig, path)
end

"Tick label for an index size: 8000 -> \"8k\", 300 -> \"300\"."
n_label(n::Integer) = n % 1000 == 0 ? "$(n ÷ 1000)k" : string(n)

"Panel annotation: rebuild time over insert time at the largest n."
speedup_label(rebuild_s::Real, insert_s::Real) = "rebuild/insert ≈ $(round(Int, rebuild_s / insert_s))×"

"ops_comparison: one panel per d, x = n (log), y = ms per operation (log), shared legend."
function plot_ops(exp3::AbstractDataFrame, path::AbstractString)
    h = exp3[exp3.method .== "hNNG", :]
    ds = sort(unique(Int.(h.d)))
    isempty(ds) && error("exp3 has no hNNG rows to plot")
    fig = Figure(size = FIG_DOUBLE, fontsize = FIG_FONT, figure_padding = FIG_PAD)
    panels = Axis[]
    for (col, d) in enumerate(ds)
        sub = h[h.d .== d, :]
        ns = sort(unique(Int.(sub.n)))
        ax = Axis(fig[1, col]; title = "d = $d", xlabel = "n",
                  ylabel = col == 1 ? "time per operation (ms)" : "",
                  xscale = log10, yscale = log10, xticks = (ns, n_label.(ns)))
        push!(panels, ax)
        lat = Dict(op => op_latency(sub, nothing, "hNNG", op) for (op, _, _, _, _) in OPS_LINES)
        for (op, label, color, marker, ls) in OPS_LINES
            s = lat[op]
            nrow(s) == 0 && continue
            scatterlines!(ax, Float64.(s.n), 1000 .* s.latency_s; color = color, marker = marker,
                          linestyle = ls, markersize = MS, linewidth = LW, label = label)
        end
        reb, ins = lat["rebuild"], lat["insert"]
        if nrow(reb) > 0 && nrow(ins) > 0
            text!(ax, Float64(reb.n[end]), 1000 * reb.latency_s[end];
                  text = speedup_label(reb.latency_s[end], ins.latency_s[end]),
                  align = (:right, :top), offset = (-3, -3), fontsize = 7)   # below the point: above it the axis clips the text
        end
    end
    length(panels) > 1 && linkyaxes!(panels...)
    Legend(fig[2, 1:length(panels)], panels[1]; orientation = :horizontal, LEGEND_STYLE...)
    rowgap!(fig.layout, 1, 4)
    return save_pdf(fig, path)
end

function write_plots(resdir::AbstractString = results_dir())
    rd(f) = CSV.read(joinpath(resdir, f), DataFrame)
    figdir = joinpath(resdir, "fig")
    plot_recall_qps(rd("exp1.csv"), joinpath(figdir, "recall_qps.pdf"))
    exp2 = rd("exp2.csv")
    plot_dist_eval(exp2, joinpath(figdir, "dist_eval_vs_dim.pdf"))
    plot_qps_dim(exp2, joinpath(figdir, "qps_vs_dim.pdf"))
    plot_ops(rd("exp3.csv"), joinpath(figdir, "ops_comparison.pdf"))
    return figdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    write_plots()
end
