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
const FIG_RECALL = (215, 197)    # recall_qps: one column, legend below the axis
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

# recall_qps only: the exact methods are horizontal lines at their QPS (they all sit at recall 1,
# where markers hid each other), one line style each; HNSW is the ef frontier. SimilaritySearch
# rows stay in the CSV but are not drawn (the manuscript does not introduce the package).
const RECALL_EXACT = ["hNNG-incremental", "hNNG-bulk", "brute-BLAS", "KD-tree", "ball-tree"]
const RECALL_LINESTYLE = Dict("hNNG-incremental" => :solid, "hNNG-bulk" => :dash, "brute-BLAS" => :dot,
                              "KD-tree" => :dashdot, "ball-tree" => :dashdotdot)
const RECALL_LABEL = Dict("hNNG-incremental" => "hNNG (incremental)", "hNNG-bulk" => "hNNG (bulk)",
                          "brute-BLAS" => "brute force", "KD-tree" => "KD-tree", "ball-tree" => "ball tree",
                          "HNSW" => "HNSW (ef sweep)")

"QPS tick label: 300 -> \"300\", 3000 -> \"3k\", 100000 -> \"100k\", 3e6 -> \"3M\"."
qps_label(v::Real) = v < 1e3 ? string(round(Int, v)) :
                     v < 1e6 ? "$(round(Int, v / 1e3))k" : "$(round(Int, v / 1e6))M"

"""
    qps_axis(lo, hi) -> (limits, (values, labels))

Log y axis for QPS data in [lo, hi]: limits pad the data by a factor 1.25 each side, ticks are
the 1-3-10 steps inside the limits, labelled like "3k", "10k", "30k", "100k" (no fractional
exponents). When fewer than two steps fall inside, the limits widen to the nearest steps
strictly below lo and above hi, so the ticks always span the data.
"""
function qps_axis(lo::Real, hi::Real)
    0 < lo <= hi || throw(ArgumentError("qps_axis needs 0 < lo <= hi, got $lo, $hi"))
    steps = [m * 10.0^e for e in floor(Int, log10(lo)) - 2:ceil(Int, log10(hi)) + 1 for m in (1, 3)]
    limits = (lo / 1.25, hi * 1.25)
    vals = filter(v -> limits[1] <= v <= limits[2], steps)
    if length(vals) < 2
        limits = (steps[findlast(<(lo), steps)], steps[findfirst(>(hi), steps)])
        vals = filter(v -> limits[1] <= v <= limits[2], steps)
    end
    return limits, (vals, qps_label.(vals))
end

"""
    recall_qps_figure(exp1) -> (fig, ax)

Recall vs QPS: HNSW traces its ef frontier; each exact method is a horizontal line at its QPS
(its recall is 1). Legend below the axis, explicit 1-3-10 QPS ticks (`qps_axis`).
"""
function recall_qps_figure(exp1::AbstractDataFrame)
    fig = Figure(size = FIG_RECALL, fontsize = FIG_FONT, figure_padding = FIG_PAD)
    shown = exp1[in.(exp1.method, Ref(vcat(RECALL_EXACT, "HNSW"))), :]
    nrow(shown) == 0 && error("exp1 has no hNNG, baseline or HNSW rows to plot")
    qps = Float64.(shown.qps)
    ylim, yticks = qps_axis(minimum(qps), maximum(qps))
    ax = Axis(fig[1, 1]; xlabel = "recall (ann-benchmarks rule, k = 10)", ylabel = "queries per second",
              yscale = log10, yticks = yticks, limits = (nothing, ylim))
    for m in RECALL_EXACT
        r = shown[shown.method .== m, :]
        nrow(r) == 0 && continue
        hlines!(ax, Float64.(r.qps); color = METHOD_STYLE[m].color, linestyle = RECALL_LINESTYLE[m],
                linewidth = LW, label = RECALL_LABEL[m])
    end
    r = sort(shown[shown.method .== "HNSW", :], :recall_ann)
    if nrow(r) > 0
        st = METHOD_STYLE["HNSW"]
        scatterlines!(ax, Float64.(r.recall_ann), Float64.(r.qps); color = st.color, marker = st.marker,
                      markersize = MS, linewidth = LW, label = RECALL_LABEL["HNSW"])
    else
        xlims!(ax, 0.95, 1.005)          # horizontal lines alone set no x range
    end
    Legend(fig[2, 1], ax; orientation = :horizontal, nbanks = 3, LEGEND_STYLE...)
    rowgap!(fig.layout, 1, 4)
    return fig, ax
end

"Recall vs QPS on random-xs (see `recall_qps_figure`), saved as a PDF of size FIG_RECALL."
plot_recall_qps(exp1::AbstractDataFrame, path::AbstractString) = save_pdf(first(recall_qps_figure(exp1)), path)

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
                      markersize = MS, linewidth = LW, label = "$(st.label), n = $(n_label(r.n[1]))")
    end
    axislegend(ax; position = :rb, LEGEND_STYLE...)
    return save_pdf(fig, path)
end

"QPS vs d for the exact methods; one panel per geometry (half a column each), shared legend below. Returns (fig, panels)."
function qps_dim_figure(exp2::AbstractDataFrame)
    ds = sort(unique(Int.(exp2.d)))
    fig = Figure(size = FIG_SINGLE, fontsize = FIG_FONT, figure_padding = FIG_PAD)
    panels = Axis[]
    for g in ("gaussian", "blobs")
        sub = exp2[exp2.geometry .== g, :]
        nrow(sub) == 0 && continue
        ax = Axis(fig[1, length(panels) + 1]; title = "$(GEOMETRY_STYLE[g].label), n = $(n_label(sub.n[1]))",
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
    return fig, panels
end
plot_qps_dim(exp2::AbstractDataFrame, path::AbstractString) = save_pdf(first(qps_dim_figure(exp2)), path)

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
