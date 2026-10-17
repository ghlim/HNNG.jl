# LaTeX tables and \newcommand macros from the experiment CSVs (spec §11).
# Run: julia -t 1 --project=bench bench/tables.jl      (reads results_dir())
isdefined(Main, :BenchCore) || include(joinpath(@__DIR__, "src", "BenchCore.jl"))
using .BenchCore
using CSV, DataFrames, Printf, Statistics

const METHOD_ROWS = [
    ("hNNG-incremental", "hNNG (incremental build)"),
    ("hNNG-bulk", "hNNG (bulk build)"),
    ("brute-BLAS", "brute force (BLAS)"),
    ("KD-tree", "KD-tree"),
    ("ball-tree", "ball tree"),
]
const SWEET_SPOT_RECALL = 0.99
const ROW_END = "\\\\"                     # the LaTeX row end \\

# ---- number formats (spec assumption A20) --------------------------------------------
"Rounded integer with LaTeX thousands separators: 51679.4 -> 51{,}679."
function fmt_int(x::Real)
    s = string(abs(round(Int, x)))
    groups = String[]
    while length(s) > 3
        pushfirst!(groups, s[end-2:end])
        s = s[1:end-3]
    end
    pushfirst!(groups, s)
    return (x < 0 ? "-" : "") * join(groups, "{,}")
end
fmt_fixed(x::Real, digits::Int) = Printf.format(Printf.Format("%.$(digits)f"), x)
fmt_qps(x::Real) = x >= 100 ? fmt_int(x) : fmt_fixed(x, 1)
"Three significant digits (integers with separators from 1000 up)."
function fmt_sig3(x::Real)
    x == 0 && return "0"
    abs(x) >= 1000 && return fmt_int(x)
    return fmt_fixed(x, max(0, 2 - floor(Int, log10(abs(x)))))
end
fmt_ms(seconds::Real) = fmt_fixed(1000 * seconds, 2) * "\\,ms"
fmt_s(seconds::Real) = fmt_sig3(seconds) * "\\,s"
or_na(f, x) = (x === nothing || x === missing) ? "n/a" : f(x)

# ---- tab:annbench and tab:fashion ----------------------------------------------------
"HNSW rows (assumption A7): max = highest recall (ties: faster); sweet = fastest with recall ≥ 0.99, else max."
function hnsw_picks(df::AbstractDataFrame; target::Real = SWEET_SPOT_RECALL)
    h = df[df.method .== "HNSW", :]
    nrow(h) == 0 && return nothing
    top = h[h.recall_ann .== maximum(h.recall_ann), :]
    best = top[argmax(top.qps), :]
    ok = h[h.recall_ann .>= target, :]
    sweet = nrow(ok) == 0 ? best : ok[argmax(ok.qps), :]
    return (max = best, sweet = sweet)
end

method_row(label, r, recall_digits) = string(label, " & ", fmt_fixed(r.recall_ann, recall_digits),
    " & ", fmt_qps(r.qps), " & ", fmt_sig3(r.build_s), ROW_END)

"tabular body with columns Method & Recall & QPS & Build (s), TKDE tab:annbench / tab:fashion."
function method_table_tex(df::AbstractDataFrame; recall_digits::Int)
    lines = String["\\begin{tabular}{lccc}", "\\hline", "Method & Recall & QPS & Build (s)" * ROW_END, "\\hline"]
    for (m, label) in METHOD_ROWS
        sel = df[df.method .== m, :]
        nrow(sel) > 0 && push!(lines, method_row(label, sel[1, :], recall_digits))
    end
    picks = hnsw_picks(df)
    if picks !== nothing
        push!(lines, method_row("HNSW (max recall attained)", picks.max, recall_digits))
        push!(lines, method_row("HNSW (sweet spot)", picks.sweet, recall_digits))
    else
        sg = df[df.method .== "SS-SearchGraph", :]
        nrow(sg) > 0 && push!(lines, method_row("SearchGraph (SimilaritySearch.jl)", sg[1, :], recall_digits))
    end
    append!(lines, ["\\hline", "\\end{tabular}"])
    return join(lines, "\n") * "\n"
end

# ---- tab:ops -------------------------------------------------------------------------
"One tab:ops row; speedup = rebuild / insert (assumption A8)."
ops_row(n, d, ins, del, reb) = string(fmt_int(n), " (\$", d, "\$) & ", fmt_ms(ins), " & ", fmt_ms(del),
    " & ", fmt_s(reb), " \\;\\;(\$\\sim\$", fmt_int(reb / ins), "\$\\times\$)", ROW_END)

"Per table configuration, sorted by (n, d): median over runs of the per-run mean latency."
function ops_summary(exp3::AbstractDataFrame)
    lat(op) = op_latency(exp3, "table", "hNNG", op)
    ins, del, reb, bulk = lat("insert"), lat("delete"), lat("rebuild"), lat("rebuild-bulk")
    (ins.n == del.n == reb.n == bulk.n && ins.d == del.d == reb.d == bulk.d) ||
        error("exp3.csv: the table configurations lack some hNNG insert/delete/rebuild rows")
    return DataFrame(n = ins.n, d = ins.d, insert_s = ins.latency_s, delete_s = del.latency_s,
                     rebuild_s = reb.latency_s, rebuild_bulk_s = bulk.latency_s)
end

function ops_tex(exp3::AbstractDataFrame)
    lines = String["\\begin{tabular}{lccc}", "\\hline",
                   "\$n\$ (\$d\$) & insert/op & delete/op & rebuild/op \\;(speedup)" * ROW_END, "\\hline"]
    for r in eachrow(ops_summary(exp3))
        push!(lines, ops_row(r.n, r.d, r.insert_s, r.delete_s, r.rebuild_s))
    end
    append!(lines, ["\\hline", "\\end{tabular}"])
    return join(lines, "\n") * "\n"
end

# ---- macros.tex -------------------------------------------------------------------------
macro_line(name, value) = "\\newcommand{\\$name}{$value}"
function first_row(df::AbstractDataFrame, method::AbstractString)
    sel = df[df.method .== method, :]
    return nrow(sel) == 0 ? nothing : sel[1, :]
end
function rho_at(exp2::AbstractDataFrame, d::Int)
    sel = exp2[(exp2.geometry .== "gaussian") .& (exp2.method .== "hNNG-bulk") .& (exp2.d .== d), :]
    return nrow(sel) == 0 ? nothing : sel.frac_evaluated[1]
end
ratio_or_na(a, b, field, digits) =
    (a === nothing || b === nothing) ? "n/a" : fmt_fixed(getproperty(a, field) / getproperty(b, field), digits)

"Every number the manuscript prose quotes, as \\newcommand lines."
function macros_tex(exp1, exp1b, exp2, exp3)
    m = Pair{String,String}[]
    # Exp.2: ρ(d) on Gaussian data (TKDE lines 91-95)
    for (name, d) in (("rhoDtwo", 2), ("rhoDeight", 8), ("rhoDsixteen", 16), ("rhoDthirtytwo", 32))
        push!(m, name => or_na(x -> fmt_fixed(x, 3), rho_at(exp2, d)))
    end
    push!(m, "pruneDtwo" => or_na(x -> fmt_fixed(100 * (1 - x), 1), rho_at(exp2, 2)))
    # Exp.1: random-xs headline
    inc, bulk, brute = first_row(exp1, "hNNG-incremental"), first_row(exp1, "hNNG-bulk"), first_row(exp1, "brute-BLAS")
    push!(m, "hnngQpsRandomXs" => or_na(r -> fmt_qps(r.qps), inc))
    push!(m, "hnngBulkQpsRandomXs" => or_na(r -> fmt_qps(r.qps), bulk))
    push!(m, "bruteQpsRandomXs" => or_na(r -> fmt_qps(r.qps), brute))
    push!(m, "hnngOverBruteRandomXs" => ratio_or_na(inc, brute, :qps, 1))
    hn = exp1[exp1.method .== "HNSW", :]
    peak = nrow(hn) == 0 ? nothing : hn[argmax(hn.qps), :]
    ex = hn[hn.recall_ann .>= 1.0, :]
    exact = nrow(ex) == 0 ? nothing : ex[argmax(ex.qps), :]
    push!(m, "hnswPeakQpsRandomXs" => or_na(r -> fmt_qps(r.qps), peak))
    push!(m, "hnswPeakRecallRandomXs" => or_na(r -> fmt_fixed(r.recall_ann, 2), peak))
    push!(m, "hnswExactQpsRandomXs" => or_na(r -> fmt_qps(r.qps), exact))
    # Exp.1b: Fashion-MNIST
    finc, fbulk, fbrute = first_row(exp1b, "hNNG-incremental"), first_row(exp1b, "hNNG-bulk"), first_row(exp1b, "brute-BLAS")
    push!(m, "hnngQpsFashion" => or_na(r -> fmt_qps(r.qps), finc))
    push!(m, "hnngBulkQpsFashion" => or_na(r -> fmt_qps(r.qps), fbulk))
    push!(m, "bruteQpsFashion" => or_na(r -> fmt_qps(r.qps), fbrute))
    picks = hnsw_picks(exp1b)
    push!(m, "hnswMaxRecallFashion" => or_na(p -> fmt_fixed(p.max.recall_ann, 5), picks))
    push!(m, "hnswMaxRecallQpsFashion" => or_na(p -> fmt_qps(p.max.qps), picks))
    push!(m, "hnswSweetRecallFashion" => or_na(p -> fmt_fixed(p.sweet.recall_ann, 3), picks))
    push!(m, "hnswSweetQpsFashion" => or_na(p -> fmt_qps(p.sweet.qps), picks))
    push!(m, "hnngIncBuildFashion" => or_na(r -> fmt_sig3(r.build_s), finc))
    push!(m, "hnngBulkBuildFashion" => or_na(r -> fmt_sig3(r.build_s), fbulk))
    push!(m, "bulkBuildSpeedupFashion" => ratio_or_na(finc, fbulk, :build_s, 1))
    # Exp.3: rewires over every hNNG insert and delete of the table configurations
    ops = exp3[(exp3.config .== "table") .& (exp3.method .== "hNNG") .& in.(exp3.op, Ref(("insert", "delete"))), :]
    rw = collect(skipmissing(ops.rewired))
    push!(m, "rewireMean" => isempty(rw) ? "n/a" : fmt_fixed(mean(rw), 2))
    push!(m, "rewireMax" => isempty(rw) ? "n/a" : string(maximum(rw)))
    s = ops_summary(exp3)
    largest = nrow(s) == 0 ? nothing : s[end, :]
    push!(m, "insertMsLargest" => or_na(r -> fmt_fixed(1000 * r.insert_s, 2), largest))
    push!(m, "deleteMsLargest" => or_na(r -> fmt_fixed(1000 * r.delete_s, 2), largest))
    push!(m, "rebuildSLargest" => or_na(r -> fmt_sig3(r.rebuild_s), largest))
    push!(m, "bulkRebuildSLargest" => or_na(r -> fmt_sig3(r.rebuild_bulk_s), largest))
    push!(m, "speedupLargest" => or_na(r -> fmt_int(r.rebuild_s / r.insert_s), largest))
    push!(m, "speedupBulkLargest" => or_na(r -> fmt_int(r.rebuild_bulk_s / r.insert_s), largest))
    return join((macro_line(k, v) for (k, v) in m), "\n") * "\n"
end

# ---- files ------------------------------------------------------------------------------
function read_result(resdir::AbstractString, file::AbstractString)
    p = joinpath(resdir, file)
    isfile(p) || error("$p is missing; run the experiment that writes it first")
    return CSV.read(p, DataFrame)
end

function write_tables(resdir::AbstractString = results_dir())
    exp1, exp1b, exp2, exp3 = (read_result(resdir, f) for f in ("exp1.csv", "exp1b.csv", "exp2.csv", "exp3.csv"))
    texdir = joinpath(resdir, "tex")
    mkpath(texdir)
    header = "% generated by bench/tables.jl from the CSV files next to this directory; do not edit\n"
    write(joinpath(texdir, "tab_annbench.tex"), header * method_table_tex(exp1; recall_digits = 3))
    write(joinpath(texdir, "tab_fashion.tex"), header * method_table_tex(exp1b; recall_digits = 5))
    write(joinpath(texdir, "tab_ops.tex"), header * ops_tex(exp3))
    write(joinpath(texdir, "macros.tex"), header * macros_tex(exp1, exp1b, exp2, exp3))
    return texdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    write_tables()
end
