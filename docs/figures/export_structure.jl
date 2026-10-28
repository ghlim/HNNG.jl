# Export the structure of a small, fixed 2-D HNNG index and one traced exact search,
# as plain TSV tables for the manuscript's TikZ figures.
#
#   julia --project=. docs/figures/export_structure.jl OUTDIR          # write the tables
#   julia --project=. docs/figures/export_structure.jl --check OUTDIR  # fail if they differ
#
# The index is built with the public API (`HNNGIndex`, `add!`) and every structural
# field written out is read from it (`outlink`, `outlink_distance`, `cluster`,
# `members`, `representative`, `radius`, `children`, and the stored `repdist` of each
# cluster). Nothing structural is recomputed here.
#
# The search trace comes from `_traced_knn!`, a copy of the plain best-first engine
# `HNNG._knn!` (src/search.jl; the engine `knn` runs on a 2-D index, where the
# parent-distance filter is off) that records each decision. It is checked against
# the library before anything is written:
#   1. its ids, distances and evaluation count equal `HNNG.knn_counted` on the index;
#   2. an ObjectStore twin of the index under a logging `UserMetric` evaluates, inside
#      the library's own `knn`, exactly the elements the trace evaluates, in that order.
# If the engine changes, the check fails and the copy must be updated with it.
#
# Determinism: the points come from a self-contained SplitMix64 generator (not Julia's
# default RNG, whose stream may change between versions) and are written with fixed
# precision, so the tables are byte-for-byte reproducible. The publications repository
# (Hierarchical-Nearest-Neighbor-Graph) commits the tables in figures/data and runs this
# script through `make figdata`; `make figures` there renders them with TikZ.

using HNNG
using HNNG: ClusterRef
using Printf

# ---------------------------------------------------------------- fixed configuration

const SEED    = 0x0000000000000145   # SplitMix64 state (325)
const CENTRES = ((2.0, 4.5), (7.0, 6.5), (6.0, 1.5))   # left, upper right, lower
const PER     = 12                   # points per blob: n = 36
const SPREAD  = 0.9                  # blob standard deviation
const QUERY   = (4.10, 6.50)       # with SEED: a single-path greedy descent misses the
                                     # nearest neighbour and one level-1 cluster is pruned
const K       = 1
const PREFIX  = "hnng_example"

# ---------------------------------------------------------------- data

mutable struct SplitMix64
    s::UInt64
end
function _next!(r::SplitMix64)
    r.s += 0x9e3779b97f4a7c15
    z = r.s
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end
_u01(r) = (_next!(r) >> 11) * (1.0 / 2.0^53)
_gauss(r) = sqrt(-2 * log(1 - _u01(r))) * cos(2π * _u01(r))

"""`2 × n` points: `per` Gaussian points around each centre, rounded to 1e-3."""
function example_points(seed=SEED; centres=CENTRES, per=PER, spread=SPREAD)
    r = SplitMix64(UInt64(seed))
    X = Matrix{Float64}(undef, 2, 0)
    for c in centres, _ in 1:per
        p = [c[1] + spread * _gauss(r), c[2] + spread * _gauss(r)]
        X = hcat(X, round.(p; digits=3))
    end
    X
end

function example_index(X)
    idx = HNNGIndex(Euclidean(), 2; T=Float64)
    add!(idx, X)
    verify(idx) || error("verify failed")
    idx
end

# ---------------------------------------------------------------- traced search

# Status codes written to the search table (integers keep the TikZ side simple).
const UNREACHED = 0   # its parent cluster was never expanded
const EXPANDED  = 1   # popped from the frontier and expanded
const PRUNED    = 2   # bound above the k-th distance when it would have been pushed
const LEFTOVER  = 3   # pushed, but still on the frontier when the search stopped

struct Decision
    level::Int
    ci::Int
    dqrep::Float64      # distance from q to the representative (internal units)
    lb::Float64         # guarded lower bound HNNG._lb(dqrep, radius, guard)
    tau::Float64        # k-th distance when the decision was taken (Inf: heap not full)
    status::Int
    order::Int          # expansion rank for EXPANDED, else 0
end

"""
    _traced_knn!(idx, q, k) -> (ids, dists, nevals, decisions, evals)

`HNNG._knn!` with `pf=false`, statement for statement, plus a record of every cluster
decision and every distance evaluation (`evals`: `(id, distance)` in evaluation order).
"""
function _traced_knn!(idx::HNNGIndex, q, k::Int)
    memo = HNNG._begin!(HNNG.DistMemo(), HNNG._ntotal(idx))
    res, front = memo.res, memo.front
    met = idx.is_metric
    T = nlevels(idx) - 1
    dec = Dict{Tuple{Int,Int},Decision}()
    evals = Tuple{Int,Float64}[]
    dist(j) = begin
        seen = memo.seen[j] == memo.epoch
        d = HNNG._mdist(memo, idx, q, j)
        seen || push!(evals, (j, d))
        d
    end
    tau() = length(res) == k ? res[1][1] : Inf
    for (ci, c) in enumerate(idx.levels[T+1].clusters)
        c.rep == 0 && continue
        dr = dist(c.rep)
        lb = met ? HNNG._lb(dr, c.radius, idx.guard) : 0.0
        dec[(T, ci)] = Decision(T, ci, dr, lb, tau(), LEFTOVER, 0)
        HNNG.heappush!(front, (lb, T, ci), HNNG.FWD)
    end
    rank = 0
    while !isempty(front)
        lb, L, ci = HNNG.heappop!(front, HNNG.FWD)
        if met && length(res) == k && lb > res[1][1]
            HNNG.heappush!(front, (lb, L, ci), HNNG.FWD)   # back, for the record below
            break
        end
        rank += 1
        d0 = dec[(L, ci)]
        dec[(L, ci)] = Decision(L, ci, d0.dqrep, d0.lb, tau(), EXPANDED, rank)
        c = idx.levels[L+1].clusters[ci]
        below = idx.levels[max(L, 1)]
        for m in c.members
            d = dist(m)
            if memo.mark[m] != memo.epoch
                memo.mark[m] = memo.epoch
                HNNG._offer!(res, d, m, k)
            end
            L == 0 && continue
            cj = below.cluster_of[m]
            ch = below.clusters[cj]
            dr = ch.rep == m ? d : dist(ch.rep)
            lbc = met ? HNNG._lb(dr, ch.radius, idx.guard) : 0.0
            if met && length(res) == k && lbc > res[1][1]
                dec[(L - 1, cj)] = Decision(L - 1, cj, dr, lbc, res[1][1], PRUNED, 0)
            else
                dec[(L - 1, cj)] = Decision(L - 1, cj, dr, lbc, tau(), LEFTOVER, 0)
                HNNG.heappush!(front, (lbc, L - 1, cj), HNNG.FWD)
            end
        end
    end
    # Clusters still on the frontier keep status LEFTOVER; their tau becomes the final one.
    for (_, L, ci) in front
        d0 = dec[(L, ci)]
        dec[(L, ci)] = Decision(L, ci, d0.dqrep, d0.lb, tau(), LEFTOVER, 0)
    end
    ids, ds = HNNG._sorted_results(res)
    ids, ds, memo.nevals, dec, evals
end

"""Run the traced search and check it against the library (see the file header)."""
function traced_search(idx::HNNGIndex, X, q, k)
    HNNG._use_pf(idx) && error("the parent-distance filter is on: trace the _knn_pf! engine instead")
    qp = HNNG._prepare(idx, q)
    ids, ds, nev, dec, evals = _traced_knn!(idx, qp, k)
    lids, ldists, lnev = HNNG.knn_counted(idx, q, k)
    (ids == lids && HNNG._to_user.(Ref(idx), ds) == ldists && nev == lnev) ||
        error("trace disagrees with knn_counted: $((ids, ds, nev)) vs $((lids, ldists, lnev))")
    # ObjectStore twin under a logging metric: the library's own evaluation sequence.
    log = Vector{Vector{Float64}}()
    f = (a, b) -> (push!(log, b); sqrt(sum(abs2, a .- b)))
    tw = HNNGIndex(UserMetric(f; is_metric=true), Vector{Float64})
    for j in axes(X, 2)
        add!(tw, X[:, j])
    end
    for (a, b) in zip(idx.levels, tw.levels)
        (a.outlink == b.outlink && [c.rep for c in a.clusters] == [c.rep for c in b.clusters] &&
         [c.members for c in a.clusters] == [c.members for c in b.clusters]) ||
            error("the ObjectStore twin built a different structure")
    end
    empty!(log)
    tids, _ = knn(tw, collect(Float64, q), k)
    tids == ids || error("twin result differs")
    byval = Dict(X[:, j] => j for j in axes(X, 2))
    [byval[p] for p in log] == first.(evals) ||
        error("the library evaluated $([byval[p] for p in log]); the trace says $(first.(evals))")
    ids, ds, dec, evals
end

# ---------------------------------------------------------------- tables

fmt(x::Real) = isfinite(x) ? @sprintf("%.6f", x) : "-1"

function write_tsv(path, header::Vector{String}, rows, comment)
    open(path, "w") do io
        println(io, "# ", comment)
        println(io, join(header, '\t'))
        for r in rows
            println(io, join((v isa AbstractFloat ? fmt(v) : string(v) for v in r), '\t'))
        end
    end
    path
end

"""Write the tables into `outdir`; returns their paths."""
function export_tables(outdir; seed=SEED, query=QUERY)
    X = example_points(seed)
    idx = example_index(X)
    n = size(X, 2)
    q = collect(Float64, query)
    ids, ds, dec, evals = traced_search(idx, X, q, K)
    T = nlevels(idx) - 1
    note = "generated by HNNG.jl docs/figures/export_structure.jl (n=$n, seed=$(repr(UInt64(seed))), " *
           "Euclidean); do not edit by hand"

    # Display names: clusters numbered per level in order of their representative's x.
    lvl_clusters = [[ci for (ci, c) in enumerate(idx.levels[l+1].clusters) if c.rep != 0] for l in 0:T]
    name = Dict{Tuple{Int,Int},Int}()
    for l in 0:T
        order = sort(lvl_clusters[l+1]; by=ci -> (X[1, idx.levels[l+1].clusters[ci].rep], ci))
        for (i, ci) in enumerate(order)
            name[(l, ci)] = i
        end
    end

    # elements: one row per (level, element) the element reaches.
    erows = Any[]
    for l in 0:T, id in 1:n
        cr = cluster(idx, id; level=l)
        cr === nothing && continue
        o = outlink(idx, id; level=l)
        push!(erows, (l, id, X[1, id], X[2, id], o, o == 0 ? Inf : outlink_distance(idx, id; level=l),
                      name[(l, cr.index)], Int(representative(idx, cr) == id)))
    end
    # clusters: radius as stored; its arg max member from the stored repdist and the
    # child radius sigma (zero at level 0), checked against the stored radius.
    crows = Any[]
    mrows = Any[]
    for l in 0:T, ci in lvl_clusters[l+1]
        cr = ClusterRef(l, ci)
        c = idx.levels[l+1].clusters[ci]
        rep = representative(idx, cr)
        mem = members(idx, cr)
        mem == c.members || error("members order")
        kids = children(idx, cr)
        sig = [l == 0 ? 0.0 : radius(idx, kids[i]) for i in eachindex(mem)]
        tot = c.repdist .+ sig
        am = argmax(tot)
        isapprox(tot[am], radius(idx, cr); rtol=1e-12, atol=1e-12) ||
            error("radius of level-$l cluster $ci is not max(repdist + sigma)")
        par = l == T ? 0 : name[(l + 1, cluster(idx, rep; level=l + 1).index)]
        push!(crows, (l, name[(l, ci)], rep, X[1, rep], X[2, rep], length(mem), radius(idx, cr),
                      mem[am], c.repdist[am], sig[am], par))
        for i in eachindex(mem)
            push!(mrows, (l, name[(l, ci)], mem[i], c.repdist[i], sig[i], tot[i], Int(i == am)))
        end
    end
    # search: one row per cluster, decision as traced.
    srows = Any[]
    for l in T:-1:0, ci in lvl_clusters[l+1]
        c = idx.levels[l+1].clusters[ci]
        d = get(dec, (l, ci), nothing)
        push!(srows, d === nothing ?
              (l, name[(l, ci)], c.rep, Inf, Inf, Inf, UNREACHED, 0) :
              (l, name[(l, ci)], c.rep, d.dqrep, d.lb, d.tau, d.status, d.order))
    end
    vrows = [(i, j, X[1, j], X[2, j], d, Int(j in ids)) for (i, (j, d)) in enumerate(evals)]
    # query: the exact answer and the answer of a single-path greedy descent (which
    # follows the closest representative at each level) for the text's comparison.
    g = greedy_descent(idx, q)
    qrows = [(q[1], q[2], K, ids[1], ds[1], g, sqrt(sum(abs2, X[:, g] .- q)), length(evals), n)]

    mkpath(outdir)
    p(s) = joinpath(outdir, "$(PREFIX)_$(s).tsv")
    [write_tsv(p("elements"), ["level", "id", "x", "y", "out", "outdist", "cluster", "isrep"], erows, note),
     write_tsv(p("clusters"), ["level", "cluster", "rep", "x", "y", "size", "radius", "argmax",
                               "repdist", "sigma", "parent"], crows, note),
     write_tsv(p("members"), ["level", "cluster", "id", "repdist", "sigma", "total", "isargmax"], mrows, note),
     write_tsv(p("search"), ["level", "cluster", "rep", "dqrep", "lb", "tau", "status", "order"], srows,
               note * "; status 0 unreached, 1 expanded, 2 pruned, 3 left on the frontier; -1 = none"),
     write_tsv(p("evals"), ["order", "id", "x", "y", "dist", "result"], vrows, note),
     write_tsv(p("query"), ["x", "y", "k", "nn", "nndist", "greedy", "greedydist", "nevals", "n"], qrows, note)]
end

"""Single-path greedy descent over the stored clusters: the baseline the text contrasts."""
function greedy_descent(idx::HNNGIndex, q)
    d(j) = sqrt(sum(abs2, element(idx, j) .- q))
    c = root(idx)
    while true
        mem = members(idx, c)
        best = mem[argmin(d.(mem))]
        c.level == 0 && return best
        c = cluster(idx, best; level=c.level - 1)
    end
end

function main(args)
    check = "--check" in args
    rest = filter(!=("--check"), args)
    length(rest) == 1 || (println(stderr, "usage: export_structure.jl [--check] OUTDIR"); exit(2))
    outdir = rest[1]
    if check
        tmp = mktempdir()
        bad = String[]
        for f in export_tables(tmp)
            ref = joinpath(outdir, basename(f))
            (isfile(ref) && read(ref) == read(f)) || push!(bad, basename(f))
        end
        isempty(bad) || (println(stderr, "out of date: ", join(bad, ", ")); exit(1))
        println("OK: tables in $outdir match the index")
    else
        for f in export_tables(outdir)
            println("wrote ", f)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__()
    main(ARGS)
end
