# Shared test helpers: brute-force oracle, data generators, structure snapshots.
using Test, HNNG, StableRNGs, Random, LinearAlgebra

"""Exhaustive k-NN over the live elements with the package kernel: (distance, id) order, user units."""
function brute_knn(idx, q, k)
    qp = HNNG._prepare(idx, q)
    cand = Tuple{Float64,Int}[]
    for i in 1:HNNG._ntotal(idx)
        idx.alive[i] || continue
        push!(cand, (HNNG._rawdist(idx, qp, HNNG._point(idx.store, i)), i))
    end
    sort!(cand)
    top = cand[1:min(k, length(cand))]
    [p[2] for p in top], [HNNG._to_user(idx, p[1]) for p in top]
end

gauss(rng, T, d, n) = randn(rng, T, d, n)

"""`n` points around `nc` centres uniform in [-10, 10]^d with unit spread."""
function blobs(rng, T, d, n; nc=8)
    C = 20 .* rand(rng, T, d, nc) .- 10
    X = Matrix{T}(undef, d, n)
    for j in 1:n
        X[:, j] = C[:, rand(rng, 1:nc)] .+ randn(rng, T, d)
    end
    X
end

"""Insert the columns of `X` one at a time; `verify` after each insert (throws on the first violation)."""
function stream!(idx, X)
    ids = Int[]
    for j in axes(X, 2)
        push!(ids, add!(idx, view(X, :, j)))
        verify(idx)
    end
    ids
end

"""
Snapshot of the hierarchy with ids mapped through `key` (key[id] = point index):
per level, sorted `(key(x), key(outlink), outdist)` triples and sorted
`(sorted member keys, key(rep), radius, maxout)` tuples.
"""
function structure(idx, key=collect(1:HNNG._ntotal(idx)))
    snap = []
    for lv in idx.levels
        outs = sort!([(key[x], lv.outlink[x] == 0 ? 0 : key[lv.outlink[x]], lv.outdist[x])
                      for x in findall(lv.present)])
        cls = sort!([(sort!(key[c.members]), key[c.rep], c.radius, copy(c.maxout))
                     for c in lv.clusters if c.rep != 0])
        push!(snap, (outs=outs, clusters=cls))
    end
    snap
end

"""Integer fields equal, float fields equal to `rtol`."""
function same_structure(a, b; rtol=1e-9)
    length(a) == length(b) || return false
    for (la, lb) in zip(a, b)
        length(la.outs) == length(lb.outs) || return false
        for (x, y) in zip(la.outs, lb.outs)
            (x[1] == y[1] && x[2] == y[2] && isapprox(x[3], y[3]; rtol)) || return false
        end
        length(la.clusters) == length(lb.clusters) || return false
        for (x, y) in zip(la.clusters, lb.clusters)
            (x[1] == y[1] && x[2] == y[2] && isapprox(x[3], y[3]; rtol) &&
             length(x[4]) == length(y[4]) && all(isapprox.(x[4], y[4]; rtol))) || return false
        end
    end
    true
end

"""Representative of the root cluster."""
top_rep(idx) = first(c.rep for c in idx.levels[end].clusters if c.rep != 0)
