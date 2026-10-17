# Shared test helpers: brute-force oracle, data generators, structure snapshots.
using Test, HNNG, StableRNGs, Random, LinearAlgebra

"""`HNNG_SLOW=1` also runs the largest parameter cases (spec §10: the default suite stays well under a minute)."""
const HNNG_SLOW = get(ENV, "HNNG_SLOW", "0") == "1"

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

"""knn equals the brute-force oracle: same ids and the same distances (same kernel, same bits)."""
function check_knn(idx, q, k)
    ids, ds = knn(idx, q, k)
    bids, bds = brute_knn(idx, q, k)
    ids == bids && ds == bds
end

"""`n` descriptor sets: `d × m` matrices, `m` drawn from `mmin:mmax`, each set around its own centre."""
descriptor_sets(rng, d, n; mmin=2, mmax=9) =
    [randn(rng, d, rand(rng, mmin:mmax)) .+ 3 .* randn(rng, d) for _ in 1:n]

_pd(a, b) = sqrt(sum(abs2, a .- b))
_sq(a, b) = sum(abs2, a .- b)

"""Hausdorff distance between point sets (columns), Euclidean point distance. A metric."""
function hausdorff(A::AbstractMatrix, B::AbstractMatrix)
    h(X, Y) = maximum(minimum(_pd(view(X, :, i), view(Y, :, j)) for j in axes(Y, 2)) for i in axes(X, 2))
    max(h(A, B), h(B, A))
end

"""
The 2015 ROS view distance `max(mean_i min_j ‖a_i − b_j‖², mean_j min_i ‖a_i − b_j‖²)`
(race_feature_extraction/src/spin_image.cpp:252–312, view_clustering_utils.cpp:1957–1969).
Not a metric.
"""
function ros_view_distance(A::AbstractMatrix, B::AbstractMatrix)
    ab = sum(minimum(_sq(view(A, :, i), view(B, :, j)) for j in axes(B, 2)) for i in axes(A, 2)) / size(A, 2)
    ba = sum(minimum(_sq(view(A, :, i), view(B, :, j)) for i in axes(A, 2)) for j in axes(B, 2)) / size(B, 2)
    max(ab, ba)
end
