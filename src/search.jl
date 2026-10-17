# Exact k-NN search (spec §6.5): best-first branch-and-bound over the cluster tree.

@inline function _offer!(res::Vector{Tuple{Float64,Int}}, d::Float64, m::Int, k::Int)
    if length(res) < k
        heappush!(res, (d, m), REV)
    elseif (d, m) < res[1]
        heappop!(res, REV)
        heappush!(res, (d, m), REV)
    end
    nothing
end

function _sorted_results(res::Vector{Tuple{Float64,Int}})
    sort!(res)
    [p[2] for p in res], [p[1] for p in res]
end

"""
    _knn!(memo, idx, q, k) -> (ids, dists)

Best-first engine in internal units. Frontier: min-heap of clusters keyed by the
guarded lower bound; results: max-heap of at most `k` `(distance, id)`. Stops when
`k` results are held and the smallest bound is strictly above the k-th distance.
Distances are memoised for the query; a representative is offered once.
"""
function _knn!(memo::DistMemo, idx::HNNGIndex, q, k::Int)
    _begin!(memo, _ntotal(idx))
    isempty(idx.levels) && return Int[], Float64[]
    res, front = memo.res, memo.front
    met = idx.is_metric
    T = nlevels(idx) - 1
    for (ci, c) in enumerate(idx.levels[T+1].clusters)
        c.rep == 0 && continue
        dr = _mdist(memo, idx, q, c.rep)
        heappush!(front, (met ? _lb(dr, c.radius, idx.guard) : 0.0, T, ci), FWD)
    end
    while !isempty(front)
        lb, L, ci = heappop!(front, FWD)
        met && length(res) == k && lb > res[1][1] && break
        c = idx.levels[L+1].clusters[ci]
        below = idx.levels[max(L, 1)]
        for m in c.members
            d = _mdist(memo, idx, q, m)
            if memo.mark[m] != memo.epoch
                memo.mark[m] = memo.epoch
                _offer!(res, d, m, k)
            end
            L == 0 && continue
            cj = below.cluster_of[m]
            ch = below.clusters[cj]
            dr = ch.rep == m ? d : _mdist(memo, idx, q, ch.rep)
            lbc = met ? _lb(dr, ch.radius, idx.guard) : 0.0
            (met && length(res) == k && lbc > res[1][1]) || heappush!(front, (lbc, L - 1, cj), FWD)
        end
    end
    _sorted_results(res)
end

_check_k(k::Integer) = k >= 1 || throw(ArgumentError("k must be at least 1, got $k"))

"""
    knn(idx, q, k) -> (ids::Vector{Int}, dists::Vector{Float64})

Exact `k` nearest live elements of `q`, sorted by ascending distance, ties by
ascending id, distances in the metric's own units. Fewer than `k` live elements:
all of them. Empty index: two empty vectors. Counts its evaluations in `nevals`.
Ties are resolved in the index's internal Euclidean units (the units in which the
k-th element is selected); after conversion to SqEuclidean or CosineDist units two
distinct internal distances can print equal, and such a pair keeps its internal order.
"""
function knn(idx::HNNGIndex, q, k::Integer)
    _check_k(k)
    ids, d = _knn!(idx.memo, idx, _prepare(idx, q), Int(k))
    _flush!(idx)
    ids, _to_user.(Ref(idx), d)
end

"""
    HNNG.knn_counted(idx, q, k) -> (ids, dists, nevals_this_query)

`knn` plus the number of distance evaluations this query made (the benchmark's
fraction evaluated is that number divided by `length(idx)`).
"""
function knn_counted(idx::HNNGIndex, q, k::Integer)
    _check_k(k)
    # idx.memo.nevals is 0 here after any public call: add!, delete!, bulk_add!, knn,
    # knn_counted and the serial knn_batch all end with _flush!. It is not zeroed: a
    # residue (a pass that threw inside a user metric, or a direct internal call in a
    # test) must still reach idx.nevals, and is then counted in this query.
    ids, d = _knn!(idx.memo, idx, _prepare(idx, q), Int(k))
    n = idx.memo.nevals
    _flush!(idx)
    ids, _to_user.(Ref(idx), d), n
end
