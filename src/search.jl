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
`seed != 0` names the stored element `q` is (see `_seed!`); `0` for an ordinary query,
which costs one branch per call.
"""
function _knn!(memo::DistMemo, idx::HNNGIndex, q, k::Int, seed::Int=0)
    _begin!(memo, _ntotal(idx))
    seed == 0 || _seed!(memo, idx, seed)
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

"""
    _seed!(memo, idx, id)

Pre-fill the pass memo for a query that is the stored element `id` with the distances
the index already holds: `id` itself (0), and at every level `id` reaches, its outlink
and each of its inlinkers (the stored outlink distances, by symmetry). Only `seen`/`val`
are set, never `mark`, so a seeded element is still offered and expanded like any
other; nothing is counted. A no-op unless the metric is a metric, since the seeds
assume `d(a, b) == d(b, a)` with `d(id, id) == 0`.
"""
function _seed!(memo::DistMemo, idx::HNNGIndex, id::Int)
    idx.is_metric || return memo
    ep = memo.epoch
    memo.seen[id] = ep
    memo.val[id] = 0.0
    for lv in idx.levels
        lv.present[id] || continue
        o = lv.outlink[id]
        if o != 0
            memo.seen[o] = ep
            memo.val[o] = lv.outdist[id]
        end
        for x in lv.inlinks[id]
            memo.seen[x] = ep
            memo.val[x] = lv.outdist[x]
        end
    end
    memo
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

Holds the index lock for its whole duration: `knn` calls on one index from several
threads (for example inside `Threads.@threads`) are serialized, correct but not
parallel. Use `knn_batch(idx, Q, k; threads=true)` for parallel queries.
"""
knn(idx::HNNGIndex, q, k::Integer) = @_locked idx _knn(idx, q, k)

@noinline function _knn(idx::HNNGIndex, q, k::Integer)
    _check_k(k)
    ids, d = _knn!(idx.memo, idx, _prepare(idx, q), Int(k))
    _flush!(idx)
    ids, _to_user.(Ref(idx), d)
end

"""
    knn_stored(idx, id, k) -> (ids::Vector{Int}, dists::Vector{Float64})

Exact `k` nearest live elements of the stored element `id`, `id` itself excluded:
ascending distance, ties by ascending id, distances in the metric's own units, as
`knn` returns them. Other elements identical to `id` (distance 0) are ordinary
neighbours and are returned. Fewer than `k` other live elements: all of them; none
(`id` is the only live element): two empty vectors. An unknown or deleted `id`, or
`k < 1`, throws `ArgumentError`. Counts its evaluations in `nevals`.

`k == 1` reads the level-0 outlink and its stored distance and evaluates nothing. For
`k > 1` the search starts with the distances the index already stores for `id` (to its
outlinks and inlinkers) in its memo, so it never evaluates more than
`knn(idx, element(idx, id), k + 1)`; a non-metric index runs without those seeds.
Holds the index lock, like `knn`.
"""
knn_stored(idx::HNNGIndex, id::Integer, k::Integer) = @_locked idx _knn_stored(idx, id, k)

@noinline function _knn_stored(idx::HNNGIndex, id::Integer, k::Integer)
    i = _check_live(idx, id)
    _check_k(k)
    if k == 1
        o = idx.levels[1].outlink[i]
        return o == 0 ? (Int[], Float64[]) : ([o], [_to_user(idx, idx.levels[1].outdist[i])])
    end
    # The stored point is already prepared (unit length under CosineDist): no _prepare.
    ids, d = _knn!(idx.memo, idx, _point(idx.store, i), Int(k) + 1, i)
    _flush!(idx)
    # `i` is among the k+1 nearest unless k+1 or more duplicates with smaller ids precede
    # it (or a non-metric d(i, i) > 0 pushes it out); then all k+1 are others: drop the last.
    p = findfirst(==(i), ids)
    deleteat!(ids, p === nothing ? length(ids) : p)
    deleteat!(d, p === nothing ? length(d) : p)
    ids, _to_user.(Ref(idx), d)
end

"""
    HNNG.knn_counted(idx, q, k) -> (ids, dists, nevals_this_query)

`knn` plus the number of distance evaluations this query made (the benchmark's
fraction evaluated is that number divided by `length(idx)`). Holds the index lock,
like `knn`.
"""
knn_counted(idx::HNNGIndex, q, k::Integer) = @_locked idx _knn_counted(idx, q, k)

@noinline function _knn_counted(idx::HNNGIndex, q, k::Integer)
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

"""
    HNNG.knn_levelsync(idx, q, k) -> (ids, dists)

The level-synchronous descent of the 2015 algorithm with the corrected, strict and
guarded prune: at each level offer every candidate cluster's representative, prune
clusters whose bound exceeds the k-th distance, descend into the survivors' children;
at level 0 evaluate the survivors' members. A second oracle for the tests only.
Holds the index lock, like `knn`.
"""
knn_levelsync(idx::HNNGIndex, q, k::Integer) = @_locked idx _knn_levelsync(idx, q, k)

@noinline function _knn_levelsync(idx::HNNGIndex, q, k::Integer)
    _check_k(k)
    qp = _prepare(idx, q)
    memo = _begin!(DistMemo(), _ntotal(idx))
    isempty(idx.levels) && return Int[], Float64[]
    res = memo.res
    met = idx.is_metric
    T = nlevels(idx) - 1
    list = [ci for (ci, c) in enumerate(idx.levels[T+1].clusters) if c.rep != 0]
    function offer(m)
        d = _mdist(memo, idx, qp, m)
        if memo.mark[m] != memo.epoch
            memo.mark[m] = memo.epoch
            _offer!(res, d, m, Int(k))
        end
        d
    end
    for L in T:-1:0
        lv = idx.levels[L+1]
        for ci in list
            offer(lv.clusters[ci].rep)
        end
        keep = Int[]
        for ci in list
            c = lv.clusters[ci]
            lb = met ? _lb(offer(c.rep), c.radius, idx.guard) : 0.0
            (met && length(res) == k && lb > res[1][1]) || push!(keep, ci)
        end
        if L == 0
            for ci in keep, m in lv.clusters[ci].members
                offer(m)
            end
        else
            below = idx.levels[L]
            list = [below.cluster_of[m] for ci in keep for m in lv.clusters[ci].members]
        end
    end
    ids, d = _sorted_results(res)
    ids, _to_user.(Ref(idx), d)
end

_queries(idx::HNNGIndex{<:VectorStore}, Q::AbstractMatrix{<:Real}) =
    [_prepare(idx, view(Q, :, j)) for j in axes(Q, 2)]
_queries(idx::HNNGIndex{<:ObjectStore}, Q::AbstractVector) = [_prepare(idx, q) for q in Q]

function _store_col!(ids::Matrix{Int}, ds::Matrix{Float64}, j::Int, a, b, idx::HNNGIndex)
    for i in eachindex(a)
        ids[i, j] = a[i]
        ds[i, j] = _to_user(idx, b[i])
    end
end

"""
    knn_batch(idx, Q, k; threads=false) -> (ids::Matrix{Int}, dists::Matrix{Float64})

`knn` for every query: the columns of a matrix (VectorStore) or the elements of a
vector (ObjectStore). Column `j` holds query `j`'s answer; when fewer than `k` live
elements exist the tail is padded with id `0` and distance `Inf`.

The calling task holds the index lock for the whole batch. `threads=false` answers the
queries one after another and counts their evaluations in `nevals`. `threads=true`
splits them over `Threads.nthreads()` tasks with private buffers, waits for every task
(the lock is held until the last one finishes, so nothing mutates the index meanwhile),
and does not update the evaluation counter; the metric is then called from several
threads at once, so a `UserMetric` must be thread-safe. If a task throws, the first
exception is rethrown after all tasks have finished.
"""
knn_batch(idx::HNNGIndex, Q, k::Integer; threads::Bool=false) =
    @_locked idx _knn_batch(idx, Q, k, threads)

"""
    _wait_all(tasks) -> first exception or nothing

Wait until every task has finished, even when a task fails or the caller is
interrupted, so a threaded batch never outlives the lock its caller holds.
"""
function _wait_all(tasks)
    failed = nothing
    for t in tasks
        while true
            try
                wait(t)
                break
            catch err
                failed === nothing && (failed = err)
                istaskdone(t) && break
            end
        end
    end
    failed
end

@noinline function _knn_batch(idx::HNNGIndex, Q, k::Integer, threads::Bool)
    _check_k(k)
    qs = _queries(idx, Q)
    nq = length(qs)
    ids = zeros(Int, k, nq)
    ds = fill(Inf, k, nq)
    nq == 0 && return ids, ds
    if threads
        chunks = collect(Iterators.partition(1:nq, cld(nq, Threads.nthreads())))
        # The calling task holds idx.lock until every child has finished; children
        # call internal functions only and never take the lock.
        tasks = map(chunks) do r
            Threads.@spawn begin
                memo = DistMemo()
                for j in r
                    a, b = _knn!(memo, idx, qs[j], Int(k))
                    _store_col!(ids, ds, j, a, b, idx)
                end
            end
        end
        err = _wait_all(tasks)
        err === nothing || throw(err isa TaskFailedException ? err.task.exception : err)
    else
        for j in 1:nq
            a, b = _knn!(idx.memo, idx, qs[j], Int(k))
            _store_col!(ids, ds, j, a, b, idx)
        end
        _flush!(idx)
    end
    ids, ds
end
