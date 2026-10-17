# Removal from a level (spec §6.3) and, from Task 10, the public delete!.

"""
    _relink_search(idx, e, l, floating) -> Vector{Tuple{Int,Float64}}

The new `(outlink, outdist)` of every inlinker of `e` at level `l` once `e` leaves:
its nearest neighbour with `e` excluded, or `(0, Inf)` when one element would remain.
Writes nothing but `idx.memo`. The searches do not read what `unlink!` writes
(outlinks, inlinks, `present` and `npop` of level `l`; `e` keeps its cluster pointer),
so running them all before the writes gives the same results and evaluations.
"""
function _relink_search(idx::HNNGIndex, e::Int, l::Int, floating::Vector{Int})
    lv = idx.levels[l+1]
    lv.present[e] || error("HNNG internal error: $e is not present at level $l")
    inl = lv.inlinks[e]
    out = Vector{Tuple{Int,Float64}}(undef, length(inl))
    for (k, x) in enumerate(inl)
        if lv.npop - 1 == 1
            out[k] = (0, Inf)
            continue
        end
        # e keeps its cluster pointer during the descents: its cluster is reached through e
        nn, nnd, _ = _nn_search(idx, _point(idx.store, x), l, e, x, floating, false)
        nn == 0 && error("HNNG internal error: no neighbour found for $x at level $l")
        out[k] = (nn, nnd)
    end
    out
end

"""
    unlink!(idx, e, l, floating[, relinks]) -> (removed, added)

Remove `e` from the level-`l` population: every inlinker re-finds its nearest
neighbour with `e` excluded (no rewire sweep is needed), then `e`'s former cluster
and the clusters the inlinkers now point into are repaired. `relinks` is
`_relink_search(idx, e, l, floating)` made just before; the 4-argument form searches
first.
"""
unlink!(idx::HNNGIndex, e::Int, l::Int, floating::Vector{Int}) =
    unlink!(idx, e, l, floating, _relink_search(idx, e, l, floating))

function unlink!(idx::HNNGIndex, e::Int, l::Int, floating::Vector{Int},
                 relinks::Vector{Tuple{Int,Float64}})
    lv = idx.levels[l+1]
    lv.present[e] || error("HNNG internal error: $e is not present at level $l")
    affected = [lv.cluster_of[e]]
    old = lv.outlink[e]
    old == 0 || _remove_sorted!(lv.inlinks[old], e)
    lv.outlink[e] = 0
    lv.outdist[e] = Inf
    lv.present[e] = false
    lv.npop -= 1
    inl = lv.inlinks[e]
    lv.inlinks[e] = Int[]
    length(relinks) == length(inl) || error("HNNG internal error: stale re-search for $e at level $l")
    for (x, (nn, nnd)) in zip(inl, relinks)
        l == 0 && (idx.last_rewired += 1)          # read by HNNG.last_update_stats
        if lv.npop == 1
            lv.outlink[x] = 0
            lv.outdist[x] = Inf
            continue
        end
        lv.outlink[x] = nn
        lv.outdist[x] = nnd
        _insert_sorted!(lv.inlinks[nn], x)
        push!(affected, lv.cluster_of[nn])
    end
    lv.cluster_of[e] = 0
    _repair!(idx, l, affected, Int[], e, floating)
end

"""Cold path of a rolled-back `delete!`: restore `last_update_stats` and count the evaluations made."""
function _restore_stats!(@nospecialize(idx::HNNGIndex), rew0::Int, top0::Int)
    disable_sigint() do
        idx.last_rewired, idx.last_top = rew0, top0
        _flush!(idx)
    end
    nothing
end

"""
    delete!(idx, id) -> idx

Remove a live element exactly (spec §6.4): tombstone the id, `unlink!` it from
level 0 and propagate the representative delta. The id is never reused and its
storage is kept. Deleting the last live element leaves an index with no levels.

Exception safety: the level-0 re-search of `id`'s inlinkers runs before any write. An
exception there (a metric that throws or returns NaN, an `InterruptException`, …)
leaves the index exactly as before, `id` still live; an exception after it marks the
index invalid, as for `add!`. The exception is always rethrown.
"""
Base.delete!(idx::HNNGIndex, id::Integer) = @_locked idx _delete!(idx, id)

function _delete!(idx::HNNGIndex, id::Integer)
    i = _check_live(idx, id)
    rew0, top0 = idx.last_rewired, idx.last_top
    phase = 0                     # 0: nothing structural written, 1: writing, 2: done
    try
        idx.last_rewired = 0
        idx.last_top = 0
        floating = Int[]
        found = _relink_search(idx, i, 0, floating)
        phase = 1
        idx.alive[i] = false
        idx.nlive -= 1
        delta = unlink!(idx, i, 0, floating, found)
        if idx.levels[1].npop == 0
            idx.last_top = nlevels(idx) - 1
            empty!(idx.levels)
        else
            propagate!(idx, delta..., 1)
        end
        phase = 2
        _flush!(idx)
    catch
        if phase == 0
            _restore_stats!(idx, rew0, top0)
        elseif phase == 1
            idx.invalid = true
        end
        rethrow()
    end
    idx
end
