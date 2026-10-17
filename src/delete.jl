# Removal from a level (spec §6.3) and, from Task 10, the public delete!.

"""
    unlink!(idx, e, l, floating) -> (removed, added)

Remove `e` from the level-`l` population: every inlinker re-finds its nearest
neighbour with `e` excluded (no rewire sweep is needed), then `e`'s former cluster
and the clusters the inlinkers now point into are repaired.
"""
function unlink!(idx::HNNGIndex, e::Int, l::Int, floating::Vector{Int})
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
    for x in inl
        l == 0 && (idx.last_rewired += 1)          # read by HNNG.last_update_stats
        if lv.npop == 1
            lv.outlink[x] = 0
            lv.outdist[x] = Inf
            continue
        end
        # e keeps its cluster pointer during the descents: its cluster is reached through e
        nn, nnd, _ = _nn_search(idx, _point(idx.store, x), l, e, x, floating, false)
        nn == 0 && error("HNNG internal error: no neighbour found for $x at level $l")
        lv.outlink[x] = nn
        lv.outdist[x] = nnd
        _insert_sorted!(lv.inlinks[nn], x)
        push!(affected, lv.cluster_of[nn])
    end
    lv.cluster_of[e] = 0
    _repair!(idx, l, affected, Int[], e, floating)
end

"""
    delete!(idx, id) -> idx

Remove a live element exactly (spec §6.4): tombstone the id, `unlink!` it from
level 0 and propagate the representative delta. The id is never reused and its
storage is kept. Deleting the last live element leaves an index with no levels.
"""
Base.delete!(idx::HNNGIndex, id::Integer) = _locked(() -> _delete!(idx, id), idx)

function _delete!(idx::HNNGIndex, id::Integer)
    i = _check_live(idx, id)
    idx.last_rewired = 0
    idx.last_top = 0
    idx.alive[i] = false
    idx.nlive -= 1
    delta = unlink!(idx, i, 0, Int[])
    if idx.levels[1].npop == 0
        idx.last_top = nlevels(idx) - 1
        empty!(idx.levels)
    else
        propagate!(idx, delta..., 1)
    end
    _flush!(idx)
    idx
end
