# Navigation along stored links (spec §7). No call here evaluates a distance.

function _check_live(idx::HNNGIndex, id::Integer)
    n = _ntotal(idx)
    1 <= id <= n || throw(ArgumentError("unknown id $id: the index holds ids 1:$n"))
    idx.alive[id] || throw(ArgumentError("id $id is deleted"))
    Int(id)
end

"""
    element(idx, id)

The stored element: a column view for a VectorStore (unit-normalised under
`CosineDist`), the object itself for an ObjectStore.
"""
element(idx::HNNGIndex, id::Integer) = _point(idx.store, _check_live(idx, id))
