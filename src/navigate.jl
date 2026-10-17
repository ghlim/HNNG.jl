# Navigation along stored links (spec §7). No call here evaluates a distance.

function _check_live(idx::HNNGIndex, id::Integer)
    _check_valid(idx)
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

"""
    ClusterRef(level, index)

Handle of a cluster: its level and slot. Valid until the next `add!`/`delete!`.
"""
struct ClusterRef
    level::Int
    index::Int
end

function _check_level(idx::HNNGIndex, level::Integer)
    _check_valid(idx)
    0 <= level < nlevels(idx) ||
        throw(ArgumentError("level $level is out of range 0:$(nlevels(idx) - 1)"))
    Int(level)
end

function _cl(idx::HNNGIndex, c::ClusterRef)
    _check_level(idx, c.level)
    cls = idx.levels[c.level+1].clusters
    (1 <= c.index <= length(cls) && cls[c.index].rep != 0) ||
        throw(ArgumentError("stale or invalid cluster handle $c"))
    cls[c.index]
end

"""Outlink of `id` at `level` (its nearest neighbour there), `0` if none."""
outlink(idx::HNNGIndex, id::Integer; level::Integer=0) =
    idx.levels[_check_level(idx, level)+1].outlink[_check_live(idx, id)]

"""Stored outlink distance of `id` at `level`, in the metric's units; `Inf` if none."""
outlink_distance(idx::HNNGIndex, id::Integer; level::Integer=0) =
    _to_user(idx, idx.levels[_check_level(idx, level)+1].outdist[_check_live(idx, id)])

"""Ids whose outlink at `level` is `id` (a sorted copy)."""
inlinks(idx::HNNGIndex, id::Integer; level::Integer=0) =
    copy(idx.levels[_check_level(idx, level)+1].inlinks[_check_live(idx, id)])

"""Handle of the cluster holding `id` at `level`; `nothing` if `id` does not reach that level."""
function cluster(idx::HNNGIndex, id::Integer; level::Integer=0)
    i = _check_live(idx, id)
    l = _check_level(idx, level)
    ci = idx.levels[l+1].cluster_of[i]
    ci == 0 ? nothing : ClusterRef(l, ci)
end

"""Member ids of a cluster (sorted copy)."""
members(idx::HNNGIndex, c::ClusterRef) = copy(_cl(idx, c).members)

"""Representative id of a cluster."""
representative(idx::HNNGIndex, c::ClusterRef) = _cl(idx, c).rep

"""Radius of a cluster in the metric's units: no level-0 element under it is farther from its representative."""
radius(idx::HNNGIndex, c::ClusterRef) = _to_user(idx, _cl(idx, c).radius)

"""The cluster one level up that contains the representative of `c`; `nothing` at the top."""
function Base.parent(idx::HNNGIndex, c::ClusterRef)
    r = _cl(idx, c).rep
    c.level + 1 < nlevels(idx) || return nothing
    ClusterRef(c.level + 1, idx.levels[c.level+2].cluster_of[r])
end

"""Child clusters: the level-`(l-1)` cluster of every member, in member order. Empty at level 0."""
function children(idx::HNNGIndex, c::ClusterRef)
    cl = _cl(idx, c)
    c.level == 0 && return ClusterRef[]
    below = idx.levels[c.level]
    [ClusterRef(c.level - 1, below.cluster_of[m]) for m in cl.members]
end

"""Cluster handles from `id`'s level-0 cluster up to the root."""
function path_to_root(idx::HNNGIndex, id::Integer)
    i = _check_live(idx, id)
    p = [ClusterRef(0, idx.levels[1].cluster_of[i])]
    while (q = parent(idx, p[end])) !== nothing
        push!(p, q)
    end
    p
end

"""Handle of the single top-level cluster; `nothing` for an empty index."""
function root(idx::HNNGIndex)
    _check_valid(idx)
    isempty(idx.levels) && return nothing
    ClusterRef(nlevels(idx) - 1, findfirst(c -> c.rep != 0, idx.levels[end].clusters))
end

"""Level-0 ids under `c`, sorted (the ROS `getLeafElementswithCluster`)."""
function leaves(idx::HNNGIndex, c::ClusterRef)
    cl = _cl(idx, c)
    c.level == 0 && return copy(cl.members)
    out = Int[]
    for ch in children(idx, c)
        append!(out, leaves(idx, ch))
    end
    sort!(out)
end

function _newick_name(s::AbstractString)
    occursin(r"[\s(),:;'\[\]]", s) ? "'" * replace(s, "'" => "''") * "'" : String(s)
end

function _newick!(io::IO, idx::HNNGIndex, c::ClusterRef, labels)
    cl = _cl(idx, c)
    print(io, "(")
    if c.level == 0
        for (i, m) in enumerate(cl.members)
            i > 1 && print(io, ",")
            name = labels === nothing ? string(m) : string(labels[m])
            print(io, _newick_name(name), ":", _to_user(idx, cl.repdist[i]))
        end
    else
        for (i, ch) in enumerate(children(idx, c))
            i > 1 && print(io, ",")
            _newick!(io, idx, ch, labels)
        end
    end
    print(io, "):", _to_user(idx, cl.radius))
end

"""
    newick(idx; labels=nothing) -> String

The hierarchy as a Newick tree (the ROS eTree export): one node per cluster with its
radius as branch length, one leaf per level-0 element named `labels[id]` (or the id)
with its distance to its cluster's representative as branch length, all in the
metric's units. `";"` for an empty index.
"""
function newick(idx::HNNGIndex; labels=nothing)
    r = root(idx)
    r === nothing && return ";"
    io = IOBuffer()
    _newick!(io, idx, r, labels)
    print(io, ";")
    String(take!(io))
end

"""
    level_graph(idx, l) -> (g::SimpleDiGraph, weights::SparseMatrixCSC, ids::Vector{Int})

Requires `using Graphs`. One vertex per level-`l` element (vertex `i` is `ids[i]`),
one edge per outlink, weighted by the stored outlink distance (metric units), so
`Graphs.dijkstra_shortest_paths(g, s, weights)` runs without a distance evaluation.
"""
function level_graph end
