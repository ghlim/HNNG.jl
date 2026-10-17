# Persistence and statistics (spec §4, §4.3).

const FORMAT = "HNNG.jl index"
const FORMAT_VERSION = 1

"""
    HNNG.save(path, idx) -> path

Write the index with `Serialization`: store, tombstones, levels, guard and, for a
VectorStore with a Distances.jl metric, the metric. An ObjectStore or a `UserMetric`
index is saved without its metric; `load` then needs `metric=`.
"""
function save(path::AbstractString, idx::HNNGIndex)
    keep = idx.store isa VectorStore && !(idx.metric isa UserMetric)
    payload = (format=FORMAT, version=FORMAT_VERSION, store=idx.store,
               metric=keep ? idx.metric : nothing, guard=idx.guard, alive=idx.alive, levels=idx.levels)
    open(io -> serialize(io, payload), path, "w")
    path
end

"""
    HNNG.load(path; metric=nothing) -> HNNGIndex

Read an index written by `HNNG.save`. `metric=` is required for an ObjectStore or a
`UserMetric` index and overrides the saved metric otherwise. `nevals` starts at 0; the
live counter `nlive` is not in the file and is recomputed from the tombstones. A file
written with another `FORMAT_VERSION` throws `ArgumentError`. Deserializing a file can
construct arbitrary Julia objects, so load only files you trust (`Serialization` is not
a safe format for untrusted input).
"""
function load(path::AbstractString; metric=nothing)
    p = open(deserialize, path)
    (p isa NamedTuple && get(p, :format, nothing) == FORMAT) ||
        throw(ArgumentError("$path is not an HNNG.jl index file"))
    get(p, :version, 0) == FORMAT_VERSION || throw(ArgumentError(
        "$path has HNNG.jl index format version $(get(p, :version, 0)); this package reads version $FORMAT_VERSION"))
    m = metric === nothing ? p.metric : _wrap_metric(metric)
    m === nothing && throw(ArgumentError(
        "the index in $path has an ObjectStore or a UserMetric; pass metric= to load it"))
    HNNGIndex(p.store, m, _is_metric_for(p.store, m), p.guard, p.alive, count(p.alive), p.levels,
              0, DistMemo(), 0, 0)
end

"""
    stats(idx) -> NamedTuple

`(n_live, n_total, nlevels, population, nclusters, mean_cluster_size, nevals)`:
per-level population and cluster counts (index `l+1` is level `l`), and the distance
evaluations made by `add!`, `delete!`, `bulk_add!` and the serial `knn` path since
the last `reset_evals!`.
"""
function stats(idx::HNNGIndex)
    pop = [lv.npop for lv in idx.levels]
    ncl = [lv.nclusters for lv in idx.levels]
    (n_live=length(idx), n_total=_ntotal(idx), nlevels=nlevels(idx), population=pop,
     nclusters=ncl, mean_cluster_size=[c == 0 ? 0.0 : p / c for (p, c) in zip(pop, ncl)],
     nevals=idx.nevals)
end

"""Reset the distance-evaluation counter to 0."""
reset_evals!(idx::HNNGIndex) = (idx.nevals = 0; idx)

"""
    HNNG.last_update_stats(idx) -> (rewired, top_level_touched)

For the most recent `add!` or `delete!`: the number of live level-0 elements whose
outlink changed (excluding the inserted element's own outlink and the deleted
element), and the highest level whose outlinks, clusters or representatives changed
(for an update that dropped levels or emptied the index, the highest level that
existed before). A cleared outlink counts as changed: deleting the second-to-last
element counts the survivor, whose outlink becomes 0. For a matrix `add!(idx, X)`
the result reflects the last inserted column only (the most recent single
insertion). `bulk_add!` leaves it unchanged. `(0, 0)` before any update.
"""
last_update_stats(idx::HNNGIndex) = (rewired=idx.last_rewired, top_level_touched=idx.last_top)
