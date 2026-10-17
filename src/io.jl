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
