# Persistence and statistics (spec §4, §4.3).

const FORMAT = "HNNG.jl index"
const FORMAT_VERSION = 1

"""
    HNNG.save(path, idx) -> path

Write the index with `Serialization` as two objects: a plain header
`(FORMAT, FORMAT_VERSION)`, a `String` and an `Int` that read back without any HNNG.jl
type, then the payload: the store (for a VectorStore only the live columns `1:n`, no
spare capacity), tombstones, levels, guard and, for a VectorStore with a Distances.jl
metric, the metric. An ObjectStore or a `UserMetric` index is saved without its metric;
`load` then needs `metric=`. Holds the index lock; throws `ArgumentError` on an index
invalidated by an exception in `add!`/`delete!`.
"""
save(path::AbstractString, idx::HNNGIndex) = @_locked idx _save(path, idx)

_live_store(s::VectorStore) = VectorStore(s.data[:, 1:s.n], s.n)
_live_store(s::ObjectStore) = s

function _save(path::AbstractString, idx::HNNGIndex)
    keep = idx.store isa VectorStore && !(idx.metric isa UserMetric)
    payload = (store=_live_store(idx.store), metric=keep ? idx.metric : nothing, guard=idx.guard,
               alive=idx.alive, levels=idx.levels)
    open(path, "w") do io
        serialize(io, (FORMAT, FORMAT_VERSION))
        serialize(io, payload)
    end
    path
end

"""Deserialize the next object of `path`; any failure (an empty, truncated or foreign file) is an `ArgumentError`."""
function _read_object(io::IO, path::AbstractString)
    try
        deserialize(io)
    catch e
        e isa InterruptException && rethrow()
        throw(ArgumentError("$path is not an HNNG.jl index file or is damaged: " * sprint(showerror, e)))
    end
end

_payload_ok(p) = p isa NamedTuple && all(k -> haskey(p, k), (:store, :metric, :guard, :alive, :levels)) &&
    p.store isa Union{VectorStore,ObjectStore} && p.levels isa Vector{Level} &&
    p.alive isa BitVector && p.guard isa Float64 && length(p.alive) == length(p.store)

"""
    HNNG.load(path; metric=nothing) -> HNNGIndex

Read an index written by `HNNG.save`. The header is read and checked before the
payload: a file written with another `FORMAT_VERSION` throws `ArgumentError` naming the
version, and an empty, truncated or foreign file throws `ArgumentError` ("not an HNNG.jl
index file or is damaged", with the underlying error). `metric=` is required for an
ObjectStore or a `UserMetric` index. When the file holds a metric, `metric=` may only
replace it by one of the same type (other parameters, e.g. `Minkowski(4.0)` for
`Minkowski(3.0)`, are accepted, and the caller answers for them); another type throws
`ArgumentError`. `nevals` starts at 0; the live counter `nlive` is not in the file and is
recomputed from the tombstones. Deserializing a file can construct arbitrary Julia
objects, so load only files you trust (`Serialization` is not a safe format for
untrusted input).
"""
function load(path::AbstractString; metric=nothing)
    p = open(path) do io
        h = _read_object(io, path)
        (h isa Tuple{String,Int} && h[1] == FORMAT) ||
            throw(ArgumentError("$path is not an HNNG.jl index file"))
        h[2] == FORMAT_VERSION || throw(ArgumentError(
            "$path has HNNG.jl index format version $(h[2]); this package reads version $FORMAT_VERSION"))
        _read_object(io, path)
    end
    _payload_ok(p) ||
        throw(ArgumentError("$path is not an HNNG.jl index file or is damaged: unexpected payload"))
    m = metric === nothing ? p.metric : _wrap_metric(metric)
    m === nothing && throw(ArgumentError(
        "the index in $path has an ObjectStore or a UserMetric; pass metric= to load it"))
    p.metric === nothing || Base.typename(typeof(m)) === Base.typename(typeof(p.metric)) ||
        throw(ArgumentError("the index in $path was saved with $(nameof(typeof(p.metric))); " *
                            "metric= gives $(nameof(typeof(m))), whose distances would not match the stored structure"))
    HNNGIndex(p.store, m, _is_metric_for(p.store, m), p.guard, p.alive, count(p.alive), p.levels,
              0, DistMemo(), 0, 0, false, ReentrantLock())
end

"""
    stats(idx) -> NamedTuple

`(n_live, n_total, nlevels, population, nclusters, mean_cluster_size, nevals)`:
per-level population and cluster counts (index `l+1` is level `l`), and the distance
evaluations made by `add!`, `delete!`, `bulk_add!` and the serial query path (`knn`,
`HNNG.knn_counted`, `knn_batch` with `threads=false`) since the last `reset_evals!`.
`knn_batch(...; threads=true)` and `HNNG.knn_levelsync` are not counted.
"""
stats(idx::HNNGIndex) = lock(() -> _stats(idx), idx.lock)

function _stats(idx::HNNGIndex)
    pop = [lv.npop for lv in idx.levels]
    ncl = [lv.nclusters for lv in idx.levels]
    (n_live=length(idx), n_total=_ntotal(idx), nlevels=nlevels(idx), population=pop,
     nclusters=ncl, mean_cluster_size=[c == 0 ? 0.0 : p / c for (p, c) in zip(pop, ncl)],
     nevals=idx.nevals)
end

"""Reset the distance-evaluation counter to 0."""
reset_evals!(idx::HNNGIndex) = lock(() -> (idx.nevals = 0; idx), idx.lock)

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
