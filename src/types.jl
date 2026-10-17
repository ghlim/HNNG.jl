# Storage, index, level and cluster types (spec §5).

"""
    VectorStore{T}

Elements are the columns `1:n` of a `d × capacity` matrix that doubles when full.
"""
mutable struct VectorStore{T<:AbstractFloat}
    data::Matrix{T}
    n::Int
end
VectorStore{T}(dim::Int) where {T<:AbstractFloat} = VectorStore{T}(Matrix{T}(undef, dim, 0), 0)

"""
    ObjectStore{E}

Elements are arbitrary Julia objects of type `E`.
"""
struct ObjectStore{E}
    objs::Vector{E}
end
ObjectStore{E}() where {E} = ObjectStore{E}(E[])

Base.length(s::VectorStore) = s.n
Base.length(s::ObjectStore) = length(s.objs)
dim(s::VectorStore) = size(s.data, 1)

@inline _point(s::VectorStore, i::Int) = view(s.data, :, i)
@inline _point(s::ObjectStore, i::Int) = @inbounds s.objs[i]

function _append!(s::VectorStore{T}, x::AbstractVector{T}) where {T}
    if s.n == size(s.data, 2)
        grown = Matrix{T}(undef, size(s.data, 1), max(16, 2 * s.n))
        copyto!(grown, 1, s.data, 1, length(s.data))
        s.data = grown
    end
    s.n += 1
    copyto!(view(s.data, :, s.n), x)
    s.n
end
_append!(s::ObjectStore{E}, x::E) where {E} = (push!(s.objs, x); length(s.objs))

"""
    Cluster

One connected component of a level's nearest-neighbour graph. `members` is sorted
by id and `repdist[i]` is the distance from `rep` to `members[i]` (internal units).
`radius` and `maxout[t+1]` (t = 0..level) follow spec §6.2. A dead slot has `rep == 0`.
"""
mutable struct Cluster
    rep::Int
    members::Vector{Int}
    repdist::Vector{Float64}
    radius::Float64
    maxout::Vector{Float64}
end
_dead_cluster() = Cluster(0, Int[], Float64[], 0.0, Float64[])

"""
    Level

One level, indexed by id (length = ids ever assigned). `outlink[i] == 0` and
`outdist[i] == Inf` mean no outlink; `inlinks[i]` is sorted; `cluster_of[i] == 0`
means absent. Dead cluster slots are listed in `free` and reused last in, first out.
`npop` is the population size, `nclusters` the number of live clusters.
"""
mutable struct Level
    outlink::Vector{Int}
    outdist::Vector{Float64}
    inlinks::Vector{Vector{Int}}
    present::BitVector
    cluster_of::Vector{Int}
    clusters::Vector{Cluster}
    free::Vector{Int}
    npop::Int
    nclusters::Int
end
Level(n::Int) = Level(zeros(Int, n), fill(Inf, n), [Int[] for _ in 1:n], falses(n),
                      zeros(Int, n), Cluster[], Int[], 0, 0)

function _grow!(lv::Level, n::Int)
    while length(lv.outlink) < n
        push!(lv.outlink, 0)
        push!(lv.outdist, Inf)
        push!(lv.inlinks, Int[])
        push!(lv.present, false)
        push!(lv.cluster_of, 0)
    end
    lv
end

"""
    DistMemo

Per-pass distance memo: `val[i]` is valid iff `seen[i] == epoch`; `mark[i] == epoch`
records that `i` was offered or expanded in this pass. `front` (min-heap of
`(LB, level, cluster index)`) and `res` (max-heap of `(distance, id)`) are reused
buffers. `nevals` accumulates until the owner folds it into the index.
"""
mutable struct DistMemo
    val::Vector{Float64}
    seen::Vector{UInt32}
    mark::Vector{UInt32}
    epoch::UInt32
    nevals::Int
    front::Vector{Tuple{Float64,Int,Int}}
    res::Vector{Tuple{Float64,Int}}
end
DistMemo() = DistMemo(Float64[], UInt32[], UInt32[], UInt32(0), 0,
                      Tuple{Float64,Int,Int}[], Tuple{Float64,Int}[])

"""Start a new pass over at most `n` ids: bump the epoch and empty the heap buffers."""
function _begin!(m::DistMemo, n::Int)
    old = length(m.seen)
    if old < n
        resize!(m.val, n)
        resize!(m.seen, n)
        resize!(m.mark, n)
        fill!(view(m.seen, old+1:n), UInt32(0))
        fill!(view(m.mark, old+1:n), UInt32(0))
    end
    if m.epoch == typemax(UInt32)
        fill!(m.seen, UInt32(0))
        fill!(m.mark, UInt32(0))
        m.epoch = UInt32(0)
    end
    m.epoch += UInt32(1)
    empty!(m.front)
    empty!(m.res)
    m
end

"""
    HNNGIndex(metric, dim::Int; T=Float32)   # VectorStore: columns of T
    HNNGIndex(metric, ::Type{E})             # ObjectStore{E}

An exact, parameter-free hierarchical nearest-neighbour-graph index. `metric` is a
Distances.jl `PreMetric`, a `UserMetric`, or any callable `f(a, b) -> Real` (taken as
`UserMetric(f; is_metric=false)`). `Euclidean`, `SqEuclidean` and `CosineDist` on a
VectorStore use the specialised kernel and work in Euclidean units internally;
results are always in the metric's own units. `guard` is the relative prune guard
γ: `nothing` gives `max(1e-6, 4·d·eps(T))` for a VectorStore and `1e-6` for an
ObjectStore.
"""
mutable struct HNNGIndex{S<:Union{VectorStore,ObjectStore},M}
    store::S
    metric::M
    is_metric::Bool
    guard::Float64              # relative prune guard γ (spec §5)
    alive::BitVector
    nlive::Int                  # live elements, == count(alive); length(idx) is O(1)
    levels::Vector{Level}       # levels[l+1] is level l
    nevals::Int
    memo::DistMemo
    last_rewired::Int           # see HNNG.last_update_stats (Task 18)
    last_top::Int
end

const GUARD = 1e-6

"""Default relative prune guard of a VectorStore index: `max(1e-6, 4·d·eps(T))`."""
_default_guard(::Type{T}, d::Int) where {T<:AbstractFloat} = max(GUARD, 4 * d * Float64(eps(T)))

function HNNGIndex(metric, dim::Integer; T::Type{<:AbstractFloat}=Float32,
                   guard::Union{Nothing,Real}=nothing)
    dim >= 1 || throw(ArgumentError("dim must be at least 1, got $dim"))
    m = _wrap_metric(metric)
    s = VectorStore{T}(Int(dim))
    g = guard === nothing ? _default_guard(T, Int(dim)) : Float64(guard)
    HNNGIndex(s, m, _is_metric_for(s, m), g, falses(0), 0, Level[], 0, DistMemo(), 0, 0)
end

function HNNGIndex(metric, ::Type{E}; guard::Union{Nothing,Real}=nothing) where {E}
    m = _wrap_metric(metric)
    s = ObjectStore{E}()
    g = guard === nothing ? GUARD : Float64(guard)
    HNNGIndex(s, m, _is_metric_for(s, m), g, falses(0), 0, Level[], 0, DistMemo(), 0, 0)
end

_ntotal(idx::HNNGIndex) = length(idx.store)

"""Number of live (not deleted) elements; O(1), read from the counter `nlive`."""
Base.length(idx::HNNGIndex) = idx.nlive

"""Number of levels; 0 for an empty index."""
nlevels(idx::HNNGIndex) = length(idx.levels)

"""`true` iff pruning runs: the metric satisfies (or is declared to satisfy) the triangle inequality."""
is_metric(idx::HNNGIndex) = idx.is_metric

function Base.show(io::IO, idx::HNNGIndex)
    print(io, "HNNGIndex(", nameof(typeof(idx.metric)), ", ", length(idx), " live, ",
          nlevels(idx), " levels)")
end
