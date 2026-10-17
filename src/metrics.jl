# Metric wrappers and the is_metric rules (spec §4.2).

"""
    UserMetric(f; is_metric=false)

Wrap any callable `f(a, b) -> Real` as the index distance, for either store. Set
`is_metric=true` only when `f` satisfies the triangle inequality: pruning then runs.
Declaring a non-metric `f` with `is_metric=true` is the opt-in heuristic mode of the
2015 ROS system: queries and inserts may miss neighbours and no error is raised.
"""
struct UserMetric{F}
    f::F
    is_metric::Bool
end
UserMetric(f; is_metric::Bool=false) = UserMetric(f, is_metric)

const EuclidFamily = Union{Euclidean,SqEuclidean,CosineDist}

_wrap_metric(m::PreMetric) = m
_wrap_metric(m::UserMetric) = m
_wrap_metric(f) = UserMetric(f; is_metric=false)

_is_metric_for(::VectorStore, ::Union{SqEuclidean,CosineDist}) = true
_is_metric_for(::Union{VectorStore,ObjectStore}, ::Metric) = true
_is_metric_for(::Union{VectorStore,ObjectStore}, ::PreMetric) = false
_is_metric_for(::Union{VectorStore,ObjectStore}, m::UserMetric) = m.is_metric

# Raw distance in internal units: Euclidean for the Euclidean family on a
# VectorStore, the metric's own value otherwise. Not counted.
@inline _rawdist(::HNNGIndex{<:VectorStore,<:EuclidFamily}, a, b) = _euclid(a, b)
@inline _rawdist(idx::HNNGIndex{<:Union{VectorStore,ObjectStore},<:UserMetric}, a, b) = Float64(idx.metric.f(a, b))
@inline _rawdist(idx::HNNGIndex, a, b) = Float64(idx.metric(a, b))

# Internal units -> the metric's own units. Monotone, so bounds convert too.
@inline _to_user(::HNNGIndex{<:VectorStore,<:SqEuclidean}, d::Float64) = d * d
@inline _to_user(::HNNGIndex{<:VectorStore,<:CosineDist}, d::Float64) = d * d / 2
@inline _to_user(::HNNGIndex, d::Float64) = d

"""Counted distance between two stored ids (internal units)."""
@inline function _d(idx::HNNGIndex, a::Int, b::Int)
    idx.nevals += 1
    _rawdist(idx, _point(idx.store, a), _point(idx.store, b))
end

"""Memoised distance from the pass query `q` to stored id `j`, counted in `memo.nevals`."""
@inline function _mdist(memo::DistMemo, idx::HNNGIndex, q, j::Int)
    @inbounds if memo.seen[j] == memo.epoch
        return memo.val[j]
    end
    d = _rawdist(idx, q, _point(idx.store, j))
    memo.nevals += 1
    @inbounds memo.seen[j] = memo.epoch
    @inbounds memo.val[j] = d
    d
end

"""
    _prepare(idx, x)

VectorStore: check the length and finiteness of `x`, convert to `Vector{T}`, and
normalise to unit length under `CosineDist` (`x / (‖x‖ + 1e-30)`, so a zero vector
stays zero; the sum of squares runs in at least `Float32`). ObjectStore: `x` unchanged.
"""
function _prepare(idx::HNNGIndex{VectorStore{T}}, x::AbstractVector{<:Real}) where {T}
    d = dim(idx.store)
    length(x) == d ||
        throw(ArgumentError("dimension mismatch: the index has dim $d, the input has length $(length(x))"))
    v = Vector{T}(undef, d)
    for (j, xj) in enumerate(x)
        vj = T(xj)
        isfinite(vj) || throw(ArgumentError("non-finite coordinate $xj at position $j"))
        v[j] = vj
    end
    _normalize!(idx, v)
end
_prepare(::HNNGIndex{<:ObjectStore}, x) = x

_normalize!(::HNNGIndex, v) = v
function _normalize!(::HNNGIndex{VectorStore{T},<:CosineDist}, v::Vector{T}) where {T<:AbstractFloat}
    S = promote_type(T, Float32)             # Float16 sums in Float32; S == T otherwise
    s = zero(S)
    for x in v
        s += S(x) * S(x)
    end
    v .*= one(S) / (sqrt(s) + S(1e-30))
    v
end

"""
    HNNG.distance(idx, a, b) -> Float64

Distance between two inputs as the index sees them (converted, normalised under
cosine), in the metric's own units. Not counted in `nevals`.
"""
distance(idx::HNNGIndex, a, b) = _to_user(idx, _rawdist(idx, _prepare(idx, a), _prepare(idx, b)))
