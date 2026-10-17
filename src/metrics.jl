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
