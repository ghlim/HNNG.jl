module HNNG

using Distances: Distances, PreMetric, Metric, Euclidean, SqEuclidean, CosineDist

export HNNGIndex, UserMetric, Euclidean, SqEuclidean, CosineDist
export is_metric, nlevels

include("types.jl")
include("metrics.jl")

end # module
