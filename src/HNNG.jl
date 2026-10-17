module HNNG

using Distances: Distances, PreMetric, Metric, Euclidean, SqEuclidean, CosineDist

export HNNGIndex, UserMetric, Euclidean, SqEuclidean, CosineDist
export is_metric, nlevels, element

include("types.jl")
include("kernels.jl")
include("metrics.jl")
include("navigate.jl")

end # module
