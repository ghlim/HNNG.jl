module HNNG

using Distances: Distances, PreMetric, Metric, Euclidean, SqEuclidean, CosineDist
using DataStructures: heappush!, heappop!

export HNNGIndex, UserMetric, Euclidean, SqEuclidean, CosineDist
export is_metric, nlevels, element, verify, add!, knn, knn_batch
export outlink, outlink_distance, inlinks, cluster, members, representative, radius,
       children, path_to_root, root

include("types.jl")
include("kernels.jl")
include("metrics.jl")
include("insert.jl")
include("delete.jl")
include("search.jl")
include("navigate.jl")
include("verify.jl")

end # module
