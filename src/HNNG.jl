module HNNG

using Distances: Distances, PreMetric, Metric, Euclidean, SqEuclidean, CosineDist
using DataStructures: heappush!, heappop!
using Serialization: serialize, deserialize

export HNNGIndex, UserMetric, Euclidean, SqEuclidean, CosineDist
export is_metric, nlevels, element, verify, add!, knn, knn_batch, stats, reset_evals!, bulk_add!
export outlink, outlink_distance, inlinks, cluster, members, representative, radius,
       children, path_to_root, leaves, root, newick, level_graph

include("types.jl")
include("kernels.jl")
include("metrics.jl")
include("insert.jl")
include("delete.jl")
include("search.jl")
include("bulk.jl")
include("navigate.jl")
include("verify.jl")
include("io.jl")

end # module
