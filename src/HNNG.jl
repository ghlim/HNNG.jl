module HNNG

using Distances: Distances, PreMetric, Metric, Euclidean, SqEuclidean, CosineDist
using DataStructures: heappush!, heappop!
using Serialization: serialize, deserialize
using LinearAlgebra: mul!

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

# Precompile workload: keeps spec §10's test-time budget and cuts first-call latency.
if ccall(:jl_generating_output, Cint, ()) == 1
    let
        for M in (Euclidean(), SqEuclidean(), CosineDist()), T in (Float32, Float64, Float16)
            X = T[sin(0.7 * i + 1.3 * j) + cos(0.37 * i * j) for i in 1:4, j in 1:60]
            q = T[cos(0.5 * i) for i in 1:4]
            idx = HNNGIndex(M, 4; T=T)
            add!(idx, X)
            add!(idx, q .+ one(T))
            knn(idx, q, 3)
            knn_levelsync(idx, q, 3)
            knn_batch(idx, X[:, 1:2], 2)
            delete!(idx, 3)
            verify(idx) || error("precompile workload: verify failed")
            stats(idx)
            b = HNNGIndex(M, 4; T=T)
            bulk_add!(b, X)
            verify(b) || error("precompile workload: verify failed")
        end
    end
end

end # module
