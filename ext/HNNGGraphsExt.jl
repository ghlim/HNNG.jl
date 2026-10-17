module HNNGGraphsExt

using HNNG
using HNNG: HNNGIndex, _check_level, _to_user
using Graphs: SimpleDiGraph, add_edge!
using SparseArrays: sparse

function HNNG.level_graph(idx::HNNGIndex, l::Integer)
    lv = idx.levels[_check_level(idx, l)+1]
    ids = findall(lv.present)
    pos = Dict(x => i for (i, x) in enumerate(ids))
    g = SimpleDiGraph(length(ids))
    I, J, V = Int[], Int[], Float64[]
    for (i, x) in enumerate(ids)
        y = lv.outlink[x]
        y == 0 && continue
        add_edge!(g, i, pos[y])
        push!(I, i)
        push!(J, pos[y])
        push!(V, _to_user(idx, lv.outdist[x]))
    end
    g, sparse(I, J, V, length(ids), length(ids)), ids
end

end # module
