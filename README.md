# HNNG.jl

Exact, parameter-free nearest-neighbour search with a hierarchy of nearest-neighbour
graphs (hNNG). Every level is the exact 1-NN graph of its population; its connected
components are clusters with an elected representative and a radius; the
representatives form the next level until one cluster remains. Search is exact
(recall 1.0 by construction), there is nothing to tune (no `M`, `ef`, leaf size or
bucket size), elements can be any Julia object under any distance, and insert and
delete are exact and incremental.

This package ports the C++ reference implementation `hnnglib` (header `hnnalg.h`)
with the divergences listed in its design spec (§12).

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/ghlim/HNNG.jl")
```

Julia 1.10 or later.

## Quick start

```julia
using HNNG
idx = HNNGIndex(Euclidean(), 20)          # columns of Float32; T=Float64 also works
ids = add!(idx, rand(Float32, 20, 1000))  # 1:1000, assigned in order, never reused
nn, d = knn(idx, rand(Float32, 20), 10)   # exact; ties broken by smaller id
delete!(idx, 17)                          # exact, no rebuild
verify(idx)                               # checks every invariant from scratch
stats(idx)                                # levels, cluster counts, distance evaluations
```

`Euclidean()`, `SqEuclidean()` and `CosineDist()` use a specialised kernel; results
come back in the metric's own units (`CosineDist` returns `1 - cos`). Ties are
resolved in the index's internal Euclidean units (the units in which the k-th element
is selected); after conversion to SqEuclidean or CosineDist units two distinct
internal distances can print equal, and such a pair keeps its internal order. Any other
Distances.jl metric works too; a `SemiMetric` or `PreMetric` runs unpruned and stays
exact. `bulk_add!(idx, X)` builds the same hierarchy as `add!(idx, X)` on an empty
index with BLAS. `HNNG.save(path, idx)` and `HNNG.load(path)` persist an index.
`HNNG.load` reads Serialization files: load only files you trust, and files are not
guaranteed to load across Julia versions.

A metric that returns NaN raises an `ArgumentError` naming the pair. When the metric
throws (or returns NaN) or the call is interrupted during `add!` or `delete!`, the
exception propagates; if that happens during the first nearest-neighbour search (the
usual case: the metric fails on the new element) the index is left exactly as it was
and no id is used. A failure after that point marks the index invalid: further calls
throw `ArgumentError`, and the index must be rebuilt or loaded from a saved copy.
`add!(idx, X)` inserts column by column, so the columns before the failing one stay.
A failed `bulk_add!` leaves the index empty.

`knn_batch(idx, Q, k)` answers the columns of `Q` at once. With `threads=true` it
spreads them over `Threads.nthreads()` tasks, which needs a thread-safe `UserMetric`,
and it does not update the distance-evaluation counter. Each index has a lock that
`add!`, `delete!`, `bulk_add!`, `knn`, `knn_batch`, `verify` and `HNNG.save` hold for
their whole duration, so calls on one index from several threads (for example `knn`
inside `Threads.@threads`) are serialized: correct, not parallel. Use
`knn_batch(idx, Q, k; threads=true)` for parallel queries. The navigation functions
below read the stored structure without the lock: they are safe alongside queries,
not alongside `add!`, `delete!` or `bulk_add!`.

## Any element, any distance

```julia
hausdorff(A, B) = max(maximum(minimum(sqrt(sum(abs2, a .- b)) for b in eachcol(B)) for a in eachcol(A)),
                      maximum(minimum(sqrt(sum(abs2, a .- b)) for a in eachcol(A)) for b in eachcol(B)))
views = HNNGIndex(UserMetric(hausdorff; is_metric=true), Matrix{Float64})
add!(views, [rand(3, rand(2:9)) for _ in 1:100])   # descriptor sets of varying size
knn(views, rand(3, 4), 5)
```

`UserMetric(f; is_metric=true)` turns pruning on and is exact when `f` satisfies the
triangle inequality. With `is_metric=false` (the default, and what a bare function
gets) the index is exact but unpruned. Declaring a non-metric distance as a metric is
the heuristic mode of the 2015 ROS system: faster, and it may miss neighbours.

## Navigation without distance evaluations

`outlink`, `outlink_distance`, `inlinks`, `cluster`, `members`, `representative`,
`radius`, `parent`, `children`, `path_to_root`, `leaves`, `root` and `newick` read
the stored structure only. Cluster handles are `HNNG.ClusterRef` values; the type is
not exported, so write `HNNG.ClusterRef` to construct or dispatch on one. With
`using Graphs`, `level_graph(idx, l)` returns a `SimpleDiGraph`, its weight matrix
and the vertex ids, so graph algorithms run on stored distances. With `using Graphs`,
write `HNNG.radius` (Graphs exports its own `radius`).

### Recipe 1: recognition by the nearest stored view

The caller keeps a `labels` vector indexed by id:

```julia
recognise(idx, labels, q) = labels[first(first(knn(idx, q, 1)))]
```

### Recipe 2: maximal single-label subtrees of an object

The parts of the hierarchy that belong to one label only (the 2015 ROS
`ObjectRepresentation`), found by walking `children` and `leaves`:

```julia
function single_label_subtrees(idx, labels, L)
    out = Vector{Vector{Int}}()
    stack = [root(idx)]
    while !isempty(stack)
        c = pop!(stack)
        ls = leaves(idx, c)
        if all(i -> labels[i] == L, ls)
            push!(out, ls)
        elseif any(i -> labels[i] == L, ls)
            if c.level == 0
                append!(out, [[i] for i in ls if labels[i] == L])
            else
                append!(stack, children(idx, c))
            end
        end
    end
    out
end
```

### Nearest-neighbour chains and shortest paths

```julia
using Graphs
chain = [1]                                   # follow outlinks to the mutual pair
while outlink(idx, outlink(idx, chain[end])) != chain[end]
    push!(chain, outlink(idx, chain[end]))
end
g, W, ids = level_graph(idx, 0)
dijkstra_shortest_paths(g, 1, W).dists        # no new distance evaluation
```

## How hNNG compares

| Property | hNNG | HNSW | KD-tree | Ball tree | Cover tree | Brute force |
|---|---|---|---|---|---|---|
| Exact result guaranteed | yes | no (approximate) | yes | yes | yes | yes |
| Parameter-free | yes | no (M, efConstruction, ef) | leaf size | leaf size | base | yes |
| Variable-size elements, any metric | yes | metric by distance function, fixed dim | fixed dim, Minkowski | fixed dim, metric | any metric | any |
| Online insert | yes, exact | yes | rebuild | rebuild | yes | trivial |
| Delete without rebuild | yes, exact | mark only | rebuild | rebuild | partial | trivial |
| Hierarchy of data clusters as a by-product | yes | random levels | splits, not clusters | balls, not clusters | nested covers | none |
| Navigation along stored links | yes (NN chains, shortest paths) | approximate links | none | none | tree only | none |
| Query cost versus intrinsic dimension | degrades (exact pruning weakens) | mild | degrades | degrades | degrades | flat |

## Exactness

Exactness is relative to the package's own distance kernel. Every prune applies a
relative guard γ in favour of visiting, which absorbs rounding in the triangle
inequality: `max(1e-6, 4·d·eps(T))` for vectors of dimension `d` and type `T`, `1e-6`
for objects, or the value passed as `HNNGIndex(...; guard=γ)`. `Float32` data
accumulates in `Float32`; every stored distance, radius and bound is `Float64`.

## Tests

`julia --project=. -e 'using Pkg; Pkg.test()'` runs the default suite (under a
minute). `HNNG_SLOW=1` adds the runs on the ann-benchmarks HDF5 files, the larger
knn cases (n = 2000) and the full-size ObjectStore cases.
