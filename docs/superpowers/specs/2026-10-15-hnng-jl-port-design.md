# HNNG.jl — design spec (Julia port of the hNNG index)

Date: 2026-10-15. Status: draft for author review. Companion spec: the ISR
manuscript spec in
`publications/Hierarchical-Nearest-Neighbor-Graph/docs/superpowers/specs/2026-10-15-isr2026-manuscript-design.md`.

## 1. Purpose and intent

HNNG.jl is the implementation that the Intelligent Service Robotics (ISR)
submission reports. It ports the C++ reference at `/home/coral/working/hnng`
(commit c73f09d, header `hnnglib/hnnalg.h`) to idiomatic Julia with
**identical algorithm semantics** (approach B agreed in brainstorming:
Julia style and data layout, same exact structure and same exact results).

The 2015 ROS implementation (`archives/race/raceua/race_view_clustering`) was
reviewed against this spec on 2026-10-16; the review with line citations is
`docs/notes/2026-10-16-ros-review.md` and its findings are folded in below.

The package must make three merits of hNNG concrete and testable, because the
manuscript builds its story on them:

1. **Non-fixed-dimension elements and arbitrary metrics.** An element can be
   any Julia object (for example a set of SIFT descriptors whose count varies
   per object; in the 2015 ROS system an element was one object view, a
   variable-size set of 153-value spin images) under any user distance. The
   algorithm only ever calls `distance(idx, a, b)`.
2. **Online and incremental learning.** Insert and delete are first-class and
   exact; the hierarchy is never rebuilt from scratch. Updates touch the
   root-to-leaf path and the affected clusters only (memory rule
   `hnng_incremental_design`).
3. **Navigation along stored links.** Outlinks, inlinks, cluster membership,
   representatives, radii and parents are readable without computing any
   distance, so a caller can follow nearest-neighbour chains or run shortest
   paths over the stored graph.

Plus the two properties the C++ work established: search is **exact** (recall
1.0 by construction) and the index is **parameter-free** (no `M`, `ef`,
`efConstruction`, leaf size or bucket size).

Success criteria: every test in §10 passes; the benchmark harness in §11
produces the CSV, LaTeX and PDF outputs that the manuscript inputs; the
package builds on Julia 1.10 LTS and the current stable release.

## 2. Scope and non-goals

In scope: the index (`src/`), its tests (`test/`), the paper benchmark
harness (`bench/`), a README with the comparison table of §8, a GitHub
release with a Zenodo DOI for the availability statement.

Out of scope (decided during brainstorming): million-scale datasets
(SIFT-1M, GloVe), the scan-fallback budget and latches of the C++ engine,
intra-query threading and prefetch, the engine ablation, registration in the
General registry, hnswlib-compatible on-disk format, inner-product
(unnormalised dot product) space, and any change to the C++ repository, which
stays the reference implementation.

The implementation plan may split `bench/` into its own phase once the index
and its tests are green; the index, tests and harness share this one spec.

## 3. Repository, layout, dependencies

Repository: `/home/coral/working/HNNG.jl`, its own git repository, to be
published at `github.com/ghlim/HNNG.jl`. The package name `HNNG` is free in
the General registry (checked 2026-10-15).

```
HNNG.jl/
├── Project.toml            name = "HNNG", julia = "1.10"
├── src/
│   ├── HNNG.jl             module, exports, includes
│   ├── metrics.jl          metric wrappers, is_metric, unit conversion, kernels
│   ├── types.jl            HNNGIndex, VectorStore, ObjectStore, Level, Cluster
│   ├── kernels.jl          SIMD Euclidean kernel, tiled all-pairs for bulk
│   ├── insert.jl           add!, link!, repair, bound refresh, rep deltas
│   ├── delete.jl           delete!, unlink!
│   ├── search.jl           knn, knn_batch, best-first engine, level-sync oracle
│   ├── bulk.jl             bulk loader (VectorStore only)
│   ├── navigate.jl         outlink, inlinks, cluster, members, parent, ...
│   ├── verify.jl           verify(idx) invariant checker
│   └── io.jl               save, load, stats
├── ext/HNNGGraphsExt.jl    package extension on Graphs.jl
├── test/
│   ├── runtests.jl, oracle.jl, invariants.jl, order.jl, updates.jl,
│   │   objects.jl, navigate.jl, io.jl, parity_cpp.jl, recall_rule.jl
│   └── fixtures/           C++ parity fixtures (generated once, committed)
├── bench/                  own Project.toml (see §11)
└── docs/superpowers/specs/ this file
```

Dependencies: `Distances` (metric types and `Metric`/`SemiMetric`
hierarchy), `LinearAlgebra` (BLAS for the bulk loader and norms),
`DataStructures` (binary heaps), `Serialization` (stdlib, for save/load).
Weak dependency: `Graphs` (extension, which also uses the `SparseArrays` stdlib). Test dependencies: `Test`,
`StableRNGs`, `Graphs`, `HDF5` (slow tests only), `JSON3` (fixtures).

Julia is not installed on this machine. Install the stable release through
juliaup (`curl -fsSL https://install.julialang.org | sh`, user-level); the
package `compat` lower bound is 1.10 so the LTS also works.

## 4. Public API

```julia
HNNGIndex(metric, dim::Int; T=Float32)      # VectorStore: d×n matrix of T
HNNGIndex(metric, ::Type{E})                # ObjectStore{E}: Vector{E}

add!(idx, x::AbstractVector)        -> Int            # one column (VectorStore)
add!(idx, X::AbstractMatrix)        -> UnitRange{Int} # columns, in order
add!(idx, obj::E)                   -> Int            # ObjectStore
add!(idx, objs::AbstractVector{E})  -> UnitRange{Int}
delete!(idx, id::Int)               -> idx

knn(idx, q, k::Int)                 -> (ids::Vector{Int}, dists::Vector{Float64})
knn_batch(idx, Q, k; threads=false) -> (ids::Matrix{Int}, dists::Matrix{Float64})
                                    # Q: matrix columns (VectorStore) or Vector{E} (ObjectStore)

bulk_add!(idx, X::AbstractMatrix)   -> UnitRange{Int} # VectorStore only, empty index only

length(idx)        # live elements
nlevels(idx)       # number of levels (0 for an empty index)
stats(idx)         # NamedTuple, see §4.3
verify(idx)        # throws on the first violated invariant, returns true

save(path, idx); load(path; metric=nothing)
```

### 4.1 Ids, elements, order

Ids are `Int`, assigned 1, 2, 3, … in `add!` order and never reused. A
deleted id stays a tombstone: `element(idx, id)` and navigation calls throw
`ArgumentError`, `knn` never returns it, and `length` excludes it. Column
storage for deleted elements is kept (no compaction in this version).

`knn` returns ids sorted by ascending distance, ties by ascending id. With
fewer than `k` live elements it returns all of them. `knn` on an empty index
returns two empty vectors. `k < 1` throws.

### 4.2 Metrics and units

- `Euclidean()`, `SqEuclidean()`, `CosineDist()` from Distances.jl get the
  specialised VectorStore kernels. The index works in Euclidean units
  internally for all three. `SqEuclidean` results are squared on output;
  `CosineDist` normalises every inserted vector and every query to unit
  length and returns `e^2/2` where `e` is the Euclidean distance between the
  unit vectors (this equals `1 - cos`). The C++ returns raw squared L2; the
  Julia package returns **the metric's own units** instead.
- Any other Distances.jl `PreMetric` is called on column views.
  `is_metric(idx)` is `true` iff the metric is a `Distances.Metric`
  (triangle inequality holds); a `SemiMetric` or `PreMetric` runs unpruned.
- `UserMetric(f; is_metric=false)` wraps any callable `f(a, b) -> Real` for
  either store. The user flips `is_metric=true` when `f` satisfies the
  triangle inequality. A bare function passed as the metric is treated as
  `UserMetric(f; is_metric=false)`.
- Non-metric distances give an **exact but unpruned** index: every lower
  bound is 0, so inserts and queries evaluate every element at the levels
  they visit. Declaring `is_metric=true` for a distance that is not a metric
  is the opt-in heuristic mode: pruning runs and results may miss neighbours,
  with no error raised. That is how the 2015 ROS system ran (it pruned with a
  non-metric view distance), and the docstring says so. No separate mode
  exists.
- Labels (object names) are not stored in the index. The caller keeps a
  `labels` vector indexed by id; recognition of a query view is
  `labels[first(first(knn(idx, q, 1)))]`, shown in the README.

### 4.3 `stats(idx)`

Returns `(n_live, n_total, nlevels, population::Vector{Int},
nclusters::Vector{Int}, mean_cluster_size::Vector{Float64}, nevals::Int)`.
`nevals` counts every distance evaluation made by `add!`, `delete!`,
`bulk_add!` and the serial `knn` path since the last `reset_evals!(idx)`.
The counter is a plain `Int` increment and is always on. The internal
`HNNG.knn_counted(idx, q, k)` returns `(ids, dists, nevals_this_query)` and
is what the benchmark harness calls; `knn_batch(...; threads=true)` does not
update the index counter.

## 5. Data model

```julia
struct VectorStore{T}   # d×n matrix, columns are elements, grows by append
struct ObjectStore{E}   # Vector{E}
mutable struct HNNGIndex{S<:Union{VectorStore,ObjectStore}, M}
    store::S; metric::M; is_metric::Bool
    alive::BitVector                      # tombstones
    levels::Vector{Level}                 # levels[l+1] is level l
    nevals::Int
    memo::DistMemo                        # per-query memo for the serial path
end
mutable struct Level
    outlink::Vector{Int}      # 0 = none; indexed by id, length n_total
    outdist::Vector{Float64}  # metric units; Inf when outlink == 0
    inlinks::Vector{Vector{Int}}
    present::BitVector        # id is in this level's population
    cluster_of::Vector{Int}   # cluster index at this level, 0 if absent
    clusters::Vector{Cluster} # slots may be free-listed after merges
end
mutable struct Cluster
    rep::Int
    members::Vector{Int}
    repdist::Vector{Float64}  # distance rep -> member, same order as members
    radius::Float64           # §6.2
    maxout::Vector{Float64}   # maxout[t+1] for t = 0..level, §6.2
end
```

All stored distances, radii and bounds are `Float64`, whatever `T` is. The
Euclidean kernel for `Float32` data accumulates in `Float32` (as BLAS and the
C++ do); the result is widened on store. Every prune comparison applies a
relative guard of `1e-6` in favour of visiting, to absorb rounding in the
triangle inequality. Exactness is defined relative to the package's own
kernel; the brute-force oracle in the tests uses the same kernel.

Memory: level 0 holds every element; upper levels are sparse and indexed by
id through `present`/`cluster_of`, which is simple and costs `O(n)` ints per
level. Levels are few (cluster size ≈ 6 shrinks the population about 6× per
level), so this is acceptable for the scales in scope.

## 6. Algorithm semantics

### 6.1 Invariants (hold at rest, checked by `verify`)

For every level `l` with population `P_l`:

1. `P_0` = all live elements. `P_{l+1}` = representatives of the level-`l`
   clusters.
2. Every element of `P_l` has exactly one outlink when `|P_l| ≥ 2`, and it
   is its exact nearest neighbour in `P_l \ {self}`; ties go to the smallest
   id. With `|P_l| = 1` the element has no outlink.
3. `inlinks` is the exact reverse index of `outlink`.
4. Clusters are the connected components of the level-`l` graph with
   outlinks taken as undirected edges. Each element of `P_l` is in exactly
   one cluster.
5. The representative of a cluster is chosen by: most inlinks at this level;
   then smallest sum of the inlinkers' outlink distances (metric units);
   then smallest id.
6. `radius` and `maxout` equal the definitions of §6.2.
7. The top level has exactly one cluster; no level above it exists. A level
   with one cluster is the top. An empty index has no levels.

Because every tie is broken canonically, the hierarchy at rest is a pure
function of the live point set (up to id relabelling). Streaming inserts in
any order, deletes followed by re-inserts, and the bulk loader must produce
the same structure; §10 tests this. Order independence is new relative to
both earlier implementations: the ROS code breaks ties by key or member order
with an unstable sort, its rewire sweep can miss candidates and it never
refreshes ancestor radii (review D4, D8, D12).

### 6.2 Bounds

For a cluster `C` at level `L` with representative `r`, and for a member
`m`, let `child(m)` be the level-`(L-1)` cluster that contains `m` (at rest
`rep(child(m)) == m`; during an update it may not be). Then

```
radius(C)   = max over m in C of ( M(r, m) + M(m, rep(child(m))) + radius(child(m)) )
radius at level 0 = max over m of M(r, m)
maxout[t](C) = max over m in C of ( t == L ? outdist_L(m) : maxout[t](child(m)) ),
               +Inf if any element counted has no outlink at level t
```

`radius(C)` bounds the distance from `r` to every level-0 element reachable
by descent from `C`, and stays valid in the intermediate states of an update
because the middle term covers a member that is no longer its child's
representative. `maxout[t](C)` bounds the level-`t` outlink distance of every
level-`t` element in `C`'s subtree and generalises the C++ `max_outlink_m`
rewire bound (commit fe7641f) from level 0 to every level.

`repdist` is stored so that refreshing an ancestor whose members and
representative did not change costs no distance evaluation.

### 6.3 One routine per level: `link!(e, l)` and `unlink!(e, l)`

Both are pruned descents. The frontier starts with every cluster of the
current top level (one cluster at rest, several while a new level is being
formed), ordered by lower bound `LB(C) = max(0, M(e, rep(C)) - radius(C))`,
with per-element distance memoisation for `e`. Clusters at levels above `l`
expand into their members' child clusters; clusters at level `l` expand into
their members, which are evaluated.

`link!(e, l)` adds `e` to `P_l`. It needs two things from the descent: the
nearest neighbour `nn` of `e` in `P_l` (canonical ties), and every `x ∈ P_l`
whose outlink would improve: `(M(x, e), id(e)) < (outdist(x), id(outlink(x)))`
lexicographically. A cluster `C` is skipped only if **both** `LB(C) >
best-so-far` and `LB(C) > maxout[l](C)` (both strict, with the §5 guard).
Every evaluated level-`l` element is recorded. Then:

1. Set `outlink(e) = nn`; rewire every recorded improving `x` to `e`;
   maintain `inlinks`.
2. Repair the affected clusters only: `nn`'s cluster and the former clusters
   of the rewired `x`. Recompute connected components over the union of their
   members plus `e`, elect representatives, recompute `repdist`, `radius`,
   `maxout`.
3. Refresh bounds up the root paths: for each repaired cluster, each member
   present at level `l+1` names an ancestor to refresh, and so on to the top.
4. Return the representative delta at level `l`: `(removed_reps, added_reps)`.

`unlink!(e, l)` removes `e` from `P_l`:

1. Each inlinker `x` of `e` re-finds its nearest neighbour in `P_l \ {e, x}`
   by the same descent with `e` excluded as a candidate (bounds that involve
   `e` remain valid because `e` is still physically present). `x`'s new
   outlink cannot change anybody else's nearest neighbour, so no rewire
   sweep is needed.
2. Remove `e` from its cluster and from `present`; drop its outlink and
   inlink entries at this level.
3. Repair the affected clusters only: `e`'s former cluster (it may split)
   and the clusters that the re-linked inlinkers now point into (pieces may
   merge into them). Refresh ancestors, return the delta.

Reachability contract for implementers: while level `l` is being processed,
every element at levels `≥ l` is reachable from the top by descent, and every
cluster's `radius` and `maxout` are valid for its reachable subtree. Clusters
at levels `< l` whose new representative has not yet been linked are
"floating" and are reattached by the `link!` of that representative. `verify`
is only meaningful at rest.

### 6.4 `add!`, `delete!`, level growth

```
add!(e):    mark e present at level 0; delta = link!(e, 0);   propagate(delta, 1)
delete!(e): alive[e] = false;          delta = unlink!(e, 0); propagate(delta, 1)

propagate(delta, l):          # delta = net (removed, added) representatives of level l-1
    if delta is empty: return # cluster counts cannot change without a rep change
    if level l-1 has exactly one cluster: drop every level ≥ l; return
    if level l does not exist: create it empty; delta = (∅, all reps of level l-1)
    delta' = ∅
    for r in removed, in id order: delta' ∘= unlink!(r, l)   # ∘ = net composition
    for r in added,   in id order: delta' ∘= link!(r, l)
    propagate(delta', l+1)
```

Net composition means an element removed from and re-added to the same
level's representative set within one step is no change: level `l+1` was not
touched in between, so its structure is still a function of an unchanged
population. `link!` and `unlink!` on a level whose population is at most one
after the operation reduce to bookkeeping (no outlink exists).

Deleting the last live element leaves an empty index with no levels. A level
whose population drops to one element has no outlinks there and is the top.
The first insert creates level 0 with one singleton cluster; the second
insert links the two elements, forms one cluster, and the index stays at one
level.

### 6.5 Search

One exact engine: best-first branch-and-bound.

- Frontier: min-heap of clusters keyed by `LB(C) = max(0, M(q, rep(C)) -
  radius(C))`, seeded with the root cluster. Result set: max-heap of at most
  `k` `(distance, id)` pairs. Distances are memoised per query because a
  representative is a member at every level it reaches.
- Pop the cluster with the smallest `LB`. If the result set holds `k` entries
  and `LB` is strictly greater than the k-th distance (with the guard), stop.
  Otherwise evaluate every member `m`, offer `(M(q, m), m)` to the result set,
  and when the cluster is above level 0 push `child(m)` with `LB =
  max(0, M(q, m) - radius(child(m)))`.
- Non-metric index: `LB = 0` everywhere; the engine visits every cluster and
  returns the exact answer.
- Output sorted by `(distance, id)`.

`HNNG.knn_levelsync(idx, q, k)` keeps the level-synchronous descent of the
2015 algorithm unexported, as a second oracle for the tests only, with the
corrected prune rule: the ROS `getNNElementwithPH` has the keep/prune sign
flipped (`view_clustering_utils.cpp:2260`), the correct non-strict form is in
`getNNClusterswithRadius` (line 2338), and the port uses the strict form with
the guard in both engines.

The C++ budget fallback (`max(64, n/32)` brute-force scan) and its latches
are not ported.

### 6.6 Bulk loader (VectorStore only, empty index only)

Per level: tiled all-pairs squared distances with BLAS (`G = XᵀX` tiles,
norms accumulated in `Float64`), per-row argmin with canonical ties,
union-find for components, representative election, bounds. Every argmin
candidate within `1e-5` relative distance of the tile minimum is re-ranked
with the direct kernel so that outlinks agree with the streaming path on
tie-free data. The level-`(l+1)` population is the set of representatives and
the same procedure repeats until one cluster remains. `bulk_add!` on a
non-empty index throws.

## 7. Navigation API (no distance evaluation)

```julia
element(idx, id)                 # the stored vector (a view) or object
outlink(idx, id; level=0)        # Int, 0 if none
outlink_distance(idx, id; level=0)
inlinks(idx, id; level=0)        # Vector{Int} (a copy)
cluster(idx, id; level=0)        # cluster handle (level, index)
members(idx, c); representative(idx, c); radius(idx, c)
parent(idx, c)                   # cluster at level+1 containing rep(c), nothing at the top
children(idx, c)                 # child clusters of the members
path_to_root(idx, id)            # Vector of cluster handles, level 0 up
leaves(idx, c)                   # level-0 ids under c (the ROS getLeafElementswithCluster)
nlevels(idx); root(idx)          # root cluster handle
newick(idx; labels=nothing)      # Newick string of the hierarchy with radii as branch
                                 # lengths and labels[id] as leaf names (ROS eTree export)
```

The README shows two application recipes built on these calls and an external
`labels` vector: recognition by the label of the 1-NN, and the maximal
single-label subtrees of an object (the ROS `ObjectRepresentation`), found by
walking `children` and `leaves` without a distance evaluation. Neither is a
package function.

Package extension `HNNGGraphsExt` (loaded when `Graphs` is present):
`Graphs.SimpleDiGraph`-compatible export `level_graph(idx, l) ->
(g::SimpleDiGraph, weights::SparseMatrixCSC, ids::Vector{Int})` with one
vertex per level-`l` element and one weighted edge per outlink. The weights
are the stored outlink distances, so `Graphs.dijkstra_shortest_paths` runs
without a single new distance evaluation. The README shows a nearest-chain
walk and a shortest-path example.

## 8. Comparison table (README, spec, manuscript Section 2)

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

Every cell is checked against the package documentation saved under
`/home/coral/working/.workflow/scratch/julia-eco/` before it appears in the
manuscript.

## 9. Error handling

`ArgumentError` with the offending value for: dimension mismatch on `add!`
or `knn`; a non-finite coordinate; `k < 1`; `delete!` of an unknown or
already deleted id; navigation on a dead id; `bulk_add!` on a non-empty index
or an ObjectStore; `load` of an ObjectStore or `UserMetric` index without
`metric=`. No other exceptions are part of the contract. The package never
silently drops an operation.

## 10. Testing and verification

The default suite runs in well under a minute on small inputs; `HNNG_SLOW=1`
enables the HDF5-based runs. All randomness uses `StableRNGs` with fixed
seeds.

- **Brute-force oracle** (`oracle.jl`): `knn` equals exhaustive search for
  Euclidean, squared Euclidean and cosine on random Float32 and Float64 data
  with `d ∈ {2, 8, 32}`, `n ∈ {1, 2, 50, 2000}`, `k ∈ {1, 10, n+5}`; also
  for an ObjectStore of variable-size descriptor sets (`d×m` matrices with
  `m` varying per element) under a Hausdorff `UserMetric(is_metric=true)`
  with Euclidean point distance, and under the 2015 ROS view distance
  `max(mean_i min_j ‖a_i − b_j‖², mean_j min_i ‖a_i − b_j‖²)`
  (`race_feature_extraction/src/spin_image.cpp:252–312`,
  `view_clustering_utils.cpp:1957–1969`), which is not a metric and runs
  with `is_metric=false`. Best-first and level-sync oracle agree.
- **Invariants** (`invariants.jl`): `verify` after every `add!` and `delete!`
  in every test that mutates an index. `verify` checks §6.1 items 1–7, with
  the 1-NN test done by brute force over each level's population and the
  bounds recomputed from scratch.
- **Order independence** (`order.jl`): five random insertion orders and the
  bulk loader produce identical hierarchies (compared after mapping ids back
  to point indices) on tie-free data.
- **Updates** (`updates.jl`): random interleaved inserts and deletes with
  the oracle and `verify` after each step; deleting a representative; delete
  everything and re-insert gives the same structure; `length`, `stats`.
- **C++ parity** (`parity_cpp.jl`): fixtures generated once with the Python
  bindings in `hnng/python_bindings` on the first 2,000 train points and 100
  test queries of random-xs-20-euclidean, `k = 10`; stored as JSON in
  `test/fixtures/`. The C++ squared distances are square-rooted. Gate:
  per-query distance multisets equal to relative tolerance `1e-5`; ids equal
  wherever the true k-th and (k+1)-th distances differ by more than `1e-6`
  relative. Level count and per-level cluster counts are printed, not gated,
  because representative election sums metric rather than squared distances
  (§12).
- **Navigation and extension** (`navigate.jl`): outlink chains terminate at
  a mutual pair; `path_to_root` ends at `root`; `level_graph` edge weights
  equal `outlink_distance`; a Dijkstra path over the level-0 graph is
  reproduced from stored distances alone (eval counter unchanged).
- **IO** (`io.jl`): save/load round-trip reproduces `verify`-clean structure
  and identical `knn` results for both stores.
- **Recall rule** (`recall_rule.jl`): the harness's two recall definitions on
  a hand-made tie case.
- **Errors**: every case in §9.

## 11. Benchmark harness (`bench/`)

Own `Project.toml`: `HNNG` (dev path), `NearestNeighbors`, `HNSW`, `HDF5`,
`BenchmarkTools`, `StableRNGs`, `CSV`, `DataFrames`, `CairoMakie`,
`SimilaritySearch` (optional, ExhaustiveSearch and SearchGraph).

Protocol for every experiment: `julia -t 1`, `BLAS.set_num_threads(1)`,
process pinned with `taskset`, one query at a time, the best of three
timed passes, `k = 10`. Two recall definitions are reported: the
ann-benchmarks rule (a returned distance counts if it is `≤` the true k-th
distance `+ 1e-3`) and strict id recall. `HNNG.knn_counted` supplies the
fraction evaluated: distance evaluations during the query divided by the
number of live elements. `env.txt` records Julia version, the
`Manifest.toml` hash, CPU model, governor, and date.

- **Exp.1 / Exp.1b** (`exp1.jl`): `random-xs-20-euclidean.hdf5` and
  `fashion-mnist-784-euclidean.hdf5` from `/home/coral/working/benchmarks/data/`
  (`train`, `test`, `neighbors`, `distances` datasets). Rows: hNNG streaming
  build, hNNG bulk build, KD-tree and ball tree (NearestNeighbors.jl),
  hand-written BLAS brute force (Gram trick with `gemv`, exact re-rank of the
  top candidates), HNSW.jl with `M = 16`, `efConstruction = 200` and an `ef`
  sweep over `{10, 20, 40, 80, 160, 320, 640}`; SimilaritySearch rows when
  the package loads. Columns: build time, QPS, both recalls, fraction
  evaluated, levels and cluster counts for hNNG.
- **Exp.2** (`exp2.jl`): synthetic `d ∈ {2, 4, 8, 16, 32, 64, 128}` with
  1,000 queries, two geometries: i.i.d. Gaussian `N(0, I_d)` with
  `n = 5,000` (the TKDE setting, so the manuscript's ρ(d) sentence keeps its
  form with Julia numbers) and 100-centre blobs with `n = 10,000` (centres
  uniform in `[-10, 10]^d`, unit isotropic standard deviation, matching the
  ann-benchmarks `random-xs` generator). Columns: fraction evaluated ρ, QPS
  for hNNG, KD-tree, ball tree and brute force, build time, number of levels.
- **Exp.3** (`exp3.jl`): Gaussian data. Table configurations are the TKDE
  rows `(n, d) ∈ {(8000, 16), (32000, 16), (32000, 128)}`; the figure sweeps
  `n ∈ {1k, 2k, 5k, 10k, 20k, 60k}` at `d = 16`. After the build, 200 inserts
  of new points, then 200 deletes of random live ids; the table reports the
  median per-operation latency of three runs. Per operation: latency, distance
  evaluations, elements rewired at level 0, highest level touched. After every
  operation 20 random queries are checked against brute force (an assertion,
  not timed). Baselines: full hNNG rebuild time per operation (the TKDE
  "rebuild" column and speedup), HNSW.jl `add_to_graph!` latency (delete
  reported as "mark only"), KD-tree and ball tree full rebuild time, brute
  force (no update cost).
- Outputs: `bench/results/{exp1,exp1b,exp2,exp3}.csv`, `env.txt`.
  `tables.jl` writes `bench/results/tex/tab_annbench.tex`,
  `tab_fashion.tex` and `tab_ops.tex` in the TKDE layouts (columns
  `Method & Recall & QPS & Build (s)` for the first two, with hNNG
  incremental, hNNG bulk, brute force, KD-tree, ball tree, HNSW at maximum
  recall and HNSW at its sweet spot as rows; `n (d) & insert/op & delete/op &
  rebuild/op (speedup)` for the third), plus `macros.tex` with one
  `
ewcommand` per number quoted in the manuscript prose (the ρ(d) values
  at `d = 2, 8, 16, 32`, the rewire counts, the recall–QPS headline).
  `plots.jl` writes `bench/results/fig/recall_qps.pdf`,
  `dist_eval_vs_dim.pdf`, `qps_vs_dim.pdf` and `ops_comparison.pdf` with
  CairoMakie, the four TKDE benchmark figures. The manuscript inputs these
  files and types no number by hand. `run_all.jl` runs everything in order.

Throughput parity with the C++ numbers in the TKDE draft is not a goal and
is not promised; the manuscript reports the Julia numbers as such.

## 12. Accepted divergences from the C++ reference

| Topic | C++ (`hnnalg.h`) | HNNG.jl |
|---|---|---|
| Returned distances | raw squared L2 | metric's own units |
| Representative tiebreak | sum of raw squared outlink distances, then member order | sum of metric-unit outlink distances, then smallest id |
| Upper levels | brute force over the representative set, full re-cluster of dirty levels | same `link!`/`unlink!` descent at every level, affected clusters only |
| Search fallback | budget scan `max(64, n/32)` with latches | none |
| Level-sync prune | `>=` | strict `>` with guard, like best-first |
| Inner-product space | pruning disabled | not offered; cosine via normalisation |
| Rewire bound | level 0 only (`max_outlink_m`) | `maxout[t]` at every level |

The first two change tie resolution and the hierarchy shape, not the set of
distances a query returns; the parity test is gated accordingly.

Against the 2015 ROS implementation (review D1–D23): it agrees with the C++
on the representative rule (raw units, then member order) and on the
non-strict level-sync prune; it already updated upper levels level by level
with the same insert routine, which this spec follows and the C++ replaced by
brute force over the representative set; its rewire bound is the pre-fe7641f
level-local form; it never refreshed ancestor radii; it dropped one level
only when the hierarchy shrank; it had no deletion of a view; its recognition
service returned the label of the single nearest view with the flipped sign.
The manuscript's online-delete claim therefore rests on the C++ and Julia
implementations, not on the 2015 system.

## 13. Risks and open items

- The path-local upper-level update (§6.3–6.4) is the hardest code. Guards:
  `verify` after every operation in the tests, the brute-force oracle, and
  the order-independence test.
- Julia throughput may sit below the C++ figures; accepted and stated.
- HNSW.jl is lightly maintained (last commit 2025-11); if it fails to load
  on the chosen Julia version, SimilaritySearch's SearchGraph takes its row
  and the manuscript names the substitution.
- Memory of the per-level `O(n)` id-indexed vectors is fine at 60k elements;
  million-scale would need a sparse map (out of scope).
- Rep election in metric units is a documented departure from the C++ and
  from the manuscript's older "HDE" wording; the manuscript adopts the
  Julia rule.

## 14. Availability

Tag `v0.1.0` on GitHub after the benchmarks are frozen; archive the release on
Zenodo and cite the DOI in the manuscript's availability statement. The C++
repository is cited as the reference implementation.
