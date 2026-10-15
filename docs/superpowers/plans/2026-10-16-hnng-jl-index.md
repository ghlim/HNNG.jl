# HNNG.jl Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Julia package `HNNG` (index, tests, Graphs extension, README): an exact, parameter-free hierarchical nearest-neighbour-graph index with first-class insert and delete, arbitrary elements and metrics, and navigation along stored links.

**Architecture:** One mutable `HNNGIndex` holds a store (`VectorStore` columns or `ObjectStore` objects), tombstones, and a vector of `Level`s; every level is an exact 1-NN graph over its population, clustered into connected components with an elected representative, a recursive radius and per-level `maxout` bounds. Every update runs the same two routines at every level (`link!`, `unlink!`: pruned best-first descents) and climbs with a net representative delta (`propagate!`). Queries run one exact best-first branch-and-bound engine; a level-synchronous engine and a brute-force oracle exist for the tests.

**Tech Stack:** Julia 1.13.x (stable) and 1.10.x (LTS) via juliaup; Distances.jl, DataStructures.jl, LinearAlgebra (BLAS), Serialization; Graphs.jl + SparseArrays as a package extension; Test, StableRNGs, JSON3, HDF5 (slow tests), Base64, Random for tests; Python 3.12 + h5py + the existing `hnnglib` bindings for the one-time C++ parity fixture.

**Spec:** `/home/coral/working/HNNG.jl/docs/superpowers/specs/2026-10-15-hnng-jl-port-design.md` (sections 3 to 10 and 12). The benchmark harness `bench/` and `test/recall_rule.jl` are covered by the sibling plan `docs/superpowers/plans/2026-10-16-hnng-jl-bench.md`.

## Global Constraints

- Package `name = "HNNG"`, `version = "0.1.0"`, `[compat] julia = "1.10"`; the suite passes on the LTS (1.10.x) and on the current stable release (1.13.x on 2026-10-16).
- Dependencies exactly: `Distances`, `LinearAlgebra`, `DataStructures`, `Serialization`. Weak dependency `Graphs` (extension `HNNGGraphsExt`, which also uses the `SparseArrays` stdlib). Test dependencies: `Test`, `StableRNGs`, `Graphs`, `HDF5` (slow tests only), `JSON3` (fixtures), plus the stdlibs `Base64`, `Random`, `SparseArrays`.
- Ids are `Int`, assigned 1, 2, 3, … in `add!` order and never reused; a deleted id is a tombstone; `knn` returns ids sorted by ascending distance, ties by ascending id, in **the metric's own units**.
- "All stored distances, radii and bounds are `Float64`, whatever `T` is." The Euclidean kernel for `Float32` data accumulates in `Float32` and widens the result.
- "Every prune comparison applies a relative guard of `1e-6` in favour of visiting." Exactness is relative to the package's own kernel; the test oracle uses the same kernel.
- Canonical ties everywhere: nearest neighbour by `(distance, id)`; representative by most inlinks, then smallest sum of the inlinkers' outlink distances (metric units), then smallest id.
- Exceptions: `ArgumentError` with the offending value for exactly the cases of spec §9; "The package never silently drops an operation."
- `nevals` counts every distance evaluation of `add!`, `delete!`, `bulk_add!` and the serial `knn` path; it is a plain always-on `Int`; `knn_batch(...; threads=true)` does not update it.
- Tests: `StableRNGs` with fixed seeds; `verify(idx)` after every mutating call in every test from Task 5 on (a matrix `add!` is one call); the default suite runs well under a minute; HDF5 runs only with `HNNG_SLOW=1`.
- No change to the C++ repository `/home/coral/working/hnng`; it is read-only reference.
- Shell: every command assumes `export PATH="$HOME/.juliaup/bin:$PATH"` and `cd /home/coral/working/HNNG.jl`. Agent shells reset between calls, so prefix each command with both.
- Every commit message ends with the trailer line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; commits use the repo-local identity set in Task 1 (`Gi Hyun Lim <hhmetal@gmail.com>`).

## Review Focus

1. **Exact duplicate points** (distance-0 ties) inserted, queried and deleted one copy at a time: the structure verifies after every step, `knn` returns the copies in ascending id order, and deleting one copy leaves the others findable. Pinned in Task 8 (search) and Task 13 (deletes).
2. **Integer-lattice data with massive distance ties**: `verify` passes and `knn` equals the oracle id-for-id with smallest-id tiebreaks. Pinned in Task 8.
3. **Non-`Float32` input on the default `Float32` index** (`Int` and `Float16` vectors, `Float64` matrices, views, row slices, `Int` queries): converted to `T`, stored and queried like `Float32` input, never a `MethodError`. Pinned in Task 5 (`add!`) and Task 8 (`knn` with an `Int` query).
4. **The zero vector under `CosineDist`**: stored as the zero vector (no `NaN`), at cosine distance 0.5 from every non-zero vector, usable as a query. Pinned in Task 3 and Task 8.
5. **A `UserMetric` that returns an `Int`** (Hamming distance on strings in an `ObjectStore`, `is_metric=true`): distances widen to `Float64`, the many ties go to the smallest id, results are exact. Pinned in Task 11.

## Spec assumptions

The spec was not changed. Where it is silent or self-contradictory, this plan assumes:

1. **Order independence (spec §6.1, §10).** The third representative tiebreak (smallest id) fires on every two-member mutual pair, because both members have one inlink and the same sum (the distance is symmetric). So the hierarchy is a pure function of the live set of `(id, point)` pairs, invariant under any order-preserving relabelling of ids, but not under arbitrary insertion orders. `test/order.jl` therefore checks: five random insertion orders give the identical level-0 graph and identical level-0 member sets (as point sets); the full hierarchy is identical for (a) streaming versus `bulk_add!` in the same column order, (b) a stream with interleaved noise inserts and deletes versus a clean stream, (c) delete-and-re-insert versus a fresh build with the same relative id order.
2. **Reachability during one `propagate!` step (spec §6.3 "Reachability contract").** It is not true that every element at levels ≥ `l` is reachable from the top while level `l` is processed: an earlier `link!`/`unlink!` of the same step can split a level-`l` cluster, and the piece without the old representative has no member at level `l+1`. Every descent at level `l` is therefore seeded with the top-level clusters plus the level-`l` clusters repaired earlier in the same step that have no member present at level `l+1` (the `floating` list).
3. **A member without a child cluster (spec §6.2).** During a delete cascade an element is present at level `L` but already absent at level `L-1`. Its radius term is `repdist` alone and it contributes nothing to `maxout[t]` for `t < L`. At rest this never happens.
4. **Guard form.** `LB(C) = max(0, M(q, rep) - radius - 1e-6 * (M(q, rep) + radius))`; every prune compares this guarded `LB` strictly (`>`). Non-metric indexes never prune (`LB = 0` and the comparison is gated on `is_metric`).
5. **Child bounds inside `link!`/`unlink!`.** A child cluster's `LB` uses `M(e, rep(child))`, not `M(e, m)`: the two differ during an update. Search at rest uses the same formula; there `rep(child(m)) == m`.
6. **Canonical sums.** The inlinker-distance sum of the representative rule is summed in ascending value order, so it does not depend on ids or history. `inlinks[x]` and `members` are kept sorted by id.
7. **`knn_batch` shape.** `ids` and `dists` are `k × nq` (one column per query); missing results are padded with id `0` and distance `Inf`.
8. **`bulk_add!` "empty".** Requires that no id was ever assigned (a drained index has tombstones and is not empty, as in the C++). An empty matrix returns `1:0`. Its `nevals` counts every Gram pair plus every direct re-rank.
9. **Navigation on a live id at a level it does not reach** returns `0`, `Inf`, `Int[]` or `nothing`; a `level` outside `0:nlevels-1` and a stale or invalid cluster handle throw `ArgumentError`. `parent` extends `Base.parent`; `save`, `load`, `distance`, `knn_counted`, `knn_levelsync` stay unexported; `Euclidean`, `SqEuclidean`, `CosineDist` are re-exported.
10. **Newick format.** Level-0 leaves carry `repdist` (distance to their cluster's representative) as branch length, every cluster carries its radius; all lengths are in the metric's units; no internal node names; leaf names with Newick-special characters are single-quoted.
11. **Cosine zero vector.** Normalisation is `x / (‖x‖ + 1e-30)` like the C++ bindings, so a zero vector stays zero.
12. **Test file layout.** `test/oracle.jl` holds the brute-force oracle and shared helpers only; the spec's oracle grid lives in `test/search.jl`; `test/metrics.jl`, `test/errors.jl`, `test/slow.jl` and `test/recipes.jl` are added to the spec's list.
13. **C++ parity counts.** The bindings expose no level or cluster getters, so the fixture script reads `maxlevel` and the per-level cluster counts from the C++ `save_index` file (format in the brief §7, `hnnalg.h:2471-2526`). The id gate compares sorted id sets (C++ labels + 1).
14. **Internal layout.** `VectorStore` is a `mutable struct` (capacity doubling); `HNNGIndex` gains `last_rewired` and `last_top` (read by the unexported `HNNG.last_update_stats`, requested by the benchmark plan for Exp.3); `Level` gains `free`, `npop`, `nclusters`; `DistMemo` carries the reusable heap buffers; `save`/`load` use `Serialization` of a tagged `NamedTuple`, store the metric only for a `VectorStore` with a Distances metric, and do not persist `nevals`. `HNNGIndex(metric, dim)` with `dim < 1` throws `ArgumentError`.
15. **Graphs extension loading.** `SparseArrays` sits in `[weakdeps]` next to `Graphs` and the extension triggers on `"Graphs"` alone (extensions may load their parent's weak deps; `Graphs` itself loads `SparseArrays`).
16. **`verify`** assumes a symmetric distance (the 1-NN check recomputes `d(x, y)`); it throws `ErrorException("HNNG.verify: ...")`, which is not one of the §9 user errors.

## Decisions fixed by this plan

- **Memo layout:** dense epoch-stamped arrays indexed by id (`val`, `seen`, `mark`, `epoch::UInt32`), resized lazily, one `DistMemo` in the index for every serial pass and one per task for threaded batches.
- **Heaps:** `DataStructures.heappush!`/`heappop!` on plain vectors held in the memo: frontier `Vector{Tuple{Float64,Int,Int}}` = `(LB, level, cluster index)` as a min-heap (`Base.Order.Forward`), results `Vector{Tuple{Float64,Int}}` = `(distance, id)` as a max-heap (`Base.Order.Reverse`). Tuple order gives canonical ties.
- **Cluster slots:** a freed slot gets `rep = 0` and goes on `Level.free`; new clusters pop from `free` (last in, first out) or append. Cluster indices are valid only between mutations.
- **Components:** breadth-first search over outlinks and inlinks (`_component`), in the local repair and in the bulk loader; it yields the same components as the union-find of spec §6.6.
- **Descent frontier order inside `link!`/`unlink!`:** min-heap on `(LB, level, cluster index)`; the skip test runs at push and again at pop; level-`l` clusters reached twice in one pass are expanded once (the representative's `mark`).
- **Internal names used across tasks:** `_point`, `_append!`, `_ntotal`, `dim`, `_grow!`, `_begin!`, `_euclid`, `_rawdist`, `_to_user`, `_prepare`, `_normalize!`, `_d`, `_mdist`, `_flush!`, `_better`, `_insum`, `_elect`, `_component`, `_insert_sorted!`, `_remove_sorted!`, `_scan_level`, `_nn_search`, `_lb`, `_skip`, `_push_cluster!`, `_repdist`, `_bounds!`, `_new_slot!`, `_free_slot!`, `_refresh_up!`, `_repair!`, `link!`, `unlink!`, `_compose!`, `propagate!`, `_insert!`, `_knn!`, `_offer!`, `_sorted_results`, `_check_live`, `_check_level`, `_cl`, `_verify_level`, `_verify_top`, `_fresh_bounds`, `_rep_mask`, `_fail`, `_components`, `_allpairs_nn`, `_allpairs_nn_direct`, `_allpairs_nn_blas`, `last_update_stats`. Constants: `GUARD = 1e-6`, `FWD`, `REV`, `BULK_TILE = 512`, `RERANK = 1e-5`, `FORMAT`, `FORMAT_VERSION`.

## File structure

| File | Responsibility | Created in |
|---|---|---|
| `Project.toml`, `.gitignore` | package metadata, deps, weak deps, extension, test target | Task 1 |
| `src/HNNG.jl` | module, imports, exports, includes (edited as tasks add files) | Task 1 |
| `src/types.jl` | `VectorStore`, `ObjectStore`, `Cluster`, `Level`, `DistMemo`, `HNNGIndex`, `length`, `nlevels`, `is_metric`, `show` | Task 2 |
| `src/metrics.jl` | `UserMetric`, metric wrapping, `is_metric` rules, `_rawdist`, `_to_user`, `_prepare`, `_d`, `_mdist`, `distance` | Tasks 2–3 |
| `src/kernels.jl` | `_euclid`; BLAS tiled all-pairs 1-NN for the bulk loader | Tasks 3, 20 |
| `src/insert.jl` | canonical comparisons, election, components, `_nn_search`, repair, bounds, `link!`, `propagate!`, `add!` | Tasks 4–7 |
| `src/delete.jl` | `unlink!`, `delete!` | Tasks 6, 10 |
| `src/search.jl` | `knn`, `HNNG.knn_counted`, `knn_batch`, best-first engine, `HNNG.knn_levelsync` | Tasks 8–9 |
| `src/bulk.jl` | `bulk_add!` | Tasks 19–20 |
| `src/navigate.jl` | `element`, `ClusterRef`, navigation calls, `leaves`, `newick`, `level_graph` stub | Tasks 3, 14–16 |
| `src/verify.jl` | `verify` | Task 4 |
| `src/io.jl` | `save`, `load`, `stats`, `reset_evals!`, `HNNG.last_update_stats` | Tasks 17–18 |
| `ext/HNNGGraphsExt.jl` | `level_graph` on Graphs.jl | Tasks 1, 16 |
| `test/runtests.jl` | entry point, includes every test file, `HNNG_SLOW` gate | Task 1, edited per task |
| `test/oracle.jl` | brute-force oracle, generators, structure snapshots, `stream!` | Task 4, extended in 8, 11 |
| `test/metrics.jl` | stores, metric rules, kernel, units, preparation | Tasks 2–3 |
| `test/invariants.jl` | `verify` itself, level 0, hierarchy, descent | Tasks 4–7 |
| `test/search.jl` | oracle grid, ties, duplicates, cosine zero vector, engines, batch | Tasks 8–9 |
| `test/updates.jl` | deletes, interleaved updates, drain and refill, `stats` | Tasks 10, 13, 18 |
| `test/objects.jl` | ObjectStore: Hausdorff, ROS view distance, Hamming | Task 11 |
| `test/order.jl` | order and history independence, bulk loader | Tasks 12, 19–20 |
| `test/navigate.jl` | navigation, `leaves`, `newick`, Graphs extension | Tasks 14–16 |
| `test/io.jl` | save/load round trips | Task 17 |
| `test/fixtures/make_fixtures.py`, `test/fixtures/cpp_parity_random_xs_20.json` | C++ parity fixture (generated once, committed) | Task 21 |
| `test/parity_cpp.jl` | C++ parity gate | Task 22 |
| `test/errors.jl` | every error of spec §9 | Task 23 |
| `test/slow.jl` | HDF5 runs behind `HNNG_SLOW=1` | Task 24 |
| `test/recipes.jl`, `README.md` | README recipes as tests; README | Task 25 |

Layer order: (a) Tasks 1–3, (b) Task 4, (c) Task 5, (d) Tasks 6–7, (e) Tasks 8–9, (f) Task 10, (g) Tasks 11–13, (h) Tasks 14–16, (i) Tasks 17–18, (j) Tasks 19–20, (k) Tasks 21–22, (l) Task 23, slow runs Task 24, (m) Task 25. `unlink!` lands in layer (d) rather than (f) because an insert that merges two clusters removes a representative, so insert-only propagation already needs it; layer (f) adds the public `delete!`.

**Run commands used throughout.** Full suite: `julia --project=. -e 'using Pkg; Pkg.test()'`. Single file: `julia --project=. -e 'using Test, HNNG; include("test/<file>.jl")'` (the test packages are also installed in the default environment in Task 1, which Julia stacks on the load path). Every test file from Task 4 on starts with `isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))`, so it runs alone or from `runtests.jl`.

---

### Task 1: Toolchain and package skeleton

**Files:**
- Create: `Project.toml`, `.gitignore`, `src/HNNG.jl`, `ext/HNNGGraphsExt.jl`, `test/runtests.jl`

**Interfaces:**
- Consumes: nothing.
- Produces: a loadable empty module `HNNG`; the test target `[extras]`/`[targets]`; `julia` (stable 1.13.x) and `julia +lts` (1.10.x) on `PATH`; StableRNGs, JSON3 and Graphs in the default environment for single-file runs; repo-local git identity.

- [ ] **Step 1: Install Julia through juliaup at user level**

```bash
curl -fsSL https://install.julialang.org | sh -s -- --yes
export PATH="$HOME/.juliaup/bin:$PATH"
juliaup add lts
julia --version
julia +lts --version
```

Expected: the installer finishes without a prompt and installs into `~/.juliaup`; `julia --version` prints `julia version 1.13.1` (or a later 1.13.x); `julia +lts --version` prints `julia version 1.10.12` (or a later 1.10.x). The installer also appends the `PATH` line to the shell startup files; agent shells still need the `export` line above in every command.

- [ ] **Step 2: Set the repo-local git identity**

```bash
cd /home/coral/working/HNNG.jl
git config user.name "Gi Hyun Lim"
git config user.email "hhmetal@gmail.com"
git config user.name; git config user.email
```

Expected:
```
Gi Hyun Lim
hhmetal@gmail.com
```

- [ ] **Step 3: Write `Project.toml` with a fresh UUID**

```bash
export PATH="$HOME/.juliaup/bin:$PATH"; cd /home/coral/working/HNNG.jl
UUID=$(julia -e 'using UUIDs; print(uuid4())')
echo "$UUID"
cat > Project.toml <<EOF
name = "HNNG"
uuid = "$UUID"
authors = ["Gi Hyun Lim <hhmetal@gmail.com>"]
version = "0.1.0"

[deps]
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[weakdeps]
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[extensions]
HNNGGraphsExt = "Graphs"

[compat]
DataStructures = "0.18, 0.19"
Distances = "0.10"
Graphs = "1"
HDF5 = "0.17"
JSON3 = "1"
StableRNGs = "1"
julia = "1.10"

[extras]
Base64 = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
StableRNGs = "860ef19b-820b-49d6-a774-d7a799459cd3"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Base64", "Graphs", "HDF5", "JSON3", "Random", "SparseArrays", "StableRNGs", "Test"]
EOF
grep -c '=' Project.toml
```

Expected: a UUID such as `3f1c...` is printed and the file exists. A wrong registry UUID would surface in Step 6 as `expected package ... to be registered`; if that happens, fix the entry from `julia -e 'using Pkg; Pkg.add("<Name>")'` output in a scratch environment, not by guessing.

- [ ] **Step 4: Write the module, the extension stub, `.gitignore` and a one-test suite**

`src/HNNG.jl`:
```julia
module HNNG

end # module
```

`ext/HNNGGraphsExt.jl` (filled in Task 16; it must exist because `Project.toml` declares it):
```julia
module HNNGGraphsExt

using HNNG, Graphs

end # module
```

`.gitignore`:
```
Manifest.toml
Manifest-v*.toml
/test/fixtures/__pycache__/
```

`test/runtests.jl`:
```julia
using Test, HNNG

@testset "HNNG.jl" begin
    @test HNNG isa Module
end
```

```bash
cd /home/coral/working/HNNG.jl && mkdir -p src ext test
# write the four files above with the editor
```

- [ ] **Step 5: Install the test packages in the default environment (single-file runs)**

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
julia -e 'using Pkg; Pkg.add(["StableRNGs", "JSON3", "Graphs"])'
```

Expected: `Updating ~/.julia/environments/v1.13/Project.toml` listing the three packages. `julia --project=.` stacks this environment on the load path, so `using StableRNGs` works in single-file runs; `Pkg.test()` uses the `[targets]` list instead.

- [ ] **Step 6: Run the suite on stable and on the LTS**

```bash
export PATH="$HOME/.juliaup/bin:$PATH"; cd /home/coral/working/HNNG.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
rm -f Manifest.toml
julia +lts --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
rm -f Manifest.toml
```

Expected for both runs (the first one downloads HDF5_jll and takes a few minutes):
```
Test Summary: | Pass  Total  Time
HNNG.jl       |    1      1  0.0s
     Testing HNNG tests passed
```

- [ ] **Step 7: Commit**

```bash
cd /home/coral/working/HNNG.jl
git add Project.toml .gitignore src/HNNG.jl ext/HNNGGraphsExt.jl test/runtests.jl
git commit -m "build: package skeleton, test target and Graphs extension stub" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Index types, stores and metric rules

**Files:**
- Create: `src/types.jl`, `src/metrics.jl`, `test/metrics.jl`
- Modify: `src/HNNG.jl`, `test/runtests.jl`

**Interfaces:**
- Consumes: Task 1 skeleton.
- Produces:
  - `mutable struct VectorStore{T<:AbstractFloat}; data::Matrix{T}; n::Int; end`, `VectorStore{T}(dim::Int)`
  - `struct ObjectStore{E}; objs::Vector{E}; end`, `ObjectStore{E}()`
  - `Base.length(::VectorStore)`, `Base.length(::ObjectStore)`, `dim(::VectorStore) -> Int`
  - `_point(store, i::Int)` (column view or object), `_append!(store, x) -> Int` (new id)
  - `mutable struct Cluster; rep::Int; members::Vector{Int}; repdist::Vector{Float64}; radius::Float64; maxout::Vector{Float64}; end`, `_dead_cluster() -> Cluster`
  - `mutable struct Level` with fields `outlink, outdist, inlinks, present, cluster_of, clusters, free, npop, nclusters`; `Level(n::Int)`; `_grow!(lv::Level, n::Int)`
  - `mutable struct DistMemo` with fields `val, seen, mark, epoch, nevals, front, res`; `DistMemo()`; `_begin!(m::DistMemo, n::Int) -> DistMemo`
  - `mutable struct HNNGIndex{S,M}` with fields `store, metric, is_metric, alive, levels, nevals, memo, last_rewired, last_top`
  - `HNNGIndex(metric, dim::Integer; T=Float32)`, `HNNGIndex(metric, ::Type{E})`
  - `_ntotal(idx) -> Int`, `Base.length(idx)` (live), `nlevels(idx)`, `is_metric(idx)`
  - `struct UserMetric{F}; f::F; is_metric::Bool; end`, `UserMetric(f; is_metric=false)`
  - `const EuclidFamily = Union{Euclidean,SqEuclidean,CosineDist}`, `_wrap_metric(m)`, `_is_metric_for(store, metric) -> Bool`

- [ ] **Step 1: Write the failing test**

`test/metrics.jl`:
```julia
using Test, HNNG, StableRNGs
using Distances: Cityblock, CorrDist

@testset "stores and index construction" begin
    idx = HNNGIndex(Euclidean(), 3)
    @test idx.store isa HNNG.VectorStore{Float32}
    @test length(idx) == 0 && nlevels(idx) == 0 && idx.nevals == 0
    @test is_metric(idx)
    idx64 = HNNGIndex(SqEuclidean(), 5; T=Float64)
    @test idx64.store isa HNNG.VectorStore{Float64}
    @test HNNG.dim(idx64.store) == 5
    s = HNNG.VectorStore{Float32}(2)
    for i in 1:40
        @test HNNG._append!(s, Float32[i, -i]) == i
    end
    @test length(s) == 40
    @test HNNG._point(s, 17) == Float32[17, -17]
    @test HNNG._point(s, 40) == Float32[40, -40]
    o = HNNGIndex((a, b) -> abs(a - b), Int)
    @test o.store isa HNNG.ObjectStore{Int}
    @test HNNG._append!(o.store, 7) == 1
    @test HNNG._point(o.store, 1) == 7
    @test_throws ArgumentError HNNGIndex(Euclidean(), 0)
    @test sprint(show, idx) == "HNNGIndex(Euclidean, 0 live, 0 levels)"
end

@testset "is_metric rules" begin
    @test is_metric(HNNGIndex(Euclidean(), 2))
    @test is_metric(HNNGIndex(SqEuclidean(), 2))
    @test is_metric(HNNGIndex(CosineDist(), 2))
    @test is_metric(HNNGIndex(Cityblock(), 2))
    @test !is_metric(HNNGIndex(CorrDist(), 2))
    @test !is_metric(HNNGIndex(SqEuclidean(), Vector{Float64}))
    @test is_metric(HNNGIndex(Euclidean(), Vector{Float64}))
    f(a, b) = abs(a - b)
    @test !is_metric(HNNGIndex(f, Float64))
    @test HNNGIndex(f, Float64).metric isa UserMetric
    @test is_metric(HNNGIndex(UserMetric(f; is_metric=true), Float64))
    @test !is_metric(HNNGIndex(UserMetric(f), Float64))
end

@testset "distance memo epochs" begin
    m = HNNG.DistMemo()
    HNNG._begin!(m, 10)
    @test length(m.seen) == 10 && m.epoch == 1
    m.seen[3] = m.epoch
    HNNG._begin!(m, 12)
    @test length(m.val) == 12 && m.epoch == 2 && m.seen[3] != m.epoch
    m.epoch = typemax(UInt32)
    m.seen[5] = typemax(UInt32)
    HNNG._begin!(m, 12)
    @test m.epoch == 1 && m.seen[5] == 0
end
```

`test/runtests.jl`:
```julia
using Test, HNNG

@testset "HNNG.jl" begin
    include("metrics.jl")
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. -e 'using Test, HNNG; include("test/metrics.jl")'`
Expected: FAIL with `UndefVarError: \`HNNGIndex\` not defined`.

- [ ] **Step 3: Write the implementation**

`src/types.jl`:
```julia
# Storage, index, level and cluster types (spec §5).

"""
    VectorStore{T}

Elements are the columns `1:n` of a `d × capacity` matrix that doubles when full.
"""
mutable struct VectorStore{T<:AbstractFloat}
    data::Matrix{T}
    n::Int
end
VectorStore{T}(dim::Int) where {T<:AbstractFloat} = VectorStore{T}(Matrix{T}(undef, dim, 0), 0)

"""
    ObjectStore{E}

Elements are arbitrary Julia objects of type `E`.
"""
struct ObjectStore{E}
    objs::Vector{E}
end
ObjectStore{E}() where {E} = ObjectStore{E}(E[])

Base.length(s::VectorStore) = s.n
Base.length(s::ObjectStore) = length(s.objs)
dim(s::VectorStore) = size(s.data, 1)

@inline _point(s::VectorStore, i::Int) = view(s.data, :, i)
@inline _point(s::ObjectStore, i::Int) = @inbounds s.objs[i]

function _append!(s::VectorStore{T}, x::AbstractVector{T}) where {T}
    if s.n == size(s.data, 2)
        grown = Matrix{T}(undef, size(s.data, 1), max(16, 2 * s.n))
        copyto!(grown, 1, s.data, 1, length(s.data))
        s.data = grown
    end
    s.n += 1
    copyto!(view(s.data, :, s.n), x)
    s.n
end
_append!(s::ObjectStore{E}, x::E) where {E} = (push!(s.objs, x); length(s.objs))

"""
    Cluster

One connected component of a level's nearest-neighbour graph. `members` is sorted
by id and `repdist[i]` is the distance from `rep` to `members[i]` (internal units).
`radius` and `maxout[t+1]` (t = 0..level) follow spec §6.2. A dead slot has `rep == 0`.
"""
mutable struct Cluster
    rep::Int
    members::Vector{Int}
    repdist::Vector{Float64}
    radius::Float64
    maxout::Vector{Float64}
end
_dead_cluster() = Cluster(0, Int[], Float64[], 0.0, Float64[])

"""
    Level

One level, indexed by id (length = ids ever assigned). `outlink[i] == 0` and
`outdist[i] == Inf` mean no outlink; `inlinks[i]` is sorted; `cluster_of[i] == 0`
means absent. Dead cluster slots are listed in `free` and reused last in, first out.
`npop` is the population size, `nclusters` the number of live clusters.
"""
mutable struct Level
    outlink::Vector{Int}
    outdist::Vector{Float64}
    inlinks::Vector{Vector{Int}}
    present::BitVector
    cluster_of::Vector{Int}
    clusters::Vector{Cluster}
    free::Vector{Int}
    npop::Int
    nclusters::Int
end
Level(n::Int) = Level(zeros(Int, n), fill(Inf, n), [Int[] for _ in 1:n], falses(n),
                      zeros(Int, n), Cluster[], Int[], 0, 0)

function _grow!(lv::Level, n::Int)
    while length(lv.outlink) < n
        push!(lv.outlink, 0)
        push!(lv.outdist, Inf)
        push!(lv.inlinks, Int[])
        push!(lv.present, false)
        push!(lv.cluster_of, 0)
    end
    lv
end

"""
    DistMemo

Per-pass distance memo: `val[i]` is valid iff `seen[i] == epoch`; `mark[i] == epoch`
records that `i` was offered or expanded in this pass. `front` (min-heap of
`(LB, level, cluster index)`) and `res` (max-heap of `(distance, id)`) are reused
buffers. `nevals` accumulates until the owner folds it into the index.
"""
mutable struct DistMemo
    val::Vector{Float64}
    seen::Vector{UInt32}
    mark::Vector{UInt32}
    epoch::UInt32
    nevals::Int
    front::Vector{Tuple{Float64,Int,Int}}
    res::Vector{Tuple{Float64,Int}}
end
DistMemo() = DistMemo(Float64[], UInt32[], UInt32[], UInt32(0), 0,
                      Tuple{Float64,Int,Int}[], Tuple{Float64,Int}[])

"""Start a new pass over at most `n` ids: bump the epoch and empty the heap buffers."""
function _begin!(m::DistMemo, n::Int)
    old = length(m.seen)
    if old < n
        resize!(m.val, n)
        resize!(m.seen, n)
        resize!(m.mark, n)
        fill!(view(m.seen, old+1:n), UInt32(0))
        fill!(view(m.mark, old+1:n), UInt32(0))
    end
    if m.epoch == typemax(UInt32)
        fill!(m.seen, UInt32(0))
        fill!(m.mark, UInt32(0))
        m.epoch = UInt32(0)
    end
    m.epoch += UInt32(1)
    empty!(m.front)
    empty!(m.res)
    m
end

"""
    HNNGIndex(metric, dim::Int; T=Float32)   # VectorStore: columns of T
    HNNGIndex(metric, ::Type{E})             # ObjectStore{E}

An exact, parameter-free hierarchical nearest-neighbour-graph index. `metric` is a
Distances.jl `PreMetric`, a `UserMetric`, or any callable `f(a, b) -> Real` (taken as
`UserMetric(f; is_metric=false)`). `Euclidean`, `SqEuclidean` and `CosineDist` on a
VectorStore use the specialised kernel and work in Euclidean units internally;
results are always in the metric's own units.
"""
mutable struct HNNGIndex{S<:Union{VectorStore,ObjectStore},M}
    store::S
    metric::M
    is_metric::Bool
    alive::BitVector
    levels::Vector{Level}       # levels[l+1] is level l
    nevals::Int
    memo::DistMemo
    last_rewired::Int           # see HNNG.last_update_stats (Task 18)
    last_top::Int
end

function HNNGIndex(metric, dim::Integer; T::Type{<:AbstractFloat}=Float32)
    dim >= 1 || throw(ArgumentError("dim must be at least 1, got $dim"))
    m = _wrap_metric(metric)
    s = VectorStore{T}(Int(dim))
    HNNGIndex(s, m, _is_metric_for(s, m), falses(0), Level[], 0, DistMemo(), 0, 0)
end

function HNNGIndex(metric, ::Type{E}) where {E}
    m = _wrap_metric(metric)
    s = ObjectStore{E}()
    HNNGIndex(s, m, _is_metric_for(s, m), falses(0), Level[], 0, DistMemo(), 0, 0)
end

_ntotal(idx::HNNGIndex) = length(idx.store)

"""Number of live (not deleted) elements."""
Base.length(idx::HNNGIndex) = count(idx.alive)

"""Number of levels; 0 for an empty index."""
nlevels(idx::HNNGIndex) = length(idx.levels)

"""`true` iff pruning runs: the metric satisfies (or is declared to satisfy) the triangle inequality."""
is_metric(idx::HNNGIndex) = idx.is_metric

function Base.show(io::IO, idx::HNNGIndex)
    print(io, "HNNGIndex(", nameof(typeof(idx.metric)), ", ", length(idx), " live, ",
          nlevels(idx), " levels)")
end
```

`src/metrics.jl`:
```julia
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
```

`src/HNNG.jl`:
```julia
module HNNG

using Distances: Distances, PreMetric, Metric, Euclidean, SqEuclidean, CosineDist

export HNNGIndex, UserMetric, Euclidean, SqEuclidean, CosineDist
export is_metric, nlevels

include("types.jl")
include("metrics.jl")

end # module
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Test, HNNG; include("test/metrics.jl")'`
Expected: `stores and index construction`, `is_metric rules` and `distance memo epochs` all pass.
Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: `Testing HNNG tests passed`.

- [ ] **Step 5: Commit**

```bash
git add src/HNNG.jl src/types.jl src/metrics.jl test/metrics.jl test/runtests.jl
git commit -m "feat: index, store, level and cluster types; metric rules" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Kernel, units, input preparation, `element`

**Files:**
- Create: `src/kernels.jl`, `src/navigate.jl`
- Modify: `src/metrics.jl` (append), `src/HNNG.jl`, `test/metrics.jl` (append)

**Interfaces:**
- Consumes: Task 2 types, `_point`, `_append!`, `dim`, `DistMemo`, `_begin!`.
- Produces:
  - `_euclid(a::AbstractVector{T}, b::AbstractVector{T}) -> Float64` (accumulates in `T`, bitwise symmetric)
  - `_rawdist(idx, a, b) -> Float64` (internal units; not counted)
  - `_to_user(idx, d::Float64) -> Float64` (internal → metric units; monotone)
  - `_prepare(idx, x)` → `Vector{T}` for a VectorStore (validated, converted, normalised under cosine), `x` itself for an ObjectStore
  - `_normalize!(idx, v)`
  - `_d(idx, a::Int, b::Int) -> Float64` (counted in `idx.nevals`)
  - `_mdist(memo::DistMemo, idx, q, j::Int) -> Float64` (memoised, counted in `memo.nevals`)
  - `HNNG.distance(idx, a, b) -> Float64` (user units, not counted, unexported)
  - `_check_live(idx, id::Integer) -> Int` (throws `ArgumentError`), exported `element(idx, id)`

- [ ] **Step 1: Write the failing test**

Append to `test/metrics.jl`:
```julia
@testset "Euclidean kernel" begin
    rng = StableRNG(3)
    for T in (Float32, Float64), d in (1, 3, 33)
        a = randn(rng, T, d)
        b = randn(rng, T, d)
        k = HNNG._euclid(a, b)
        @test k isa Float64
        @test k == HNNG._euclid(b, a)
        @test isapprox(k, sqrt(sum(abs2, Float64.(a) .- Float64.(b))); rtol = T == Float32 ? 1e-5 : 1e-12)
        @test HNNG._euclid(a, a) == 0.0
    end
end

@testset "units and input preparation" begin
    e = HNNGIndex(Euclidean(), 2; T=Float64)
    s = HNNGIndex(SqEuclidean(), 2; T=Float64)
    c = HNNGIndex(CosineDist(), 2; T=Float64)
    @test HNNG.distance(e, [0, 0], [3, 4]) == 5.0
    @test HNNG.distance(s, [0, 0], [3, 4]) ≈ 25.0
    @test HNNG.distance(c, [1, 0], [0, 2]) ≈ 1.0
    @test HNNG.distance(c, [1, 1], [2, 2]) ≈ 0.0 atol = 1e-12
    @test HNNG.distance(c, [1, 0], [-3, 0]) ≈ 2.0
    @test HNNG._prepare(c, [3.0, 4.0]) ≈ [0.6, 0.8]
    # Review Focus 4: the zero vector stays zero under cosine and is 0.5 from any unit vector
    z = HNNG._prepare(c, [0.0, 0.0])
    @test z == [0.0, 0.0] && all(isfinite, z)
    @test HNNG.distance(c, [0, 0], [5, 0]) ≈ 0.5
    f32 = HNNGIndex(Euclidean(), 3)
    v = HNNG._prepare(f32, [1, 2, 3])
    @test v isa Vector{Float32} && v == Float32[1, 2, 3]
    @test HNNG._prepare(f32, view([1.0 2 3; 4 5 6], 2, :)) == Float32[4, 5, 6]
    @test_throws "dimension mismatch" HNNG._prepare(f32, [1.0, 2.0])
    @test_throws "non-finite coordinate NaN" HNNG._prepare(f32, [1.0, NaN, 2.0])
    @test_throws "non-finite coordinate 1.0e300" HNNG._prepare(f32, [1.0, 1e300, 2.0])
    u = HNNGIndex(UserMetric((a, b) -> abs(a - b); is_metric=true), Int)
    @test HNNG.distance(u, 3, 10) === 7.0
end

@testset "element and counted distances" begin
    idx = HNNGIndex(Euclidean(), 2; T=Float64)
    for x in ([0.0, 0.0], [3.0, 4.0])
        HNNG._append!(idx.store, HNNG._prepare(idx, x))
        push!(idx.alive, true)
    end
    @test element(idx, 2) == [3.0, 4.0]
    @test HNNG._d(idx, 1, 2) == 5.0 && idx.nevals == 1
    m = HNNG._begin!(HNNG.DistMemo(), 2)
    q = [0.0, 4.0]
    @test HNNG._mdist(m, idx, q, 2) == 3.0
    @test HNNG._mdist(m, idx, q, 2) == 3.0
    @test m.nevals == 1
    @test_throws "unknown id 3" element(idx, 3)
    idx.alive[1] = false
    @test_throws "id 1 is deleted" element(idx, 1)
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. -e 'using Test, HNNG; include("test/metrics.jl")'`
Expected: the three new testsets error with `UndefVarError: \`_euclid\` not defined` (and `distance`, `element`); the Task 2 testsets still pass.

- [ ] **Step 3: Write the implementation**

`src/kernels.jl`:
```julia
# Distance kernels (spec §5): Float32 data accumulates in Float32, the result is widened.

"""
    _euclid(a, b) -> Float64

Euclidean distance accumulated in the element type, widened to `Float64`.
Swapping `a` and `b` gives the same bits (each term is the square of an exact negation).
"""
@inline function _euclid(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    s = zero(T)
    @inbounds @simd for i in eachindex(a, b)
        t = a[i] - b[i]
        s += t * t
    end
    sqrt(Float64(s))
end
```

Append to `src/metrics.jl`:
```julia
# Raw distance in internal units: Euclidean for the Euclidean family on a
# VectorStore, the metric's own value otherwise. Not counted.
@inline _rawdist(::HNNGIndex{<:VectorStore,<:EuclidFamily}, a, b) = _euclid(a, b)
@inline _rawdist(idx::HNNGIndex{<:Any,<:UserMetric}, a, b) = Float64(idx.metric.f(a, b))
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
stays zero). ObjectStore: `x` unchanged.
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
function _normalize!(::HNNGIndex{VectorStore{T},<:CosineDist}, v::Vector{T}) where {T}
    s = zero(T)
    for x in v
        s += x * x
    end
    v .*= one(T) / (sqrt(s) + T(1e-30))
    v
end

"""
    HNNG.distance(idx, a, b) -> Float64

Distance between two inputs as the index sees them (converted, normalised under
cosine), in the metric's own units. Not counted in `nevals`.
"""
distance(idx::HNNGIndex, a, b) = _to_user(idx, _rawdist(idx, _prepare(idx, a), _prepare(idx, b)))
```

`src/navigate.jl`:
```julia
# Navigation along stored links (spec §7). No call here evaluates a distance.

function _check_live(idx::HNNGIndex, id::Integer)
    n = _ntotal(idx)
    1 <= id <= n || throw(ArgumentError("unknown id $id: the index holds ids 1:$n"))
    idx.alive[id] || throw(ArgumentError("id $id is deleted"))
    Int(id)
end

"""
    element(idx, id)

The stored element: a column view for a VectorStore (unit-normalised under
`CosineDist`), the object itself for an ObjectStore.
"""
element(idx::HNNGIndex, id::Integer) = _point(idx.store, _check_live(idx, id))
```

`src/HNNG.jl`:
```julia
module HNNG

using Distances: Distances, PreMetric, Metric, Euclidean, SqEuclidean, CosineDist

export HNNGIndex, UserMetric, Euclidean, SqEuclidean, CosineDist
export is_metric, nlevels, element

include("types.jl")
include("kernels.jl")
include("metrics.jl")
include("navigate.jl")

end # module
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Test, HNNG; include("test/metrics.jl")'`
Expected: all six testsets pass.

- [ ] **Step 5: Commit**

```bash
git add src/HNNG.jl src/kernels.jl src/metrics.jl src/navigate.jl test/metrics.jl
git commit -m "feat: Euclidean kernel, unit conversion, input preparation, element" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `verify` and the brute-force oracle helpers

**Files:**
- Create: `src/insert.jl` (shared helpers only), `src/verify.jl`, `test/oracle.jl`, `test/invariants.jl`
- Modify: `src/HNNG.jl`, `test/runtests.jl`

**Interfaces:**
- Consumes: Tasks 2–3 (`_rawdist`, `_point`, `_ntotal`, `Level`, `Cluster`, `_to_user`, `_prepare`).
- Produces:
  - `_better(d::Float64, i::Int, bd::Float64, bi::Int) -> Bool` — `(d, i) < (bd, bi)`, with `bi == 0` meaning "none yet"
  - `_insum(lv::Level, m::Int) -> Float64` — sum of the inlinkers' outdists in ascending value order
  - `_elect(lv::Level, members::Vector{Int}) -> Int` — representative rule of spec §6.1 item 5 (`members` sorted)
  - `_component(lv::Level, r::Int) -> Vector{Int}` — sorted connected component (outlinks as undirected edges)
  - `_insert_sorted!(v::Vector{Int}, x::Int)`, `_remove_sorted!(v::Vector{Int}, x::Int)`
  - `verify(idx) -> true` (exported), `_verify_level(idx, l::Int) -> true`, `_verify_top(idx) -> true`, `_fresh_bounds(idx, l::Int, ci::Int) -> (radius::Float64, maxout::Vector{Float64})`, `_rep_mask(idx, l) -> BitVector`, `_fail(msg)`
  - test helpers in `test/oracle.jl`: `brute_knn(idx, q, k) -> (ids, dists)`, `gauss(rng, T, d, n)`, `blobs(rng, T, d, n; nc=8)`, `stream!(idx, X) -> Vector{Int}`, `structure(idx, key=1:n)`, `same_structure(a, b; rtol=1e-9) -> Bool`, `top_rep(idx) -> Int`

- [ ] **Step 1: Write the oracle helpers and the failing tests**

`test/oracle.jl`:
```julia
# Shared test helpers: brute-force oracle, data generators, structure snapshots.
using Test, HNNG, StableRNGs, Random, LinearAlgebra

"""Exhaustive k-NN over the live elements with the package kernel: (distance, id) order, user units."""
function brute_knn(idx, q, k)
    qp = HNNG._prepare(idx, q)
    cand = Tuple{Float64,Int}[]
    for i in 1:HNNG._ntotal(idx)
        idx.alive[i] || continue
        push!(cand, (HNNG._rawdist(idx, qp, HNNG._point(idx.store, i)), i))
    end
    sort!(cand)
    top = cand[1:min(k, length(cand))]
    [p[2] for p in top], [HNNG._to_user(idx, p[1]) for p in top]
end

gauss(rng, T, d, n) = randn(rng, T, d, n)

"""`n` points around `nc` centres uniform in [-10, 10]^d with unit spread."""
function blobs(rng, T, d, n; nc=8)
    C = 20 .* rand(rng, T, d, nc) .- 10
    X = Matrix{T}(undef, d, n)
    for j in 1:n
        X[:, j] = C[:, rand(rng, 1:nc)] .+ randn(rng, T, d)
    end
    X
end

"""Insert the columns of `X` one at a time; `verify` after each insert (throws on the first violation)."""
function stream!(idx, X)
    ids = Int[]
    for j in axes(X, 2)
        push!(ids, add!(idx, view(X, :, j)))
        verify(idx)
    end
    ids
end

"""
Snapshot of the hierarchy with ids mapped through `key` (key[id] = point index):
per level, sorted `(key(x), key(outlink), outdist)` triples and sorted
`(sorted member keys, key(rep), radius, maxout)` tuples.
"""
function structure(idx, key=collect(1:HNNG._ntotal(idx)))
    snap = []
    for lv in idx.levels
        outs = sort!([(key[x], lv.outlink[x] == 0 ? 0 : key[lv.outlink[x]], lv.outdist[x])
                      for x in findall(lv.present)])
        cls = sort!([(sort!(key[c.members]), key[c.rep], c.radius, copy(c.maxout))
                     for c in lv.clusters if c.rep != 0])
        push!(snap, (outs=outs, clusters=cls))
    end
    snap
end

"""Integer fields equal, float fields equal to `rtol`."""
function same_structure(a, b; rtol=1e-9)
    length(a) == length(b) || return false
    for (la, lb) in zip(a, b)
        length(la.outs) == length(lb.outs) || return false
        for (x, y) in zip(la.outs, lb.outs)
            (x[1] == y[1] && x[2] == y[2] && isapprox(x[3], y[3]; rtol)) || return false
        end
        length(la.clusters) == length(lb.clusters) || return false
        for (x, y) in zip(la.clusters, lb.clusters)
            (x[1] == y[1] && x[2] == y[2] && isapprox(x[3], y[3]; rtol) &&
             length(x[4]) == length(y[4]) && all(isapprox.(x[4], y[4]; rtol))) || return false
        end
    end
    true
end

"""Representative of the root cluster."""
top_rep(idx) = first(c.rep for c in idx.levels[end].clusters if c.rep != 0)
```

`test/invariants.jl`:
```julia
isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))

# Hand-built level 0 over the points 0, 1, 5 on a line: 1 <-> 2 is the mutual pair,
# 3 -> 2; one cluster with representative 2 (two inlinks).
function three_point_index()
    idx = HNNGIndex(Euclidean(), 1; T=Float64)
    for x in (0.0, 1.0, 5.0)
        HNNG._append!(idx.store, [x])
        push!(idx.alive, true)
    end
    lv = HNNG.Level(3)
    lv.present .= true
    lv.npop = 3
    lv.outlink .= [2, 1, 2]
    lv.outdist .= [1.0, 1.0, 4.0]
    lv.inlinks .= [[2], [1, 3], Int[]]
    push!(lv.clusters, HNNG.Cluster(2, [1, 2, 3], [1.0, 0.0, 4.0], 4.0, [4.0]))
    lv.cluster_of .= 1
    lv.nclusters = 1
    push!(idx.levels, lv)
    idx
end

# Hand-built level 0 over 0, 1, 10, 11: two clusters, so level 0 cannot be the top.
function two_cluster_level0()
    idx = HNNGIndex(Euclidean(), 1; T=Float64)
    for x in (0.0, 1.0, 10.0, 11.0)
        HNNG._append!(idx.store, [x])
        push!(idx.alive, true)
    end
    lv = HNNG.Level(4)
    lv.present .= true
    lv.npop = 4
    lv.outlink .= [2, 1, 4, 3]
    lv.outdist .= 1.0
    lv.inlinks .= [[2], [1], [4], [3]]
    push!(lv.clusters, HNNG.Cluster(1, [1, 2], [0.0, 1.0], 1.0, [1.0]))
    push!(lv.clusters, HNNG.Cluster(3, [3, 4], [0.0, 1.0], 1.0, [1.0]))
    lv.cluster_of .= [1, 1, 2, 2]
    lv.nclusters = 2
    push!(idx.levels, lv)
    idx
end

@testset "verify accepts consistent indexes" begin
    @test verify(HNNGIndex(Euclidean(), 2))
    @test verify(three_point_index())
    @test HNNG._verify_level(two_cluster_level0(), 0)
end

@testset "verify rejects each broken invariant" begin
    function broken(f)
        idx = three_point_index()
        f(idx)
        idx
    end
    @test_throws "population differs from the live set" verify(broken(i -> i.alive[3] = false))
    @test_throws "outlink of 1 is 3 but its nearest neighbour is 2" verify(broken(function (i)
        lv = i.levels[1]
        lv.outlink[1] = 3; lv.inlinks[3] = [1]; lv.inlinks[2] = [3]
    end))
    @test_throws "outdist of 1" verify(broken(i -> i.levels[1].outdist[1] = 0.5))
    @test_throws "inlinks of 1 are not sorted and unique" verify(broken(i -> i.levels[1].inlinks[1] = [2, 2]))
    @test_throws "1 is missing from the inlinks of 2" verify(broken(i -> i.levels[1].inlinks[2] = [3]))
    @test_throws "is not the connected component" verify(broken(function (i)
        lv = i.levels[1]
        lv.clusters[1] = HNNG.Cluster(2, [1, 2], [1.0, 0.0], 1.0, [1.0])
        push!(lv.clusters, HNNG.Cluster(3, [3], [0.0], 0.0, [4.0]))
        lv.cluster_of[3] = 2
        lv.nclusters = 2
    end))
    @test_throws "has representative 1, the election gives 2" verify(broken(i -> i.levels[1].clusters[1].rep = 1))
    @test_throws "repdist of 3" verify(broken(i -> i.levels[1].clusters[1].repdist[3] = 4.5))
    @test_throws "radius is 3.0, expected 4.0" verify(broken(i -> i.levels[1].clusters[1].radius = 3.0))
    @test_throws "maxout[0]" verify(broken(i -> i.levels[1].clusters[1].maxout = [1.0]))
    @test_throws "top level 0 has 2 clusters" verify(two_cluster_level0())
    nolevels = HNNGIndex(Euclidean(), 1)
    HNNG._append!(nolevels.store, Float32[1])
    push!(nolevels.alive, true)
    @test_throws "live elements but no levels" verify(nolevels)
end
```

`test/runtests.jl`:
```julia
using Test, HNNG

@testset "HNNG.jl" begin
    include("oracle.jl")
    include("metrics.jl")
    include("invariants.jl")
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. -e 'using Test, HNNG; include("test/invariants.jl")'`
Expected: FAIL with `UndefVarError: \`verify\` not defined`.

- [ ] **Step 3: Write the implementation**

`src/insert.jl`:
```julia
# Level maintenance (spec §6). This task adds the shared canonical helpers.

"""`(d, i) < (bd, bi)` lexicographically; `bi == 0` means no candidate yet."""
@inline _better(d::Float64, i::Int, bd::Float64, bi::Int) =
    bi == 0 || d < bd || (d == bd && i < bi)

"""Sum of the outlink distances of `m`'s inlinkers, added in ascending value order
so the result does not depend on ids or history."""
function _insum(lv::Level, m::Int)
    inl = lv.inlinks[m]
    k = length(inl)
    k == 0 && return 0.0
    k == 1 && return lv.outdist[inl[1]]
    k == 2 && return lv.outdist[inl[1]] + lv.outdist[inl[2]]
    s = 0.0
    for x in sort!([lv.outdist[i] for i in inl])
        s += x
    end
    s
end

"""Representative of a component (`members` sorted): most inlinks, then smallest
inlinker-distance sum, then smallest id."""
function _elect(lv::Level, members::Vector{Int})
    best, bdeg, bsum = 0, -1, Inf
    for m in members
        deg = length(lv.inlinks[m])
        if deg > bdeg
            best, bdeg, bsum = m, deg, _insum(lv, m)
        elseif deg == bdeg
            s = _insum(lv, m)
            if s < bsum
                best, bsum = m, s
            end
        end
    end
    best
end

"""Sorted connected component of `r`, outlinks taken as undirected edges."""
function _component(lv::Level, r::Int)
    comp = [r]
    seen = Set{Int}((r,))
    i = 1
    while i <= length(comp)
        x = comp[i]
        i += 1
        y = lv.outlink[x]
        if y != 0 && !(y in seen)
            push!(seen, y)
            push!(comp, y)
        end
        for z in lv.inlinks[x]
            if !(z in seen)
                push!(seen, z)
                push!(comp, z)
            end
        end
    end
    sort!(comp)
end

@inline _insert_sorted!(v::Vector{Int}, x::Int) = insert!(v, searchsortedfirst(v, x), x)

@inline function _remove_sorted!(v::Vector{Int}, x::Int)
    k = searchsortedfirst(v, x)
    (k <= length(v) && v[k] == x) || error("HNNG internal error: $x is not in the inlink list")
    deleteat!(v, k)
end
```

`src/verify.jl`:
```julia
# verify(idx): the invariants of spec §6.1, recomputed from scratch.

_fail(msg::AbstractString) = error("HNNG.verify: " * msg)

"""
    verify(idx) -> true

Check every invariant of spec §6.1 from scratch: populations; exact 1-NN outlinks
with `(distance, id)` ties (brute force over each level's population); inlinks as
the sorted reverse index; clusters as connected components; the representative
rule; `repdist`, `radius` and `maxout` recomputed bottom-up; one cluster at the
top. Throws `ErrorException("HNNG.verify: ...")` naming the first violation.
Distance evaluations made here are not counted. Assumes a symmetric distance.
"""
function verify(idx::HNNGIndex)
    n = _ntotal(idx)
    length(idx.alive) == n || _fail("alive has length $(length(idx.alive)) but the store holds $n")
    if isempty(idx.levels)
        any(idx.alive) && _fail("the index has live elements but no levels")
        return true
    end
    for l in 0:nlevels(idx)-1
        _verify_level(idx, l)
    end
    _verify_top(idx)
end

function _verify_top(idx::HNNGIndex)
    L = nlevels(idx)
    idx.levels[L].nclusters == 1 ||
        _fail("top level $(L-1) has $(idx.levels[L].nclusters) clusters, expected 1")
    for l in 0:L-2
        idx.levels[l+1].nclusters >= 2 || _fail("level $l has one cluster but is not the top level")
    end
    true
end

function _rep_mask(idx::HNNGIndex, l::Int)
    mask = falses(_ntotal(idx))
    for c in idx.levels[l+1].clusters
        c.rep == 0 || (mask[c.rep] = true)
    end
    mask
end

"""Radius and maxout of cluster `ci` at level `l`, recomputed with fresh distances."""
function _fresh_bounds(idx::HNNGIndex, l::Int, ci::Int)
    lv = idx.levels[l+1]
    c = lv.clusters[ci]
    pr = _point(idx.store, c.rep)
    mo = fill(-Inf, l + 1)
    rad = 0.0
    for m in c.members
        drm = m == c.rep ? 0.0 : _rawdist(idx, pr, _point(idx.store, m))
        mo[l+1] = max(mo[l+1], lv.outdist[m])
        if l == 0
            rad = max(rad, drm)
            continue
        end
        below = idx.levels[l]
        cj = below.cluster_of[m]
        cj == 0 && _fail("level $l: member $m of cluster $ci has no child cluster")
        below.clusters[cj].rep == m ||
            _fail("level $l: member $m is not the representative of its child cluster")
        r2, mo2 = _fresh_bounds(idx, l - 1, cj)
        rad = max(rad, drm + r2)
        for t in 1:l
            mo[t] = max(mo[t], mo2[t])
        end
    end
    rad, mo
end

function _verify_level(idx::HNNGIndex, l::Int)
    lv = idx.levels[l+1]
    n = _ntotal(idx)
    for v in (lv.outlink, lv.outdist, lv.inlinks, lv.present, lv.cluster_of)
        length(v) == n || _fail("level $l: an id-indexed array has length $(length(v)), expected $n")
    end
    want = l == 0 ? idx.alive : _rep_mask(idx, l - 1)
    lv.present == want || _fail(l == 0 ? "level 0: population differs from the live set" :
                                "level $l: population differs from the representatives of level $(l-1)")
    pop = findall(lv.present)
    lv.npop == length(pop) || _fail("level $l: npop is $(lv.npop) but $(length(pop)) elements are present")
    for x in 1:n
        lv.present[x] && continue
        (lv.outlink[x] == 0 && isempty(lv.inlinks[x]) && lv.cluster_of[x] == 0) ||
            _fail("level $l: absent element $x still has links or a cluster")
    end
    for x in pop
        bd, bj = Inf, 0
        px = _point(idx.store, x)
        for y in pop
            y == x && continue
            dy = _rawdist(idx, px, _point(idx.store, y))
            if _better(dy, y, bd, bj)
                bd, bj = dy, y
            end
        end
        lv.outlink[x] == bj ||
            _fail("level $l: outlink of $x is $(lv.outlink[x]) but its nearest neighbour is $bj")
        (bj == 0 ? lv.outdist[x] == Inf : isapprox(lv.outdist[x], bd; rtol=1e-12)) ||
            _fail("level $l: outdist of $x is $(lv.outdist[x]), expected $bd")
        inl = lv.inlinks[x]
        (issorted(inl) && allunique(inl)) || _fail("level $l: inlinks of $x are not sorted and unique")
        for y in inl
            lv.outlink[y] == x || _fail("level $l: $y is an inlink of $x but its outlink is $(lv.outlink[y])")
        end
        bj == 0 || insorted(x, lv.inlinks[bj]) || _fail("level $l: $x is missing from the inlinks of $bj")
    end
    seen = falses(n)
    live = 0
    for (ci, c) in enumerate(lv.clusters)
        if c.rep == 0
            (isempty(c.members) && ci in lv.free) || _fail("level $l: dead cluster slot $ci is not on the free list")
            continue
        end
        live += 1
        (issorted(c.members) && allunique(c.members)) || _fail("level $l: members of cluster $ci are not sorted and unique")
        for m in c.members
            lv.present[m] || _fail("level $l: cluster $ci holds absent element $m")
            lv.cluster_of[m] == ci || _fail("level $l: cluster_of[$m] is $(lv.cluster_of[m]), expected $ci")
            seen[m] && _fail("level $l: element $m is in two clusters")
            seen[m] = true
        end
        _component(lv, c.rep) == c.members ||
            _fail("level $l: cluster $ci is not the connected component of its representative $(c.rep)")
        e = _elect(lv, c.members)
        e == c.rep || _fail("level $l: cluster $ci has representative $(c.rep), the election gives $e")
        length(c.repdist) == length(c.members) || _fail("level $l: cluster $ci repdist has the wrong length")
        pr = _point(idx.store, c.rep)
        for (i, m) in enumerate(c.members)
            dm = m == c.rep ? 0.0 : _rawdist(idx, pr, _point(idx.store, m))
            isapprox(c.repdist[i], dm; rtol=1e-12) ||
                _fail("level $l: cluster $ci repdist of $m is $(c.repdist[i]), expected $dm")
        end
        r, mo = _fresh_bounds(idx, l, ci)
        isapprox(c.radius, r; rtol=1e-9) || _fail("level $l: cluster $ci radius is $(c.radius), expected $r")
        length(c.maxout) == l + 1 || _fail("level $l: cluster $ci maxout has length $(length(c.maxout))")
        for t in 1:l+1
            (c.maxout[t] == mo[t] || isapprox(c.maxout[t], mo[t]; rtol=1e-9)) ||
                _fail("level $l: cluster $ci maxout[$(t-1)] is $(c.maxout[t]), expected $(mo[t])")
        end
    end
    for x in pop
        seen[x] || _fail("level $l: present element $x is in no cluster")
    end
    live == lv.nclusters || _fail("level $l: nclusters is $(lv.nclusters) but $live clusters are live")
    for ci in lv.free
        lv.clusters[ci].rep == 0 || _fail("level $l: free slot $ci holds a live cluster")
    end
    true
end
```

`src/HNNG.jl`:
```julia
module HNNG

using Distances: Distances, PreMetric, Metric, Euclidean, SqEuclidean, CosineDist

export HNNGIndex, UserMetric, Euclidean, SqEuclidean, CosineDist
export is_metric, nlevels, element, verify

include("types.jl")
include("kernels.jl")
include("metrics.jl")
include("insert.jl")
include("navigate.jl")
include("verify.jl")

end # module
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Test, HNNG; include("test/invariants.jl")'`
Expected: both testsets pass (3 + 12 tests).
Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: `Testing HNNG tests passed`.

- [ ] **Step 5: Commit**

```bash
git add src/HNNG.jl src/insert.jl src/verify.jl test/oracle.jl test/invariants.jl test/runtests.jl
git commit -m "feat: verify checks every invariant from scratch; brute-force oracle helpers" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Level 0 — `link!` by full scan, repair, election, bounds, `add!`

This task builds level 0 only, with the nearest-neighbour search done by an exact scan of the level population, so that outlinks, inlinks, clusters, representatives and bounds are right before any pruning exists. `add!` does not climb yet; tests check level 0 with `HNNG._verify_level(idx, 0)`.

**Files:**
- Modify: `src/insert.jl` (append), `src/HNNG.jl`, `test/invariants.jl` (append)

**Interfaces:**
- Consumes: Task 4 helpers (`_better`, `_elect`, `_component`, `_insert_sorted!`, `_remove_sorted!`), Task 3 (`_mdist`, `_d`, `_prepare`, `_point`), Task 2 (`Level`, `Cluster`, `_begin!`, `_grow!`).
- Produces:
  - `_scan_level(idx, q, l::Int, ex1::Int, ex2::Int) -> (nn::Int, nnd::Float64, cands::Vector{Int})` — exact scan; distances of `cands` are in `idx.memo.val`
  - `_nn_search(idx, q, l::Int, ex1::Int, ex2::Int, floating::Vector{Int}, record::Bool) -> (nn, nnd, cands)` — same contract (scan body here; Task 7 swaps in the descent)
  - `_repdist(idx, lv::Level, m::Int, r::Int) -> Float64`
  - `_bounds!(idx, l::Int, c::Cluster) -> Cluster` (sets `radius`, `maxout`)
  - `_new_slot!(lv::Level, c::Cluster) -> Int`, `_free_slot!(lv::Level, ci::Int)`
  - `_refresh_up!(idx, l::Int, cis::Vector{Int})`
  - `_repair!(idx, l::Int, affected::Vector{Int}, extra::Vector{Int}, gone::Int, floating::Vector{Int}) -> (removed::Vector{Int}, added::Vector{Int})`
  - `link!(idx, e::Int, l::Int, floating::Vector{Int}) -> (removed, added)`; at `l == 0` it adds one to `idx.last_rewired` per rewired element
  - `_insert!(idx, v) -> Int`, `_flush!(idx)`
  - exported `add!(idx, x::AbstractVector) -> Int`, `add!(idx, X::AbstractMatrix) -> UnitRange{Int}`, `add!(idx, obj::E) -> Int`, `add!(idx, objs::AbstractVector{<:E}) -> UnitRange{Int}`

- [ ] **Step 1: Write the failing tests**

Append to `test/invariants.jl`:
```julia
@testset "level 0: streaming inserts keep the exact 1-NN graph" begin
    rng = StableRNG(5)
    for (T, d, n) in ((Float64, 2, 120), (Float32, 8, 100))
        idx = HNNGIndex(Euclidean(), d; T=T)
        X = gauss(rng, T, d, n)
        for j in 1:n
            @test add!(idx, view(X, :, j)) == j
            @test HNNG._verify_level(idx, 0)
        end
        @test length(idx) == n
        @test idx.levels[1].npop == n
    end
end

@testset "level 0: four points on a line" begin
    idx = HNNGIndex(Euclidean(), 1; T=Float64)
    for x in (0.0, 1.0, 10.0, 11.0)
        add!(idx, [x])
        @test HNNG._verify_level(idx, 0)
    end
    lv = idx.levels[1]
    @test lv.outlink == [2, 1, 4, 3]
    @test lv.inlinks == [[2], [1], [4], [3]]
    @test sort([c.rep for c in lv.clusters if c.rep != 0]) == [1, 3]
end

@testset "add! accepts any real input on a Float32 index" begin   # Review Focus 3
    idx = HNNGIndex(Euclidean(), 3)
    @test add!(idx, [1, 2, 3]) == 1
    @test add!(idx, [0.5, 0.25, 2.0]) == 2
    M = [1.0 4.0; 2.0 5.0; 3.5 6.0]
    @test add!(idx, M) == 3:4
    @test add!(idx, view(permutedims(M), 2, :)) == 5
    @test add!(idx, Float16[1, 1, 1]) == 6
    @test HNNG._verify_level(idx, 0)
    @test element(idx, 1) == Float32[1, 2, 3]
    @test element(idx, 5) == Float32[4, 5, 6]
    @test eltype(element(idx, 2)) == Float32
end

@testset "add! on an ObjectStore" begin
    idx = HNNGIndex(UserMetric((a, b) -> abs(a - b); is_metric=true), Float64)
    @test add!(idx, 3.0) == 1
    @test add!(idx, [1.0, 7.5, 2.0]) == 2:4
    @test HNNG._verify_level(idx, 0)
    @test element(idx, 3) == 7.5
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. -e 'using Test, HNNG; include("test/invariants.jl")'`
Expected: the four new testsets error with `UndefVarError: \`add!\` not defined`; the Task 4 testsets pass.

- [ ] **Step 3: Write the implementation**

Append to `src/insert.jl`:
```julia
"""
    _scan_level(idx, q, l, ex1, ex2) -> (nn, nnd, cands)

Exact scan of the level-`l` population: the `(distance, id)`-nearest element to `q`
other than `ex1` and `ex2`, and every element evaluated. Distances of `cands` are in
`idx.memo.val`. Used by `link!`/`unlink!` until Task 7 and by the tests as an oracle.
"""
function _scan_level(idx::HNNGIndex, q, l::Int, ex1::Int, ex2::Int)
    memo = _begin!(idx.memo, _ntotal(idx))
    lv = idx.levels[l+1]
    nn, nnd = 0, Inf
    cands = Int[]
    for x in 1:_ntotal(idx)
        (lv.present[x] && x != ex1 && x != ex2) || continue
        d = _mdist(memo, idx, q, x)
        push!(cands, x)
        if _better(d, x, nnd, nn)
            nn, nnd = x, d
        end
    end
    nn, nnd, cands
end

"""
    _nn_search(idx, q, l, ex1, ex2, floating, record) -> (nn, nnd, cands)

Nearest level-`l` element to `q` excluding `ex1`, `ex2`; with `record`, also every
evaluated level-`l` element (a superset of the elements whose outlink a new element
at `q` would improve). Distances of `cands` are in `idx.memo.val`.
"""
function _nn_search(idx::HNNGIndex, q, l::Int, ex1::Int, ex2::Int,
                    floating::Vector{Int}, record::Bool)
    nn, nnd, cands = _scan_level(idx, q, l, ex1, ex2)
    nn, nnd, record ? cands : Int[]
end

"""Distance from representative `r` to member `m`, reused from `m`'s current cluster
when that cluster already has representative `r`."""
@inline function _repdist(idx::HNNGIndex, lv::Level, m::Int, r::Int)
    ci = lv.cluster_of[m]
    if ci != 0
        old = lv.clusters[ci]
        if old.rep == r
            return old.repdist[searchsortedfirst(old.members, m)]
        end
    end
    _d(idx, r, m)
end

"""
    _bounds!(idx, l, c) -> c

Set `radius` and `maxout` of the level-`l` cluster `c` from `repdist`, the level-`l`
outlink distances and the level-`(l-1)` child clusters (spec §6.2). A member whose
child is not the member's own cluster adds the middle term `M(m, rep(child))`. A
member without a child cluster (mid-cascade only) contributes `repdist` alone.
"""
function _bounds!(idx::HNNGIndex, l::Int, c::Cluster)
    lv = idx.levels[l+1]
    mo = fill(-Inf, l + 1)
    rad = 0.0
    for (i, m) in enumerate(c.members)
        mo[l+1] = max(mo[l+1], lv.outdist[m])
        if l == 0
            rad = max(rad, c.repdist[i])
            continue
        end
        below = idx.levels[l]
        cj = below.cluster_of[m]
        if cj == 0
            rad = max(rad, c.repdist[i])
            continue
        end
        ch = below.clusters[cj]
        mid = ch.rep == m ? 0.0 : _d(idx, m, ch.rep)
        rad = max(rad, c.repdist[i] + mid + ch.radius)
        for t in 1:l
            mo[t] = max(mo[t], ch.maxout[t])
        end
    end
    c.radius = rad
    c.maxout = mo
    c
end

function _new_slot!(lv::Level, c::Cluster)
    lv.nclusters += 1
    if isempty(lv.free)
        push!(lv.clusters, c)
        return length(lv.clusters)
    end
    ci = pop!(lv.free)
    lv.clusters[ci] = c
    ci
end

function _free_slot!(lv::Level, ci::Int)
    lv.clusters[ci] = _dead_cluster()
    push!(lv.free, ci)
    lv.nclusters -= 1
    nothing
end

"""
    _refresh_up!(idx, l, cis)

Recompute `radius` and `maxout` of every ancestor of the level-`l` clusters `cis`:
each member present at level `l+1` names a parent to refresh, and so on to the top.
"""
function _refresh_up!(idx::HNNGIndex, l::Int, cis::Vector{Int})
    cur = cis
    L = l
    while L + 1 < nlevels(idx)
        lv, up = idx.levels[L+1], idx.levels[L+2]
        nxt = Int[]
        for ci in cur
            c = lv.clusters[ci]
            c.rep == 0 && continue
            for m in c.members
                if up.present[m]
                    pj = up.cluster_of[m]
                    pj == 0 || push!(nxt, pj)
                end
            end
        end
        isempty(nxt) && return nothing
        unique!(sort!(nxt))
        for pj in nxt
            _bounds!(idx, L + 1, up.clusters[pj])
        end
        cur = nxt
        L += 1
    end
    nothing
end

"""
    _repair!(idx, l, affected, extra, gone, floating) -> (removed, added)

Replace the level-`l` clusters `affected` by the connected components of their
members plus `extra`, minus `gone` (the pool is closed under links). Elects
representatives, fills `repdist`, `radius` and `maxout`, appends the new cluster
indices to `floating`, refreshes ancestor bounds, and returns the representative
delta of level `l`.
"""
function _repair!(idx::HNNGIndex, l::Int, affected::Vector{Int}, extra::Vector{Int},
                  gone::Int, floating::Vector{Int})
    lv = idx.levels[l+1]
    affected = unique!(sort(affected))
    oldreps = Int[]
    pool = copy(extra)
    for ci in affected
        c = lv.clusters[ci]
        push!(oldreps, c.rep)
        for m in c.members
            m == gone || push!(pool, m)
        end
    end
    sort!(pool)
    done = Set{Int}()
    newcl = Cluster[]
    for s in pool
        s in done && continue
        comp = _component(lv, s)
        union!(done, comp)
        r = _elect(lv, comp)
        rd = [m == r ? 0.0 : _repdist(idx, lv, m, r) for m in comp]
        push!(newcl, Cluster(r, comp, rd, 0.0, Float64[]))
    end
    for ci in affected
        _free_slot!(lv, ci)
    end
    newidx = Int[]
    for c in newcl
        ci = _new_slot!(lv, c)
        for m in c.members
            lv.cluster_of[m] = ci
        end
        _bounds!(idx, l, c)
        push!(newidx, ci)
    end
    append!(floating, newidx)
    _refresh_up!(idx, l, newidx)
    newreps = [c.rep for c in newcl]
    setdiff(oldreps, newreps), setdiff(newreps, oldreps)
end

"""
    link!(idx, e, l, floating) -> (removed, added)

Add `e` to the level-`l` population (spec §6.3): outlink to its exact nearest
neighbour, rewire every element `x` with `(M(x, e), e) < (outdist(x), outlink(x))`,
repair the touched clusters, refresh ancestors, return the representative delta.
"""
function link!(idx::HNNGIndex, e::Int, l::Int, floating::Vector{Int})
    lv = idx.levels[l+1]
    lv.present[e] && error("HNNG internal error: $e is already present at level $l")
    lv.present[e] = true
    lv.npop += 1
    affected = Int[]
    if lv.npop >= 2
        nn, nnd, cands = _nn_search(idx, _point(idx.store, e), l, e, 0, floating, true)
        nn == 0 && error("HNNG internal error: no neighbour found for $e at level $l")
        val = idx.memo.val
        lv.outlink[e] = nn
        lv.outdist[e] = nnd
        _insert_sorted!(lv.inlinks[nn], e)
        push!(affected, lv.cluster_of[nn])
        for x in cands
            dx = val[x]
            _better(dx, e, lv.outdist[x], lv.outlink[x]) || continue
            old = lv.outlink[x]
            old == 0 || _remove_sorted!(lv.inlinks[old], x)
            push!(affected, lv.cluster_of[x])
            lv.outlink[x] = e
            lv.outdist[x] = dx
            _insert_sorted!(lv.inlinks[e], x)
            l == 0 && (idx.last_rewired += 1)      # read by HNNG.last_update_stats
        end
    end
    _repair!(idx, l, affected, [e], 0, floating)
end

_flush!(idx::HNNGIndex) = (idx.nevals += idx.memo.nevals; idx.memo.nevals = 0; idx)

function _insert!(idx::HNNGIndex, v)
    idx.last_rewired = 0
    idx.last_top = 0
    id = _append!(idx.store, v)
    push!(idx.alive, true)
    for lv in idx.levels
        _grow!(lv, id)
    end
    isempty(idx.levels) && push!(idx.levels, Level(id))
    link!(idx, id, 0, Int[])
    _flush!(idx)
    id
end

"""
    add!(idx, x::AbstractVector) -> Int              # VectorStore: one element
    add!(idx, X::AbstractMatrix) -> UnitRange{Int}   # VectorStore: the columns, in order
    add!(idx, obj::E) -> Int                         # ObjectStore{E}
    add!(idx, objs::AbstractVector{<:E}) -> UnitRange{Int}

Insert and return the new id(s). Vector input of any `Real` type is converted to the
store's element type; a matrix is validated in full before anything is inserted.
"""
add!(idx::HNNGIndex{<:VectorStore}, x::AbstractVector{<:Real}) = _insert!(idx, _prepare(idx, x))

function add!(idx::HNNGIndex{<:VectorStore}, X::AbstractMatrix{<:Real})
    d = dim(idx.store)
    size(X, 1) == d ||
        throw(ArgumentError("dimension mismatch: the index has dim $d, the matrix has $(size(X, 1)) rows"))
    cols = [_prepare(idx, view(X, :, j)) for j in axes(X, 2)]
    first = _ntotal(idx) + 1
    for v in cols
        _insert!(idx, v)
    end
    first:_ntotal(idx)
end

add!(idx::HNNGIndex{ObjectStore{E}}, obj::E) where {E} = _insert!(idx, obj)

function add!(idx::HNNGIndex{ObjectStore{E}}, objs::AbstractVector{<:E}) where {E}
    first = _ntotal(idx) + 1
    for o in objs
        _insert!(idx, o)
    end
    first:_ntotal(idx)
end
```

In `src/HNNG.jl` change the export line to:
```julia
export is_metric, nlevels, element, verify, add!
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Test, HNNG; include("test/invariants.jl")'`
Expected: all six testsets pass.

- [ ] **Step 5: Commit**

```bash
git add src/HNNG.jl src/insert.jl test/invariants.jl
git commit -m "feat: level-0 link! by exact scan with local cluster repair, election and bounds" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Upper levels — `unlink!`, net delta composition, `propagate!`

**Files:**
- Create: `src/delete.jl`
- Modify: `src/insert.jl` (append `_compose!`, `propagate!`; change `_insert!`), `src/HNNG.jl`, `test/invariants.jl` (append)

**Interfaces:**
- Consumes: Task 5 (`_nn_search`, `_repair!`, `link!`, `_insert_sorted!`, `_remove_sorted!`, `Level`).
- Produces:
  - `unlink!(idx, e::Int, l::Int, floating::Vector{Int}) -> (removed, added)`; at `l == 0` it adds one to `idx.last_rewired` per relinked inlinker
  - `_compose!(rm::Vector{Int}, ad::Vector{Int}, delta::Tuple{Vector{Int},Vector{Int}}) -> (rm, ad)` (net composition, in place)
  - `propagate!(idx, removed::Vector{Int}, added::Vector{Int}, l::Int) -> idx`; raises `idx.last_top` to every level it processes, and to the old top when it drops levels
  - `_insert!` now climbs: `propagate!(idx, link!(idx, id, 0, Int[])..., 1)`

- [ ] **Step 1: Write the failing tests**

Append to `test/invariants.jl`:
```julia
@testset "hierarchy: four points on a line" begin
    idx = HNNGIndex(Euclidean(), 1; T=Float64)
    stream!(idx, reshape([0.0, 1.0, 10.0, 11.0], 1, :))
    @test nlevels(idx) == 2
    top = idx.levels[2]
    c = only([c for c in top.clusters if c.rep != 0])
    @test c.members == [1, 3] && c.rep == 1
    @test c.radius == 11.0
    @test c.maxout == [1.0, 10.0]
    @test top.outlink[1] == 3 && top.outlink[3] == 1
end

@testset "hierarchy: streaming inserts keep every invariant at every level" begin
    rng = StableRNG(6)
    for (T, d, n, gen) in ((Float64, 2, 150, gauss), (Float32, 3, 120, blobs), (Float64, 8, 80, gauss))
        idx = HNNGIndex(Euclidean(), d; T=T)
        stream!(idx, gen(rng, T, d, n))
        @test length(idx) == n
        @test nlevels(idx) >= 2
    end
end

@testset "hierarchy: unlink! then link! of any element is the identity" begin
    rng = StableRNG(66)
    idx = HNNGIndex(Euclidean(), 2; T=Float64)
    stream!(idx, gauss(rng, Float64, 2, 40))
    snap = structure(idx)
    for e in 1:40
        HNNG.propagate!(idx, HNNG.unlink!(idx, e, 0, Int[])..., 1)
        HNNG.propagate!(idx, HNNG.link!(idx, e, 0, Int[])..., 1)
        @test verify(idx)
        @test same_structure(structure(idx), snap)
    end
end

@testset "net composition of representative deltas" begin
    rm, ad = Int[], Int[]
    HNNG._compose!(rm, ad, ([5], [7]))
    HNNG._compose!(rm, ad, ([7, 9], [5, 11]))
    @test sort(rm) == [9] && sort(ad) == [11]
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. -e 'using Test, HNNG; include("test/invariants.jl")'`
Expected: the three `hierarchy:` testsets error inside `stream!` with `HNNG.verify: top level 0 has 2 clusters, expected 1` (inserts do not climb yet); `net composition of representative deltas` errors with `UndefVarError: \`_compose!\` not defined`.

- [ ] **Step 3: Write the implementation**

`src/delete.jl`:
```julia
# Removal from a level (spec §6.3) and, from Task 10, the public delete!.

"""
    unlink!(idx, e, l, floating) -> (removed, added)

Remove `e` from the level-`l` population: every inlinker re-finds its nearest
neighbour with `e` excluded (no rewire sweep is needed), then `e`'s former cluster
and the clusters the inlinkers now point into are repaired.
"""
function unlink!(idx::HNNGIndex, e::Int, l::Int, floating::Vector{Int})
    lv = idx.levels[l+1]
    lv.present[e] || error("HNNG internal error: $e is not present at level $l")
    affected = [lv.cluster_of[e]]
    old = lv.outlink[e]
    old == 0 || _remove_sorted!(lv.inlinks[old], e)
    lv.outlink[e] = 0
    lv.outdist[e] = Inf
    lv.present[e] = false
    lv.npop -= 1
    inl = lv.inlinks[e]
    lv.inlinks[e] = Int[]
    for x in inl
        l == 0 && (idx.last_rewired += 1)          # read by HNNG.last_update_stats
        if lv.npop == 1
            lv.outlink[x] = 0
            lv.outdist[x] = Inf
            continue
        end
        # e keeps its cluster pointer during the descents: its cluster is reached through e
        nn, nnd, _ = _nn_search(idx, _point(idx.store, x), l, e, x, floating, false)
        nn == 0 && error("HNNG internal error: no neighbour found for $x at level $l")
        lv.outlink[x] = nn
        lv.outdist[x] = nnd
        _insert_sorted!(lv.inlinks[nn], x)
        push!(affected, lv.cluster_of[nn])
    end
    lv.cluster_of[e] = 0
    _repair!(idx, l, affected, Int[], e, floating)
end
```

Append to `src/insert.jl`:
```julia
"""
    _compose!(rm, ad, delta) -> (rm, ad)

Net composition of representative deltas (spec §6.4): an element removed and
re-added (or added and removed) within one step is no change.
"""
function _compose!(rm::Vector{Int}, ad::Vector{Int}, delta::Tuple{Vector{Int},Vector{Int}})
    drm, dad = delta
    for r in drm
        k = findfirst(==(r), ad)
        k === nothing ? push!(rm, r) : deleteat!(ad, k)
    end
    for a in dad
        k = findfirst(==(a), rm)
        k === nothing ? push!(ad, a) : deleteat!(rm, k)
    end
    rm, ad
end

"""
    propagate!(idx, removed, added, l) -> idx

Apply the net representative delta of level `l-1` to level `l` and climb (spec §6.4):
stop on an empty delta; drop every level ≥ `l` when level `l-1` has one cluster;
create level `l` from all representatives of level `l-1` when it does not exist;
otherwise `unlink!` the removed and `link!` the added representatives in id order.
"""
function propagate!(idx::HNNGIndex, removed::Vector{Int}, added::Vector{Int}, l::Int)
    while !(isempty(removed) && isempty(added))
        below = idx.levels[l]
        if below.nclusters == 1
            idx.last_top = max(idx.last_top, nlevels(idx) - 1)
            resize!(idx.levels, l)
            return idx
        end
        if nlevels(idx) == l
            push!(idx.levels, Level(_ntotal(idx)))
            removed = Int[]
            added = sort!([c.rep for c in below.clusters if c.rep != 0])
        end
        idx.last_top = max(idx.last_top, l)
        floating = Int[]
        rm, ad = Int[], Int[]
        for r in sort(removed)
            _compose!(rm, ad, unlink!(idx, r, l, floating))
        end
        for a in sort(added)
            _compose!(rm, ad, link!(idx, a, l, floating))
        end
        removed, added = rm, ad
        l += 1
    end
    idx
end
```

In `src/insert.jl` replace the body of `_insert!` with:
```julia
function _insert!(idx::HNNGIndex, v)
    idx.last_rewired = 0
    idx.last_top = 0
    id = _append!(idx.store, v)
    push!(idx.alive, true)
    for lv in idx.levels
        _grow!(lv, id)
    end
    isempty(idx.levels) && push!(idx.levels, Level(id))
    propagate!(idx, link!(idx, id, 0, Int[])..., 1)
    _flush!(idx)
    id
end
```

In `src/HNNG.jl` add `include("delete.jl")` after `include("insert.jl")`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Test, HNNG; include("test/invariants.jl")'`
Expected: all ten testsets pass.
Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: `Testing HNNG tests passed`.

- [ ] **Step 5: Commit**

```bash
git add src/HNNG.jl src/insert.jl src/delete.jl test/invariants.jl
git commit -m "feat: unlink!, net delta composition and level-by-level propagation" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Pruned descent with floating seeds

Replace the scan inside `_nn_search` by the best-first pruned descent of spec §6.3: frontier = every live cluster of the top level plus, below the top, the level-`l` clusters repaired earlier in the same step that have no member at level `l+1` (Spec assumption 2); skip a cluster only if `LB > best` and, when recording rewire candidates, also `LB > maxout[l]` (both strict, guarded `LB`).

**Files:**
- Modify: `src/insert.jl` (append constants and helpers, replace `_nn_search`), `src/HNNG.jl`, `test/invariants.jl` (append)

**Interfaces:**
- Consumes: Task 5 `_scan_level` (kept as the test oracle), `Cluster.maxout`, `DistMemo.front`, `DistMemo.mark`.
- Produces:
  - `const GUARD = 1e-6`, `const FWD = Base.Order.Forward`, `const REV = Base.Order.Reverse`
  - `_lb(dr::Float64, r::Float64) -> Float64` — `max(0, dr - r - GUARD*(dr + r))`
  - `_skip(idx, lb, nnd, c::Cluster, l::Int, record::Bool) -> Bool`
  - `_push_cluster!(memo, idx, q, L::Int, ci::Int, nnd::Float64, l::Int, record::Bool)`
  - `_nn_search` (same signature and contract as Task 5)

- [ ] **Step 1: Write the failing tests**

Append to `test/invariants.jl`:
```julia
@testset "descent: same neighbour and a superset of the rewire candidates" begin
    rng = StableRNG(7)
    for (T, d, gen) in ((Float64, 2, gauss), (Float32, 4, blobs), (Float64, 16, gauss))
        idx = HNNGIndex(Euclidean(), d; T=T)
        add!(idx, gen(rng, T, d, 400))
        @test verify(idx)
        Q = gen(rng, T, d, 30)
        for j in axes(Q, 2), l in 0:nlevels(idx)-1
            q = HNNG._prepare(idx, view(Q, :, j))
            nn, nnd, cands = HNNG._nn_search(idx, q, l, 0, 0, Int[], true)
            snn, snnd, scands = HNNG._scan_level(idx, q, l, 0, 0)
            @test (nn, nnd) == (snn, snnd)
            lv = idx.levels[l+1]
            need = [x for x in scands if HNNG._euclid(q, HNNG._point(idx.store, x)) < lv.outdist[x]]
            @test issubset(need, cands)
            nn2, nnd2, _ = HNNG._nn_search(idx, q, l, snn, 0, Int[], false)
            s2 = HNNG._scan_level(idx, q, l, snn, 0)
            @test (nn2, nnd2) == (s2[1], s2[2])
        end
    end
end

@testset "descent: pruning evaluates far fewer elements than a scan" begin
    rng = StableRNG(77)
    idx = HNNGIndex(Euclidean(), 2; T=Float64)
    add!(idx, blobs(rng, Float64, 2, 2000; nc=20))
    @test verify(idx)
    q = HNNG._prepare(idx, [0.1, -0.2])
    _, _, cands = HNNG._nn_search(idx, q, 0, 0, 0, Int[], true)
    @test length(cands) < 1000
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. -e 'using Test, HNNG; include("test/invariants.jl")'`
Expected: `descent: pruning evaluates far fewer elements than a scan` fails with `Evaluated: 2000 < 1000` (the scan records every element); the other testsets pass.

- [ ] **Step 3: Write the implementation**

Add `using DataStructures: heappush!, heappop!` to the `using` lines of `src/HNNG.jl`.

Append to `src/insert.jl`:
```julia
const GUARD = 1e-6
const FWD = Base.Order.Forward
const REV = Base.Order.Reverse

"""
    _lb(dr, r) -> Float64

Lower bound on the distance from the pass query to every element reachable from a
cluster whose representative is `dr` away and whose radius is `r`, shrunk by the
spec §5 relative guard (1e-6) so that rounding in `dr` and `r` never prunes.
"""
@inline _lb(dr::Float64, r::Float64) = max(0.0, dr - r - GUARD * (dr + r))

# Skip rule of link!/unlink! (spec §6.3): hopeless for the nearest neighbour and,
# when recording rewire candidates, hopeless for every rewire under the cluster.
@inline _skip(idx::HNNGIndex, lb::Float64, nnd::Float64, c::Cluster, l::Int, record::Bool) =
    idx.is_metric && lb > nnd && (!record || lb > c.maxout[l+1])

@inline function _push_cluster!(memo::DistMemo, idx::HNNGIndex, q, L::Int, ci::Int,
                                nnd::Float64, l::Int, record::Bool)
    c = idx.levels[L+1].clusters[ci]
    dr = _mdist(memo, idx, q, c.rep)
    lb = idx.is_metric ? _lb(dr, c.radius) : 0.0
    _skip(idx, lb, nnd, c, l, record) || heappush!(memo.front, (lb, L, ci), FWD)
    nothing
end
```

In `src/insert.jl` replace the Task 5 `_nn_search` (keep its docstring) with:
```julia
function _nn_search(idx::HNNGIndex, q, l::Int, ex1::Int, ex2::Int,
                    floating::Vector{Int}, record::Bool)
    memo = _begin!(idx.memo, _ntotal(idx))
    T = nlevels(idx) - 1
    nn, nnd = 0, Inf
    cands = Int[]
    for (ci, c) in enumerate(idx.levels[T+1].clusters)
        c.rep == 0 || _push_cluster!(memo, idx, q, T, ci, nnd, l, record)
    end
    if l < T
        lv, up = idx.levels[l+1], idx.levels[l+2]
        for ci in floating
            c = lv.clusters[ci]
            (c.rep == 0 || any(m -> up.present[m], c.members)) && continue
            _push_cluster!(memo, idx, q, l, ci, nnd, l, record)
        end
    end
    while !isempty(memo.front)
        lb, L, ci = heappop!(memo.front, FWD)
        c = idx.levels[L+1].clusters[ci]
        _skip(idx, lb, nnd, c, l, record) && continue
        if L == l
            memo.mark[c.rep] == memo.epoch && continue      # reached twice after a merge
            for m in c.members
                memo.mark[m] = memo.epoch
                (m == ex1 || m == ex2) && continue
                d = _mdist(memo, idx, q, m)
                record && push!(cands, m)
                if _better(d, m, nnd, nn)
                    nn, nnd = m, d
                end
            end
        else
            below = idx.levels[L]
            for m in c.members
                cj = below.cluster_of[m]
                cj == 0 || _push_cluster!(memo, idx, q, L - 1, cj, nnd, l, record)
            end
        end
    end
    nn, nnd, cands
end
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Test, HNNG; include("test/invariants.jl")'`
Expected: all twelve testsets pass (the Task 5–6 streaming tests now run on the descent).
Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: `Testing HNNG tests passed`.

- [ ] **Step 5: Commit**

```bash
git add src/HNNG.jl src/insert.jl test/invariants.jl
git commit -m "feat: pruned best-first descent for link!/unlink! with maxout rewire bound and floating seeds" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: `knn` — best-first branch-and-bound, `HNNG.knn_counted`

**Files:**
- Create: `src/search.jl`, `test/search.jl`
- Modify: `src/HNNG.jl`, `test/oracle.jl` (append `check_knn`), `test/runtests.jl`

**Interfaces:**
- Consumes: Task 7 (`_lb`, `FWD`, `REV`, `heappush!`, `heappop!`), Task 3 (`_mdist`, `_prepare`, `_to_user`), Task 5 (`_flush!`).
- Produces:
  - `_offer!(res::Vector{Tuple{Float64,Int}}, d::Float64, m::Int, k::Int)`
  - `_sorted_results(res) -> (ids::Vector{Int}, dists::Vector{Float64})` (internal units)
  - `_knn!(memo::DistMemo, idx, q, k::Int) -> (ids, dists)` (internal units; `q` prepared; counts in `memo.nevals`)
  - exported `knn(idx, q, k::Integer) -> (ids::Vector{Int}, dists::Vector{Float64})`
  - unexported `HNNG.knn_counted(idx, q, k) -> (ids, dists, nevals_this_query::Int)`
  - test helper `check_knn(idx, q, k) -> Bool` (ids and distances equal to `brute_knn`)

- [ ] **Step 1: Write the failing tests**

Append to `test/oracle.jl`:
```julia
"""knn equals the brute-force oracle: same ids and the same distances (same kernel, same bits)."""
function check_knn(idx, q, k)
    ids, ds = knn(idx, q, k)
    bids, bds = brute_knn(idx, q, k)
    ids == bids && ds == bds
end
```

`test/search.jl`:
```julia
isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))

@testset "knn equals the brute-force oracle (Euclidean family)" begin
    rng = StableRNG(8)
    for M in (Euclidean(), SqEuclidean(), CosineDist()), T in (Float32, Float64),
        d in (2, 8, 32), n in (1, 2, 50, 2000)
        idx = HNNGIndex(M, d; T=T)
        add!(idx, gauss(rng, T, d, n))
        @test verify(idx)
        for _ in 1:5, k in (1, 10, n + 5)
            @test check_knn(idx, randn(rng, T, d), k)
        end
    end
end

@testset "knn basics" begin
    idx = HNNGIndex(Euclidean(), 2; T=Float64)
    @test knn(idx, [0.0, 0.0], 3) == (Int[], Float64[])
    add!(idx, [1.0 2.0 4.0; 0.0 0.0 0.0])
    @test verify(idx)
    @test knn(idx, [0.0, 0.0], 2) == ([1, 2], [1.0, 2.0])
    @test knn(idx, [0, 0], 10) == ([1, 2, 3], [1.0, 2.0, 4.0])          # Int query, k > n
    s = HNNGIndex(SqEuclidean(), 2; T=Float64)
    add!(s, [1.0 2.0; 0.0 0.0])
    @test verify(s)
    @test knn(s, [0.0, 0.0], 2)[2] ≈ [1.0, 4.0]
end

@testset "nevals counts the serial knn path" begin
    rng = StableRNG(81)
    idx = HNNGIndex(Euclidean(), 4)
    add!(idx, gauss(rng, Float32, 4, 300))
    @test verify(idx)
    before = idx.nevals
    @test before > 0
    ids, ds, n1 = HNNG.knn_counted(idx, randn(rng, Float32, 4), 5)
    @test 1 <= n1 <= 300                    # memoised: every element at most once
    @test idx.nevals == before + n1
    @test length(ids) == 5 && issorted(ds)
    knn(idx, randn(rng, Float32, 4), 5)
    @test idx.nevals > before + n1
end

@testset "exact duplicates: ties go to the smallest id" begin    # Review Focus 1
    rng = StableRNG(82)
    base = gauss(rng, Float64, 3, 25)
    idx = HNNGIndex(Euclidean(), 3; T=Float64)
    stream!(idx, hcat(base, base, base))
    for j in 1:25
        ids, ds = knn(idx, base[:, j], 3)
        @test ids == [j, j + 25, j + 50] && ds == [0.0, 0.0, 0.0]
        @test check_knn(idx, base[:, j] .+ 0.01, 7)
    end
end

@testset "integer lattice: massive distance ties" begin           # Review Focus 2
    pts = reduce(hcat, [Float32[i, j] for i in 0:7 for j in 0:7])
    idx = HNNGIndex(Euclidean(), 2)
    stream!(idx, pts)
    for q in ([0.5, 0.5], [3, 3], [7.5, -1], [3.5, 3.5]), k in (1, 4, 9, 64)
        @test check_knn(idx, q, k)
    end
end

@testset "cosine: zero vectors are stored and queried" begin       # Review Focus 4
    idx = HNNGIndex(CosineDist(), 3; T=Float64)
    stream!(idx, [0.0 1.0 0.0 2.0; 0.0 0.0 1.0 2.0; 0.0 0.0 0.0 0.0])
    ids, ds = knn(idx, [0.0, 0.0, 0.0], 4)
    @test ids[1] == 1 && sort(ids) == [1, 2, 3, 4]
    @test ds[1] == 0.0 && all(isapprox.(ds[2:end], 0.5; atol=1e-12))
    @test check_knn(idx, [0.0, 0.0, 0.0], 4)
    @test check_knn(idx, [1.0, 1.0, 0.0], 4)
end
```

`test/runtests.jl`:
```julia
using Test, HNNG

@testset "HNNG.jl" begin
    include("oracle.jl")
    include("metrics.jl")
    include("invariants.jl")
    include("search.jl")
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. -e 'using Test, HNNG; include("test/search.jl")'`
Expected: every testset errors with `UndefVarError: \`knn\` not defined`.

- [ ] **Step 3: Write the implementation**

`src/search.jl`:
```julia
# Exact k-NN search (spec §6.5): best-first branch-and-bound over the cluster tree.

@inline function _offer!(res::Vector{Tuple{Float64,Int}}, d::Float64, m::Int, k::Int)
    if length(res) < k
        heappush!(res, (d, m), REV)
    elseif (d, m) < res[1]
        heappop!(res, REV)
        heappush!(res, (d, m), REV)
    end
    nothing
end

function _sorted_results(res::Vector{Tuple{Float64,Int}})
    sort!(res)
    [p[2] for p in res], [p[1] for p in res]
end

"""
    _knn!(memo, idx, q, k) -> (ids, dists)

Best-first engine in internal units. Frontier: min-heap of clusters keyed by the
guarded lower bound; results: max-heap of at most `k` `(distance, id)`. Stops when
`k` results are held and the smallest bound is strictly above the k-th distance.
Distances are memoised for the query; a representative is offered once.
"""
function _knn!(memo::DistMemo, idx::HNNGIndex, q, k::Int)
    _begin!(memo, _ntotal(idx))
    isempty(idx.levels) && return Int[], Float64[]
    res, front = memo.res, memo.front
    met = idx.is_metric
    T = nlevels(idx) - 1
    for (ci, c) in enumerate(idx.levels[T+1].clusters)
        c.rep == 0 && continue
        dr = _mdist(memo, idx, q, c.rep)
        heappush!(front, (met ? _lb(dr, c.radius) : 0.0, T, ci), FWD)
    end
    while !isempty(front)
        lb, L, ci = heappop!(front, FWD)
        met && length(res) == k && lb > res[1][1] && break
        c = idx.levels[L+1].clusters[ci]
        below = idx.levels[max(L, 1)]
        for m in c.members
            d = _mdist(memo, idx, q, m)
            if memo.mark[m] != memo.epoch
                memo.mark[m] = memo.epoch
                _offer!(res, d, m, k)
            end
            L == 0 && continue
            cj = below.cluster_of[m]
            ch = below.clusters[cj]
            dr = ch.rep == m ? d : _mdist(memo, idx, q, ch.rep)
            lbc = met ? _lb(dr, ch.radius) : 0.0
            (met && length(res) == k && lbc > res[1][1]) || heappush!(front, (lbc, L - 1, cj), FWD)
        end
    end
    _sorted_results(res)
end

_check_k(k::Integer) = k >= 1 || throw(ArgumentError("k must be at least 1, got $k"))

"""
    knn(idx, q, k) -> (ids::Vector{Int}, dists::Vector{Float64})

Exact `k` nearest live elements of `q`, sorted by ascending distance, ties by
ascending id, distances in the metric's own units. Fewer than `k` live elements:
all of them. Empty index: two empty vectors. Counts its evaluations in `nevals`.
"""
function knn(idx::HNNGIndex, q, k::Integer)
    _check_k(k)
    ids, d = _knn!(idx.memo, idx, _prepare(idx, q), Int(k))
    _flush!(idx)
    ids, _to_user.(Ref(idx), d)
end

"""
    HNNG.knn_counted(idx, q, k) -> (ids, dists, nevals_this_query)

`knn` plus the number of distance evaluations this query made (the benchmark's
fraction evaluated is that number divided by `length(idx)`).
"""
function knn_counted(idx::HNNGIndex, q, k::Integer)
    _check_k(k)
    ids, d = _knn!(idx.memo, idx, _prepare(idx, q), Int(k))
    n = idx.memo.nevals
    _flush!(idx)
    ids, _to_user.(Ref(idx), d), n
end
```

In `src/HNNG.jl` add `include("search.jl")` after `include("delete.jl")` and change the export line to:
```julia
export is_metric, nlevels, element, verify, add!, knn
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Test, HNNG; include("test/search.jl")'`
Expected: all six testsets pass.
Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: `Testing HNNG tests passed`, total time well under a minute.

- [ ] **Step 5: Commit**

```bash
git add src/HNNG.jl src/search.jl test/oracle.jl test/search.jl test/runtests.jl
git commit -m "feat: exact best-first knn with memoised distances and per-query eval count" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: `HNNG.knn_levelsync` oracle and `knn_batch`

**Files:**
- Modify: `src/search.jl` (append), `src/HNNG.jl`, `test/search.jl` (append)

**Interfaces:**
- Consumes: Task 8 (`_knn!`, `_offer!`, `_sorted_results`, `_check_k`), Task 7 (`_lb`).
- Produces:
  - unexported `HNNG.knn_levelsync(idx, q, k) -> (ids, dists)` (test oracle; not counted)
  - `_queries(idx, Q) -> Vector` of prepared queries
  - exported `knn_batch(idx, Q, k; threads=false) -> (ids::Matrix{Int}, dists::Matrix{Float64})`, both `k × nq`, padded with `0` / `Inf` (Spec assumption 7); `threads=true` uses one `DistMemo` per task and leaves `idx.nevals` unchanged

- [ ] **Step 1: Write the failing tests**

Append to `test/search.jl`:
```julia
@testset "best-first and level-sync engines agree" begin
    rng = StableRNG(9)
    for (M, T, d) in ((Euclidean(), Float32, 8), (SqEuclidean(), Float64, 3), (CosineDist(), Float32, 16))
        idx = HNNGIndex(M, d; T=T)
        add!(idx, blobs(rng, T, d, 600))
        @test verify(idx)
        for _ in 1:20, k in (1, 10)
            q = 5 .* randn(rng, T, d)
            @test HNNG.knn_levelsync(idx, q, k) == knn(idx, q, k)
        end
    end
end

@testset "knn_batch" begin
    rng = StableRNG(91)
    idx = HNNGIndex(Euclidean(), 5)
    add!(idx, gauss(rng, Float32, 5, 400))
    @test verify(idx)
    Q = gauss(rng, Float32, 5, 37)
    ids, ds = knn_batch(idx, Q, 6)
    @test size(ids) == (6, 37) && size(ds) == (6, 37)
    for j in 1:37
        @test (ids[:, j], ds[:, j]) == knn(idx, Q[:, j], 6)
    end
    before = idx.nevals
    tids, tds = knn_batch(idx, Q, 6; threads=true)
    @test tids == ids && tds == ds
    @test idx.nevals == before
    small = HNNGIndex(Euclidean(), 5)
    add!(small, Q[:, 1:3])
    @test verify(small)
    pids, pds = knn_batch(small, Q[:, 4:5], 5)
    @test pids[4:5, :] == zeros(Int, 2, 2) && all(isinf, pds[4:5, :])
    @test knn_batch(small, zeros(Float32, 5, 0), 2) == (zeros(Int, 2, 0), zeros(2, 0))
    objs = HNNGIndex(UserMetric((a, b) -> abs(a - b); is_metric=true), Float64)
    add!(objs, [1.0, 4.0, 9.0])
    @test verify(objs)
    @test knn_batch(objs, [0.0, 10.0], 2) == ([1 3; 2 2], [1.0 1.0; 4.0 6.0])
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. -e 'using Test, HNNG; include("test/search.jl")'`
Expected: the two new testsets error with `UndefVarError: \`knn_levelsync\` not defined in \`HNNG\`` and `UndefVarError: \`knn_batch\` not defined`.

- [ ] **Step 3: Write the implementation**

Append to `src/search.jl`:
```julia
"""
    HNNG.knn_levelsync(idx, q, k) -> (ids, dists)

The level-synchronous descent of the 2015 algorithm with the corrected, strict and
guarded prune: at each level offer every candidate cluster's representative, prune
clusters whose bound exceeds the k-th distance, descend into the survivors' children;
at level 0 evaluate the survivors' members. A second oracle for the tests only.
"""
function knn_levelsync(idx::HNNGIndex, q, k::Integer)
    _check_k(k)
    qp = _prepare(idx, q)
    memo = _begin!(DistMemo(), _ntotal(idx))
    isempty(idx.levels) && return Int[], Float64[]
    res = memo.res
    met = idx.is_metric
    T = nlevels(idx) - 1
    list = [ci for (ci, c) in enumerate(idx.levels[T+1].clusters) if c.rep != 0]
    function offer(m)
        d = _mdist(memo, idx, qp, m)
        if memo.mark[m] != memo.epoch
            memo.mark[m] = memo.epoch
            _offer!(res, d, m, Int(k))
        end
        d
    end
    for L in T:-1:0
        lv = idx.levels[L+1]
        for ci in list
            offer(lv.clusters[ci].rep)
        end
        keep = Int[]
        for ci in list
            c = lv.clusters[ci]
            lb = met ? _lb(offer(c.rep), c.radius) : 0.0
            (met && length(res) == k && lb > res[1][1]) || push!(keep, ci)
        end
        if L == 0
            for ci in keep, m in lv.clusters[ci].members
                offer(m)
            end
        else
            below = idx.levels[L]
            list = [below.cluster_of[m] for ci in keep for m in lv.clusters[ci].members]
        end
    end
    ids, d = _sorted_results(res)
    ids, _to_user.(Ref(idx), d)
end

_queries(idx::HNNGIndex{<:VectorStore}, Q::AbstractMatrix{<:Real}) =
    [_prepare(idx, view(Q, :, j)) for j in axes(Q, 2)]
_queries(idx::HNNGIndex{<:ObjectStore}, Q::AbstractVector) = [_prepare(idx, q) for q in Q]

function _store_col!(ids::Matrix{Int}, ds::Matrix{Float64}, j::Int, a, b, idx::HNNGIndex)
    for i in eachindex(a)
        ids[i, j] = a[i]
        ds[i, j] = _to_user(idx, b[i])
    end
end

"""
    knn_batch(idx, Q, k; threads=false) -> (ids::Matrix{Int}, dists::Matrix{Float64})

`knn` for every query: the columns of a matrix (VectorStore) or the elements of a
vector (ObjectStore). Column `j` holds query `j`'s answer; when fewer than `k` live
elements exist the tail is padded with id `0` and distance `Inf`. `threads=true`
splits the queries over `Threads.nthreads()` tasks with private memos and does not
update the index's evaluation counter.
"""
function knn_batch(idx::HNNGIndex, Q, k::Integer; threads::Bool=false)
    _check_k(k)
    qs = _queries(idx, Q)
    nq = length(qs)
    ids = zeros(Int, k, nq)
    ds = fill(Inf, k, nq)
    nq == 0 && return ids, ds
    if threads
        chunks = collect(Iterators.partition(1:nq, cld(nq, Threads.nthreads())))
        tasks = map(chunks) do r
            Threads.@spawn begin
                memo = DistMemo()
                for j in r
                    a, b = _knn!(memo, idx, qs[j], Int(k))
                    _store_col!(ids, ds, j, a, b, idx)
                end
            end
        end
        foreach(wait, tasks)
    else
        for j in 1:nq
            a, b = _knn!(idx.memo, idx, qs[j], Int(k))
            _store_col!(ids, ds, j, a, b, idx)
        end
        _flush!(idx)
    end
    ids, ds
end
```

In `src/HNNG.jl` change the export line to:
```julia
export is_metric, nlevels, element, verify, add!, knn, knn_batch
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Test, HNNG; include("test/search.jl")'`
Expected: all eight testsets pass.
Run: `julia -t 4 --project=. -e 'using Test, HNNG; include("test/search.jl")'`
Expected: all eight testsets pass with four threads.

- [ ] **Step 5: Commit**

```bash
git add src/HNNG.jl src/search.jl test/search.jl
git commit -m "feat: level-synchronous oracle engine and knn_batch with optional threads" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: `delete!`

**Files:**
- Modify: `src/delete.jl` (append), `test/updates.jl` (create), `test/runtests.jl`

**Interfaces:**
- Consumes: Task 6 (`unlink!`, `propagate!`), Task 3 (`_check_live`), Task 5 (`_flush!`).
- Produces: `Base.delete!(idx::HNNGIndex, id::Integer) -> idx` (resets `last_rewired`/`last_top`; a drain sets `last_top` to the old top level and leaves no levels).

- [ ] **Step 1: Write the failing tests**

`test/updates.jl`:
```julia
isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))

@testset "delete!: four points on a line" begin
    idx = HNNGIndex(Euclidean(), 1; T=Float64)
    stream!(idx, reshape([0.0, 1.0, 10.0, 11.0], 1, :))
    @test delete!(idx, 3) === idx
    @test verify(idx)
    @test length(idx) == 3 && nlevels(idx) == 1
    @test idx.levels[1].outlink[4] == 2
    @test knn(idx, [10.0], 4) == ([4, 2, 1], [1.0, 9.0, 10.0])
end

@testset "delete!: drain to empty and refill" begin
    rng = StableRNG(10)
    X = gauss(rng, Float64, 2, 30)
    idx = HNNGIndex(Euclidean(), 2; T=Float64)
    stream!(idx, X)
    for id in randperm(rng, 30)
        delete!(idx, id)
        @test verify(idx)
        @test check_knn(idx, randn(rng, 2), 3)
    end
    @test length(idx) == 0 && nlevels(idx) == 0
    @test knn(idx, [0.0, 0.0], 1) == (Int[], Float64[])
    @test stream!(idx, X) == 31:60
    @test check_knn(idx, randn(rng, 2), 5)
end

@testset "delete!: removing the root representative again and again" begin
    rng = StableRNG(101)
    idx = HNNGIndex(Euclidean(), 3)
    stream!(idx, blobs(rng, Float32, 3, 150))
    while length(idx) > 1
        delete!(idx, top_rep(idx))
        @test verify(idx)
        @test check_knn(idx, 4 .* randn(rng, Float32, 3), 5)
    end
    @test nlevels(idx) == 1
end
```

`test/runtests.jl`: add `include("updates.jl")` after `include("search.jl")`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. -e 'using Test, HNNG; include("test/updates.jl")'`
Expected: FAIL with `MethodError: no method matching delete!(::HNNGIndex{...}, ::Int64)`.

- [ ] **Step 3: Write the implementation**

Append to `src/delete.jl`:
```julia
"""
    delete!(idx, id) -> idx

Remove a live element exactly (spec §6.4): tombstone the id, `unlink!` it from
level 0 and propagate the representative delta. The id is never reused and its
storage is kept. Deleting the last live element leaves an index with no levels.
"""
function Base.delete!(idx::HNNGIndex, id::Integer)
    i = _check_live(idx, id)
    idx.last_rewired = 0
    idx.last_top = 0
    idx.alive[i] = false
    delta = unlink!(idx, i, 0, Int[])
    if idx.levels[1].npop == 0
        idx.last_top = nlevels(idx) - 1
        empty!(idx.levels)
    else
        propagate!(idx, delta..., 1)
    end
    _flush!(idx)
    idx
end
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Test, HNNG; include("test/updates.jl")'`
Expected: all three testsets pass.
Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: `Testing HNNG tests passed`.

- [ ] **Step 5: Commit**

```bash
git add src/delete.jl test/updates.jl test/runtests.jl
git commit -m "feat: exact delete! with tombstones and level dropping" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: ObjectStore — Hausdorff, the 2015 ROS view distance, Hamming

No new package code is expected: this task proves merit 1 of spec §1 (variable-size elements, arbitrary metrics) end to end. If a test fails, fix the package code the failure points to, inside this task.

**Files:**
- Create: `test/objects.jl`
- Modify: `test/oracle.jl` (append), `test/runtests.jl`

**Interfaces:**
- Consumes: `add!`, `delete!`, `knn`, `HNNG.knn_counted`, `HNNG.knn_levelsync`, `verify`, `check_knn`, `top_rep`.
- Produces (test helpers): `descriptor_sets(rng, d, n; mmin=2, mmax=9) -> Vector{Matrix{Float64}}`, `hausdorff(A, B) -> Float64`, `ros_view_distance(A, B) -> Float64`.

- [ ] **Step 1: Write the tests**

Append to `test/oracle.jl`:
```julia
"""`n` descriptor sets: `d × m` matrices, `m` drawn from `mmin:mmax`, each set around its own centre."""
descriptor_sets(rng, d, n; mmin=2, mmax=9) =
    [randn(rng, d, rand(rng, mmin:mmax)) .+ 3 .* randn(rng, d) for _ in 1:n]

_pd(a, b) = sqrt(sum(abs2, a .- b))
_sq(a, b) = sum(abs2, a .- b)

"""Hausdorff distance between point sets (columns), Euclidean point distance. A metric."""
function hausdorff(A::AbstractMatrix, B::AbstractMatrix)
    h(X, Y) = maximum(minimum(_pd(view(X, :, i), view(Y, :, j)) for j in axes(Y, 2)) for i in axes(X, 2))
    max(h(A, B), h(B, A))
end

"""
The 2015 ROS view distance `max(mean_i min_j ‖a_i − b_j‖², mean_j min_i ‖a_i − b_j‖²)`
(race_feature_extraction/src/spin_image.cpp:252–312, view_clustering_utils.cpp:1957–1969).
Not a metric.
"""
function ros_view_distance(A::AbstractMatrix, B::AbstractMatrix)
    ab = sum(minimum(_sq(view(A, :, i), view(B, :, j)) for j in axes(B, 2)) for i in axes(A, 2)) / size(A, 2)
    ba = sum(minimum(_sq(view(A, :, i), view(B, :, j)) for i in axes(A, 2)) for j in axes(B, 2)) / size(B, 2)
    max(ab, ba)
end
```

`test/objects.jl`:
```julia
isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))

@testset "ObjectStore: Hausdorff on variable-size descriptor sets (pruned, exact)" begin
    rng = StableRNG(11)
    S = descriptor_sets(rng, 3, 80)
    idx = HNNGIndex(UserMetric(hausdorff; is_metric=true), Matrix{Float64})
    @test is_metric(idx)
    for s in S
        add!(idx, s)
        @test verify(idx)
    end
    for q in descriptor_sets(rng, 3, 10), k in (1, 5, 100)
        @test check_knn(idx, q, k)
        @test HNNG.knn_levelsync(idx, q, k) == knn(idx, q, k)
    end
    delete!(idx, 7)
    @test verify(idx)
    delete!(idx, top_rep(idx))
    @test verify(idx)
    @test check_knn(idx, S[7], 3)
end

@testset "ObjectStore: 2015 ROS view distance (not a metric: unpruned, exact)" begin
    rng = StableRNG(111)
    S = descriptor_sets(rng, 4, 70)
    idx = HNNGIndex(ros_view_distance, Matrix{Float64})   # bare function -> UserMetric(is_metric=false)
    @test !is_metric(idx)
    for s in S
        add!(idx, s)
        @test verify(idx)
    end
    for q in descriptor_sets(rng, 4, 10), k in (1, 5)
        @test check_knn(idx, q, k)
        _, _, ne = HNNG.knn_counted(idx, q, k)
        @test ne == length(idx)                        # unpruned: every live element once
    end
end

@testset "ObjectStore: integer-valued Hamming distance on strings" begin   # Review Focus 5
    rng = StableRNG(112)
    hamming(a, b) = count(i -> a[i] != b[i], eachindex(a))
    alphabet = ['a', 'c', 'g', 't']
    words = [String(rand(rng, alphabet, 6)) for _ in 1:90]
    idx = HNNGIndex(UserMetric(hamming; is_metric=true), String)
    for w in words
        add!(idx, w)
        @test verify(idx)
    end
    ids, ds = knn(idx, words[5], 4)
    @test ds isa Vector{Float64} && ds[1] == 0.0 && ids[1] == findfirst(==(words[5]), words)
    for _ in 1:15
        @test check_knn(idx, String(rand(rng, alphabet, 6)), 6)
    end
end
```

`test/runtests.jl`: add `include("objects.jl")` after `include("updates.jl")`.

- [ ] **Step 2: Run the tests**

Run: `julia --project=. -e 'using Test, HNNG; include("test/objects.jl")'`
Expected: all three testsets pass. A failure here is a package bug: reproduce it with the smallest `S` prefix that fails, fix it in `src/`, rerun, and keep the regression in this file.

- [ ] **Step 3: Run the full suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: `Testing HNNG tests passed`.

- [ ] **Step 4: Commit**

```bash
git add test/oracle.jl test/objects.jl test/runtests.jl
git commit -m "test: ObjectStore with Hausdorff, the ROS view distance and integer Hamming" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: Order and history independence

Tests the canonical-structure claim in the form of Spec assumption 1. No new package code is expected; a failure is a package bug to fix inside this task.

**Files:**
- Create: `test/order.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `add!`, `delete!`, `verify`, `stream!`, `structure`, `same_structure`, `gauss`, `blobs`.
- Produces: `test/order.jl` (extended by Tasks 19–20).

- [ ] **Step 1: Write the tests**

`test/order.jl`:
```julia
isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))
using Distances: Cityblock

@testset "five insertion orders give the same level-0 graph and clusters" begin
    rng = StableRNG(12)
    X = gauss(rng, Float64, 3, 200)
    ref = nothing
    for t in 1:5
        p = randperm(rng, 200)
        idx = HNNGIndex(Euclidean(), 3; T=Float64)
        stream!(idx, X[:, p])
        s0 = structure(idx, p)[1]                      # id j holds point p[j]
        g = ([(o[1], o[2]) for o in s0.outs], sort!([c[1] for c in s0.clusters]))
        if t == 1
            ref = g
        else
            @test g == ref
        end
    end
end

@testset "the hierarchy depends only on the live (id, point) set" begin
    rng = StableRNG(121)
    n = 150
    X = blobs(rng, Float64, 2, n)
    A = HNNGIndex(Euclidean(), 2; T=Float64)
    stream!(A, X)
    B = HNNGIndex(Euclidean(), 2; T=Float64)
    key = Int[]                    # key[id] = point index, 0 for noise
    noise = Int[]
    for j in 1:n
        add!(B, X[:, j]); push!(key, j); verify(B)
        if rand(rng) < 0.5
            push!(noise, add!(B, 20 .* rand(rng, 2) .- 10)); push!(key, 0); verify(B)
        end
        if !isempty(noise) && rand(rng) < 0.4
            delete!(B, popat!(noise, rand(rng, 1:length(noise)))); verify(B)
        end
    end
    for id in noise
        delete!(B, id); verify(B)
    end
    @test same_structure(structure(B, key), structure(A))
end

@testset "delete and re-insert equals a fresh build in the same id order" begin
    rng = StableRNG(122)
    n = 120
    X = gauss(rng, Float64, 4, n)
    S = sort!(randperm(rng, n)[1:40])
    C = HNNGIndex(Euclidean(), 4; T=Float64)
    stream!(C, X)
    for i in S
        delete!(C, i); verify(C)
    end
    for i in S
        add!(C, X[:, i]); verify(C)
    end
    rest = setdiff(1:n, S)
    D = HNNGIndex(Euclidean(), 4; T=Float64)
    stream!(D, X[:, vcat(rest, S)])
    @test same_structure(structure(C, vcat(collect(1:n), S)), structure(D, vcat(rest, S)))
end
```

`test/runtests.jl`: add `include("order.jl")` after `include("objects.jl")`.

- [ ] **Step 2: Run the tests**

Run: `julia --project=. -e 'using Test, HNNG; include("test/order.jl")'`
Expected: all three testsets pass.

- [ ] **Step 3: Commit**

```bash
git add test/order.jl test/runtests.jl
git commit -m "test: order and history independence of the hierarchy" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: Update stress — interleaved operations, duplicate deletes, drain and re-insert

**Files:**
- Modify: `test/updates.jl` (append)

**Interfaces:**
- Consumes: `add!`, `delete!`, `knn`, `verify`, `check_knn`, `stream!`, `structure`, `same_structure`.
- Produces: nothing new.

- [ ] **Step 1: Write the tests**

Append to `test/updates.jl`:
```julia
@testset "random interleaved inserts and deletes stay exact" begin
    rng = StableRNG(13)
    for (M, T, d) in ((Euclidean(), Float64, 2), (SqEuclidean(), Float32, 5), (CosineDist(), Float64, 3))
        idx = HNNGIndex(M, d; T=T)
        live = Int[]
        for _ in 1:300
            if isempty(live) || rand(rng) < 0.6
                push!(live, add!(idx, randn(rng, T, d)))
            else
                delete!(idx, popat!(live, rand(rng, 1:length(live))))
            end
            @test verify(idx)
            @test length(idx) == length(live)
            @test check_knn(idx, randn(rng, T, d), 4)
        end
    end
end

@testset "deleting copies of duplicated points" begin      # Review Focus 1
    rng = StableRNG(131)
    base = gauss(rng, Float32, 2, 20)
    idx = HNNGIndex(Euclidean(), 2)
    stream!(idx, hcat(base, base, base))
    for j in 1:20
        delete!(idx, j + 20 * rand(rng, 0:2))
        @test verify(idx)
    end
    for j in 1:20
        ids, ds = knn(idx, base[:, j], 2)
        @test ds == [0.0, 0.0] && all(i -> mod1(i, 20) == j, ids) && issorted(ids)
        @test check_knn(idx, base[:, j], 5)
    end
end

@testset "delete everything and re-insert gives the same structure" begin
    rng = StableRNG(132)
    X = blobs(rng, Float64, 3, 90)
    idx = HNNGIndex(Euclidean(), 3; T=Float64)
    stream!(idx, X)
    ref = structure(idx)
    for id in randperm(rng, 90)
        delete!(idx, id); verify(idx)
    end
    @test nlevels(idx) == 0
    stream!(idx, X)
    @test same_structure(structure(idx, vcat(1:90, 1:90)), ref)
end
```

- [ ] **Step 2: Run the tests**

Run: `julia --project=. -e 'using Test, HNNG; include("test/updates.jl")'`
Expected: all six testsets pass. A failure is a package bug: shrink the step count until it reproduces, fix `src/`, keep the regression.

- [ ] **Step 3: Commit**

```bash
git add test/updates.jl
git commit -m "test: interleaved updates, duplicate deletes, drain and re-insert" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 14: Navigation — links, cluster handles, parent, children, path to root

**Files:**
- Modify: `src/navigate.jl` (append), `src/HNNG.jl`, `test/navigate.jl` (create), `test/runtests.jl`

**Interfaces:**
- Consumes: `_check_live` (Task 3), `_to_user`, `Level`, `Cluster`.
- Produces:
  - `struct ClusterRef; level::Int; index::Int; end`
  - `_check_level(idx, level::Integer) -> Int`, `_cl(idx, c::ClusterRef) -> Cluster` (both throw `ArgumentError`)
  - exported `outlink(idx, id; level=0) -> Int`, `outlink_distance(idx, id; level=0) -> Float64`, `inlinks(idx, id; level=0) -> Vector{Int}`, `cluster(idx, id; level=0) -> Union{ClusterRef,Nothing}`, `members(idx, c) -> Vector{Int}`, `representative(idx, c) -> Int`, `radius(idx, c) -> Float64`, `children(idx, c) -> Vector{ClusterRef}`, `path_to_root(idx, id) -> Vector{ClusterRef}`, `root(idx) -> Union{ClusterRef,Nothing}`
  - `Base.parent(idx::HNNGIndex, c::ClusterRef) -> Union{ClusterRef,Nothing}`
  - test helper `line4()` in `test/navigate.jl`

- [ ] **Step 1: Write the failing tests**

`test/navigate.jl`:
```julia
isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))

# Points 0, 1, 10, 11: level 0 = {1,2} (rep 1) and {3,4} (rep 3); level 1 = {1,3} (rep 1, radius 11).
function line4()
    idx = HNNGIndex(Euclidean(), 1; T=Float64)
    stream!(idx, reshape([0.0, 1.0, 10.0, 11.0], 1, :))
    idx
end

@testset "navigation on four points on a line" begin
    idx = line4()
    n0 = idx.nevals
    @test outlink(idx, 3) == 4 && outlink_distance(idx, 3) == 1.0
    @test inlinks(idx, 2) == [1]
    c1 = cluster(idx, 2)
    @test members(idx, c1) == [1, 2] && representative(idx, c1) == 1 && HNNG.radius(idx, c1) == 1.0
    r = root(idx)
    @test r.level == 1 && members(idx, r) == [1, 3]
    @test HNNG.radius(idx, r) == 11.0
    @test parent(idx, cluster(idx, 4)) == r && parent(idx, r) === nothing
    @test children(idx, r) == [cluster(idx, 1), cluster(idx, 3)]
    @test children(idx, c1) == HNNG.ClusterRef[]
    @test path_to_root(idx, 4) == [cluster(idx, 4), r]
    @test outlink(idx, 1; level=1) == 3 && outlink_distance(idx, 1; level=1) == 10.0
    @test outlink(idx, 2; level=1) == 0 && cluster(idx, 2; level=1) === nothing
    @test inlinks(idx, 2; level=1) == Int[] && outlink_distance(idx, 2; level=1) == Inf
    @test idx.nevals == n0                                  # no distance evaluated
    @test root(HNNGIndex(Euclidean(), 2)) === nothing
end

@testset "navigation reports the metric's units" begin
    s = HNNGIndex(SqEuclidean(), 1; T=Float64)
    stream!(s, reshape([0.0, 1.0, 10.0, 11.0], 1, :))
    @test outlink_distance(s, 1; level=1) == 100.0
    @test HNNG.radius(s, root(s)) == 121.0
end

@testset "outlink chains end at a mutual pair; paths end at the root" begin
    rng = StableRNG(14)
    idx = HNNGIndex(Euclidean(), 2; T=Float64)
    stream!(idx, blobs(rng, Float64, 2, 200))
    for s in 1:200
        x, seen = s, Set{Int}()
        while !(x in seen)
            push!(seen, x)
            x = outlink(idx, x)
        end
        @test outlink(idx, outlink(idx, x)) == x
        p = path_to_root(idx, s)
        @test p[end] == root(idx) && length(p) == nlevels(idx)
        @test all(i -> representative(idx, p[i]) in members(idx, p[i+1]), 1:length(p)-1)
    end
end
```

`test/runtests.jl`: add `include("navigate.jl")` after `include("order.jl")`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. -e 'using Test, HNNG; include("test/navigate.jl")'`
Expected: FAIL with `UndefVarError: \`outlink\` not defined`.

- [ ] **Step 3: Write the implementation**

Append to `src/navigate.jl`:
```julia
"""
    ClusterRef(level, index)

Handle of a cluster: its level and slot. Valid until the next `add!`/`delete!`.
"""
struct ClusterRef
    level::Int
    index::Int
end

function _check_level(idx::HNNGIndex, level::Integer)
    0 <= level < nlevels(idx) ||
        throw(ArgumentError("level $level is out of range 0:$(nlevels(idx) - 1)"))
    Int(level)
end

function _cl(idx::HNNGIndex, c::ClusterRef)
    _check_level(idx, c.level)
    cls = idx.levels[c.level+1].clusters
    (1 <= c.index <= length(cls) && cls[c.index].rep != 0) ||
        throw(ArgumentError("stale or invalid cluster handle $c"))
    cls[c.index]
end

"""Outlink of `id` at `level` (its nearest neighbour there), `0` if none."""
outlink(idx::HNNGIndex, id::Integer; level::Integer=0) =
    idx.levels[_check_level(idx, level)+1].outlink[_check_live(idx, id)]

"""Stored outlink distance of `id` at `level`, in the metric's units; `Inf` if none."""
outlink_distance(idx::HNNGIndex, id::Integer; level::Integer=0) =
    _to_user(idx, idx.levels[_check_level(idx, level)+1].outdist[_check_live(idx, id)])

"""Ids whose outlink at `level` is `id` (a sorted copy)."""
inlinks(idx::HNNGIndex, id::Integer; level::Integer=0) =
    copy(idx.levels[_check_level(idx, level)+1].inlinks[_check_live(idx, id)])

"""Handle of the cluster holding `id` at `level`; `nothing` if `id` does not reach that level."""
function cluster(idx::HNNGIndex, id::Integer; level::Integer=0)
    i = _check_live(idx, id)
    l = _check_level(idx, level)
    ci = idx.levels[l+1].cluster_of[i]
    ci == 0 ? nothing : ClusterRef(l, ci)
end

"""Member ids of a cluster (sorted copy)."""
members(idx::HNNGIndex, c::ClusterRef) = copy(_cl(idx, c).members)

"""Representative id of a cluster."""
representative(idx::HNNGIndex, c::ClusterRef) = _cl(idx, c).rep

"""Radius of a cluster in the metric's units: no level-0 element under it is farther from its representative."""
radius(idx::HNNGIndex, c::ClusterRef) = _to_user(idx, _cl(idx, c).radius)

"""The cluster one level up that contains the representative of `c`; `nothing` at the top."""
function Base.parent(idx::HNNGIndex, c::ClusterRef)
    r = _cl(idx, c).rep
    c.level + 1 < nlevels(idx) || return nothing
    ClusterRef(c.level + 1, idx.levels[c.level+2].cluster_of[r])
end

"""Child clusters: the level-`(l-1)` cluster of every member, in member order. Empty at level 0."""
function children(idx::HNNGIndex, c::ClusterRef)
    cl = _cl(idx, c)
    c.level == 0 && return ClusterRef[]
    below = idx.levels[c.level]
    [ClusterRef(c.level - 1, below.cluster_of[m]) for m in cl.members]
end

"""Cluster handles from `id`'s level-0 cluster up to the root."""
function path_to_root(idx::HNNGIndex, id::Integer)
    i = _check_live(idx, id)
    p = [ClusterRef(0, idx.levels[1].cluster_of[i])]
    while (q = parent(idx, p[end])) !== nothing
        push!(p, q)
    end
    p
end

"""Handle of the single top-level cluster; `nothing` for an empty index."""
function root(idx::HNNGIndex)
    isempty(idx.levels) && return nothing
    ClusterRef(nlevels(idx) - 1, findfirst(c -> c.rep != 0, idx.levels[end].clusters))
end
```

In `src/HNNG.jl` change the export lines to:
```julia
export is_metric, nlevels, element, verify, add!, knn, knn_batch
export outlink, outlink_distance, inlinks, cluster, members, representative, radius,
       children, path_to_root, root
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Test, HNNG; include("test/navigate.jl")'`
Expected: all three testsets pass.

- [ ] **Step 5: Commit**

```bash
git add src/HNNG.jl src/navigate.jl test/navigate.jl test/runtests.jl
git commit -m "feat: navigation along stored links and cluster handles" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 15: `leaves` and `newick`

**Files:**
- Modify: `src/navigate.jl` (append), `src/HNNG.jl`, `test/navigate.jl` (append)

**Interfaces:**
- Consumes: Task 14 (`ClusterRef`, `_cl`, `children`, `root`), `_to_user`.
- Produces: exported `leaves(idx, c) -> Vector{Int}` (sorted level-0 ids), `newick(idx; labels=nothing) -> String`; `_newick!(io, idx, c, labels)`, `_newick_name(s::AbstractString) -> String`.

- [ ] **Step 1: Write the failing tests**

Append to `test/navigate.jl`:
```julia
@testset "leaves and newick" begin
    idx = line4()
    @test leaves(idx, root(idx)) == [1, 2, 3, 4]
    @test leaves(idx, cluster(idx, 4)) == [3, 4]
    @test newick(idx) == "((1:0.0,2:1.0):1.0,(3:0.0,4:1.0):1.0):11.0;"
    @test newick(idx; labels=["a", "b", "c d", "e'f"]) ==
          "((a:0.0,b:1.0):1.0,('c d':0.0,'e''f':1.0):1.0):11.0;"
    @test newick(HNNGIndex(Euclidean(), 1)) == ";"
    rng = StableRNG(15)
    big = HNNGIndex(Euclidean(), 3)
    stream!(big, blobs(rng, Float32, 3, 120))
    @test leaves(big, root(big)) == 1:120
    @test count(==(','), newick(big)) == 119
    n0 = big.nevals
    newick(big)
    leaves(big, root(big))
    @test big.nevals == n0
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. -e 'using Test, HNNG; include("test/navigate.jl")'`
Expected: FAIL with `UndefVarError: \`leaves\` not defined`.

- [ ] **Step 3: Write the implementation**

Append to `src/navigate.jl`:
```julia
"""Level-0 ids under `c`, sorted (the ROS `getLeafElementswithCluster`)."""
function leaves(idx::HNNGIndex, c::ClusterRef)
    cl = _cl(idx, c)
    c.level == 0 && return copy(cl.members)
    out = Int[]
    for ch in children(idx, c)
        append!(out, leaves(idx, ch))
    end
    sort!(out)
end

function _newick_name(s::AbstractString)
    occursin(r"[\s(),:;'\[\]]", s) ? "'" * replace(s, "'" => "''") * "'" : String(s)
end

function _newick!(io::IO, idx::HNNGIndex, c::ClusterRef, labels)
    cl = _cl(idx, c)
    print(io, "(")
    if c.level == 0
        for (i, m) in enumerate(cl.members)
            i > 1 && print(io, ",")
            name = labels === nothing ? string(m) : string(labels[m])
            print(io, _newick_name(name), ":", _to_user(idx, cl.repdist[i]))
        end
    else
        for (i, ch) in enumerate(children(idx, c))
            i > 1 && print(io, ",")
            _newick!(io, idx, ch, labels)
        end
    end
    print(io, "):", _to_user(idx, cl.radius))
end

"""
    newick(idx; labels=nothing) -> String

The hierarchy as a Newick tree (the ROS eTree export): one node per cluster with its
radius as branch length, one leaf per level-0 element named `labels[id]` (or the id)
with its distance to its cluster's representative as branch length, all in the
metric's units. `";"` for an empty index.
"""
function newick(idx::HNNGIndex; labels=nothing)
    r = root(idx)
    r === nothing && return ";"
    io = IOBuffer()
    _newick!(io, idx, r, labels)
    print(io, ";")
    String(take!(io))
end
```

In `src/HNNG.jl` change the second export line to:
```julia
export outlink, outlink_distance, inlinks, cluster, members, representative, radius,
       children, path_to_root, leaves, root, newick
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Test, HNNG; include("test/navigate.jl")'`
Expected: all four testsets pass.

- [ ] **Step 5: Commit**

```bash
git add src/HNNG.jl src/navigate.jl test/navigate.jl
git commit -m "feat: leaves and Newick export of the hierarchy" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 16: Graphs.jl extension — `level_graph`

**Files:**
- Modify: `ext/HNNGGraphsExt.jl`, `src/navigate.jl` (append the generic function), `src/HNNG.jl`, `test/navigate.jl` (append)

**Interfaces:**
- Consumes: `_check_level`, `_to_user`, `outlink`, `outlink_distance`.
- Produces: exported generic `level_graph`; method `HNNG.level_graph(idx, l) -> (g::SimpleDiGraph, weights::SparseMatrixCSC{Float64,Int}, ids::Vector{Int})` in the extension; vertex `i` is `ids[i]`; one edge per outlink, weighted by `outlink_distance`.

- [ ] **Step 1: Write the failing test**

Append to `test/navigate.jl`:
```julia
using Graphs, SparseArrays

@testset "Graphs extension: level_graph and Dijkstra over stored distances" begin
    rng = StableRNG(16)
    idx = HNNGIndex(Euclidean(), 2; T=Float64)
    stream!(idx, blobs(rng, Float64, 2, 80))
    for l in 0:nlevels(idx)-1
        g, W, ids = level_graph(idx, l)
        @test nv(g) == length(ids) == idx.levels[l+1].npop
        @test ne(g) == length(ids)
        @test W isa SparseMatrixCSC
        for (i, x) in enumerate(ids)
            j = findfirst(==(outlink(idx, x; level=l)), ids)
            @test has_edge(g, i, j) && W[i, j] == outlink_distance(idx, x; level=l)
        end
    end
    g, W, ids = level_graph(idx, 0)
    @test ids == 1:80
    n0 = idx.nevals
    st = dijkstra_shortest_paths(g, 1, W)
    x, acc, seen = 1, 0.0, Set([1])
    while true
        y = outlink(idx, x)
        acc += outlink_distance(idx, x)
        y in seen && break
        @test st.dists[y] ≈ acc
        push!(seen, y)
        x = y
    end
    @test idx.nevals == n0
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: the new testset errors with `UndefVarError: \`level_graph\` not defined`.

- [ ] **Step 3: Write the implementation**

Append to `src/navigate.jl`:
```julia
"""
    level_graph(idx, l) -> (g::SimpleDiGraph, weights::SparseMatrixCSC, ids::Vector{Int})

Requires `using Graphs`. One vertex per level-`l` element (vertex `i` is `ids[i]`),
one edge per outlink, weighted by the stored outlink distance (metric units), so
`Graphs.dijkstra_shortest_paths(g, s, weights)` runs without a distance evaluation.
"""
function level_graph end
```

`ext/HNNGGraphsExt.jl`:
```julia
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
```

In `src/HNNG.jl` change the second export line to:
```julia
export outlink, outlink_distance, inlinks, cluster, members, representative, radius,
       children, path_to_root, leaves, root, newick, level_graph
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: `Testing HNNG tests passed`. If loading the extension fails with `Package SparseArrays not found` (an older Pkg that does not let extensions load non-trigger weak deps), change the `[extensions]` entry to `HNNGGraphsExt = ["Graphs", "SparseArrays"]` (Graphs loads SparseArrays, so the extension still triggers on `using Graphs`) and rerun; record the change in the commit message.

- [ ] **Step 5: Commit**

```bash
git add src/HNNG.jl src/navigate.jl ext/HNNGGraphsExt.jl test/navigate.jl Project.toml
git commit -m "feat: Graphs.jl extension exports each level as a weighted digraph" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 17: `save` and `load`

**Files:**
- Create: `src/io.jl`, `test/io.jl`
- Modify: `src/HNNG.jl`, `test/runtests.jl`

**Interfaces:**
- Consumes: `HNNGIndex` fields, `_wrap_metric`, `_is_metric_for`, `DistMemo`.
- Produces: `const FORMAT = "HNNG.jl index"`, `const FORMAT_VERSION = 1`; unexported `HNNG.save(path, idx) -> path`, `HNNG.load(path; metric=nothing) -> HNNGIndex` (`nevals`, `last_rewired`, `last_top` start at 0).

- [ ] **Step 1: Write the failing tests**

`test/io.jl`:
```julia
isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))

@testset "save/load round trip: VectorStore" begin
    rng = StableRNG(17)
    for M in (Euclidean(), SqEuclidean(), CosineDist())
        idx = HNNGIndex(M, 4)
        add!(idx, blobs(rng, Float32, 4, 300))
        @test verify(idx)
        for id in (5, 77, 123)
            delete!(idx, id)
            @test verify(idx)
        end
        path = joinpath(mktempdir(), "idx.hnng")
        @test HNNG.save(path, idx) == path
        idx2 = HNNG.load(path)
        @test verify(idx2)
        @test idx2.metric == idx.metric && length(idx2) == 297 && idx2.nevals == 0
        @test same_structure(structure(idx2), structure(idx))
        for _ in 1:20
            q = randn(rng, Float32, 4)
            @test knn(idx2, q, 7) == knn(idx, q, 7)
        end
        @test add!(idx2, randn(rng, Float32, 4)) == 301
        @test verify(idx2)
    end
end

@testset "save/load round trip: ObjectStore needs metric=" begin
    rng = StableRNG(171)
    S = descriptor_sets(rng, 3, 60)
    idx = HNNGIndex(UserMetric(hausdorff; is_metric=true), Matrix{Float64})
    add!(idx, S)
    @test verify(idx)
    path = joinpath(mktempdir(), "obj.hnng")
    HNNG.save(path, idx)
    @test_throws ArgumentError HNNG.load(path)
    idx2 = HNNG.load(path; metric=UserMetric(hausdorff; is_metric=true))
    @test verify(idx2) && is_metric(idx2)
    for q in descriptor_sets(rng, 3, 5)
        @test knn(idx2, q, 4) == knn(idx, q, 4)
    end
end

@testset "load rejects other files" begin
    path = joinpath(mktempdir(), "other.jls")
    open(io -> HNNG.serialize(io, (a = 1,)), path, "w")
    @test_throws "not an HNNG.jl index file" HNNG.load(path)
end
```

`test/runtests.jl`: add `include("io.jl")` after `include("navigate.jl")`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. -e 'using Test, HNNG; include("test/io.jl")'`
Expected: FAIL with `UndefVarError: \`save\` not defined in \`HNNG\``.

- [ ] **Step 3: Write the implementation**

`src/io.jl`:
```julia
# Persistence and statistics (spec §4, §4.3).

const FORMAT = "HNNG.jl index"
const FORMAT_VERSION = 1

"""
    HNNG.save(path, idx) -> path

Write the index with `Serialization`: store, tombstones, levels and, for a
VectorStore with a Distances.jl metric, the metric. An ObjectStore or a `UserMetric`
index is saved without its metric; `load` then needs `metric=`.
"""
function save(path::AbstractString, idx::HNNGIndex)
    keep = idx.store isa VectorStore && !(idx.metric isa UserMetric)
    payload = (format=FORMAT, version=FORMAT_VERSION, store=idx.store,
               metric=keep ? idx.metric : nothing, alive=idx.alive, levels=idx.levels)
    open(io -> serialize(io, payload), path, "w")
    path
end

"""
    HNNG.load(path; metric=nothing) -> HNNGIndex

Read an index written by `HNNG.save`. `metric=` is required for an ObjectStore or a
`UserMetric` index and overrides the saved metric otherwise. `nevals` starts at 0.
"""
function load(path::AbstractString; metric=nothing)
    p = open(deserialize, path)
    (p isa NamedTuple && get(p, :format, nothing) == FORMAT) ||
        throw(ArgumentError("$path is not an HNNG.jl index file"))
    m = metric === nothing ? p.metric : _wrap_metric(metric)
    m === nothing && throw(ArgumentError(
        "the index in $path has an ObjectStore or a UserMetric; pass metric= to load it"))
    HNNGIndex(p.store, m, _is_metric_for(p.store, m), p.alive, p.levels, 0, DistMemo(), 0, 0)
end
```

In `src/HNNG.jl` add `using Serialization: serialize, deserialize` to the `using` lines and `include("io.jl")` after `include("verify.jl")`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Test, HNNG; include("test/io.jl")'`
Expected: all three testsets pass.

- [ ] **Step 5: Commit**

```bash
git add src/HNNG.jl src/io.jl test/io.jl test/runtests.jl
git commit -m "feat: save and load with Serialization" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 18: `stats`, `reset_evals!`, `HNNG.last_update_stats`

`HNNG.last_update_stats` was requested by the benchmark plan (Exp.3 per-operation rewire counts and highest level touched). Its two fields have been maintained since Tasks 5, 6 and 10 (`link!`/`unlink!` at level 0 count rewired elements; `propagate!` and `delete!` raise `last_top`); this task exposes and tests them.

**Files:**
- Modify: `src/io.jl` (append), `src/HNNG.jl`, `test/updates.jl` (append)

**Interfaces:**
- Consumes: `Level.npop`, `Level.nclusters`, `idx.nevals`, `idx.last_rewired`, `idx.last_top`.
- Produces:
  - exported `stats(idx) -> (n_live, n_total, nlevels, population::Vector{Int}, nclusters::Vector{Int}, mean_cluster_size::Vector{Float64}, nevals::Int)`
  - exported `reset_evals!(idx) -> idx`
  - unexported `HNNG.last_update_stats(idx) -> (rewired::Int, top_level_touched::Int)`: `rewired` = live level-0 elements whose outlink changed in the most recent `add!` or `delete!` (not the inserted element's own outlink, not the deleted element); `top_level_touched` = highest 0-based level whose outlinks, clusters or representatives changed (levels created count; for an operation that drops levels or empties the index, the highest level that existed before); `(0, 0)` before any update. `bulk_add!` leaves it unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `test/updates.jl`:
```julia
@testset "length and stats" begin
    rng = StableRNG(18)
    idx = HNNGIndex(Euclidean(), 2; T=Float64)
    @test stats(idx) == (n_live=0, n_total=0, nlevels=0, population=Int[], nclusters=Int[],
                         mean_cluster_size=Float64[], nevals=0)
    stream!(idx, blobs(rng, Float64, 2, 200))
    for id in 1:10
        delete!(idx, id); verify(idx)
    end
    s = stats(idx)
    @test s.n_live == 190 == length(idx) && s.n_total == 200 && s.nlevels == nlevels(idx)
    @test s.population[1] == 190
    @test all(s.population[l+1] == s.nclusters[l] for l in 1:s.nlevels-1)
    @test s.nclusters[end] == 1
    @test s.mean_cluster_size ≈ s.population ./ s.nclusters
    @test s.nevals == idx.nevals > 0
    @test reset_evals!(idx) === idx && stats(idx).nevals == 0
    knn(idx, [0.0, 0.0], 3)
    @test stats(idx).nevals > 0
end

@testset "HNNG.last_update_stats" begin
    snapshot(idx) = [(copy(lv.outlink), sort!([(c.rep, copy(c.members)) for c in lv.clusters if c.rep != 0]))
                     for lv in idx.levels]
    function differs(a, b)
        (oa, ca), (ob, cb) = a, b
        m = min(length(oa), length(ob))
        oa[1:m] != ob[1:m] || any(!=(0), oa[m+1:end]) || any(!=(0), ob[m+1:end]) || ca != cb
    end
    function touched(a, b)
        top = 0
        for l in 0:max(length(a), length(b))-1
            (l >= length(a) || l >= length(b) || differs(a[l+1], b[l+1])) && (top = l)
        end
        top
    end
    rng = StableRNG(183)
    idx = HNNGIndex(Euclidean(), 2; T=Float64)
    @test HNNG.last_update_stats(idx) == (rewired=0, top_level_touched=0)
    X = blobs(rng, Float64, 2, 120)
    for j in 1:120
        before = snapshot(idx)
        add!(idx, X[:, j])
        @test verify(idx)
        after = snapshot(idx)
        st = HNNG.last_update_stats(idx)
        @test st.rewired == (j == 1 ? 0 : count(i -> before[1][1][i] != after[1][1][i], 1:j-1))
        @test st.top_level_touched == touched(before, after)
    end
    for id in randperm(rng, 120)
        before = snapshot(idx)
        delete!(idx, id)
        @test verify(idx)
        after = snapshot(idx)
        st = HNNG.last_update_stats(idx)
        @test st.rewired == (isempty(after) ? count(i -> i != id && before[1][1][i] != 0, 1:120) :
                             count(i -> i != id && before[1][1][i] != after[1][1][i], 1:120))
        @test st.top_level_touched == touched(before, after)
    end
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. -e 'using Test, HNNG; include("test/updates.jl")'`
Expected: the two new testsets error with `UndefVarError: \`stats\` not defined` and `UndefVarError: \`last_update_stats\` not defined in \`HNNG\``.

- [ ] **Step 3: Write the implementation**

Append to `src/io.jl`:
```julia
"""
    stats(idx) -> NamedTuple

`(n_live, n_total, nlevels, population, nclusters, mean_cluster_size, nevals)`:
per-level population and cluster counts (index `l+1` is level `l`), and the distance
evaluations made by `add!`, `delete!`, `bulk_add!` and the serial `knn` path since
the last `reset_evals!`.
"""
function stats(idx::HNNGIndex)
    pop = [lv.npop for lv in idx.levels]
    ncl = [lv.nclusters for lv in idx.levels]
    (n_live=length(idx), n_total=_ntotal(idx), nlevels=nlevels(idx), population=pop,
     nclusters=ncl, mean_cluster_size=[c == 0 ? 0.0 : p / c for (p, c) in zip(pop, ncl)],
     nevals=idx.nevals)
end

"""Reset the distance-evaluation counter to 0."""
reset_evals!(idx::HNNGIndex) = (idx.nevals = 0; idx)

"""
    HNNG.last_update_stats(idx) -> (rewired, top_level_touched)

For the most recent `add!` or `delete!`: the number of live level-0 elements whose
outlink changed (excluding the inserted element's own outlink and the deleted
element), and the highest level whose outlinks, clusters or representatives changed
(for an update that dropped levels, the highest level that existed before).
`(0, 0)` before any update.
"""
last_update_stats(idx::HNNGIndex) = (rewired=idx.last_rewired, top_level_touched=idx.last_top)
```

In `src/HNNG.jl` change the first export line to:
```julia
export is_metric, nlevels, element, verify, add!, knn, knn_batch, stats, reset_evals!
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Test, HNNG; include("test/updates.jl")'`
Expected: all eight testsets pass. If `top_level_touched` disagrees with the diff, the bug is in the `last_top` bookkeeping of `propagate!` (Task 6) or `delete!` (Task 10): every level `propagate!` processes must raise it, and so must every drop.

- [ ] **Step 5: Commit**

```bash
git add src/HNNG.jl src/io.jl test/updates.jl
git commit -m "feat: stats, reset_evals! and last_update_stats for the update benchmarks" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 19: `bulk_add!` — direct all-pairs path

Correctness first: every level's 1-NN graph comes from an exact all-pairs scan with the package kernel; Task 20 swaps in BLAS tiles for the Euclidean family.

**Files:**
- Create: `src/bulk.jl`
- Modify: `src/HNNG.jl`, `test/order.jl` (append)

**Interfaces:**
- Consumes: `_prepare`, `_append!`, `Level`, `Cluster`, `_elect`, `_component`, `_new_slot!`, `_bounds!`, `_d`, `_better`, `_rawdist`, `_flush!`.
- Produces:
  - `_allpairs_nn_direct(idx, pop::Vector{Int}) -> (nn::Vector{Int}, nnd::Vector{Float64})` (ids and internal-unit distances, canonical ties; adds `m*(m-1)` to `idx.nevals`)
  - `_allpairs_nn(idx, pop)` (dispatch point; generic method = direct)
  - `_components(lv::Level, pop::Vector{Int}) -> Vector{Vector{Int}}` (sorted members, ordered by smallest id)
  - exported `bulk_add!(idx, X::AbstractMatrix) -> UnitRange{Int}`; throws `ArgumentError` on an ObjectStore or when any id was ever assigned (Spec assumption 8)

- [ ] **Step 1: Write the failing tests**

Append to `test/order.jl`:
```julia
@testset "bulk_add! equals streaming add! in the same column order" begin
    rng = StableRNG(19)
    for (M, T, d, n, gen) in ((Euclidean(), Float64, 2, 300, blobs), (SqEuclidean(), Float32, 6, 250, gauss),
                              (CosineDist(), Float32, 5, 200, gauss), (Cityblock(), Float64, 3, 150, blobs))
        X = gen(rng, T, d, n)
        S = HNNGIndex(M, d; T=T)
        add!(S, X)
        @test verify(S)
        B = HNNGIndex(M, d; T=T)
        @test bulk_add!(B, X) == 1:n
        @test verify(B)
        @test same_structure(structure(B), structure(S))
        q = randn(rng, T, d)
        @test knn(B, q, 5) == knn(S, q, 5)
        @test B.nevals > 0
    end
    base = gauss(StableRNG(191), Float64, 3, 40)
    S = HNNGIndex(Euclidean(), 3; T=Float64)
    add!(S, hcat(base, base))                       # exact duplicates
    @test verify(S)
    B = HNNGIndex(Euclidean(), 3; T=Float64)
    bulk_add!(B, hcat(base, base))
    @test verify(B)
    @test same_structure(structure(B), structure(S))
    for p in (randperm(StableRNG(193), 80), randperm(StableRNG(194), 80))
        Xp = hcat(base, base)[:, p]
        S2 = HNNGIndex(Euclidean(), 3; T=Float64); add!(S2, Xp); @test verify(S2)
        B2 = HNNGIndex(Euclidean(), 3; T=Float64); bulk_add!(B2, Xp); @test verify(B2)
        @test same_structure(structure(B2), structure(S2))
    end
    @test bulk_add!(HNNGIndex(Euclidean(), 3), zeros(Float32, 3, 0)) == 1:0
    one = HNNGIndex(Euclidean(), 3)
    @test bulk_add!(one, ones(Float32, 3, 1)) == 1:1 && verify(one) && nlevels(one) == 1
end

@testset "bulk_add! then streaming updates" begin
    rng = StableRNG(192)
    B = HNNGIndex(Euclidean(), 4)
    bulk_add!(B, gauss(rng, Float32, 4, 400))
    @test verify(B)
    for _ in 1:30
        add!(B, randn(rng, Float32, 4))
        @test verify(B)
    end
    for id in 1:15
        delete!(B, id)
        @test verify(B)
    end
    @test check_knn(B, randn(rng, Float32, 4), 10)
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. -e 'using Test, HNNG; include("test/order.jl")'`
Expected: the two new testsets error with `UndefVarError: \`bulk_add!\` not defined`.

- [ ] **Step 3: Write the implementation**

`src/bulk.jl`:
```julia
# Bulk loader (spec §6.6): VectorStore only, empty index only.

"""Exact all-pairs 1-NN inside `pop` with the package kernel, canonical `(distance, id)` ties."""
function _allpairs_nn_direct(idx::HNNGIndex, pop::Vector{Int})
    m = length(pop)
    nn = zeros(Int, m)
    nnd = fill(Inf, m)
    for i in 1:m
        p = _point(idx.store, pop[i])
        for j in 1:m
            j == i && continue
            d = _rawdist(idx, p, _point(idx.store, pop[j]))
            if _better(d, pop[j], nnd[i], nn[i])
                nn[i], nnd[i] = pop[j], d
            end
        end
    end
    idx.nevals += m * (m - 1)
    nn, nnd
end

_allpairs_nn(idx::HNNGIndex, pop::Vector{Int}) = _allpairs_nn_direct(idx, pop)

"""Connected components of the level graph restricted to `pop` (sorted), ordered by smallest id."""
function _components(lv::Level, pop::Vector{Int})
    seen = falses(length(lv.present))
    comps = Vector{Vector{Int}}()
    for s in pop
        seen[s] && continue
        comp = _component(lv, s)
        seen[comp] .= true
        push!(comps, comp)
    end
    comps
end

"""
    bulk_add!(idx, X::AbstractMatrix) -> UnitRange{Int}

Build the whole hierarchy of the columns of `X` at once (VectorStore only, and only
on an index that never held an element). Per level: exact all-pairs 1-NN with
canonical ties, connected components, representative election, bounds; the
representatives form the next level until one cluster remains. The result is the
structure that streaming `add!(idx, X)` produces.
"""
bulk_add!(::HNNGIndex{<:ObjectStore}, X) =
    throw(ArgumentError("bulk_add! needs a VectorStore index; this index stores objects"))

function bulk_add!(idx::HNNGIndex{<:VectorStore}, X::AbstractMatrix{<:Real})
    _ntotal(idx) == 0 ||
        throw(ArgumentError("bulk_add! needs an empty index; this index has already assigned $(_ntotal(idx)) ids"))
    d = dim(idx.store)
    size(X, 1) == d ||
        throw(ArgumentError("dimension mismatch: the index has dim $d, the matrix has $(size(X, 1)) rows"))
    cols = [_prepare(idx, view(X, :, j)) for j in axes(X, 2)]
    isempty(cols) && return 1:0
    for v in cols
        _append!(idx.store, v)
        push!(idx.alive, true)
    end
    n = length(cols)
    pop = collect(1:n)
    l = 0
    while true
        lv = Level(n)
        push!(idx.levels, lv)
        lv.present[pop] .= true
        lv.npop = length(pop)
        if length(pop) >= 2
            nn, nnd = _allpairs_nn(idx, pop)
            for (i, x) in enumerate(pop)
                lv.outlink[x] = nn[i]
                lv.outdist[x] = nnd[i]
            end
            for x in pop                      # ascending, so every inlink list is sorted
                push!(lv.inlinks[lv.outlink[x]], x)
            end
        end
        for comp in _components(lv, pop)
            r = _elect(lv, comp)
            c = Cluster(r, comp, [m == r ? 0.0 : _d(idx, r, m) for m in comp], 0.0, Float64[])
            ci = _new_slot!(lv, c)
            lv.cluster_of[comp] .= ci
            _bounds!(idx, l, c)
        end
        lv.nclusters == 1 && break
        pop = sort!([c.rep for c in lv.clusters])
        l += 1
    end
    _flush!(idx)
    1:n
end
```

In `src/HNNG.jl` add `include("bulk.jl")` after `include("search.jl")` and change the first export line to:
```julia
export is_metric, nlevels, element, verify, add!, knn, knn_batch, stats, reset_evals!, bulk_add!
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Test, HNNG; include("test/order.jl")'`
Expected: all five testsets pass.

- [ ] **Step 5: Commit**

```bash
git add src/HNNG.jl src/bulk.jl test/order.jl
git commit -m "feat: bulk_add! builds the canonical hierarchy level by level" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 20: `bulk_add!` — BLAS Gram tiles for the Euclidean family

**Files:**
- Modify: `src/kernels.jl` (append), `src/bulk.jl` (append one method), `src/HNNG.jl`, `test/order.jl` (append)

**Interfaces:**
- Consumes: Task 19 `_allpairs_nn_direct`, `_euclid`, `_better`, `dim`, `_point`.
- Produces: `const BULK_TILE = 512`, `const RERANK = 1e-5`; `_allpairs_nn_blas(idx::HNNGIndex{VectorStore{T}}, pop::Vector{Int}) -> (nn, nnd)` (same contract as `_allpairs_nn_direct`; adds `m*(m-1)` Gram pairs plus every direct re-rank to `idx.nevals`); `_allpairs_nn(idx::HNNGIndex{<:VectorStore,<:EuclidFamily}, pop) = _allpairs_nn_blas(idx, pop)`.

- [ ] **Step 1: Write the failing test**

Append to `test/order.jl`:
```julia
@testset "BLAS all-pairs equals the direct all-pairs kernel" begin
    rng = StableRNG(20)
    for (T, d, n) in ((Float32, 2, 700), (Float32, 20, 1100), (Float64, 7, 600), (Float32, 100, 300))
        idx = HNNGIndex(Euclidean(), d; T=T)
        X = gauss(rng, T, d, n)
        X[:, 2:2:40] .= X[:, 1:2:39]                # exact duplicates
        X .+= T(1000)                               # norms far above the distances
        for j in axes(X, 2)
            HNNG._append!(idx.store, HNNG._prepare(idx, view(X, :, j)))
            push!(idx.alive, true)
        end
        pop = collect(1:n)
        @test HNNG._allpairs_nn_blas(idx, pop) == HNNG._allpairs_nn_direct(idx, pop)
        sub = sort!(randperm(rng, n)[1:150])
        @test HNNG._allpairs_nn_blas(idx, sub) == HNNG._allpairs_nn_direct(idx, sub)
    end
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. -e 'using Test, HNNG; include("test/order.jl")'`
Expected: the new testset errors with `UndefVarError: \`_allpairs_nn_blas\` not defined in \`HNNG\``.

- [ ] **Step 3: Write the implementation**

Append to `src/kernels.jl`:
```julia
const BULK_TILE = 512
const RERANK = 1e-5

"""
    _allpairs_nn_blas(idx, pop) -> (nn, nnd)

Exact 1-NN inside `pop` for the Euclidean family. Squared distances come from BLAS
Gram tiles (`G = PᵢᵀPⱼ` in `T`, norms accumulated in `Float64`); every candidate
within `RERANK` relative of its row minimum, widened by a bound on the Gram rounding
(`2·d·eps(T)·(‖xᵢ‖² + max‖x‖²)`), is re-ranked with `_euclid` under the canonical
`(distance, id)` order, so the result equals `_allpairs_nn_direct`.
"""
function _allpairs_nn_blas(idx::HNNGIndex{VectorStore{T}}, pop::Vector{Int}) where {T}
    m = length(pop)
    d = dim(idx.store)
    P = Matrix{T}(undef, d, m)
    for (i, x) in enumerate(pop)
        copyto!(view(P, :, i), _point(idx.store, x))
    end
    nrm = zeros(Float64, m)
    for i in 1:m, k in 1:d
        nrm[i] += Float64(P[k, i])^2
    end
    slack = 2 * d * eps(T)
    nmax = maximum(nrm)
    nn = zeros(Int, m)
    nnd = fill(Inf, m)
    G = Matrix{T}(undef, BULK_TILE, BULK_TILE)
    amin = fill(Inf, BULK_TILE)
    cand = [Tuple{Float64,Int}[] for _ in 1:BULK_TILE]
    nexact = 0
    limit(ii, i) = amin[ii] + RERANK * abs(amin[ii]) + slack * (nrm[i] + nmax)
    for I in Iterators.partition(1:m, BULK_TILE)
        fill!(amin, Inf)
        foreach(empty!, cand)
        for J in Iterators.partition(1:m, BULK_TILE)
            Gv = view(G, 1:length(I), 1:length(J))
            mul!(Gv, transpose(view(P, :, I)), view(P, :, J))
            for (ii, i) in enumerate(I)
                for (jj, j) in enumerate(J)
                    j == i && continue
                    sq = nrm[i] + nrm[j] - 2 * Float64(Gv[ii, jj])
                    sq < amin[ii] && (amin[ii] = sq)
                    sq <= limit(ii, i) && push!(cand[ii], (sq, j))
                end
                if length(cand[ii]) > 64
                    lim = limit(ii, i)
                    filter!(p -> p[1] <= lim, cand[ii])
                end
            end
        end
        for (ii, i) in enumerate(I)
            lim = limit(ii, i)
            pi_ = view(P, :, i)
            for (sq, j) in cand[ii]
                sq <= lim || continue
                dj = _euclid(pi_, view(P, :, j))
                nexact += 1
                if _better(dj, pop[j], nnd[i], nn[i])
                    nn[i], nnd[i] = pop[j], dj
                end
            end
        end
    end
    idx.nevals += m * (m - 1) + nexact
    nn, nnd
end
```

Append to `src/bulk.jl`:
```julia
_allpairs_nn(idx::HNNGIndex{<:VectorStore,<:EuclidFamily}, pop::Vector{Int}) = _allpairs_nn_blas(idx, pop)
```

In `src/HNNG.jl` add `using LinearAlgebra: mul!` to the `using` lines.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Test, HNNG; include("test/order.jl")'`
Expected: all six testsets pass (the Task 19 equality tests now run on the BLAS path for the Euclidean family and on the direct path for `Cityblock`).

- [ ] **Step 5: Commit**

```bash
git add src/HNNG.jl src/kernels.jl src/bulk.jl test/order.jl
git commit -m "perf: BLAS Gram tiles with exact re-rank for the bulk loader" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 21: C++ parity fixture (generated once, committed)

**Files:**
- Create: `test/fixtures/make_fixtures.py`, `test/fixtures/cpp_parity_random_xs_20.json` (generated)

**Interfaces:**
- Consumes: the compiled C++ bindings `/home/coral/working/hnng/python_bindings/hnnglib.cpython-312-x86_64-linux-gnu.so` (read-only), `/home/coral/working/benchmarks/data/random-xs-20-euclidean.hdf5`.
- Produces: JSON with keys `dataset`, `n_train` (2000), `n_test` (100), `dim` (20), `k` (10), `train_f32_b64` and `test_f32_b64` (little-endian float32, point-major, base64), `cpp_ids` (100 × 10, 0-based labels), `cpp_sqdists` (100 × 10, C++ squared L2), `bf_dists` (100 × 11, float64 brute-force Euclidean distances on the float32 points, ascending), `cpp_nlevels`, `cpp_nclusters` (per level, from the C++ save file; Spec assumption 13).

- [ ] **Step 1: Check the Python prerequisites**

```bash
python3 -c "import h5py, numpy; print(h5py.__version__, numpy.__version__)" || pip install --user h5py
python3 -c "import sys; sys.path.insert(0, '/home/coral/working/hnng/python_bindings'); import hnnglib; print(hnnglib.Index)"
```

Expected: `3.16.0 2.4.6` (or later versions) and `<class 'hnnglib.Index'>`.

- [ ] **Step 2: Write the fixture script**

`test/fixtures/make_fixtures.py`:
```python
#!/usr/bin/env python3
"""Generate the C++ parity fixture for HNNG.jl (spec section 10, test/parity_cpp.jl).

Run once from the repository root:  python3 test/fixtures/make_fixtures.py
The JSON output is committed; the C++ repository is only read.
"""
import base64
import json
import os
import struct
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, "/home/coral/working/hnng/python_bindings")
import hnnglib  # noqa: E402

DATA = "/home/coral/working/benchmarks/data/random-xs-20-euclidean.hdf5"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp_parity_random_xs_20.json")
N_TRAIN, N_TEST, K = 2000, 100, 10


def cpp_level_counts(path):
    """Read maxlevel and per-level cluster counts from a C++ save file (hnnalg.h:2471-2526)."""
    with open(path, "rb") as f:
        buf = f.read()
    off = 0

    def take(fmt):
        nonlocal off
        vals = struct.unpack_from("<" + fmt, buf, off)
        off += struct.calcsize("<" + fmt)
        return vals

    _max_el, n, _data_size, size_per = take("QQQQ")
    (maxlevel,) = take("i")
    take("I")                                   # enterpoint
    off += n * size_per                         # level-0 block
    levels = take(f"{n}i")                      # element_levels
    off += n                                    # deleted flags, one byte each
    for _ in range(n):                          # level-0 inlinks
        (c,) = take("I")
        off += 4 * c
    (nlev,) = take("i")
    for lvl in range(nlev):                     # upper-level links
        for i in range(n):
            if levels[i] >= lvl + 1:
                take("I")
                take("f")
                (c,) = take("I")
                off += 4 * c
    (ncl,) = take("i")
    counts = []
    for _ in range(ncl):
        (nc,) = take("I")
        for _ in range(nc):
            take("I")                           # rep
            take("f")                           # radius
            take("I")                           # parent
            take("B")                           # c_status
            (nm,) = take("I")
            off += 4 * nm
        counts.append(nc)
    assert off == len(buf), f"parsed {off} of {len(buf)} bytes"
    return maxlevel + 1, counts[: maxlevel + 1]


def main():
    with h5py.File(DATA, "r") as f:
        train = np.ascontiguousarray(f["train"][:N_TRAIN], dtype="<f4")
        test = np.ascontiguousarray(f["test"][:N_TEST], dtype="<f4")
    dim = train.shape[1]

    idx = hnnglib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=N_TRAIN)
    idx.add_items(train, num_threads=1)       # 2000 rows < 4096: incremental addPoint path
    idx.set_num_threads(1)
    labels, sq = idx.knn_query(test, k=K, num_threads=1)

    t64, q64 = train.astype(np.float64), test.astype(np.float64)
    bf = np.sqrt(((q64[:, None, :] - t64[None, :, :]) ** 2).sum(-1))
    bf_sorted = np.sort(bf, axis=1)[:, : K + 1]

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cpp.idx")
        idx.save_index(path)
        nlev, ncl = cpp_level_counts(path)

    fx = {
        "dataset": "random-xs-20-euclidean",
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "dim": dim,
        "k": K,
        "train_f32_b64": base64.b64encode(train.tobytes()).decode("ascii"),
        "test_f32_b64": base64.b64encode(test.tobytes()).decode("ascii"),
        "cpp_ids": labels.astype(np.int64).tolist(),
        "cpp_sqdists": [[float(v) for v in row] for row in sq],
        "bf_dists": bf_sorted.tolist(),
        "cpp_nlevels": nlev,
        "cpp_nclusters": ncl,
    }
    with open(OUT, "w") as f:
        json.dump(fx, f)
    print(f"wrote {OUT}: {N_TRAIN} train, {N_TEST} queries, C++ levels {nlev}, clusters {ncl}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate the fixture**

```bash
cd /home/coral/working/HNNG.jl
python3 test/fixtures/make_fixtures.py
ls -l test/fixtures/cpp_parity_random_xs_20.json
python3 -c "import json; d = json.load(open('test/fixtures/cpp_parity_random_xs_20.json')); print(len(d['cpp_ids']), len(d['cpp_ids'][0]), len(d['bf_dists'][0]), d['cpp_nlevels'], d['cpp_nclusters'])"
```

Expected: `wrote .../cpp_parity_random_xs_20.json: 2000 train, 100 queries, C++ levels L, clusters [...]` with `L` between 3 and 7 and a strictly decreasing cluster list ending in 1; the file is about 250 KB; the check prints `100 10 11 L [...]`. If the `assert off == len(buf)` fires, the save layout differs from `hnnalg.h:2471-2526`; reread those lines and fix the parser, not the C++.

- [ ] **Step 4: Commit**

```bash
git add test/fixtures/make_fixtures.py test/fixtures/cpp_parity_random_xs_20.json
git commit -m "test: C++ parity fixture on random-xs-20 (2000 train, 100 queries, k=10)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 22: C++ parity test

**Files:**
- Create: `test/parity_cpp.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: the Task 21 JSON; `add!`, `knn`, `verify`, `stats`.
- Produces: the parity gate of spec §10.

- [ ] **Step 1: Write the test**

`test/parity_cpp.jl`:
```julia
isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))
using JSON3, Base64

@testset "C++ parity: random-xs-20, n = 2000, k = 10" begin
    fx = JSON3.read(read(joinpath(@__DIR__, "fixtures", "cpp_parity_random_xs_20.json"), String))
    d, ntr, nte, k = fx.dim, fx.n_train, fx.n_test, fx.k
    tr = reshape(ltoh.(reinterpret(Float32, base64decode(fx.train_f32_b64))), d, ntr)
    te = reshape(ltoh.(reinterpret(Float32, base64decode(fx.test_f32_b64))), d, nte)
    idx = HNNGIndex(Euclidean(), d)
    add!(idx, tr)
    @test verify(idx)
    s = stats(idx)
    @info "levels and clusters (printed, not gated: representative election differs, spec §12)" julia_levels = s.nlevels julia_clusters = s.nclusters cpp_levels = fx.cpp_nlevels cpp_clusters = collect(fx.cpp_nclusters)
    for j in 1:nte
        ids, ds = knn(idx, view(te, :, j), k)
        cpp = sort(sqrt.(Float64.(fx.cpp_sqdists[j])))
        @test all(isapprox.(sort(ds), cpp; rtol=1e-5))
        bf = Float64.(fx.bf_dists[j])
        if bf[k+1] - bf[k] > 1e-6 * bf[k]
            @test sort(ids) == sort(Int.(fx.cpp_ids[j]) .+ 1)
        end
    end
end
```

`test/runtests.jl`: add `include("parity_cpp.jl")` after `include("io.jl")`.

- [ ] **Step 2: Run the test**

Run: `julia --project=. -e 'using Test, HNNG; include("test/parity_cpp.jl")'`
Expected: one `@info` block with the two level and cluster lists, then the testset passes (1 + 100 distance checks + up to 100 id checks). A distance mismatch means the Julia index is not exact: rerun `check_knn` on that query to see which side is wrong before touching code.

- [ ] **Step 3: Commit**

```bash
git add test/parity_cpp.jl test/runtests.jl
git commit -m "test: C++ parity gate on distances and well-separated id sets" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 23: Error handling (spec §9)

**Files:**
- Create: `test/errors.jl`
- Modify: `test/runtests.jl`; any `src/` file whose error path the tests show missing

**Interfaces:**
- Consumes: every public entry point.
- Produces: one test per §9 case; every message carries the offending value.

- [ ] **Step 1: Write the tests**

`test/errors.jl`:
```julia
isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))

@testset "errors (spec §9)" begin
    idx = HNNGIndex(Euclidean(), 3)
    add!(idx, Float32[1 2 3; 4 5 6; 7 8 9])
    @test verify(idx)
    # dimension mismatch on add! and knn
    @test_throws "the input has length 2" add!(idx, [1.0, 2.0])
    @test_throws "the matrix has 2 rows" add!(idx, ones(2, 4))
    @test_throws "the input has length 4" knn(idx, [1.0, 2.0, 3.0, 4.0], 1)
    @test_throws ArgumentError knn_batch(idx, ones(2, 4), 1)
    # a non-finite coordinate; nothing is inserted
    @test_throws "non-finite coordinate Inf at position 2" add!(idx, [1.0, Inf, 0.0])
    @test_throws "non-finite coordinate NaN" add!(idx, [1.0 2.0; NaN 0.0; 0.0 0.0])
    @test length(idx) == 3 && HNNG._ntotal(idx) == 3 && verify(idx)
    @test_throws ArgumentError knn(idx, [NaN, 0.0, 0.0], 1)
    # k < 1
    @test_throws "k must be at least 1, got 0" knn(idx, [1.0, 2.0, 3.0], 0)
    @test_throws "got -1" knn_batch(idx, ones(3, 2), -1)
    @test_throws ArgumentError HNNG.knn_counted(idx, [1.0, 2.0, 3.0], 0)
    @test_throws ArgumentError HNNG.knn_levelsync(idx, [1.0, 2.0, 3.0], 0)
    # delete! of an unknown or already deleted id
    @test_throws "unknown id 4" delete!(idx, 4)
    @test_throws "unknown id 0" delete!(idx, 0)
    delete!(idx, 2)
    @test verify(idx)
    @test_throws "id 2 is deleted" delete!(idx, 2)
    # navigation on a dead id
    for f in (element, outlink, outlink_distance, inlinks, cluster, path_to_root)
        @test_throws "id 2 is deleted" f(idx, 2)
    end
    @test_throws "level 5 is out of range" outlink(idx, 1; level=5)
    @test_throws "stale or invalid cluster handle" members(idx, HNNG.ClusterRef(0, 99))
    # bulk_add! on a non-empty index or an ObjectStore
    @test_throws "needs an empty index" bulk_add!(idx, ones(3, 2))
    drained = HNNGIndex(Euclidean(), 1)
    add!(drained, [1.0])
    @test verify(drained)
    delete!(drained, 1)
    @test verify(drained)
    @test_throws "needs an empty index" bulk_add!(drained, ones(1, 2))
    obj = HNNGIndex(UserMetric((a, b) -> abs(a - b)), Float64)
    @test_throws "needs a VectorStore index" bulk_add!(obj, ones(1, 2))
    # load of an ObjectStore or UserMetric index without metric=
    dir = mktempdir()
    add!(obj, [1.0, 2.0])
    @test verify(obj)
    HNNG.save(joinpath(dir, "o"), obj)
    @test_throws "pass metric=" HNNG.load(joinpath(dir, "o"))
    um = HNNGIndex(UserMetric((a, b) -> sqrt(sum(abs2, a .- b)); is_metric=true), 2)
    add!(um, [0.0 1.0; 0.0 1.0])
    @test verify(um)
    HNNG.save(joinpath(dir, "u"), um)
    @test_throws "pass metric=" HNNG.load(joinpath(dir, "u"))
    @test verify(HNNG.load(joinpath(dir, "u"); metric=um.metric))
end
```

`test/runtests.jl`: add `include("errors.jl")` after `include("parity_cpp.jl")`.

- [ ] **Step 2: Run the tests**

Run: `julia --project=. -e 'using Test, HNNG; include("test/errors.jl")'`
Expected: the testset passes. Any failure names a missing or wrong `ArgumentError`: add or fix the check at the entry point it names (messages must keep the offending value) and rerun.

- [ ] **Step 3: Run the full suite and commit**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: `Testing HNNG tests passed`.

```bash
git add test/errors.jl test/runtests.jl src
git commit -m "test: every error case of spec section 9" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 24: Slow HDF5 runs behind `HNNG_SLOW=1`

**Files:**
- Create: `test/slow.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `add!`, `bulk_add!`, `knn`, `verify`, `check_knn`, `structure`, `same_structure`; HDF5.jl (`h5open`, column-major reads: an HDF5 `(n, d)` dataset reads as a `d × n` matrix).
- Produces: the slow suite; the default suite does not load HDF5.

- [ ] **Step 1: Write the slow tests**

`test/slow.jl`:
```julia
isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))
using HDF5

const SLOW_DATA = "/home/coral/working/benchmarks/data"

@testset "slow: random-xs-20-euclidean, all 9000 train points" begin
    path = joinpath(SLOW_DATA, "random-xs-20-euclidean.hdf5")
    if isfile(path)
        train, test, gt = h5open(path) do f
            read(f["train"]), read(f["test"]), read(f["distances"])
        end
        idx = HNNGIndex(Euclidean(), 20; T=Float64)
        add!(idx, train)
        @test verify(idx)
        for j in 1:200
            _, ds = knn(idx, test[:, j], 10)
            @test all(isapprox.(ds, gt[1:10, j]; rtol=1e-9))
        end
        B = HNNGIndex(Euclidean(), 20; T=Float64)
        bulk_add!(B, train)
        @test verify(B)
        @test same_structure(structure(B), structure(idx))
    else
        @test_skip isfile(path)
    end
end

@testset "slow: fashion-mnist-784, 5000-point Float32 subset" begin
    path = joinpath(SLOW_DATA, "fashion-mnist-784-euclidean.hdf5")
    if isfile(path)
        train, test = h5open(path) do f
            f["train"][:, 1:5000], f["test"][:, 1:50]
        end
        idx = HNNGIndex(Euclidean(), 784)
        add!(idx, train)
        @test verify(idx)
        for j in 1:50
            @test check_knn(idx, test[:, j], 10)
        end
        B = HNNGIndex(Euclidean(), 784)
        bulk_add!(B, train)
        @test verify(B)
        @test same_structure(structure(B), structure(idx))
    else
        @test_skip isfile(path)
    end
end
```

`test/runtests.jl` (final form; the slow file is included only when asked for):
```julia
using Test, HNNG

@testset "HNNG.jl" begin
    include("oracle.jl")
    include("metrics.jl")
    include("invariants.jl")
    include("search.jl")
    include("updates.jl")
    include("objects.jl")
    include("order.jl")
    include("navigate.jl")
    include("io.jl")
    include("parity_cpp.jl")
    include("errors.jl")
    if get(ENV, "HNNG_SLOW", "0") == "1"
        include("slow.jl")
    end
end
```

- [ ] **Step 2: Run the slow suite**

Run: `HNNG_SLOW=1 julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: `Testing HNNG tests passed`; the two slow testsets take a few minutes in total (the fashion subset dominates).
Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: `Testing HNNG tests passed` without loading HDF5.

- [ ] **Step 3: Commit**

```bash
git add test/slow.jl test/runtests.jl
git commit -m "test: HDF5-based slow runs behind HNNG_SLOW=1" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 25: README with the comparison table and the two recipes; final runs

**Files:**
- Create: `README.md`, `test/recipes.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: the whole public API, `level_graph` (Graphs extension).
- Produces: `README.md`; `test/recipes.jl` runs the README recipes verbatim, so the README cannot rot.

- [ ] **Step 1: Write the recipe tests (the README code, executed)**

`test/recipes.jl`:
```julia
isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))
using Graphs

# README recipe 1: recognition by the label of the nearest stored view.
recognise(idx, labels, q) = labels[first(first(knn(idx, q, 1)))]

# README recipe 2: maximal single-label subtrees of an object (the ROS ObjectRepresentation).
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

@testset "README recipes" begin
    rng = StableRNG(25)
    centres = [-6.0 0.0 6.0; 0.0 5.0 0.0]
    X = reduce(hcat, [centres[:, c] .+ randn(rng, 2) for c in repeat(1:3, 40)])
    labels = repeat(["cup", "bowl", "plate"], 40)
    idx = HNNGIndex(Euclidean(), 2; T=Float64)
    add!(idx, X)
    @test verify(idx)
    @test recognise(idx, labels, [-6.0, 0.2]) == "cup"
    n0 = idx.nevals
    parts = single_label_subtrees(idx, labels, "bowl")
    @test idx.nevals == n0                                   # no distance evaluated
    @test sort!(reduce(vcat, parts)) == findall(==("bowl"), labels)
    @test all(p -> all(i -> labels[i] == "bowl", p), parts)
    # nearest-neighbour chain and shortest path over stored distances
    chain = [1]
    while outlink(idx, outlink(idx, chain[end])) != chain[end]
        push!(chain, outlink(idx, chain[end]))
    end
    g, W, ids = level_graph(idx, 0)
    st = dijkstra_shortest_paths(g, 1, W)
    @test st.dists[chain[end]] ≈ sum(outlink_distance(idx, x) for x in chain[1:end-1]; init=0.0)
end
```

`test/runtests.jl`: add `include("recipes.jl")` after `include("errors.jl")`.

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: `Testing HNNG tests passed` (the recipes use only code from Tasks 1–24, so this passes at once; if it does not, the failure is a package bug to fix before writing the README).

- [ ] **Step 2: Write `README.md`**

````markdown
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
come back in the metric's own units (`CosineDist` returns `1 - cos`). Any other
Distances.jl metric works too; a `SemiMetric` or `PreMetric` runs unpruned and stays
exact. `bulk_add!(idx, X)` builds the same hierarchy as `add!(idx, X)` on an empty
index with BLAS. `HNNG.save(path, idx)` and `HNNG.load(path)` persist an index.

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
the stored structure only. With `using Graphs`, `level_graph(idx, l)` returns a
`SimpleDiGraph`, its weight matrix and the vertex ids, so graph algorithms run on
stored distances.

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
relative guard of `1e-6` in favour of visiting, which absorbs rounding in the
triangle inequality. `Float32` data accumulates in `Float32`; every stored distance,
radius and bound is `Float64`.

## Tests

`julia --project=. -e 'using Pkg; Pkg.test()'` runs the default suite (under a
minute). `HNNG_SLOW=1` adds the runs on the ann-benchmarks HDF5 files.
````

- [ ] **Step 3: Final runs on stable and on the LTS**

```bash
export PATH="$HOME/.juliaup/bin:$PATH"; cd /home/coral/working/HNNG.jl
julia --project=. -e 'using Pkg; Pkg.test()'
rm -f Manifest.toml
julia +lts --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
rm -f Manifest.toml
julia -t 4 --project=. -e 'using Pkg; Pkg.test()'
```

Expected: `Testing HNNG tests passed` three times; the `HNNG.jl` testset time column shows well under 60 s for the default runs (excluding package precompilation). If the LTS run fails on a newer-only Julia feature, replace that construct in `src/` or `test/` with a 1.10 equivalent and rerun both.

- [ ] **Step 4: Commit**

```bash
git add README.md test/recipes.jl test/runtests.jl
git commit -m "docs: README with comparison table and navigation recipes, recipes tested" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
