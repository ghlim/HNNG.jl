# HNNG.jl Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `bench/`, the pinned single-threaded Julia harness that runs Exp.1, Exp.1b, Exp.2 and Exp.3 of spec §11 against HNNG.jl and the Julia baselines, and writes the CSV files, LaTeX tables, `\newcommand` macros and CairoMakie PDFs that the manuscript inputs, plus the README benchmark section.

**Architecture:** A bench-local module `BenchCore` (`bench/src/`) holds four single-purpose files: dataset loading and generation, the two recall rules, the timing protocol with `env.txt`, and one method interface (`build` / `query` / `build_time`) over hNNG, BLAS brute force, NearestNeighbors.jl trees, HNSW.jl and optional SimilaritySearch.jl. Thin scripts `exp1.jl`, `exp2.jl`, `exp3.jl` write fixed-schema CSVs. `tables.jl` and `plots.jl` turn those CSVs into `.tex` and `.pdf` files. `run_all.jl` runs everything and `run.sh` pins it to one core. `BENCH_SMOKE=1` shrinks every experiment to tiny sizes, so every task's tests exercise the real code path in seconds.

**Tech Stack:** Julia stable (installed by the index plan through juliaup; `[compat] julia = "1.10"`), HNNG (dev path), Distances, NearestNeighbors, HNSW, HDF5, StableRNGs, CSV, DataFrames, CairoMakie, SHA, SimilaritySearch (optional), stdlibs Pkg, LinearAlgebra, Statistics, Printf, Dates, Random, Test. Package versions are not pinned here: `bench/Manifest.toml` (committed) and the `pkg_versions` and `manifest_sha256` lines of `env.txt` are the record.

**Spec:** `/home/coral/working/HNNG.jl/docs/superpowers/specs/2026-10-15-hnng-jl-port-design.md` (sections 4, 4.2, 4.3, 8 and 11 drive this plan). Package API notes: `/home/coral/working/.workflow/scratch/julia-eco/{HNSW,NearestNeighbors,SimilaritySearch,HDF5}.md`. TKDE layouts to reproduce: `/home/coral/working/publications/Hierarchical-Nearest-Neighbor-Graph/tkde2026_draft_sections.tex` lines 88–97 (ρ(d) sentence), 172–186 (`tab:annbench`), 217–234 (`tab:fashion`), 448–463 (`tab:ops`).

**Prerequisite:** the sibling plan `docs/superpowers/plans/2026-10-16-hnng-jl-index.md` is executed first: Julia is installed, `import HNNG` works from the repository root, the package tests are green, and the repo-local git identity is set. Every task below loads `HNNG`.

**Interfaces needed from the index plan** (the chair reconciles these with the index plan):

| Call | Contract this plan relies on |
|---|---|
| `HNNG.HNNGIndex(Euclidean(), d; T = Float32)` | empty VectorStore index; `Euclidean` from Distances.jl |
| `HNNG.add!(idx, X::AbstractMatrix) -> UnitRange{Int}` | streaming insert of the columns, in order |
| `HNNG.add!(idx, x::AbstractVector) -> Int` | one column; returns the new id (`n_total + 1`) |
| `HNNG.bulk_add!(idx, X) -> UnitRange{Int}` | empty index only |
| `HNNG.delete!(idx, id)` | called qualified, so it works whether HNNG extends `Base.delete!` or not |
| `HNNG.knn(idx, q, k) -> (ids::Vector{Int}, dists::Vector{Float64})` | sorted by (distance, id); Euclidean units; all live points when fewer than `k` |
| `HNNG.knn_counted(idx, q, k) -> (ids, dists, nevals::Int)` | distance evaluations of this query |
| `HNNG.stats(idx)` | fields used: `nlevels::Int`, `nclusters::Vector{Int}` (level 0 first), `nevals::Int` (since the last reset) |
| `HNNG.reset_evals!(idx)`, `length(idx)`, `HNNG.nlevels(idx)` | as spec §4 and §4.3 |
| **`HNNG.last_update_stats(idx) -> (rewired::Int, top_level_touched::Int)`** | **not in the spec.** For the most recent `add!` or `delete!`: `rewired` = number of level-0 elements whose outlink changed (insert: existing elements rewired to the new element, excluding the new element's own outlink; delete: former inlinkers re-linked); `top_level_touched` = highest level (0-based) on which `link!`/`unlink!` ran. This plan destructures it as `rew, top = ...`, so a `Tuple` or a `NamedTuple` both work. |

**Verified facts used below.** HDF5 layout checked on 2026-10-16 with h5py: `random-xs-20-euclidean.hdf5` has `train` (9000, 20) float64, `test` (1000, 20) float64, `neighbors` (1000, 100) int64, `distances` (1000, 100) float64, attrs `distance = "euclidean"`; `neighbors[0][0] = 3618`, `distances[0][0] = 4.43531921`. `fashion-mnist-784-euclidean.hdf5` has `train` (60000, 784) float32, `test` (10000, 784) float32, `neighbors` (10000, 100) int32, `distances` (10000, 100) float32. h5py shapes are row-major; HDF5.jl reads the same bytes as d×n, which is the layout this plan wants. Ids in the file are 0-based.

## Spec assumptions

Gaps or choices the spec leaves open; these are decided here. (The review pass also corrected the spec itself where noted in A2, A11 and A15.)

- **A1.** `HNNG.last_update_stats` (table above) is needed by Exp.3 and is not in spec §4.
- **A2.** (Confirmed by the coordinator.) Spec §3 and §10 say the harness's two recall definitions are tested in `bench/test/runtests.jl` (the rules live in `bench/src/recall.jl`, inside the bench environment); the package's `test/` has no `recall_rule.jl`. This plan writes that test (Task 2).
- **A3.** `bench/Project.toml` also lists `Distances` (for `Euclidean()`), `SHA` (Manifest hash), `Pkg` (package versions in `env.txt`) and the stdlibs it uses; spec §11 names only the main packages. `BenchmarkTools` is not a dependency (A15).
- **A4.** The method supertype is `BenchMethod`, not `Method` (`Method` is a `Base` type).
- **A5.** SimilaritySearch is loaded with `try import SimilaritySearch catch end` (`import`, not `using`, so its exports cannot clash with `build`/`query`). Spec §11 names both `ExhaustiveSearch` and `SearchGraph`, so both get rows (`SS-Exhaustive`, `SS-SearchGraph`) with default parameters. The calls follow the SimilaritySearch.jl searchgraph tutorial (v1.6.5): positional constructors `SearchGraph(dist, db)` and `ExhaustiveSearch(dist, db)` with `SimilaritySearch.Dist.L2()` (metric Euclidean, so the reported distances are comparable to the other methods'), `MatrixDatabase`, `SearchGraphContext(; neighborhood = Neighborhood(; neardup = 0f0))`, `GenericContext()`, `index!`, and one query at a time through `knnqueue(ctx, k)`, `search(index, ctx, q, res)`, `expand!(G, ctx, q, res)` and `IdDistView(res)`. `expand!` returns the raw k nearest points when `neardup` folds exact duplicates into cluster members (Fashion-MNIST has repeated images; without the expansion a query returns one representative per duplicate group and under-counts recall). The Task 5 testset checks the result convention; experiment rows are skipped with a warning if any call throws.
- **A6.** (Confirmed.) The generated `.tex` files hold only the `tabular` environment. The manuscript keeps its own `table` float, caption and label and `\input`s the file.
- **A7.** The table "Recall" column uses the ann-benchmarks rule. "HNSW (max recall attained)" = the HNSW row with the highest recall (ties: higher QPS). "HNSW (sweet spot)" = the fastest HNSW row with recall ≥ 0.99, else the max-recall row. When no HNSW row exists, an `SS-SearchGraph` row takes the HNSW place (spec §13).
- **A8.** (Confirmed.) `tab_ops` "rebuild" = one full streaming rebuild (`add!` of all n points), which is what the TKDE text calls the "sequential rebuild"; speedup = rebuild / insert latency (TKDE: 0.46 s / 1.10 ms ≈ 419). The bulk rebuild is recorded as `op = rebuild-bulk` and gets its own macro.
- **A9.** "Median per-operation latency of three runs" = median over the three runs of each run's mean per-operation latency. The three runs repeat the same build, inserts and deletes (fixed seeds). `exp3.csv` keeps every operation, so other summaries stay possible.
- **A10.** (Confirmed.) Exp.2 indexes hNNG with the bulk loader (by spec §6.1 the structure is the same as a streaming build); ρ is reported for hNNG only. Exactness of all four methods is asserted during the run.
- **A11.** Build time is one timed build after an untimed warm-up build on the first 200 columns, not best of three (full-size streaming builds take minutes). Query QPS is the best of three passes. Spec §11 says exactly this (query timing best of three, build time one timed build), and the manuscript's Setup sentence is meant to say the same.
- **A12.** Recall uses distances recomputed in Float64 from the returned ids, as ann-benchmarks recomputes from ids, not each library's own float output.
- **A13.** Smoke runs subset the HDF5 data (first 300 train points, first 20 queries) and recompute ground truth by Float64 brute force. Full runs use the HDF5 `neighbors`/`distances` (+1 on ids).
- **A14.** For Exp.3, HNSW.jl gets all `n + 200` points in its data vector, links `1:n` at build time, then links one point per insert (`add_to_graph!(g, [i])`). The HNSW delete row has a missing latency and the note `mark only`.
- **A15.** BenchmarkTools is not a dependency (spec §11 no longer lists it): the timing code uses explicit `time_ns` loops because the protocol (best of three whole passes; mutating operations timed one by one) does not map onto BenchmarkTools' sampling.
- **A16.** Smoke output goes to `bench/results-smoke/` (gitignored). `bench/results/` is committed after the full run (Task 12).
- **A17.** Exp.3 runs the TKDE grid `n ∈ {8000, 16000, 32000} × d ∈ {16, 64, 128}` (spec §11 as revised in commit 6e8690d). In `exp3.csv`, `config = "table"` marks the three `tab_ops` rows and `config = "grid"` the other six; the figure uses all nine.
- **A18.** Brute-force candidates: every point whose Float32 score is within τ = max(1e-4, 2·d·eps(Float32))·(‖q‖² + max‖x‖²) of the k-th score, re-ranked in Float64 (spec: "exact re-rank of the top candidates").
- **A19.** KD-tree and ball tree use the NearestNeighbors.jl default leaf size 25.
- **A20.** The TKDE tables mix precisions; this plan fixes one rule per column: recall `%.3f` (annbench) or `%.5f` (fashion); QPS integer with `{,}` separators when ≥ 100, else one decimal; build seconds three significant digits; ms with two decimals.
- **A21.** Blobs assign point i to centre `((i-1) mod 100) + 1` (equal counts, as `make_blobs` does, in a different order).
- **A22.** (Confirmed.) HNSW.jl is loaded like SimilaritySearch (`try import HNSW catch end`, flag `HAS_HNSW`), so that spec §13's fallback (SearchGraph takes HNSW's rows when HNSW.jl fails to load) works instead of the whole harness failing to load.
- **A23.** `run_all` runs the experiments in the order exp2, exp3, exp1 (random-xs only), exp1b (Fashion-MNIST): the cheap or never-yet-run-at-full-size experiments come first and the multi-hour Fashion-MNIST run last, so a late failure costs the least. An experiment whose CSV already exists in the output directory is skipped unless `BENCH_FORCE=1`; smoke runs always rerun (their directory is scratch). `BENCH_ONLY=<exp1|exp1b|exp2|exp3>` restricts a run to one experiment (add `BENCH_FORCE=1` to rerun it when its CSV exists). Tables and plots are written only when all four CSVs exist.
- **A24.** Figures are designed at their final print size and saved with `save(path, fig; pt_per_unit = 1)`, so `Figure(size = ...)` is in pt. The `sn-jnl` `iicol` layout has a 160 mm text width and 8 mm column separation, so a column is 76 mm = 215 pt and the double-column width is 453 pt. Single-column figures (`recall_qps`, `dist_eval_vs_dim`, `qps_vs_dim`) are 215 × 165 pt; `ops_comparison` is a double-column `figure*` of 450 × 150 pt; all use font size 8 pt.
- **A25.** The ann-benchmarks recall rule (additive 1e-3) cannot see a miss at low d, where gaps between consecutive neighbour distances are of order 1e-3. For the exact methods (hNNG, brute force, KD-tree, ball tree) Exp.1, Exp.1b and Exp.2 therefore also check, per query, that the k returned distances (recomputed in Float64) equal the true k nearest distances within rtol 1e-5 (`min_match`, Task 2), and fail the run otherwise. Exp.3 does the same in `check_exact!`.
- **A26.** The Exp.3 insert, delete and HNSW-insert loops call `GC.gc(false)` immediately before each timed region (outside it), so that garbage from the untimed exactness check is not collected inside a timed operation.

## Global Constraints

- Julia: package compat lower bound `1.10`; `bench/Project.toml` has `[compat] julia = "1.10"`.
- Protocol (spec §11, verbatim): "`julia -t 1`, `BLAS.set_num_threads(1)`, process pinned with `taskset`, one query at a time, `k = 10`. Query timing is the best of three timed passes; build time is a single timed build after an untimed warm-up build; the Exp.3 table reports the median over three runs of the mean per-operation latency."
- Recall (verbatim): "the ann-benchmarks rule (a returned distance counts if it is `≤` the true k-th distance `+ 1e-3`) and strict id recall."
- Fraction evaluated (verbatim): "distance evaluations during the query divided by the number of live elements", from `HNNG.knn_counted`.
- `env.txt` records "Julia version, the `Manifest.toml` hash, CPU model, governor, and date", plus this plan's additions: allowed CPUs, thread counts, HNNG.jl commit, `dirty` (uncommitted changes outside `bench/results*`), `smoke`, the seeds, `pkg_versions` and the kernel.
- Data: `random-xs-20-euclidean.hdf5` and `fashion-mnist-784-euclidean.hdf5` from `/home/coral/working/benchmarks/data/` (`train`, `test`, `neighbors`, `distances`).
- HNSW.jl: `M = 16`, `efConstruction = 200`, `ef` sweep `{10, 20, 40, 80, 160, 320, 640}`.
- Exp.2: `d ∈ {2, 4, 8, 16, 32, 64, 128}`, 1,000 queries; Gaussian `N(0, I_d)` with `n = 5,000`; 100-centre blobs with `n = 10,000`, centres uniform in `[-10, 10]^d`, unit isotropic standard deviation.
- Exp.3: Gaussian data on the TKDE grid `n ∈ {8000, 16000, 32000} × d ∈ {16, 64, 128}`; table rows `(8000, 16)`, `(32000, 16)`, `(32000, 128)`; figure with three panels (one per `d`), `n` on a log axis, three lines (full rebuild, insert, delete); streaming and bulk rebuild times recorded; 200 inserts then 200 deletes of random live ids; median of three runs; 20 random queries checked against brute force after every operation (assertion, not timed).
- Outputs: `bench/results/{exp1,exp1b,exp2,exp3}.csv`, `env.txt`, `bench/results/tex/{tab_annbench,tab_fashion,tab_ops,macros}.tex`, `bench/results/fig/{recall_qps,dist_eval_vs_dim,qps_vs_dim,ops_comparison}.pdf`.
- "Throughput parity with the C++ numbers in the TKDE draft is not a goal and is not promised."
- All randomness uses `StableRNGs` with the fixed seeds at the top of `bench/src/datasets.jl`; HNSW.jl's internal draws use `Random.seed!(SEED_HNSW)`.
- This plan touches only `bench/`, `README.md` and `.gitignore`; it never edits `src/`, `test/` or the spec.
- Every commit message ends with the line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Plain `git commit` (the index plan set the repo-local identity).

## Review Focus

1. **Float32 cancellation in the BLAS brute force** when ‖x‖² is far larger than neighbour distances² (Fashion-MNIST pixel norms, offset data): a person expects the brute-force row to stay exact (recall 1.0). Pinned by the test "brute force stays exact under Float32 cancellation" in Task 4.
2. **A run started with more than one thread** (plain `julia`, or `JULIA_NUM_THREADS` set): a person expects the harness to refuse before timing anything, not to publish inflated QPS. Pinned by the subprocess test "check_protocol refuses more than one thread" in Task 3, which asserts the refusal message, so a child that merely failed to load `BenchCore` does not pass.
3. **HNSW never reaching recall 0.99 on a dataset, or HNSW.jl absent:** a person expects the tables still to be written, with the sweet-spot row falling back to the max-recall row and SearchGraph taking HNSW's place. Pinned by two tests in Task 9.
4. **A smoke run or a test overwriting the real results:** a person expects `bench/results/` to change only on a full run. Pinned by the `results_dir` tests in Task 1; every experiment test writes into `mktempdir()`.
5. **An approximate method returning fewer than k ids, duplicate ids, or more than k:** a person expects the recall to count each true neighbour once and to ignore extras. Pinned by the recall tests in Task 2, including a duplicated returned id through `mean_recalls`.
6. **A crash hours into the full run:** a person expects the finished experiments to survive and a rerun to continue, not to repeat hours of work. Pinned by the `run_experiments` test in Task 11 (order, skip-existing, `BENCH_ONLY`, `BENCH_FORCE`, no tables and plots until all four CSVs exist).

## File structure

| File | Responsibility |
|---|---|
| `bench/Project.toml`, `bench/Manifest.toml` | bench environment; HNNG as a dev dependency (`path = ".."`) |
| `bench/src/BenchCore.jl` | module: imports, optional SimilaritySearch, `is_smoke`, `results_dir`, CSV schemas, `op_latency` |
| `bench/src/datasets.jl` | seeds, HDF5 loader, smoke subset, Gaussian and blob generators, Float64 ground truth |
| `bench/src/recall.jl` | `recall_ann`, `recall_id`, `mean_recalls`, `distance_match`, `min_match` |
| `bench/src/timing.jl` | `check_protocol`, `best_of`, `time_queries`, `write_env` |
| `bench/src/baselines.jl` | `BenchMethod` interface and every method adapter |
| `bench/exp1.jl`, `bench/exp2.jl`, `bench/exp3.jl` | `run_exp1` and `run_exp1b` (exp1.jl, one dataset each), `run_exp2`, `run_exp3`; all `run_expN(; smoke, outdir)` |
| `bench/tables.jl` | `tab_annbench.tex`, `tab_fashion.tex`, `tab_ops.tex`, `macros.tex` |
| `bench/plots.jl` | the four PDFs |
| `bench/run_all.jl`, `bench/run.sh` | the whole pipeline (ordered, resumable); pinned wrapper |
| `bench/test/setup.jl`, `bench/test/runtests.jl`, `bench/test/test_*.jl` | tests (always smoke-sized) |
| `README.md` | "Benchmarks" section |

CSV schemas (fixed in Task 1, checked by every experiment test):

- `exp1.csv`, `exp1b.csv`: `dataset, n, d, nq, k, method, param, build_s, qps, recall_ann, recall_id, frac_evaluated, nlevels, nclusters`
- `exp2.csv`: `geometry, n, d, nq, k, method, build_s, qps, recall_ann, frac_evaluated, nlevels`
- `exp3.csv`: `config, n, d, run, method, op, seq, latency_s, nevals, rewired, top_level, note`

`method` values: `hNNG-incremental`, `hNNG-bulk`, `brute-BLAS`, `KD-tree`, `ball-tree`, `HNSW`, `SS-SearchGraph`, `SS-Exhaustive` (Exp.3 uses `hNNG` for the index under update). `nclusters` is the per-level cluster count joined with `;`, level 0 first. `op` values: `insert`, `delete`, `rebuild`, `rebuild-bulk`, `update`. Exp.3 `config` values: `table` (one of the three `tab_ops` rows) or `grid` (the other six grid points).

Every script loads `BenchCore` with `using .BenchCore` next to `using DataFrames` and, in `plots.jl`, `using CairoMakie`, all in `Main`. BenchCore's exported names therefore must not repeat theirs: that is why the smoke subset is `smoke_subset` (DataFrames exports `subset`) and the random index is `draw_index` (Makie exports `pick`). The `run_all` test (Task 11) loads every script in one session and catches any new clash.

All commands run from `/home/coral/working/HNNG.jl`. Every shell command assumes `export PATH="$HOME/.juliaup/bin:$PATH"` (the index plan's Global Constraints rule): shells reset between calls, so repeat that export at the top of every command block you run. `bench/run.sh` exports it itself.

---

### Task 1: bench environment, BenchCore skeleton, datasets

**Files:**
- Create: `bench/Project.toml`, `bench/Manifest.toml` (written by Pkg)
- Create: `bench/src/BenchCore.jl`, `bench/src/datasets.jl`
- Create: `bench/test/setup.jl`, `bench/test/runtests.jl`, `bench/test/test_datasets.jl`
- Modify: `.gitignore` (append two lines and make sure only the root manifest is ignored; create the file if absent)

**Interfaces:**
- Consumes: `HNNG` loads (index plan).
- Produces: module `BenchCore` with `BENCH_DIR::String`, `HAS_HNSW::Bool`, `HAS_SS::Bool`, `is_smoke()::Bool`, `results_dir(smoke::Bool = is_smoke())::String`, `empty_exp1()`, `empty_exp2()`, `empty_exp3()` (typed empty `DataFrame`s), `SCHEMA_EXP1/2/3::Vector{String}`; from `datasets.jl`: `struct Dataset(name::String, train::Matrix{Float32}, test::Matrix{Float32}, neighbors::Matrix{Int}, distances::Matrix{Float64})`, `data_dir()`, `load_hdf5(path) -> Dataset`, `smoke_subset(ds, n, nq; kgt = 10) -> Dataset`, `columns(Q) -> Vector{Vector{Float32}}`, `dist64(a, b) -> Float64`, `ground_truth(X, Q, kgt) -> (Matrix{Int}, Matrix{Float64})`, `gaussian(n, d, seed) -> Matrix{Float32}`, `blob_centres(d, seed) -> Matrix{Float64}`, `blobs(n, C, seed) -> Matrix{Float32}`, `synthetic(geometry, n, nq, d) -> (X, Q)`, `draw_index(rng, m) -> Int`, constants `SEED_*`. Test preamble `bench/test/setup.jl`.

- [ ] **Step 1: Create the bench environment**

```bash
mkdir -p bench/src bench/test
julia --project=bench -e 'using Pkg; Pkg.develop(path="."); Pkg.add(["Distances", "NearestNeighbors", "HDF5", "StableRNGs", "CSV", "DataFrames", "CairoMakie", "SHA", "Pkg", "Statistics", "LinearAlgebra", "Printf", "Dates", "Random", "Test"]); Pkg.compat("julia", "1.10")'
julia --project=bench -e 'using Pkg; Pkg.add("HNSW")' || echo "HNSW not added; SearchGraph takes its rows (spec §13)"
julia --project=bench -e 'using Pkg; Pkg.add("SimilaritySearch")' || echo "SimilaritySearch not added; SS rows will be skipped"
julia --project=bench -e 'import HNNG, Distances, NearestNeighbors, HDF5, StableRNGs, CSV, DataFrames, CairoMakie, SHA, Pkg; println("bench env ok")'
```

Expected: the last command prints `bench env ok`. `bench/Project.toml` lists every package above under `[deps]`, and `bench/Manifest.toml` records HNNG with `path = ".."`. If the `HNSW` or `SimilaritySearch` add fails to resolve on this Julia version, leave that package out (the code handles its absence: spec §13 lets SearchGraph take HNSW's rows) and say so in the commit message.

- [ ] **Step 2: Write the test preamble, the runner, and the failing dataset tests**

`bench/test/setup.jl`:

```julia
# Shared preamble: every bench test file starts with include("setup.jl").
using Test
ENV["BENCH_SMOKE"] = "1"   # tests never run the full-size experiments
for k in ("BENCH_ONLY", "BENCH_FORCE", "BENCH_RESULTS")   # a stray export must not change what the tests do
    delete!(ENV, k)
end
isdefined(Main, :BenchCore) || include(joinpath(@__DIR__, "..", "src", "BenchCore.jl"))
using .BenchCore
```

`bench/test/runtests.jl`:

```julia
include(joinpath(@__DIR__, "setup.jl"))

@testset "bench" begin
    include("test_datasets.jl")
end
```

`bench/test/test_datasets.jl`:

```julia
include(joinpath(@__DIR__, "setup.jl"))

@testset "datasets" begin
    @testset "results directories and CSV schemas" begin
        @test results_dir(true) != results_dir(false)
        @test endswith(results_dir(true), "results-smoke")
        @test endswith(results_dir(false), "results")
        @test withenv(() -> results_dir(true), "BENCH_RESULTS" => "/tmp/elsewhere") == "/tmp/elsewhere"
        @test SCHEMA_EXP1 == ["dataset", "n", "d", "nq", "k", "method", "param", "build_s", "qps",
                              "recall_ann", "recall_id", "frac_evaluated", "nlevels", "nclusters"]
        @test SCHEMA_EXP2 == ["geometry", "n", "d", "nq", "k", "method", "build_s", "qps",
                              "recall_ann", "frac_evaluated", "nlevels"]
        @test SCHEMA_EXP3 == ["config", "n", "d", "run", "method", "op", "seq", "latency_s",
                              "nevals", "rewired", "top_level", "note"]
    end

    @testset "gaussian: seeded, d×n Float32, prefix-stable" begin
        A = gaussian(50, 4, SEED_GAUSS_TRAIN + 4)
        @test size(A) == (4, 50) && eltype(A) == Float32
        @test A == gaussian(50, 4, SEED_GAUSS_TRAIN + 4)
        @test A != gaussian(50, 4, SEED_GAUSS_QUERY + 4)
        @test gaussian(80, 4, SEED_GAUSS_TRAIN + 4)[:, 1:50] == A   # Exp.3 relies on this
    end

    @testset "blobs: 100 centres in [-10,10]^d, unit spread, round-robin" begin
        C = blob_centres(3, SEED_BLOBS_CENTRES + 3)
        @test size(C) == (3, 100) && all(-10 .<= C .<= 10)
        X = blobs(1000, C, SEED_BLOBS_TRAIN + 3)
        @test size(X) == (3, 1000) && eltype(X) == Float32
        r = [dist64(view(X, :, i), view(C, :, (i - 1) % 100 + 1)) for i in 1:1000]
        @test 1.2 < sum(r) / 1000 < 2.0          # mean of a chi_3 variable is 1.596
        Xs, Qs = synthetic("blobs", 200, 10, 3)
        @test size(Xs) == (3, 200) && size(Qs) == (3, 10)
        Xg, Qg = synthetic("gaussian", 200, 10, 3)
        @test Xg == gaussian(200, 3, SEED_GAUSS_TRAIN + 3)
        @test_throws ArgumentError synthetic("uniform", 10, 2, 3)
    end

    @testset "ground truth: Float64, ties by smaller id" begin
        X = Float32[0 2 5 1]                     # 1×4: points at 0, 2, 5, 1
        q = reshape(Float32[1], 1, 1)            # distances 1, 1, 4, 0
        nb, gd = ground_truth(X, q, 3)
        @test nb[:, 1] == [4, 1, 2]
        @test gd[:, 1] == [0.0, 1.0, 1.0]
        nb2, _ = ground_truth(X, q, 10)          # k larger than n
        @test size(nb2) == (4, 1)
    end

    @testset "smoke_subset recomputes ground truth" begin
        ds = Dataset("toy", gaussian(40, 3, 1), gaussian(5, 3, 2), zeros(Int, 1, 5), zeros(1, 5))
        s = smoke_subset(ds, 30, 4)
        @test size(s.train) == (3, 30) && size(s.test) == (3, 4)
        nb, gd = ground_truth(s.train, s.test, 10)
        @test s.neighbors == nb && s.distances == gd
    end

    @testset "HDF5 loader: ann-benchmarks layout, 1-based ids" begin
        p = joinpath(data_dir(), "random-xs-20-euclidean.hdf5")
        if isfile(p)
            ds = load_hdf5(p)
            @test ds.name == "random-xs-20-euclidean"
            @test size(ds.train) == (20, 9000) && eltype(ds.train) == Float32
            @test size(ds.test) == (20, 1000)
            @test size(ds.neighbors) == (100, 1000) && size(ds.distances) == (100, 1000)
            @test ds.neighbors[1, 1] == 3619                 # h5py shows 3618, 0-based
            @test ds.distances[1, 1] ≈ 4.43531921 atol = 1e-6
            @test dist64(view(ds.train, :, 3619), view(ds.test, :, 1)) ≈ 4.43531921 rtol = 1e-5
        else
            @warn "random-xs HDF5 not found at $p; loader test skipped"
            @test_skip isfile(p)
        end
    end
end
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_datasets.jl`
Expected: FAIL with `SystemError: opening file ".../bench/src/BenchCore.jl": No such file or directory`.

- [ ] **Step 4: Write `bench/src/BenchCore.jl` and `bench/src/datasets.jl`**

`bench/src/BenchCore.jl`:

```julia
# Shared code for the HNNG.jl benchmark harness (spec §11).
module BenchCore

using LinearAlgebra, Statistics, Printf, Dates, Random
using HDF5, StableRNGs, CSV, DataFrames, SHA
using Distances: Euclidean
import HNNG
import Pkg
import NearestNeighbors as NN

# HNSW.jl and SimilaritySearch are optional: their rows are skipped when they do not load
# (spec §13: SearchGraph then takes HNSW's table rows). `import`, not `using`, keeps their
# exported names out of this module.
try
    import HNSW
catch err
    @warn "HNSW.jl did not load; HNSW rows will be skipped" exception = err
end
try
    import SimilaritySearch
catch err
    @warn "SimilaritySearch did not load; SS-* rows will be skipped" exception = err
end
const HAS_HNSW = isdefined(@__MODULE__, :HNSW)
const HAS_SS = isdefined(@__MODULE__, :SimilaritySearch)

export BENCH_DIR, HAS_HNSW, HAS_SS, is_smoke, results_dir,
       SCHEMA_EXP1, SCHEMA_EXP2, SCHEMA_EXP3, empty_exp1, empty_exp2, empty_exp3

const BENCH_DIR = dirname(@__DIR__)   # .../HNNG.jl/bench, no trailing slash

"BENCH_SMOKE=1 shrinks every experiment to tiny sizes."
is_smoke() = get(ENV, "BENCH_SMOKE", "0") == "1"

"Where CSV, tex, fig and env.txt land. Smoke runs never write into bench/results."
results_dir(smoke::Bool = is_smoke()) =
    get(ENV, "BENCH_RESULTS", joinpath(BENCH_DIR, smoke ? "results-smoke" : "results"))

# Fixed CSV schemas: the column order here is the column order of the files.
empty_exp1() = DataFrame(dataset = String[], n = Int[], d = Int[], nq = Int[], k = Int[],
    method = String[], param = String[], build_s = Float64[], qps = Float64[],
    recall_ann = Float64[], recall_id = Float64[],
    frac_evaluated = Union{Missing,Float64}[], nlevels = Union{Missing,Int}[],
    nclusters = Union{Missing,String}[])
empty_exp2() = DataFrame(geometry = String[], n = Int[], d = Int[], nq = Int[], k = Int[],
    method = String[], build_s = Float64[], qps = Float64[], recall_ann = Float64[],
    frac_evaluated = Union{Missing,Float64}[], nlevels = Union{Missing,Int}[])
empty_exp3() = DataFrame(config = String[], n = Int[], d = Int[], run = Int[],
    method = String[], op = String[], seq = Int[], latency_s = Union{Missing,Float64}[],
    nevals = Union{Missing,Int}[], rewired = Union{Missing,Int}[],
    top_level = Union{Missing,Int}[], note = String[])
const SCHEMA_EXP1 = names(empty_exp1())
const SCHEMA_EXP2 = names(empty_exp2())
const SCHEMA_EXP3 = names(empty_exp3())

include("datasets.jl")

end # module
```

`bench/src/datasets.jl`:

```julia
# Datasets: ann-benchmarks HDF5 loader, synthetic generators, exact ground truth.
export Dataset, data_dir, load_hdf5, smoke_subset, columns, dist64, ground_truth,
       gaussian, blob_centres, blobs, synthetic, draw_index,
       SEED_GAUSS_TRAIN, SEED_GAUSS_QUERY, SEED_BLOBS_CENTRES, SEED_BLOBS_TRAIN,
       SEED_BLOBS_QUERY, SEED_EXP3_POINTS, SEED_EXP3_DELETE, SEED_EXP3_CHECK, SEED_HNSW

# StableRNG seeds. Every random draw in bench/ uses one of these, plus the dimension d
# when the draw depends on d (seeds are 1000 apart and d ≤ 128, so no two collide).
const SEED_GAUSS_TRAIN   = 1000   # Exp.2 Gaussian points
const SEED_GAUSS_QUERY   = 2000   # Exp.2 Gaussian queries
const SEED_BLOBS_CENTRES = 3000   # Exp.2 blob centres
const SEED_BLOBS_TRAIN   = 4000   # Exp.2 blob points
const SEED_BLOBS_QUERY   = 5000   # Exp.2 blob queries
const SEED_EXP3_POINTS   = 6000   # Exp.3 build points (columns 1:n) and inserted points (n+1:n+nops)
const SEED_EXP3_DELETE   = 7000   # Exp.3 choice of deleted ids
const SEED_EXP3_CHECK    = 8000   # Exp.3 exactness-check queries
const SEED_HNSW          = 9000   # Random.seed! before every HNSW.jl build (its levels use the global RNG)

"An ann-benchmarks style dataset. Matrices are d×n (columns are points); ids are 1-based."
struct Dataset
    name::String
    train::Matrix{Float32}
    test::Matrix{Float32}
    neighbors::Matrix{Int}       # kgt × nq
    distances::Matrix{Float64}   # kgt × nq, Euclidean
end

data_dir() = get(ENV, "BENCH_DATA", "/home/coral/working/benchmarks/data")

"""
    load_hdf5(path) -> Dataset

Reads `train`, `test`, `neighbors`, `distances`. h5py stores them row-major (n×d);
HDF5.jl reads the same bytes as d×n, the layout used here. File ids are 0-based.
"""
function load_hdf5(path::AbstractString)
    h5open(path, "r") do f
        train = Float32.(read(f["train"]))
        test = Float32.(read(f["test"]))
        nb = Int.(read(f["neighbors"])) .+ 1
        gd = Float64.(read(f["distances"]))
        size(train, 1) == size(test, 1) || error("$path: train and test dimensions differ")
        (size(nb) == size(gd) && size(nb, 2) == size(test, 2)) ||
            error("$path: neighbors/distances shape does not match the queries")
        Dataset(splitext(basename(path))[1], train, test, nb, gd)
    end
end

"Columns of Q as separate Vector{Float32} (the query form every adapter takes)."
columns(Q::AbstractMatrix) = [Vector{Float32}(Q[:, j]) for j in axes(Q, 2)]

"Euclidean distance accumulated in Float64."
function dist64(a::AbstractVector, b::AbstractVector)
    s = 0.0
    @inbounds for i in eachindex(a, b)
        t = Float64(a[i]) - Float64(b[i])
        s += t * t
    end
    return sqrt(s)
end

"Exact k nearest neighbours of every column of Q by Float64 brute force; ties by smaller id."
function ground_truth(X::AbstractMatrix, Q::AbstractMatrix, kgt::Int)
    n = size(X, 2)
    kk = min(kgt, n)
    nb = Matrix{Int}(undef, kk, size(Q, 2))
    gd = Matrix{Float64}(undef, kk, size(Q, 2))
    dist = Vector{Float64}(undef, n)
    for j in axes(Q, 2)
        q = view(Q, :, j)
        for i in 1:n
            dist[i] = dist64(view(X, :, i), q)
        end
        p = partialsort!(collect(1:n), 1:kk; by = i -> (dist[i], i))
        nb[:, j] = p
        gd[:, j] = dist[p]
    end
    return nb, gd
end

"First n train points and first nq queries, ground truth recomputed (smoke runs)."
function smoke_subset(ds::Dataset, n::Int, nq::Int; kgt::Int = 10)
    X = ds.train[:, 1:min(n, size(ds.train, 2))]
    Q = ds.test[:, 1:min(nq, size(ds.test, 2))]
    nb, gd = ground_truth(X, Q, kgt)
    return Dataset(ds.name, X, Q, nb, gd)
end

"d×n i.i.d. N(0, I_d) points, drawn in Float64 (StableRNG-stable) and stored as Float32."
gaussian(n::Int, d::Int, seed::Integer) = Float32.(randn(StableRNG(seed), d, n))

"100 centres uniform in [-10, 10]^d, as the ann-benchmarks random-xs generator (make_blobs)."
blob_centres(d::Int, seed::Integer; ncentres::Int = 100, box::Float64 = 10.0) =
    rand(StableRNG(seed), d, ncentres) .* (2box) .- box

"n points; point i is N(centre ((i-1) mod m) + 1, I_d), so every centre gets the same count."
function blobs(n::Int, C::AbstractMatrix{Float64}, seed::Integer)
    d, m = size(C)
    Z = randn(StableRNG(seed), d, n)
    X = Matrix{Float32}(undef, d, n)
    for i in 1:n
        X[:, i] .= Float32.(view(C, :, (i - 1) % m + 1) .+ view(Z, :, i))
    end
    return X
end

"(train, queries) for an Exp.2 geometry, \"gaussian\" or \"blobs\"."
function synthetic(geometry::AbstractString, n::Int, nq::Int, d::Int)
    if geometry == "gaussian"
        return gaussian(n, d, SEED_GAUSS_TRAIN + d), gaussian(nq, d, SEED_GAUSS_QUERY + d)
    elseif geometry == "blobs"
        C = blob_centres(d, SEED_BLOBS_CENTRES + d)
        return blobs(n, C, SEED_BLOBS_TRAIN + d), blobs(nq, C, SEED_BLOBS_QUERY + d)
    end
    throw(ArgumentError("unknown geometry $(repr(geometry)); expected \"gaussian\" or \"blobs\""))
end

"Uniform index in 1:m using only rand(rng)::Float64, which StableRNGs keeps stable."
draw_index(rng, m::Int) = min(m, 1 + floor(Int, rand(rng) * m))
```

Append the smoke output directory and the run log to `.gitignore`, and make sure the manifest pattern ignores only the repository root's `Manifest.toml`. The index plan writes that pattern; with a bare `Manifest.toml` it would also ignore `bench/Manifest.toml`, which this plan commits as the pin for the paper numbers (`git add` would refuse it and a fresh machine would resolve new versions):

```bash
printf '%s\n' 'bench/results-smoke/' 'bench/run.log' >> .gitignore
sed -i 's|^Manifest\.toml$|/Manifest.toml|' .gitignore   # no-op when the line already has the leading slash
grep -n 'Manifest' .gitignore
git check-ignore -v bench/Manifest.toml || echo "bench/Manifest.toml is not ignored"
```

Expected: the `grep` shows `/Manifest.toml` (with the leading slash) and the last command prints `bench/Manifest.toml is not ignored`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_datasets.jl`
Expected: `Test Summary:` for `datasets` with every test passing and no `Fail` or `Error` column.

- [ ] **Step 6: Commit**

```bash
git add .gitignore bench/Project.toml bench/Manifest.toml bench/src/BenchCore.jl bench/src/datasets.jl bench/test/setup.jl bench/test/runtests.jl bench/test/test_datasets.jl
git commit -m "bench: environment, BenchCore module, dataset loaders and generators" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git ls-files bench/Manifest.toml | grep -q . && echo "bench/Manifest.toml is tracked"
```

Expected: the last line prints `bench/Manifest.toml is tracked`. If it prints nothing, the manifest was not committed: fix `.gitignore` as in Step 4 and `git add -f bench/Manifest.toml` before moving on.

---

### Task 2: recall rules

**Files:**
- Create: `bench/src/recall.jl`
- Modify: `bench/src/BenchCore.jl` (one include line)
- Modify: `bench/test/runtests.jl` (the recall testset lives here)

**Interfaces:**
- Consumes: `dist64`, `ground_truth`, `Dataset` (Task 1).
- Produces: `recall_ann(found_dists, true_dists, k; eps = 1e-3) -> Float64`, `recall_id(found_ids, true_ids, k) -> Float64`, `mean_recalls(X, Q, true_ids, true_dists, results, k) -> (recall_ann::Float64, recall_id::Float64)` with `results::Vector{<:Tuple{Vector{Int},Vector{Float64}}}` (one `(ids, dists)` per query column; only the first `k` distinct returned ids count), and `mean_recalls(ds::Dataset, results, k)`; `distance_match(found_dists, true_dists, k; rtol = 1e-5, atol = 1e-9) -> Float64` (fraction of the k ranks whose sorted found distance equals the true one; the exactness check of assumption A25); `min_match(X, Q, true_dists, results, k) -> Float64` (minimum of `distance_match` over the queries, distances recomputed in Float64 from the returned ids) and `min_match(ds::Dataset, results, k)`.

- [ ] **Step 1: Write the failing test**

Replace `bench/test/runtests.jl` with:

```julia
include(joinpath(@__DIR__, "setup.jl"))

@testset "bench" begin
    include("test_datasets.jl")

    @testset "recall rules (hand-made tie case)" begin
        # truth: ids 10, 20, 30, 40 at distances 1, 2, 2, 3; ids 20 and 30 tie for 2nd place
        tid = [10, 20, 30, 40]
        tdist = [1.0, 2.0, 2.0, 3.0]
        k = 2
        # a correct answer that took the other member of the tie
        @test recall_ann([1.0, 2.0], tdist, k) == 1.0
        @test recall_id([10, 30], tid, k) == 0.5
        # the additive 1e-3 rule: inside counts, outside does not
        @test recall_ann([1.0, 2.0009], tdist, k) == 1.0
        @test recall_ann([1.0, 2.0011], tdist, k) == 0.5
        # fewer than k results
        @test recall_ann([1.0], tdist, k) == 0.5
        @test recall_id([10], tid, k) == 0.5
        # duplicates count once; results beyond k are ignored
        @test recall_id([10, 10], tid, k) == 0.5
        @test recall_ann([1.0, 9.0, 1.5], tdist, k) == 0.5
        @test recall_id([10, 40, 20], tid, k) == 0.5
        @test_throws ArgumentError recall_ann([1.0], tdist, 0)
        @test_throws ArgumentError recall_id([1], tid, 0)
        # mean over queries, distances recomputed in Float64 from the returned ids
        X = Float32[0 1 2 3]
        Q = reshape(Float32[0.4, 2.6], 1, 2)
        nb, gd = ground_truth(X, Q, 2)                       # [1, 2] and [4, 3]
        res = [([1, 2], [0.0, 0.0]), ([4, 2], [0.0, 0.0])]   # reported distances are ignored
        ra, ri = mean_recalls(X, Q, nb, gd, res, 2)
        @test ra == 0.75 && ri == 0.75
        # a duplicated returned id is credited once by both rules, even with the right distance
        resdup = [([1, 1], [0.0, 0.0]), ([4, 3], [0.0, 0.0])]
        ra2, ri2 = mean_recalls(X, Q, nb, gd, resdup, 2)
        @test ra2 == 0.75 && ri2 == 0.75                     # query 1: 0.5, query 2: 1.0
    end

    @testset "distance_match: the multiset check the additive rule cannot make" begin
        tdist = [1.0, 2.0, 2.0, 3.0]
        @test distance_match([1.0, 2.0], tdist, 2) == 1.0
        @test distance_match([2.0, 1.0], tdist, 2) == 1.0            # order of the found distances does not matter
        @test distance_match([1.0, 2.0009], tdist, 2) == 0.5         # recall_ann calls this 1.0
        @test distance_match([1.0], tdist, 2) == 0.5                 # fewer than k results
        @test distance_match([1.0, 1.0], tdist, 2) == 0.5            # a repeated distance is not a second neighbour
        @test distance_match([1.0, 2.0, 9.0], tdist, 2) == 1.0       # results beyond k are ignored
        @test_throws ArgumentError distance_match([1.0], tdist, 0)
        X = Float32[0 1 2 3]
        Q = reshape(Float32[0.4, 2.6], 1, 2)
        nb, gd = ground_truth(X, Q, 2)
        @test min_match(X, Q, gd, [([1, 2], [0.0, 0.0]), ([4, 3], [0.0, 0.0])], 2) == 1.0
        @test min_match(X, Q, gd, [([1, 2], [0.0, 0.0]), ([4, 2], [0.0, 0.0])], 2) == 0.5
    end
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/runtests.jl`
Expected: the recall testset errors with ``UndefVarError: `recall_ann` not defined``.

- [ ] **Step 3: Write `bench/src/recall.jl` and include it**

`bench/src/recall.jl`:

```julia
# The two recall definitions reported by the harness (spec §11).
export recall_ann, recall_id, mean_recalls, distance_match, min_match

"""
    recall_ann(found_dists, true_dists, k; eps = 1e-3)

ann-benchmarks rule: the fraction of the first `k` returned distances that are
`≤ true_dists[k] + eps`. A returned neighbour that ties the k-th distance counts.
"""
function recall_ann(found_dists::AbstractVector{<:Real}, true_dists::AbstractVector{<:Real},
                    k::Int; eps::Real = 1e-3)
    k >= 1 || throw(ArgumentError("k must be ≥ 1, got $k"))
    t = true_dists[k] + eps
    return count(d -> d <= t, Iterators.take(found_dists, k)) / k
end

"Strict id recall: |first k returned ids ∩ first k true ids| / k; a duplicate counts once."
function recall_id(found_ids::AbstractVector{<:Integer}, true_ids::AbstractVector{<:Integer}, k::Int)
    k >= 1 || throw(ArgumentError("k must be ≥ 1, got $k"))
    truth = Set(Iterators.take(true_ids, k))
    return count(in(truth), unique(collect(Iterators.take(found_ids, k)))) / k
end

"""
    mean_recalls(X, Q, true_ids, true_dists, results, k) -> (recall_ann, recall_id)

`results[j] = (ids, dists)` answers query column j. Only the first `k` returned ids are
considered and a repeated id counts once, so an approximate method cannot be over-credited
by returning the same neighbour twice. Distances are recomputed in Float64 from those ids
(as ann-benchmarks recomputes them), so every method is judged with the same kernel.
"""
function mean_recalls(X::AbstractMatrix, Q::AbstractMatrix, true_ids::AbstractMatrix{<:Integer},
                      true_dists::AbstractMatrix{<:Real}, results, k::Int)
    ra = 0.0
    ri = 0.0
    for (j, (ids, _)) in enumerate(results)
        q = view(Q, :, j)
        first_k = unique(collect(Iterators.take(ids, k)))     # keep the first k, drop repeats
        rd = Float64[dist64(view(X, :, i), q) for i in first_k]
        ra += recall_ann(rd, view(true_dists, :, j), k)
        ri += recall_id(first_k, view(true_ids, :, j), k)
    end
    return ra / length(results), ri / length(results)
end

mean_recalls(ds::Dataset, results, k::Int) =
    mean_recalls(ds.train, ds.test, ds.neighbors, ds.distances, results, k)

"""
    distance_match(found_dists, true_dists, k; rtol = 1e-5, atol = 1e-9) -> Float64

Exactness check for exact methods: sort the first `k` found distances and compare them
rank by rank with `true_dists[1:k]`. Returns the fraction of ranks that agree within
`rtol`/`atol`; fewer than `k` found distances leave the missing ranks unmatched. Unlike
`recall_ann`, whose additive `1e-3` slack accepts the (k+1)-th neighbour when consecutive
neighbour distances differ by less than that (low d, small distance scales), this fails on
any wrong neighbour. A tie may return either point: only the distances are compared.
"""
function distance_match(found_dists::AbstractVector{<:Real}, true_dists::AbstractVector{<:Real},
                        k::Int; rtol::Real = 1e-5, atol::Real = 1e-9)
    k >= 1 || throw(ArgumentError("k must be ≥ 1, got $k"))
    f = sort!(Float64[x for x in Iterators.take(found_dists, k)])
    matched = count(r -> r <= length(f) && isapprox(f[r], true_dists[r]; rtol = rtol, atol = atol), 1:k)
    return matched / k
end

"""
    min_match(X, Q, true_dists, results, k) -> Float64

Smallest `distance_match` over the queries; `results[j] = (ids, dists)` answers query
column j and the distances are recomputed in Float64 from the returned ids. Exact methods
must return `1.0`; the experiments fail otherwise.
"""
function min_match(X::AbstractMatrix, Q::AbstractMatrix, true_dists::AbstractMatrix{<:Real},
                   results, k::Int)
    worst = 1.0
    for (j, (ids, _)) in enumerate(results)
        q = view(Q, :, j)
        rd = Float64[dist64(view(X, :, i), q) for i in Iterators.take(ids, k)]
        worst = min(worst, distance_match(rd, view(true_dists, :, j), k))
    end
    return worst
end

min_match(ds::Dataset, results, k::Int) = min_match(ds.train, ds.test, ds.distances, results, k)
```

In `bench/src/BenchCore.jl`, add `include("recall.jl")` on the line after `include("datasets.jl")`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/runtests.jl`
Expected: `Test Summary:` for `bench` with every test passing, no `Fail` or `Error`.

- [ ] **Step 5: Commit**

```bash
git add bench/src/recall.jl bench/src/BenchCore.jl bench/test/runtests.jl
git commit -m "bench: ann-benchmarks and strict id recall rules with tie-case tests" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: timing protocol and env.txt

**Files:**
- Create: `bench/src/timing.jl`, `bench/test/test_timing.jl`
- Modify: `bench/src/BenchCore.jl` (one include line), `bench/test/runtests.jl` (one include line)

**Interfaces:**
- Consumes: `BENCH_DIR`, `is_smoke`, the `SEED_*` constants (Task 1).
- Produces: `check_protocol() -> nothing` (errors unless `Threads.nthreads() == 1`; sets `BLAS.set_num_threads(1)`), `best_of(f; passes = 3) -> (last_result, seconds::Float64)`, `time_queries(f, qs; passes = 3) -> (results::Vector, seconds::Float64)` where `f(q)` answers one query, `write_env(path; manifest) -> path` (writes `julia_version`, `manifest_sha256`, `cpu_model`, `governor`, `cpus_allowed`, `threads`, `blas_threads`, `hnng_commit`, `dirty`, `smoke`, `seeds`, `pkg_versions`, `kernel`, `date`, one `key=value` per line).

- [ ] **Step 1: Write the failing test**

`bench/test/test_timing.jl`:

```julia
include(joinpath(@__DIR__, "setup.jl"))
using LinearAlgebra

@testset "timing protocol" begin
    @testset "check_protocol accepts -t 1 and pins BLAS to one thread" begin
        @test Threads.nthreads() == 1
        BLAS.set_num_threads(2)
        @test check_protocol() === nothing
        @test BLAS.get_num_threads() == 1
    end

    @testset "check_protocol refuses more than one thread" begin
        core = joinpath(BENCH_DIR, "src", "BenchCore.jl")
        script = "include(raw\"$core\"); using .BenchCore; check_protocol()"
        cmd = `$(Base.julia_cmd()) -t 2 --project=$(BENCH_DIR) -e $script`
        err = IOBuffer()
        proc = run(pipeline(ignorestatus(cmd); stdout = devnull, stderr = err))
        @test !success(proc)
        # the refusal message, so a child that merely failed to load BenchCore does not pass
        @test occursin("needs `julia -t 1`", String(take!(err)))
    end

    @testset "best_of keeps the fastest of three passes" begin
        durations = [0.06, 0.02, 0.04]
        calls = Ref(0)
        spans = Float64[]                 # the time each pass really took
        out, best = best_of() do
            calls[] += 1
            t0 = time_ns()
            sleep(durations[calls[]])
            push!(spans, (time_ns() - t0) / 1e9)
            calls[]
        end
        @test calls[] == 3 && out == 3
        # the fastest pass is the 0.02 s one; no tight upper bound (a loaded CI machine oversleeps)
        @test 0.019 <= best <= 0.2
        @test best <= minimum(spans[[1, 3]])
    end

    @testset "time_queries: one warm-up call, then three timed passes" begin
        calls = Ref(0)
        f = q -> (calls[] += 1; ([round(Int, q[1])], [0.0]))
        qs = [Float32[i] for i in 1:5]
        results, best = time_queries(f, qs)
        @test calls[] == 1 + 3 * 5
        @test [r[1][1] for r in results] == 1:5
        @test best >= 0
    end

    @testset "env.txt records the run environment" begin
        check_protocol()
        mktempdir() do dir
            p = write_env(joinpath(dir, "env.txt"))
            txt = read(p, String)
            for key in ("julia_version=", "manifest_sha256=", "cpu_model=", "governor=",
                        "cpus_allowed=", "threads=1", "blas_threads=1", "hnng_commit=", "date=",
                        "dirty=", "smoke=", "seeds=", "pkg_versions=", "kernel=")
                @test occursin(key, txt)
            end
            @test occursin("julia_version=$(VERSION)", txt)
            @test match(r"manifest_sha256=[0-9a-f]{64}", txt) !== nothing
            @test occursin("smoke=true", txt)                       # the test preamble sets BENCH_SMOKE=1
            @test occursin("seeds=GAUSS_TRAIN=1000,", txt) && occursin(",HNSW=9000", txt)
            @test match(r"^dirty=(true|false|unknown)$"m, txt) !== nothing
            @test occursin("pkg_versions=", txt) && occursin("NearestNeighbors=", txt)
            @test match(r"^kernel=\S+"m, txt) !== nothing
        end
    end
end
```

In `bench/test/runtests.jl`, add the line `    include("test_timing.jl")` immediately before the final `end`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_timing.jl`
Expected: errors with ``UndefVarError: `check_protocol` not defined``.

- [ ] **Step 3: Write `bench/src/timing.jl` and include it**

`bench/src/timing.jl`:

```julia
# Timing protocol (spec §11) and the env.txt record.
export check_protocol, best_of, time_queries, write_env

"""
    check_protocol()

Refuses to run unless Julia was started with `-t 1`, then sets
`BLAS.set_num_threads(1)`. Every experiment calls it first.
"""
function check_protocol()
    Threads.nthreads() == 1 ||
        error("the benchmark protocol needs `julia -t 1`; this process has $(Threads.nthreads()) threads")
    BLAS.set_num_threads(1)
    BLAS.get_num_threads() == 1 || error("BLAS.set_num_threads(1) did not take effect")
    return nothing
end

"Run `f()` `passes` times; return (result of the last pass, fastest wall time in seconds)."
function best_of(f; passes::Int = 3)
    out = nothing
    best = Inf
    for _ in 1:passes
        t0 = time_ns()
        out = f()
        best = min(best, (time_ns() - t0) / 1e9)
    end
    return out, best
end

"""
    time_queries(f, qs; passes = 3) -> (results, seconds)

One untimed warm-up call `f(qs[1])` (compilation), then `passes` timed passes that
answer every query one at a time. `seconds` is the fastest pass; QPS = length(qs) / seconds.
"""
function time_queries(f, qs::AbstractVector; passes::Int = 3)
    f(first(qs))
    return best_of(passes = passes) do
        [f(q) for q in qs]
    end
end

"Value after the first ':' on the first line of `path` that starts with `prefix`."
function first_line_value(path::AbstractString, prefix::AbstractString)
    isfile(path) || return nothing
    for line in eachline(path)
        startswith(line, prefix) && return strip(split(line, ':'; limit = 2)[2])
    end
    return nothing
end

"""
    write_env(path; manifest = joinpath(BENCH_DIR, "Manifest.toml")) -> path

Records Julia version, the Manifest.toml SHA-256, CPU model, cpufreq governor (when the
sysfs file exists), allowed CPUs (shows the taskset pinning), thread counts, the HNNG.jl
commit, whether the working tree had uncommitted changes outside `bench/results*`
(`dirty`), whether this is a smoke run, the seeds, the versions of the benchmarked
packages, the kernel and the date, one `key=value` per line.
"""
function write_env(path::AbstractString; manifest::AbstractString = joinpath(BENCH_DIR, "Manifest.toml"))
    cpu = something(first_line_value("/proc/cpuinfo", "model name"), "unknown")
    gov_file = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    governor = isfile(gov_file) ? strip(read(gov_file, String)) : "not available"
    allowed = something(first_line_value("/proc/self/status", "Cpus_allowed_list"), "unknown")
    mhash = isfile(manifest) ? bytes2hex(open(sha256, manifest)) : "no-manifest"
    root = dirname(BENCH_DIR)
    commit = try
        strip(read(`git -C $root rev-parse HEAD`, String))
    catch
        "unknown"
    end
    dirty = try
        status = read(`git -C $root status --porcelain -- . ":(exclude)bench/results" ":(exclude)bench/results-smoke" ":(exclude)bench/run.log"`, String)
        string(!isempty(strip(status)))
    catch
        "unknown"
    end
    seeds = join(("GAUSS_TRAIN=$SEED_GAUSS_TRAIN", "GAUSS_QUERY=$SEED_GAUSS_QUERY",
                  "BLOBS_CENTRES=$SEED_BLOBS_CENTRES", "BLOBS_TRAIN=$SEED_BLOBS_TRAIN",
                  "BLOBS_QUERY=$SEED_BLOBS_QUERY", "EXP3_POINTS=$SEED_EXP3_POINTS",
                  "EXP3_DELETE=$SEED_EXP3_DELETE", "EXP3_CHECK=$SEED_EXP3_CHECK",
                  "HNSW=$SEED_HNSW"), ",")
    versions = Dict(info.name => info.version for (_, info) in Pkg.dependencies())
    pkgs = join(("$name=$(get(versions, name, "absent"))" for name in
                 ("HNSW", "NearestNeighbors", "HDF5", "SimilaritySearch", "CairoMakie", "DataFrames")), ",")
    kernel = try
        "$(Sys.KERNEL) $(strip(read(`uname -r`, String)))"
    catch
        string(Sys.KERNEL)
    end
    lines = [
        "julia_version=$(VERSION)",
        "manifest_sha256=$mhash",
        "cpu_model=$cpu",
        "governor=$governor",
        "cpus_allowed=$allowed",
        "threads=$(Threads.nthreads())",
        "blas_threads=$(BLAS.get_num_threads())",
        "hnng_commit=$commit",
        "dirty=$dirty",
        "smoke=$(is_smoke())",
        "seeds=$seeds",
        "pkg_versions=$pkgs",
        "kernel=$kernel",
        "date=$(Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"))",
    ]
    mkpath(dirname(path))
    write(path, join(lines, "\n") * "\n")
    return path
end
```

In `bench/src/BenchCore.jl`, add `include("timing.jl")` on the line after `include("recall.jl")`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_timing.jl`
Expected: `Test Summary:` for `timing protocol` with every test passing (the subprocess test takes a few seconds while the child loads BenchCore and fails).

- [ ] **Step 5: Commit**

```bash
git add bench/src/timing.jl bench/src/BenchCore.jl bench/test/test_timing.jl bench/test/runtests.jl
git commit -m "bench: single-thread protocol check, best-of-three timing, env.txt" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: method interface and the exact methods

**Files:**
- Create: `bench/src/baselines.jl`, `bench/test/test_baselines.jl`
- Modify: `bench/src/BenchCore.jl` (one include line), `bench/test/runtests.jl` (one include line)

**Interfaces:**
- Consumes: `dist64`, `synthetic`, `gaussian`, `ground_truth`, `columns` (Task 1); `mean_recalls`, `min_match` (Task 2); HNNG API (header table).
- Produces: `abstract type BenchMethod`; `method_name(m)::String`; `build(m, X::Matrix{Float32}) -> handle`; `query(handle, q::AbstractVector{Float32}, k::Int) -> (Vector{Int}, Vector{Float64})` sorted by (distance, id), at most `k` entries; `build_time(m, X) -> (handle, seconds)`; `canonical(ids, dists) -> (Vector{Int}, Vector{Float64})` (returns its input converted when it is already sorted by (distance, id), sorts only otherwise); methods `HNNGIncremental()`, `HNNGBulk()` (handle `HNNGHandle` with field `idx`), `BruteBLAS()` (`BruteHandle`), `KDTreeNN(; leafsize = 25)`, `BallTreeNN(; leafsize = 25)` (`TreeHandle`); `fraction_evaluated(h::HNNGHandle, qs, k) -> Float64`.

- [ ] **Step 1: Write the failing test**

`bench/test/test_baselines.jl`:

```julia
include(joinpath(@__DIR__, "setup.jl"))

@testset "baselines (exact methods)" begin
    exact = (HNNGIncremental(), HNNGBulk(), BruteBLAS(), KDTreeNN(), BallTreeNN())
    @test [method_name(m) for m in exact] ==
          ["hNNG-incremental", "hNNG-bulk", "brute-BLAS", "KD-tree", "ball-tree"]

    @testset "exact on $geometry, d = $d" for (geometry, d) in (("gaussian", 2), ("gaussian", 8), ("blobs", 8))
        X, Q = synthetic(geometry, 300, 20, d)
        nb, gd = ground_truth(X, Q, 10)
        for m in exact
            h, t = build_time(m, X)
            @test t > 0
            res = [query(h, q, 10) for q in columns(Q)]
            @test all(length(r[1]) == 10 for r in res)
            @test all(r[1] isa Vector{Int} && r[2] isa Vector{Float64} for r in res)
            @test all(issorted(collect(zip(r[2], r[1]))) for r in res)
            ra, _ = mean_recalls(X, Q, nb, gd, res, 10)
            @test ra == 1.0
            @test min_match(X, Q, gd, res, 10) == 1.0       # all ten distances, not only the additive rule
        end
    end

    @testset "canonical sorts by (distance, id) only when needed" begin
        @test canonical([5, 3, 9], [1.0, 1.0, 0.5]) == ([9, 3, 5], [0.5, 1.0, 1.0])
        @test canonical([9, 3], [1.0, 1.0]) == ([3, 9], [1.0, 1.0])        # tie in distance: smaller id first
        ids = [3, 5, 9]
        cid, cds = canonical(ids, Float32[0.5, 1.0, 1.0])                  # already sorted: no reordering,
        @test cid === ids && cds == [0.5, 1.0, 1.0] && cds isa Vector{Float64}   # ids passed through, dists widened
        @test canonical(Int[], Float64[]) == (Int[], Float64[])
    end

    @testset "hNNG fraction evaluated" begin
        X, Q = synthetic("blobs", 300, 20, 2)
        h = build(HNNGBulk(), X)
        ρ = fraction_evaluated(h, columns(Q), 10)
        @test 0 < ρ <= 1
        @test length(h.idx) == 300
    end

    @testset "brute force stays exact under Float32 cancellation" begin
        # common offset 1000: ‖x‖² ≈ 8e6 while neighbour distances² are O(10)
        X = gaussian(500, 8, 11) .+ 1000f0
        Q = gaussian(20, 8, 12) .+ 1000f0
        nb, gd = ground_truth(X, Q, 10)
        h = build(BruteBLAS(), X)
        for (j, q) in enumerate(columns(Q))
            ids, ds = query(h, q, 10)
            @test ids == nb[:, j]
            @test ds == gd[:, j]
        end
    end

    @testset "k larger than n returns every point" begin
        X, Q = synthetic("gaussian", 7, 1, 3)
        for m in exact
            ids, _ = query(build(m, X), columns(Q)[1], 10)
            @test sort(ids) == 1:7
        end
    end
end
```

In `bench/test/runtests.jl`, add `    include("test_baselines.jl")` immediately before the final `end`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_baselines.jl`
Expected: errors with ``UndefVarError: `HNNGIncremental` not defined``.

- [ ] **Step 3: Write `bench/src/baselines.jl` and include it**

`bench/src/baselines.jl`:

```julia
# One method interface over hNNG and every baseline (spec §11).
export BenchMethod, method_name, build, query, build_time, canonical,
       HNNGIncremental, HNNGBulk, HNNGHandle, fraction_evaluated,
       BruteBLAS, BruteHandle, KDTreeNN, BallTreeNN, TreeHandle

"""
A benchmarked method. Interface:
- `method_name(m) -> String`: the CSV `method` value
- `build(m, X::Matrix{Float32}) -> handle`: X is d×n, columns are points, ids are column numbers
- `query(handle, q, k) -> (ids::Vector{Int}, dists::Vector{Float64})`: sorted by (distance, id)
- `build_time(m, X) -> (handle, seconds)`
"""
abstract type BenchMethod end

"True when (dists, ids) is already ordered by (distance, id): the common case, checked without allocating."
function is_canonical(ids::AbstractVector, dists::AbstractVector)
    @inbounds for j in 2:length(ids)
        dists[j-1] < dists[j] || (dists[j-1] == dists[j] && ids[j-1] <= ids[j]) || return false
    end
    return true
end

"""
Order by (distance, id) and convert to (Vector{Int}, Vector{Float64}). NearestNeighbors.jl,
HNSW.jl and SimilaritySearch already return sorted results, so the sort (which allocates)
runs only when `is_canonical` fails, e.g. on ties; the timed query path of the baselines is
then not charged for it.
"""
function canonical(ids::AbstractVector, dists::AbstractVector)
    if !is_canonical(ids, dists)
        p = sort(collect(eachindex(ids)); by = j -> (dists[j], ids[j]))
        ids, dists = ids[p], dists[p]
    end
    return convert(Vector{Int}, ids), convert(Vector{Float64}, dists)
end

"""
    build_time(m, X) -> (handle, seconds)

Untimed warm-up build on the first min(n, 200) columns (compiles the build path), a full
GC, then one timed full build (spec assumption A11).
"""
function build_time(m::BenchMethod, X::Matrix{Float32})
    build(m, X[:, 1:min(size(X, 2), 200)])
    GC.gc()
    t0 = time_ns()
    h = build(m, X)
    return h, (time_ns() - t0) / 1e9
end

# ---- hNNG ---------------------------------------------------------------------------
struct HNNGIncremental <: BenchMethod end
struct HNNGBulk <: BenchMethod end
struct HNNGHandle{I}
    idx::I
end
method_name(::HNNGIncremental) = "hNNG-incremental"
method_name(::HNNGBulk) = "hNNG-bulk"
new_hnng(d::Int) = HNNG.HNNGIndex(Euclidean(), d; T = Float32)

function build(::HNNGIncremental, X::Matrix{Float32})
    idx = new_hnng(size(X, 1))
    HNNG.add!(idx, X)                 # streaming inserts, one column at a time, in order
    return HNNGHandle(idx)
end
function build(::HNNGBulk, X::Matrix{Float32})
    idx = new_hnng(size(X, 1))
    HNNG.bulk_add!(idx, X)
    return HNNGHandle(idx)
end
query(h::HNNGHandle, q::AbstractVector{Float32}, k::Int) = HNNG.knn(h.idx, q, k)

"Mean over queries of (distance evaluations of the query) / (live elements)."
fraction_evaluated(h::HNNGHandle, qs, k::Int) =
    mean(HNNG.knn_counted(h.idx, q, k)[3] for q in qs) / length(h.idx)

# ---- brute force: Gram trick with gemv, exact Float64 re-rank -----------------------
struct BruteBLAS <: BenchMethod end
method_name(::BruteBLAS) = "brute-BLAS"
struct BruteHandle
    X::Matrix{Float32}
    n2::Vector{Float32}     # ‖x_i‖²
    maxn2::Float32
    s::Vector{Float32}      # Xᵀq
    a::Vector{Float32}      # ‖x_i‖² − 2 x_iᵀq  (= ‖x_i − q‖² − ‖q‖²)
    buf::Vector{Float32}
end
function build(::BruteBLAS, X::Matrix{Float32})
    n = size(X, 2)
    n2 = vec(sum(abs2, X; dims = 1))
    return BruteHandle(X, n2, maximum(n2; init = 0f0), zeros(Float32, n), zeros(Float32, n), zeros(Float32, n))
end
"""
Scores every point with one BLAS gemv. Candidates are all points whose Float32 score lies
within τ = max(1e-4, 2·d·eps(Float32))·(‖q‖² + max‖x‖²) of the k-th score, a bound on the
score's rounding; candidates are re-ranked with the Float64 kernel, so the answer is exact.
"""
function query(h::BruteHandle, q::AbstractVector{Float32}, k::Int)
    X = h.X
    d, n = size(X)
    kk = min(k, n)
    mul!(h.s, transpose(X), q)                     # BLAS sgemv
    @inbounds for i in 1:n
        h.a[i] = h.n2[i] - 2f0 * h.s[i]
    end
    copyto!(h.buf, h.a)
    t = partialsort!(h.buf, kk)
    τ = max(1f-4, 2f0 * d * eps(Float32)) * (sum(abs2, q) + h.maxn2)
    cand = findall(<=(t + τ), h.a)
    ds = Float64[dist64(view(X, :, i), q) for i in cand]
    p = first(sort(collect(eachindex(cand)); by = j -> (ds[j], cand[j])), kk)
    return cand[p], ds[p]
end

# ---- NearestNeighbors.jl trees --------------------------------------------------------
Base.@kwdef struct KDTreeNN <: BenchMethod
    leafsize::Int = 25            # NearestNeighbors.jl default
end
Base.@kwdef struct BallTreeNN <: BenchMethod
    leafsize::Int = 25
end
method_name(::KDTreeNN) = "KD-tree"
method_name(::BallTreeNN) = "ball-tree"
struct TreeHandle{T}
    tree::T
    n::Int
end
build(m::KDTreeNN, X::Matrix{Float32}) =
    TreeHandle(NN.KDTree(X, Euclidean(); leafsize = m.leafsize), size(X, 2))
build(m::BallTreeNN, X::Matrix{Float32}) =
    TreeHandle(NN.BallTree(X, Euclidean(); leafsize = m.leafsize), size(X, 2))
function query(h::TreeHandle, q::AbstractVector{Float32}, k::Int)
    ids, ds = NN.knn(h.tree, q, min(k, h.n))
    return canonical(ids, ds)
end
```

In `bench/src/BenchCore.jl`, add `include("baselines.jl")` on the line after `include("timing.jl")`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_baselines.jl`
Expected: `Test Summary:` for `baselines (exact methods)` with every test passing. If an hNNG row fails `ra == 1.0`, that is an index bug: stop and report it against the index plan; do not loosen the test.

- [ ] **Step 5: Commit**

```bash
git add bench/src/baselines.jl bench/src/BenchCore.jl bench/test/test_baselines.jl bench/test/runtests.jl
git commit -m "bench: method interface with hNNG, BLAS brute force, KD-tree and ball tree" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: HNSW.jl and optional SimilaritySearch adapters

**Files:**
- Modify: `bench/src/baselines.jl` (append), `bench/test/runtests.jl` (one include line)
- Create: `bench/test/test_baselines_approx.jl`

**Interfaces:**
- Consumes: `BenchMethod`, `canonical`, `build_time` (Task 4); `SEED_HNSW`, `HAS_HNSW`, `HAS_SS` (Task 1).
- Produces: `HNSWjl(; M = 16, efConstruction = 200, ef = 10)`, `HNSWHandle`, `build_hnsw(m::HNSWjl, P::Matrix{Float32}; nbuild = size(P, 2)) -> HNSWHandle`, `set_ef!(h::HNSWHandle, ef::Int)`, `hnsw_insert!(h::HNSWHandle, i::Int)`; `SSSearchGraph()`, `SSExhaustive()`, `SSHandle`; `optional_methods() -> Vector{BenchMethod}` (empty unless `HAS_SS`).

- [ ] **Step 1: Write the failing test**

`bench/test/test_baselines_approx.jl`:

```julia
include(joinpath(@__DIR__, "setup.jl"))

@testset "baselines (approximate and optional)" begin
    X, Q = synthetic("blobs", 300, 20, 8)
    nb, gd = ground_truth(X, Q, 10)
    qs = columns(Q)

    if HAS_HNSW
        @testset "HNSW.jl adapter" begin
            m = HNSWjl()
            @test (m.M, m.efConstruction, m.ef) == (16, 200, 10)
            @test method_name(m) == "HNSW"
            h, t = build_time(m, X)
            @test t > 0
            set_ef!(h, 200)
            res = [query(h, q, 10) for q in qs]
            @test all(r[1] isa Vector{Int} && issorted(collect(zip(r[2], r[1]))) for r in res)
            ra, _ = mean_recalls(X, Q, nb, gd, res, 10)
            @test ra >= 0.95
        end

        @testset "HNSW.jl online insert of columns already in its data vector" begin
            h = build_hnsw(HNSWjl(ef = 50), X; nbuild = 250)
            q300 = X[:, 300]
            @test 300 ∉ query(h, q300, 5)[1]
            for i in 251:300
                hnsw_insert!(h, i)
            end
            ids, ds = query(h, q300, 5)
            @test ids[1] == 300 && ds[1] == 0.0
        end
    else
        @info "HNSW.jl not loaded; HNSW rows are skipped"
        @test_skip HAS_HNSW
    end

    @testset "SimilaritySearch rows (optional)" begin
        @test (length(optional_methods()) == 2) == HAS_SS
        if HAS_SS
            @test method_name.(optional_methods()) == ["SS-SearchGraph", "SS-Exhaustive"]
            for (m, minrecall) in ((SSExhaustive(), 1.0), (SSSearchGraph(), 0.9))
                h, _ = build_time(m, X)
                res = [query(h, q, 10) for q in qs]
                # result convention, checked: only filled slots come back (no padding id),
                # every id is a column of X, and distances are Euclidean, not squared
                @test all(r -> length(r[1]) == 10 && all(i -> 1 <= i <= size(X, 2), r[1]), res)
                @test all(j -> all(isapprox.(res[j][2], [dist64(view(X, :, i), qs[j]) for i in res[j][1]];
                                             rtol = 1e-4, atol = 1e-4)), eachindex(res))
                ra, _ = mean_recalls(X, Q, nb, gd, res, 10)
                @test ra >= minrecall
            end
            # exact duplicates come back as separate neighbours (neardup = 0f0 folds them, expand! restores them)
            Xd = hcat(X[:, 1:100], X[:, 1:100])                  # every point twice
            hd = build(SSSearchGraph(), Xd)
            ids, ds = query(hd, X[:, 1], 2)
            @test sort(ids) == [1, 101] && all(<(1e-4), ds)
        else
            @info "SimilaritySearch not loaded; SS rows are skipped"
            @test_skip HAS_SS
        end
    end
end
```

In `bench/test/runtests.jl`, add `    include("test_baselines_approx.jl")` immediately before the final `end`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_baselines_approx.jl`
Expected: errors with ``UndefVarError: `HNSWjl` not defined``.

- [ ] **Step 3: Append the adapters to `bench/src/baselines.jl`**

```julia
export HNSWjl, HNSWHandle, build_hnsw, set_ef!, hnsw_insert!,
       SSSearchGraph, SSExhaustive, SSHandle, optional_methods

# ---- HNSW.jl ------------------------------------------------------------------------
"HNSW.jl with the spec's build parameters; `ef` is the search beam (`set_ef!` changes it)."
Base.@kwdef struct HNSWjl <: BenchMethod
    M::Int = 16
    efConstruction::Int = 200
    ef::Int = 10
end
method_name(::HNSWjl) = "HNSW"
struct HNSWHandle{G}
    g::G
end

"""
    build_hnsw(m, P; nbuild = size(P, 2)) -> HNSWHandle

Hands every column of P to HNSW.jl (so later inserts need no resize) and links columns
1:nbuild into the graph. Seeds the global RNG first: HNSW.jl draws levels from it.
"""
function build_hnsw(m::HNSWjl, P::Matrix{Float32}; nbuild::Int = size(P, 2))
    Random.seed!(SEED_HNSW)
    data = [P[:, i] for i in 1:size(P, 2)]
    g = HNSW.HierarchicalNSW(data; metric = Euclidean(), M = m.M,
                             efConstruction = m.efConstruction, ef = m.ef)
    HNSW.add_to_graph!(g, 1:nbuild)
    return HNSWHandle(g)
end
build(m::HNSWjl, X::Matrix{Float32}) = build_hnsw(m, X)
set_ef!(h::HNSWHandle, ef::Int) = HNSW.set_ef!(h.g, ef)

"HNSW.jl's online insert: link column i (already in the data vector) into the graph."
hnsw_insert!(h::HNSWHandle, i::Int) = HNSW.add_to_graph!(h.g, [i])

function query(h::HNSWHandle, q::AbstractVector{Float32}, k::Int)
    ids, ds = HNSW.knn_search(h.g, convert(Vector{Float32}, q), k)
    return canonical(ids, ds)
end

# ---- SimilaritySearch.jl (optional) -----------------------------------------------------
struct SSSearchGraph <: BenchMethod end
struct SSExhaustive <: BenchMethod end
method_name(::SSSearchGraph) = "SS-SearchGraph"
method_name(::SSExhaustive) = "SS-Exhaustive"
struct SSHandle{I,C}
    index::I
    ctx::C
end

"The SimilaritySearch methods, or none when the package did not load."
optional_methods() = HAS_SS ? BenchMethod[SSSearchGraph(), SSExhaustive()] : BenchMethod[]

# Calls follow the SimilaritySearch.jl searchgraph tutorial (v1.6.5): positional constructors,
# SimilaritySearch's own metric `Dist.L2()` (Euclidean, not squared, like every other method),
# and one query at a time through `search(index, ctx, q, knnqueue(ctx, k))`, read with `IdDistView`.
# `neardup = 0f0` folds exact duplicate points into cluster members and a query then returns one
# representative per duplicate group, which under-counts recall on data with repeated points
# (Fashion-MNIST); `expand!` after each search gives back the raw k nearest, members included.
function build(::SSSearchGraph, X::Matrix{Float32})
    SS = SimilaritySearch
    g = SS.SearchGraph(SS.Dist.L2(), SS.MatrixDatabase(X))
    ctx = SS.SearchGraphContext(; neighborhood = SS.Neighborhood(; neardup = 0f0))
    SS.index!(g, ctx)
    return SSHandle(g, ctx)
end
function build(::SSExhaustive, X::Matrix{Float32})
    SS = SimilaritySearch
    return SSHandle(SS.ExhaustiveSearch(SS.Dist.L2(), SS.MatrixDatabase(X)), SS.GenericContext())
end
function query(h::SSHandle, q::AbstractVector{Float32}, k::Int)
    SS = SimilaritySearch
    qv = convert(Vector{Float32}, q)
    res = SS.knnqueue(h.ctx, k)
    SS.search(h.index, h.ctx, qv, res)
    h.index isa SS.SearchGraph && SS.expand!(h.index, h.ctx, qv, res)   # raw neighbours, duplicates included
    ids = Int[]
    ds = Float64[]
    for p in SS.IdDistView(res)            # only filled slots are visited: no padding convention to decode
        push!(ids, Int(p.id))
        push!(ds, Float64(p.dist))
    end
    return canonical(ids, ds)
end
```

If an `SS.*` call throws with the installed SimilaritySearch, read its docstrings (`julia --project=bench -e 'import SimilaritySearch; @doc SimilaritySearch.SearchGraph'`, likewise `search`, `knnqueue`, `expand!`, `ExhaustiveSearch`) and adjust only these two `build` methods and `query(::SSHandle, ...)` until the optional testset passes; keep the three properties the testset checks (Euclidean distances, ids in `1:n` with no padding, duplicate points returned separately). The experiments skip SS rows on any error, so this never blocks the other rows.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_baselines_approx.jl`
Expected: `Test Summary:` with every test passing (a `Broken` entry for each skipped test when HNSW.jl or SimilaritySearch is absent).

- [ ] **Step 5: Commit**

```bash
git add bench/src/baselines.jl bench/test/test_baselines_approx.jl bench/test/runtests.jl
git commit -m "bench: HNSW.jl adapter with online insert, optional SimilaritySearch adapters" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Exp.1 and Exp.1b (`exp1.jl`)

**Files:**
- Create: `bench/exp1.jl`, `bench/test/test_exp1.jl`
- Modify: `bench/test/runtests.jl` (one include line)

**Interfaces:**
- Consumes: `load_hdf5`, `smoke_subset`, `data_dir`, `columns`, `Dataset` (Task 1); `mean_recalls`, `min_match` (Task 2); `check_protocol`, `time_queries` (Task 3); every adapter and `optional_methods` (Tasks 4–5); `empty_exp1`, `is_smoke`, `results_dir`, `HAS_HNSW` (Task 1).
- Produces: `run_exp1(; smoke::Bool = is_smoke(), outdir = results_dir(smoke)) -> nothing`, writing `outdir/exp1.csv` (random-xs-20-euclidean only) and `run_exp1b(; smoke, outdir) -> nothing`, writing `outdir/exp1b.csv` (fashion-mnist-784-euclidean), both with schema `SCHEMA_EXP1`; two separate entry points so that `run_all` can run the multi-hour Fashion-MNIST experiment last and skip a finished one (Task 11); `bench_dataset(ds::Dataset; k = 10, efs) -> DataFrame`; constants `EXP1_EFS = [10, 20, 40, 80, 160, 320, 640]`, `EXP1_EFS_SMOKE = [10, 40]`.

Smoke sizes: first 300 train points and first 20 queries of each file, ground truth recomputed; `ef ∈ {10, 40}`. For the five exact methods `bench_dataset` also requires `min_match == 1.0` (assumption A25), logs `min_match=<value>` per method and errors otherwise.

- [ ] **Step 1: Write the failing test**

`bench/test/test_exp1.jl`:

```julia
include(joinpath(@__DIR__, "setup.jl"))
using CSV, DataFrames
isdefined(Main, :run_exp1) || include(joinpath(BENCH_DIR, "exp1.jl"))

@testset "Exp.1 / Exp.1b smoke" begin
    mktempdir() do out
        run_exp1(; outdir = out)
        @test !isfile(joinpath(out, "exp1b.csv"))       # run_exp1 is random-xs only
        run_exp1b(; outdir = out)
        core = ["hNNG-incremental", "hNNG-bulk", "brute-BLAS", "KD-tree", "ball-tree"]
        for (file, name) in (("exp1.csv", "random-xs-20-euclidean"), ("exp1b.csv", "fashion-mnist-784-euclidean"))
            df = CSV.read(joinpath(out, file), DataFrame)
            @test names(df) == SCHEMA_EXP1
            @test all(df.dataset .== name)
            @test all(df.n .== 300) && all(df.nq .== 20) && all(df.k .== 10)
            @test issubset(core, df.method)
            @test count(==("HNSW"), df.method) == (HAS_HNSW ? length(EXP1_EFS_SMOKE) : 0)
            HAS_HNSW && @test sort(string.(df[df.method .== "HNSW", :param])) == ["ef=10", "ef=40"]
            @test all(df.qps .> 0) && all(df.build_s .> 0)
            exact = df[in.(df.method, Ref(core)), :]
            @test all(exact.recall_ann .== 1.0)
            h = df[startswith.(df.method, "hNNG"), :]
            @test all(0 .< h.frac_evaluated .<= 1) && all(h.nlevels .>= 1)
            @test all(occursin.(r"^\d+(;\d+)*$", string.(h.nclusters)))
        end
    end
end
```

In `bench/test/runtests.jl`, add `    include("test_exp1.jl")` immediately before the final `end`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_exp1.jl`
Expected: FAIL with `SystemError: opening file ".../bench/exp1.jl": No such file or directory`.

- [ ] **Step 3: Write `bench/exp1.jl`**

```julia
# Exp.1 (random-xs-20-euclidean) and Exp.1b (fashion-mnist-784-euclidean), spec §11.
# Run: julia -t 1 --project=bench bench/exp1.jl        (BENCH_SMOKE=1 for the tiny version; runs both)
isdefined(Main, :BenchCore) || include(joinpath(@__DIR__, "src", "BenchCore.jl"))
using .BenchCore
using CSV, DataFrames
import HNNG

const EXP1_EFS = [10, 20, 40, 80, 160, 320, 640]
const EXP1_EFS_SMOKE = [10, 40]
const EXP1_K = 10

function exp1_dataset(name::AbstractString, smoke::Bool)
    ds = load_hdf5(joinpath(data_dir(), name * ".hdf5"))
    return smoke ? smoke_subset(ds, 300, 20) : ds
end

"One row per method on `ds`; HNSW.jl gets one row per `ef`."
function bench_dataset(ds::Dataset; k::Int = EXP1_K, efs = EXP1_EFS)
    df = empty_exp1()
    qs = columns(ds.test)
    base = (dataset = ds.name, n = size(ds.train, 2), d = size(ds.train, 1), nq = length(qs), k = k)
    function row!(name, param, build_s, results, seconds; frac = missing, nlev = missing, ncl = missing)
        ra, ri = mean_recalls(ds, results, k)
        push!(df, (; base..., method = name, param = param, build_s = build_s,
                   qps = length(qs) / seconds, recall_ann = ra, recall_id = ri,
                   frac_evaluated = frac, nlevels = nlev, nclusters = ncl))
    end
    for m in BenchMethod[HNNGIncremental(), HNNGBulk(), BruteBLAS(), KDTreeNN(), BallTreeNN()]
        h, bt = build_time(m, ds.train)
        results, seconds = time_queries(q -> query(h, q, k), qs)
        mm = min_match(ds, results, k)         # exactness beyond the additive recall rule (A25)
        @info "Exp.1 exactness dataset=$(ds.name) method=$(method_name(m)) min_match=$mm"
        mm == 1.0 || error("Exp.1: $(method_name(m)) is not exact on $(ds.name) (min_match $mm)")
        if h isa HNNGHandle
            s = HNNG.stats(h.idx)
            row!(method_name(m), "", bt, results, seconds;
                 frac = fraction_evaluated(h, qs, k), nlev = s.nlevels, ncl = join(s.nclusters, ";"))
        else
            row!(method_name(m), "", bt, results, seconds)
        end
        @info "Exp.1" ds.name method_name(m) build_s = bt
    end
    if HAS_HNSW
        hh, hbt = build_time(HNSWjl(), ds.train)
        for ef in efs
            set_ef!(hh, ef)
            results, seconds = time_queries(q -> query(hh, q, k), qs)
            row!("HNSW", "ef=$ef", hbt, results, seconds)
        end
    end
    for m in optional_methods()
        try
            hs, sbt = build_time(m, ds.train)
            results, seconds = time_queries(q -> query(hs, q, k), qs)
            row!(method_name(m), "", sbt, results, seconds)
        catch err
            @warn "$(method_name(m)) row skipped on $(ds.name)" exception = err
        end
    end
    return df
end

function run_exp1_file(file::AbstractString, name::AbstractString, smoke::Bool, outdir::AbstractString)
    check_protocol()
    mkpath(outdir)
    df = bench_dataset(exp1_dataset(name, smoke); efs = smoke ? EXP1_EFS_SMOKE : EXP1_EFS)
    CSV.write(joinpath(outdir, file), df)
    return nothing
end

"Exp.1: random-xs-20-euclidean -> exp1.csv."
run_exp1(; smoke::Bool = is_smoke(), outdir::AbstractString = results_dir(smoke)) =
    run_exp1_file("exp1.csv", "random-xs-20-euclidean", smoke, outdir)

"Exp.1b: fashion-mnist-784-euclidean -> exp1b.csv (hours in a full run)."
run_exp1b(; smoke::Bool = is_smoke(), outdir::AbstractString = results_dir(smoke)) =
    run_exp1_file("exp1b.csv", "fashion-mnist-784-euclidean", smoke, outdir)

if abspath(PROGRAM_FILE) == @__FILE__
    run_exp1()
    run_exp1b()
end
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_exp1.jl`
Expected: `Test Summary:` for `Exp.1 / Exp.1b smoke` with every test passing. Also run the script directly once: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/exp1.jl && head -3 bench/results-smoke/exp1.csv && ls bench/results-smoke/exp1b.csv`; the first line printed is `dataset,n,d,nq,k,method,param,build_s,qps,recall_ann,recall_id,frac_evaluated,nlevels,nclusters`.

- [ ] **Step 5: Commit**

```bash
git add bench/exp1.jl bench/test/test_exp1.jl bench/test/runtests.jl
git commit -m "bench: Exp.1 and Exp.1b on random-xs-20 and fashion-mnist-784" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Exp.2 (`exp2.jl`)

**Files:**
- Create: `bench/exp2.jl`, `bench/test/test_exp2.jl`
- Modify: `bench/test/runtests.jl` (one include line)

**Interfaces:**
- Consumes: `synthetic`, `ground_truth`, `columns` (Task 1); `mean_recalls`, `min_match` (Task 2); `check_protocol`, `time_queries` (Task 3); `HNNGBulk`, `KDTreeNN`, `BallTreeNN`, `BruteBLAS`, `build_time`, `query`, `fraction_evaluated`, `HNNGHandle` (Task 4); `empty_exp2` (Task 1).
- Produces: `run_exp2(; smoke::Bool = is_smoke(), outdir = results_dir(smoke)) -> DataFrame`, writing `outdir/exp2.csv` with schema `SCHEMA_EXP2`; `EXP2_DIMS = [2, 4, 8, 16, 32, 64, 128]`.

Smoke sizes: `n = 300` for both geometries, 20 queries, `d ∈ {2, 8}`.

- [ ] **Step 1: Write the failing test**

`bench/test/test_exp2.jl`:

```julia
include(joinpath(@__DIR__, "setup.jl"))
using CSV, DataFrames
isdefined(Main, :run_exp2) || include(joinpath(BENCH_DIR, "exp2.jl"))

@testset "Exp.2 smoke" begin
    mktempdir() do out
        run_exp2(; outdir = out)
        df = CSV.read(joinpath(out, "exp2.csv"), DataFrame)
        @test names(df) == SCHEMA_EXP2
        @test sort(unique(string.(df.geometry))) == ["blobs", "gaussian"]
        @test sort(unique(df.d)) == [2, 8]
        @test nrow(df) == 2 * 2 * 4
        @test sort(unique(string.(df.method))) == ["KD-tree", "ball-tree", "brute-BLAS", "hNNG-bulk"]
        @test all(df.recall_ann .== 1.0) && all(df.qps .> 0) && all(df.n .== 300) && all(df.nq .== 20)
        h = df[df.method .== "hNNG-bulk", :]
        @test nrow(h) == 4 && all(0 .< h.frac_evaluated .<= 1) && all(h.nlevels .>= 1)
        @test all(ismissing, df[df.method .!= "hNNG-bulk", :frac_evaluated])
    end
end
```

In `bench/test/runtests.jl`, add `    include("test_exp2.jl")` immediately before the final `end`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_exp2.jl`
Expected: FAIL with `SystemError: opening file ".../bench/exp2.jl": No such file or directory`.

- [ ] **Step 3: Write `bench/exp2.jl`**

```julia
# Exp.2: exact-search scaling with dimension on Gaussian and blob data, spec §11.
# Run: julia -t 1 --project=bench bench/exp2.jl        (BENCH_SMOKE=1 for the tiny version)
isdefined(Main, :BenchCore) || include(joinpath(@__DIR__, "src", "BenchCore.jl"))
using .BenchCore
using CSV, DataFrames
import HNNG

const EXP2_DIMS = [2, 4, 8, 16, 32, 64, 128]
const EXP2_K = 10

exp2_settings(smoke::Bool) = smoke ?
    (dims = [2, 8], nq = 20, n = Dict("gaussian" => 300, "blobs" => 300)) :
    (dims = EXP2_DIMS, nq = 1000, n = Dict("gaussian" => 5000, "blobs" => 10000))

function run_exp2(; smoke::Bool = is_smoke(), outdir::AbstractString = results_dir(smoke))
    check_protocol()
    s = exp2_settings(smoke)
    df = empty_exp2()
    for geometry in ("gaussian", "blobs"), d in s.dims
        n = s.n[geometry]
        X, Q = synthetic(geometry, n, s.nq, d)
        nb, gd = ground_truth(X, Q, EXP2_K)
        qs = columns(Q)
        for m in BenchMethod[HNNGBulk(), KDTreeNN(), BallTreeNN(), BruteBLAS()]
            h, bt = build_time(m, X)
            results, seconds = time_queries(q -> query(h, q, EXP2_K), qs)
            ra, _ = mean_recalls(X, Q, nb, gd, results, EXP2_K)
            ra == 1.0 || error("Exp.2: $(method_name(m)) is not exact on $geometry d=$d (recall $ra)")
            # at low d the additive recall rule cannot see a (k+1)-th-neighbour answer; compare the distances (A25)
            mm = min_match(X, Q, gd, results, EXP2_K)
            @info "Exp.2 exactness geometry=$geometry d=$d method=$(method_name(m)) min_match=$mm"
            mm == 1.0 || error("Exp.2: $(method_name(m)) returns a wrong neighbour on $geometry d=$d (min_match $mm)")
            frac, nlev = h isa HNNGHandle ?
                (fraction_evaluated(h, qs, EXP2_K), HNNG.nlevels(h.idx)) : (missing, missing)
            push!(df, (geometry = geometry, n = n, d = d, nq = s.nq, k = EXP2_K,
                       method = method_name(m), build_s = bt, qps = s.nq / seconds,
                       recall_ann = ra, frac_evaluated = frac, nlevels = nlev))
        end
        @info "Exp.2" geometry d
    end
    mkpath(outdir)
    CSV.write(joinpath(outdir, "exp2.csv"), df)
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_exp2()
end
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_exp2.jl`
Expected: `Test Summary:` for `Exp.2 smoke` with every test passing.

- [ ] **Step 5: Commit**

```bash
git add bench/exp2.jl bench/test/test_exp2.jl bench/test/runtests.jl
git commit -m "bench: Exp.2 dimension sweep on Gaussian and blob data" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Exp.3 (`exp3.jl`) and the per-operation summary

**Files:**
- Create: `bench/exp3.jl`, `bench/test/test_exp3.jl`
- Modify: `bench/src/BenchCore.jl` (add `op_latency`), `bench/test/runtests.jl` (one include line)

**Interfaces:**
- Consumes: `gaussian`, `dist64`, `draw_index`, `SEED_EXP3_*`, `HAS_HNSW` (Task 1); `check_protocol` (Task 3); `KDTreeNN`, `BallTreeNN`, `build_time`, `method_name` (Task 4); `HNSWjl`, `build_hnsw`, `hnsw_insert!` (Task 5); HNNG API including `HNNG.last_update_stats` (header table).
- Produces: `op_latency(exp3::AbstractDataFrame, config::Union{Nothing,AbstractString}, method, op) -> DataFrame(n::Int, d::Int, latency_s::Float64)` in BenchCore (median over runs of each run's mean latency, rows sorted by `(n, d)`; `config = nothing` takes every configuration); `run_exp3(; smoke::Bool = is_smoke(), outdir = results_dir(smoke)) -> DataFrame`, writing `outdir/exp3.csv` with schema `SCHEMA_EXP3`; `struct Exp3Config(config, n, d, nops, runs, ncheck)`; `exp3_configs(smoke::Bool) -> Vector{Exp3Config}`; `check_exact!(idx, P, alive, Qc, qi, ncheck; k = 10) -> Int`.

Rows written per configuration and run: `hNNG` `rebuild` (streaming `add!` of the n build points) and `rebuild-bulk` (`bulk_add!`), then one `insert` row per insert and one `delete` row per delete with `nevals`, `rewired`, `top_level`; `KD-tree` and `ball-tree` `rebuild`; `brute-BLAS` `update` with latency 0 and note `no update cost`; `HNSW` one `insert` row per insert and one `delete` row with missing latency and note `mark only`.

Configurations: the nine grid points `n ∈ {8000, 16000, 32000} × d ∈ {16, 64, 128}`, each with 200 inserts, 200 deletes, 3 runs and 20 checked queries per operation; `(8000, 16)`, `(32000, 16)`, `(32000, 128)` get `config = "table"`, the rest `config = "grid"`.

Smoke sizes: grid `n ∈ {200, 300} × d ∈ {2, 8}` with table rows `(200, 2)`, `(300, 2)`, `(300, 8)` (the same corners of the grid), 20 inserts and 20 deletes, 3 runs, 5 checked queries per operation.

- [ ] **Step 1: Write the failing test**

`bench/test/test_exp3.jl`:

```julia
include(joinpath(@__DIR__, "setup.jl"))
using CSV, DataFrames
using Distances: Euclidean
import HNNG
isdefined(Main, :run_exp3) || include(joinpath(BENCH_DIR, "exp3.jl"))

@testset "Exp.3" begin
    @testset "op_latency: median over runs of each run's mean" begin
        df = empty_exp3()
        for (run, lats) in enumerate(([1.0, 3.0], [5.0, 7.0], [0.5, 1.5]))   # run means 2, 6, 1
            for (seq, t) in enumerate(lats)
                push!(df, (config = "table", n = 100, d = 4, run = run, method = "hNNG", op = "insert",
                           seq = seq, latency_s = t, nevals = 1, rewired = 0, top_level = 0, note = ""))
            end
        end
        push!(df, (config = "table", n = 100, d = 4, run = 1, method = "HNSW", op = "delete", seq = 1,
                   latency_s = missing, nevals = missing, rewired = missing, top_level = missing, note = "mark only"))
        s = op_latency(df, "table", "hNNG", "insert")
        @test nrow(s) == 1 && s.n[1] == 100 && s.d[1] == 4 && s.latency_s[1] == 2.0
        @test nrow(op_latency(df, "table", "HNSW", "delete")) == 0
        @test nrow(op_latency(df, "grid", "hNNG", "insert")) == 0
        @test nrow(op_latency(df, nothing, "hNNG", "insert")) == 1
    end

    @testset "configurations: the TKDE grid, table rows at its corners" begin
        full = exp3_configs(false)
        @test length(full) == 9
        @test sort([(c.n, c.d) for c in full]) == sort([(n, d) for n in (8000, 16000, 32000) for d in (16, 64, 128)])
        @test sort([(c.n, c.d) for c in full if c.config == "table"]) == [(8000, 16), (32000, 16), (32000, 128)]
        @test all(c -> (c.nops, c.runs, c.ncheck) == (200, 3, 20), full)
        smoke = exp3_configs(true)
        @test length(smoke) == 4
        @test sort([(c.n, c.d) for c in smoke if c.config == "table"]) == [(200, 2), (300, 2), (300, 8)]
    end

    @testset "exactness check catches a wrong answer" begin
        P = gaussian(40, 2, 1)
        idx = HNNG.HNNGIndex(Euclidean(), 2; T = Float32)
        HNNG.add!(idx, P[:, 1:30])
        alive = falses(40)
        alive[1:31] .= true          # claims column 31 is live, but it was never inserted
        Qc = P[:, 31:31]             # a query on column 31: its true nearest distance is 0
        @test_throws ErrorException check_exact!(idx, P, alive, Qc, 0, 1)
        alive[31] = false
        @test check_exact!(idx, P, alive, Qc, 0, 1) == 1
    end

    @testset "smoke run" begin
        mktempdir() do out
            run_exp3(; outdir = out)
            df = CSV.read(joinpath(out, "exp3.csv"), DataFrame)
            @test names(df) == SCHEMA_EXP3
            h = df[df.method .== "hNNG", :]
            for c in exp3_configs(true), run in 1:c.runs
                g = h[(h.config .== c.config) .& (h.n .== c.n) .& (h.d .== c.d) .& (h.run .== run), :]
                @test count(==("insert"), g.op) == c.nops
                @test count(==("delete"), g.op) == c.nops
                @test count(==("rebuild"), g.op) == 1 && count(==("rebuild-bulk"), g.op) == 1
            end
            ins = h[h.op .== "insert", :]
            del = h[h.op .== "delete", :]
            @test all(ins.latency_s .> 0) && all(ins.nevals .> 0) && all(del.nevals .>= 0)
            @test all(vcat(ins.rewired, del.rewired) .>= 0) && all(vcat(ins.top_level, del.top_level) .>= 0)
            hd = df[(df.method .== "HNSW") .& (df.op .== "delete"), :]
            @test nrow(hd) == (HAS_HNSW ? 12 : 0)
            @test all(hd.note .== "mark only") && all(ismissing, hd.latency_s)
            @test nrow(op_latency(df, "table", "hNNG", "insert")) == 3
            @test nrow(op_latency(df, "grid", "hNNG", "insert")) == 1
            @test nrow(op_latency(df, nothing, "hNNG", "rebuild")) == 4
            @test nrow(op_latency(df, nothing, "KD-tree", "rebuild")) == 4
            @test nrow(op_latency(df, nothing, "HNSW", "insert")) == (HAS_HNSW ? 4 : 0)
        end
    end
end
```

(`nrow(hd) == 12`: 4 smoke configurations × 3 runs, one `mark only` row each, when HNSW.jl loads.)

In `bench/test/runtests.jl`, add `    include("test_exp3.jl")` immediately before the final `end`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_exp3.jl`
Expected: FAIL with `SystemError: opening file ".../bench/exp3.jl": No such file or directory`.

- [ ] **Step 3: Add `op_latency` to BenchCore and write `bench/exp3.jl`**

In `bench/src/BenchCore.jl`, insert after the line `const SCHEMA_EXP3 = names(empty_exp3())`:

```julia
export op_latency

"""
    op_latency(exp3, config, method, op) -> DataFrame(n, d, latency_s)

Per (n, d) of `config` (`nothing` = every configuration): the median over runs of each
run's mean latency of `op` by `method` (spec assumption A9). Rows with a missing latency
are ignored; rows sorted by (n, d).
"""
function op_latency(exp3::AbstractDataFrame, config::Union{Nothing,AbstractString},
                    method::AbstractString, op::AbstractString)
    out = DataFrame(n = Int[], d = Int[], latency_s = Float64[])
    inconfig = config === nothing ? trues(nrow(exp3)) : (exp3.config .== config)
    sel = exp3[inconfig .& (exp3.method .== method) .& (exp3.op .== op), :]
    nrow(sel) == 0 && return out
    for g in groupby(sel, [:n, :d]; sort = true)
        per_run = Float64[]
        for r in groupby(g, :run; sort = true)
            vals = collect(skipmissing(r.latency_s))
            isempty(vals) || push!(per_run, mean(vals))
        end
        isempty(per_run) || push!(out, (n = g.n[1], d = g.d[1], latency_s = median(per_run)))
    end
    return out
end
```

`bench/exp3.jl`:

```julia
# Exp.3: incremental insert/delete latency against full rebuilds, spec §11.
# Run: julia -t 1 --project=bench bench/exp3.jl        (BENCH_SMOKE=1 for the tiny version)
isdefined(Main, :BenchCore) || include(joinpath(@__DIR__, "src", "BenchCore.jl"))
using .BenchCore
using CSV, DataFrames, StableRNGs
using Distances: Euclidean
import HNNG

const EXP3_NS = [8000, 16000, 32000]                     # TKDE grid
const EXP3_DS = [16, 64, 128]
const EXP3_TABLE = [(8000, 16), (32000, 16), (32000, 128)]  # tab_ops rows
const EXP3_K = 10

struct Exp3Config
    config::String   # "table" (also a tab_ops row) or "grid"; the figure uses both
    n::Int
    d::Int
    nops::Int        # inserts, then the same number of deletes
    runs::Int
    ncheck::Int      # brute-force-checked queries after every operation
end

function exp3_configs(smoke::Bool)
    ns, ds, table = smoke ? ([200, 300], [2, 8], [(200, 2), (300, 2), (300, 8)]) :
                            (EXP3_NS, EXP3_DS, EXP3_TABLE)
    nops, runs, ncheck = smoke ? (20, 3, 5) : (200, 3, 20)
    return [Exp3Config((n, d) in table ? "table" : "grid", n, d, nops, runs, ncheck) for d in ds for n in ns]
end

new_index3(d::Int) = HNNG.HNNGIndex(Euclidean(), d; T = Float32)
seconds_since(t0) = (time_ns() - t0) / 1e9

"""
    check_exact!(idx, P, alive, Qc, qi, ncheck; k = 10) -> qi

Answers `ncheck` queries (cycling through the columns of `Qc` after column `qi`) with
`HNNG.knn` and compares them with Float64 brute force over the live columns of `P`.
Throws on any difference beyond rtol 1e-5 or on a dead id. Not timed.
"""
function check_exact!(idx, P::Matrix{Float32}, alive::BitVector, Qc::Matrix{Float32}, qi::Int,
                      ncheck::Int; k::Int = EXP3_K)
    live = findall(alive)
    dist = Vector{Float64}(undef, length(live))
    kk = min(k, length(live))
    for _ in 1:ncheck
        qi = qi % size(Qc, 2) + 1
        q = Qc[:, qi]
        ids, ds = HNNG.knn(idx, q, k)
        for (t, i) in enumerate(live)
            dist[t] = dist64(view(P, :, i), q)
        end
        truth = collect(partialsort!(dist, 1:kk))
        ok = length(ds) == kk && all(isapprox.(ds, truth; rtol = 1e-5)) && all(alive[i] for i in ids)
        ok || error("Exp.3 exactness violated on query $qi: got $ds (ids $ids), expected $truth")
    end
    return qi
end

"Compile every timed path on a tiny index so that run 1 does not time compilation."
function exp3_warmup(d::Int)
    P = gaussian(60, d, SEED_EXP3_POINTS + d)
    idx = new_index3(d)
    HNNG.add!(idx, P[:, 1:50])
    for j in 51:60
        HNNG.add!(idx, P[:, j])
        HNNG.last_update_stats(idx)
    end
    for id in 1:10
        HNNG.delete!(idx, id)
    end
    HNNG.bulk_add!(new_index3(d), P)
    HNNG.knn(idx, P[:, 1], EXP3_K)
    HAS_HNSW && build_hnsw(HNSWjl(), P; nbuild = 50)
    return nothing
end

function exp3_hnng!(df::DataFrame, c::Exp3Config, P::Matrix{Float32}, Qc::Matrix{Float32}, run::Int)
    n, d = c.n, c.d
    base = (config = c.config, n = n, d = d, run = run, method = "hNNG")
    X = P[:, 1:n]
    GC.gc()
    idx = new_index3(d)
    t0 = time_ns()
    HNNG.add!(idx, X)
    t_stream = seconds_since(t0)
    push!(df, (; base..., op = "rebuild", seq = 1, latency_s = t_stream, nevals = HNNG.stats(idx).nevals,
               rewired = missing, top_level = missing, note = "streaming add! of n points"))
    bulk = new_index3(d)
    GC.gc()
    t0 = time_ns()
    HNNG.bulk_add!(bulk, X)
    t_bulk = seconds_since(t0)
    push!(df, (; base..., op = "rebuild-bulk", seq = 1, latency_s = t_bulk, nevals = HNNG.stats(bulk).nevals,
               rewired = missing, top_level = missing, note = "bulk_add! of n points"))
    bulk = nothing
    alive = falses(size(P, 2))
    alive[1:n] .= true
    qi = 0
    for j in 1:c.nops
        x = P[:, n + j]
        HNNG.reset_evals!(idx)
        GC.gc(false)        # outside the timed region: garbage from the untimed check is not collected inside the insert (A26)
        t0 = time_ns()
        id = HNNG.add!(idx, x)
        t = seconds_since(t0)
        id == n + j || error("HNNG gave id $id to column $(n + j)")
        rew, top = HNNG.last_update_stats(idx)
        push!(df, (; base..., op = "insert", seq = j, latency_s = t, nevals = HNNG.stats(idx).nevals,
                   rewired = rew, top_level = top, note = ""))
        alive[id] = true
        qi = check_exact!(idx, P, alive, Qc, qi, c.ncheck)
    end
    rng = StableRNG(SEED_EXP3_DELETE + d)       # same deletes in every run
    for j in 1:c.nops
        live = findall(alive)
        id = live[draw_index(rng, length(live))]
        HNNG.reset_evals!(idx)
        GC.gc(false)        # as for the insert loop (A26)
        t0 = time_ns()
        HNNG.delete!(idx, id)
        t = seconds_since(t0)
        rew, top = HNNG.last_update_stats(idx)
        alive[id] = false
        push!(df, (; base..., op = "delete", seq = j, latency_s = t, nevals = HNNG.stats(idx).nevals,
                   rewired = rew, top_level = top, note = ""))
        qi = check_exact!(idx, P, alive, Qc, qi, c.ncheck)
    end
    length(idx) == n || error("live count $(length(idx)) != $n after $(c.nops) inserts and deletes")
    return df
end

function exp3_baselines!(df::DataFrame, c::Exp3Config, P::Matrix{Float32}, run::Int)
    n, d = c.n, c.d
    base = (config = c.config, n = n, d = d, run = run)
    X = P[:, 1:n]
    for m in BenchMethod[KDTreeNN(), BallTreeNN()]
        _, t = build_time(m, X)
        push!(df, (; base..., method = method_name(m), op = "rebuild", seq = 1, latency_s = t,
                   nevals = missing, rewired = missing, top_level = missing, note = "full rebuild"))
    end
    push!(df, (; base..., method = "brute-BLAS", op = "update", seq = 1, latency_s = 0.0,
               nevals = missing, rewired = missing, top_level = missing, note = "no update cost"))
    HAS_HNSW || return df
    h = build_hnsw(HNSWjl(), P; nbuild = n)
    for j in 1:c.nops
        GC.gc(false)        # same treatment as the hNNG loops (A26)
        t0 = time_ns()
        hnsw_insert!(h, n + j)
        push!(df, (; base..., method = "HNSW", op = "insert", seq = j, latency_s = seconds_since(t0),
                   nevals = missing, rewired = missing, top_level = missing, note = ""))
    end
    push!(df, (; base..., method = "HNSW", op = "delete", seq = 1, latency_s = missing,
               nevals = missing, rewired = missing, top_level = missing, note = "mark only"))
    return df
end

function run_exp3(; smoke::Bool = is_smoke(), outdir::AbstractString = results_dir(smoke))
    check_protocol()
    df = empty_exp3()
    for c in exp3_configs(smoke)
        P = gaussian(c.n + c.nops, c.d, SEED_EXP3_POINTS + c.d)
        Qc = gaussian(2 * c.nops * c.ncheck, c.d, SEED_EXP3_CHECK + c.d)
        exp3_warmup(c.d)
        for run in 1:c.runs
            exp3_hnng!(df, c, P, Qc, run)
            exp3_baselines!(df, c, P, run)
        end
        @info "Exp.3" c.config c.n c.d
    end
    mkpath(outdir)
    CSV.write(joinpath(outdir, "exp3.csv"), df)
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_exp3()
end
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_exp3.jl`
Expected: `Test Summary:` for `Exp.3` with every test passing. An `Exp.3 exactness violated` error during the smoke run is an index bug: stop and report it against the index plan.

- [ ] **Step 5: Commit**

```bash
git add bench/exp3.jl bench/src/BenchCore.jl bench/test/test_exp3.jl bench/test/runtests.jl
git commit -m "bench: Exp.3 insert/delete latency with per-operation exactness checks" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: LaTeX tables and macros (`tables.jl`)

**Files:**
- Create: `bench/tables.jl`, `bench/test/test_tables.jl`
- Modify: `bench/test/runtests.jl` (one include line)

**Interfaces:**
- Consumes: `op_latency`, `empty_exp3`, `results_dir` (Tasks 1, 8); CSV columns of `SCHEMA_EXP1/2/3`.
- Produces: formatters `fmt_int`, `fmt_qps`, `fmt_sig3`, `fmt_ms`, `fmt_s`, `fmt_fixed`; `hnsw_picks(df) -> nothing | (max::DataFrameRow, sweet::DataFrameRow)`; `method_table_tex(df; recall_digits::Int) -> String`; `ops_row(n, d, insert_s, delete_s, rebuild_s) -> String`; `ops_summary(exp3) -> DataFrame(n, d, insert_s, delete_s, rebuild_s, rebuild_bulk_s)`; `ops_tex(exp3) -> String`; `macros_tex(exp1, exp1b, exp2, exp3) -> String`; `write_tables(resdir = results_dir()) -> texdir`, writing `tex/tab_annbench.tex`, `tex/tab_fashion.tex`, `tex/tab_ops.tex`, `tex/macros.tex`.

The files hold the `tabular` body only (assumption A6), preceded by one `%` comment line. Row sets: `tab_annbench` and `tab_fashion` list hNNG (incremental build), hNNG (bulk build), brute force (BLAS), KD-tree, ball tree, HNSW (max recall attained), HNSW (sweet spot), in that order (spec §11), with recall at 3 and 5 decimals respectively. `tab_ops` has one row per table configuration, `n (d) & insert/op & delete/op & rebuild/op (speedup)`, for example:

```
8{,}000 ($16$) & 1.10\,ms & 1.60\,ms & 0.460\,s \;\;($\sim$418$\times$)\\
```

Macros written to `macros.tex` (one `\newcommand` each; `n/a` when the source row is absent, as in smoke runs):

| Macro | Value | Source |
|---|---|---|
| `\rhoDtwo`, `\rhoDeight`, `\rhoDsixteen`, `\rhoDthirtytwo` | ρ at d = 2, 8, 16, 32, `%.3f` | exp2 Gaussian, `hNNG-bulk` |
| `\pruneDtwo` | 100·(1 − ρ(2)), `%.1f` | exp2 |
| `\hnngQpsRandomXs`, `\hnngBulkQpsRandomXs`, `\bruteQpsRandomXs` | QPS | exp1 |
| `\hnngOverBruteRandomXs` | incremental QPS / brute QPS, `%.1f` | exp1 |
| `\hnswPeakQpsRandomXs`, `\hnswPeakRecallRandomXs` | fastest HNSW row: QPS, recall `%.2f` | exp1 |
| `\hnswExactQpsRandomXs` | fastest HNSW row with recall 1.0 | exp1 |
| `\hnngQpsFashion`, `\hnngBulkQpsFashion`, `\bruteQpsFashion` | QPS | exp1b |
| `\hnswMaxRecallFashion`, `\hnswMaxRecallQpsFashion` | max-recall HNSW row: recall `%.5f`, QPS | exp1b |
| `\hnswSweetRecallFashion`, `\hnswSweetQpsFashion` | sweet-spot HNSW row: recall `%.3f`, QPS | exp1b |
| `\hnngIncBuildFashion`, `\hnngBulkBuildFashion`, `\bulkBuildSpeedupFashion` | build seconds; ratio `%.1f` | exp1b |
| `\rewireMean`, `\rewireMax` | level-0 rewires over every hNNG insert and delete of the table configurations, mean `%.2f`, max | exp3 |
| `\insertMsLargest`, `\deleteMsLargest`, `\rebuildSLargest`, `\bulkRebuildSLargest`, `\speedupLargest`, `\speedupBulkLargest` | last table configuration, (32000, 128) in the full run; `\bulkRebuildSLargest` is the bulk rebuild in seconds | exp3 |

The manuscript uses these thirteen: `\rhoDtwo \rhoDeight \rhoDsixteen \rhoDthirtytwo \rewireMean \rewireMax \hnngQpsRandomXs \bruteQpsRandomXs \hnngBulkBuildFashion \hnngIncBuildFashion \insertMsLargest \rebuildSLargest \bulkRebuildSLargest`; the test below checks that each one is written.

- [ ] **Step 1: Write the failing test**

`bench/test/test_tables.jl`:

```julia
include(joinpath(@__DIR__, "setup.jl"))
using DataFrames
isdefined(Main, :write_tables) || include(joinpath(BENCH_DIR, "tables.jl"))

bs(s) = s * "\\\\"   # append the LaTeX row end \\ (a raw string cannot end in a backslash)

@testset "tables" begin
    @testset "formatters" begin
        @test fmt_int(51679.4) == "51{,}679"
        @test fmt_int(635) == "635"
        @test fmt_int(1000) == "1{,}000"
        @test fmt_int(1234567) == "1{,}234{,}567"
        @test fmt_qps(9171.2) == "9{,}171"
        @test fmt_qps(59.74) == "59.7"
        @test fmt_sig3(552.4) == "552"
        @test fmt_sig3(43.61) == "43.6"
        @test fmt_sig3(0.0712) == "0.0712"
        @test fmt_sig3(0.46) == "0.460"
        @test fmt_sig3(1234.0) == "1{,}234"
        @test fmt_ms(0.0011) == raw"1.10\,ms"
        @test fmt_s(30.8) == raw"30.8\,s"
    end

    @testset "method table: TKDE layout, spec row set and order" begin
        df = DataFrame(
            method = ["HNSW", "ball-tree", "hNNG-bulk", "brute-BLAS", "KD-tree", "hNNG-incremental", "HNSW", "HNSW"],
            recall_ann = [0.82, 1.0, 1.0, 1.0, 1.0, 1.0, 0.995, 1.0],
            qps = [92000.0, 15000.0, 49256.0, 9171.4, 20000.0, 51679.2, 60000.0, 54000.0],
            build_s = [1.234, 0.0123, 0.0712, 0.00213, 0.0105, 0.583, 1.234, 1.234])
        expected = join([
            raw"\begin{tabular}{lccc}", raw"\hline", bs("Method & Recall & QPS & Build (s)"), raw"\hline",
            bs("hNNG (incremental build) & 1.000 & 51{,}679 & 0.583"),
            bs("hNNG (bulk build) & 1.000 & 49{,}256 & 0.0712"),
            bs("brute force (BLAS) & 1.000 & 9{,}171 & 0.00213"),
            bs("KD-tree & 1.000 & 20{,}000 & 0.0105"),
            bs("ball tree & 1.000 & 15{,}000 & 0.0123"),
            bs("HNSW (max recall attained) & 1.000 & 54{,}000 & 1.23"),
            bs("HNSW (sweet spot) & 0.995 & 60{,}000 & 1.23"),
            raw"\hline", raw"\end{tabular}"], "\n") * "\n"
        @test method_table_tex(df; recall_digits = 3) == expected
    end

    @testset "sweet spot falls back to max recall when no ef reaches 0.99" begin
        df = DataFrame(method = ["HNSW", "HNSW"], recall_ann = [0.5, 0.8], qps = [100.0, 50.0], build_s = [1.0, 1.0])
        p = hnsw_picks(df)
        @test p.max.recall_ann == 0.8 && p.sweet.recall_ann == 0.8
        @test occursin(bs("HNSW (sweet spot) & 0.80000 & 50.0 & 1.00"), method_table_tex(df; recall_digits = 5))
    end

    @testset "SearchGraph takes the HNSW place when HNSW is absent" begin
        df = DataFrame(method = ["hNNG-bulk", "SS-SearchGraph"], recall_ann = [1.0, 0.97],
                       qps = [10.0, 2000.0], build_s = [0.5, 2.0])
        tex = method_table_tex(df; recall_digits = 3)
        @test occursin(bs("SearchGraph (SimilaritySearch.jl) & 0.970 & 2{,}000 & 2.00"), tex)
        @test !occursin("HNSW", tex)
        @test hnsw_picks(df) === nothing
    end

    @testset "tab_ops: row format and median over runs" begin
        @test ops_row(8000, 16, 0.0011, 0.0016, 0.46) ==
              bs(raw"8{,}000 ($16$) & 1.10\,ms & 1.60\,ms & 0.460\,s \;\;($\sim$418$\times$)")
        df = empty_exp3()
        for run in 1:3, (op, t) in (("insert", [0.0010, 0.0011, 0.0012][run]), ("delete", 0.0016),
                                    ("rebuild", 0.46), ("rebuild-bulk", 0.05))
            push!(df, (config = "table", n = 8000, d = 16, run = run, method = "hNNG", op = op, seq = 1,
                       latency_s = t, nevals = 1, rewired = 2, top_level = 1, note = ""))
        end
        lines = split(chomp(ops_tex(df)), "\n")
        @test lines[1] == raw"\begin{tabular}{lccc}"
        @test lines[3] == bs(raw"$n$ ($d$) & insert/op & delete/op & rebuild/op \;(speedup)")
        @test lines[5] == ops_row(8000, 16, 0.0011, 0.0016, 0.46)
        @test lines[end] == raw"\end{tabular}"
    end

    @testset "macros.tex" begin
        exp1 = DataFrame(method = ["hNNG-incremental", "hNNG-bulk", "brute-BLAS", "HNSW", "HNSW"],
                         recall_ann = [1.0, 1.0, 1.0, 0.82, 1.0],
                         qps = [51679.0, 49256.0, 9171.0, 92000.0, 54000.0],
                         build_s = [0.58, 0.07, 0.002, 1.0, 1.0])
        exp1b = DataFrame(method = ["hNNG-incremental", "hNNG-bulk", "brute-BLAS", "HNSW", "HNSW"],
                          recall_ann = [1.0, 1.0, 1.0, 0.99996, 0.994],
                          qps = [70.0, 66.0, 59.7, 635.0, 5245.0],
                          build_s = [552.0, 43.6, 0.1, 6.3, 6.3])
        exp2 = DataFrame(geometry = ["gaussian", "gaussian"], method = ["hNNG-bulk", "hNNG-bulk"],
                         d = [2, 8], frac_evaluated = [0.024, 0.642])
        exp3 = empty_exp3()
        for (op, t, rw) in (("insert", 0.00586, 3), ("delete", 0.00859, 5),
                            ("rebuild", 30.8, missing), ("rebuild-bulk", 2.3, missing))
            push!(exp3, (config = "table", n = 32000, d = 128, run = 1, method = "hNNG", op = op, seq = 1,
                         latency_s = t, nevals = 1, rewired = rw, top_level = 1, note = ""))
        end
        tex = macros_tex(exp1, exp1b, exp2, exp3)
        has(name, val) = occursin("\\newcommand{\\$name}{$val}\n", tex)
        @test has("rhoDtwo", "0.024") && has("rhoDeight", "0.642")
        @test has("rhoDsixteen", "n/a") && has("rhoDthirtytwo", "n/a")
        @test has("pruneDtwo", "97.6")
        @test has("hnngQpsRandomXs", "51{,}679") && has("hnngBulkQpsRandomXs", "49{,}256")
        @test has("bruteQpsRandomXs", "9{,}171") && has("hnngOverBruteRandomXs", "5.6")
        @test has("hnswPeakQpsRandomXs", "92{,}000") && has("hnswPeakRecallRandomXs", "0.82")
        @test has("hnswExactQpsRandomXs", "54{,}000")
        @test has("hnngQpsFashion", "70.0") && has("hnngBulkQpsFashion", "66.0") && has("bruteQpsFashion", "59.7")
        @test has("hnswMaxRecallFashion", "0.99996") && has("hnswMaxRecallQpsFashion", "635")
        @test has("hnswSweetRecallFashion", "0.994") && has("hnswSweetQpsFashion", "5{,}245")
        @test has("hnngIncBuildFashion", "552") && has("hnngBulkBuildFashion", "43.6")
        @test has("bulkBuildSpeedupFashion", "12.7")
        @test has("rewireMean", "4.00") && has("rewireMax", "5")
        @test has("insertMsLargest", "5.86") && has("deleteMsLargest", "8.59")
        @test has("rebuildSLargest", "30.8") && has("speedupLargest", "5{,}256")
        @test has("speedupBulkLargest", "392") && has("bulkRebuildSLargest", "2.30")
        @test count("\\newcommand", tex) == 30
        for name in ("rhoDtwo", "rhoDeight", "rhoDsixteen", "rhoDthirtytwo", "rewireMean", "rewireMax",
                     "hnngQpsRandomXs", "bruteQpsRandomXs", "hnngBulkBuildFashion", "hnngIncBuildFashion",
                     "insertMsLargest", "rebuildSLargest", "bulkRebuildSLargest")   # used by the manuscript
            @test occursin("\\newcommand{\\$name}{", tex)
        end
    end
end
```

In `bench/test/runtests.jl`, add `    include("test_tables.jl")` immediately before the final `end`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_tables.jl`
Expected: FAIL with `SystemError: opening file ".../bench/tables.jl": No such file or directory`.

- [ ] **Step 3: Write `bench/tables.jl`**

```julia
# LaTeX tables and \newcommand macros from the experiment CSVs (spec §11).
# Run: julia -t 1 --project=bench bench/tables.jl      (reads results_dir())
isdefined(Main, :BenchCore) || include(joinpath(@__DIR__, "src", "BenchCore.jl"))
using .BenchCore
using CSV, DataFrames, Printf, Statistics

const METHOD_ROWS = [
    ("hNNG-incremental", "hNNG (incremental build)"),
    ("hNNG-bulk", "hNNG (bulk build)"),
    ("brute-BLAS", "brute force (BLAS)"),
    ("KD-tree", "KD-tree"),
    ("ball-tree", "ball tree"),
]
const SWEET_SPOT_RECALL = 0.99
const ROW_END = "\\\\"                     # the LaTeX row end \\

# ---- number formats (spec assumption A20) --------------------------------------------
"Rounded integer with LaTeX thousands separators: 51679.4 -> 51{,}679."
function fmt_int(x::Real)
    s = string(abs(round(Int, x)))
    groups = String[]
    while length(s) > 3
        pushfirst!(groups, s[end-2:end])
        s = s[1:end-3]
    end
    pushfirst!(groups, s)
    return (x < 0 ? "-" : "") * join(groups, "{,}")
end
fmt_fixed(x::Real, digits::Int) = Printf.format(Printf.Format("%.$(digits)f"), x)
fmt_qps(x::Real) = x >= 100 ? fmt_int(x) : fmt_fixed(x, 1)
"Three significant digits (integers with separators from 1000 up)."
function fmt_sig3(x::Real)
    x == 0 && return "0"
    abs(x) >= 1000 && return fmt_int(x)
    return fmt_fixed(x, max(0, 2 - floor(Int, log10(abs(x)))))
end
fmt_ms(seconds::Real) = fmt_fixed(1000 * seconds, 2) * "\\,ms"
fmt_s(seconds::Real) = fmt_sig3(seconds) * "\\,s"
or_na(f, x) = (x === nothing || x === missing) ? "n/a" : f(x)

# ---- tab:annbench and tab:fashion ----------------------------------------------------
"HNSW rows (assumption A7): max = highest recall (ties: faster); sweet = fastest with recall ≥ 0.99, else max."
function hnsw_picks(df::AbstractDataFrame; target::Real = SWEET_SPOT_RECALL)
    h = df[df.method .== "HNSW", :]
    nrow(h) == 0 && return nothing
    top = h[h.recall_ann .== maximum(h.recall_ann), :]
    best = top[argmax(top.qps), :]
    ok = h[h.recall_ann .>= target, :]
    sweet = nrow(ok) == 0 ? best : ok[argmax(ok.qps), :]
    return (max = best, sweet = sweet)
end

method_row(label, r, recall_digits) = string(label, " & ", fmt_fixed(r.recall_ann, recall_digits),
    " & ", fmt_qps(r.qps), " & ", fmt_sig3(r.build_s), ROW_END)

"tabular body with columns Method & Recall & QPS & Build (s), TKDE tab:annbench / tab:fashion."
function method_table_tex(df::AbstractDataFrame; recall_digits::Int)
    lines = String["\\begin{tabular}{lccc}", "\\hline", "Method & Recall & QPS & Build (s)" * ROW_END, "\\hline"]
    for (m, label) in METHOD_ROWS
        sel = df[df.method .== m, :]
        nrow(sel) > 0 && push!(lines, method_row(label, sel[1, :], recall_digits))
    end
    picks = hnsw_picks(df)
    if picks !== nothing
        push!(lines, method_row("HNSW (max recall attained)", picks.max, recall_digits))
        push!(lines, method_row("HNSW (sweet spot)", picks.sweet, recall_digits))
    else
        sg = df[df.method .== "SS-SearchGraph", :]
        nrow(sg) > 0 && push!(lines, method_row("SearchGraph (SimilaritySearch.jl)", sg[1, :], recall_digits))
    end
    append!(lines, ["\\hline", "\\end{tabular}"])
    return join(lines, "\n") * "\n"
end

# ---- tab:ops -------------------------------------------------------------------------
"One tab:ops row; speedup = rebuild / insert (assumption A8)."
ops_row(n, d, ins, del, reb) = string(fmt_int(n), " (\$", d, "\$) & ", fmt_ms(ins), " & ", fmt_ms(del),
    " & ", fmt_s(reb), " \\;\\;(\$\\sim\$", fmt_int(reb / ins), "\$\\times\$)", ROW_END)

"Per table configuration, sorted by (n, d): median over runs of the per-run mean latency."
function ops_summary(exp3::AbstractDataFrame)
    lat(op) = op_latency(exp3, "table", "hNNG", op)
    ins, del, reb, bulk = lat("insert"), lat("delete"), lat("rebuild"), lat("rebuild-bulk")
    (ins.n == del.n == reb.n == bulk.n && ins.d == del.d == reb.d == bulk.d) ||
        error("exp3.csv: the table configurations lack some hNNG insert/delete/rebuild rows")
    return DataFrame(n = ins.n, d = ins.d, insert_s = ins.latency_s, delete_s = del.latency_s,
                     rebuild_s = reb.latency_s, rebuild_bulk_s = bulk.latency_s)
end

function ops_tex(exp3::AbstractDataFrame)
    lines = String["\\begin{tabular}{lccc}", "\\hline",
                   "\$n\$ (\$d\$) & insert/op & delete/op & rebuild/op \\;(speedup)" * ROW_END, "\\hline"]
    for r in eachrow(ops_summary(exp3))
        push!(lines, ops_row(r.n, r.d, r.insert_s, r.delete_s, r.rebuild_s))
    end
    append!(lines, ["\\hline", "\\end{tabular}"])
    return join(lines, "\n") * "\n"
end

# ---- macros.tex -------------------------------------------------------------------------
macro_line(name, value) = "\\newcommand{\\$name}{$value}"
function first_row(df::AbstractDataFrame, method::AbstractString)
    sel = df[df.method .== method, :]
    return nrow(sel) == 0 ? nothing : sel[1, :]
end
function rho_at(exp2::AbstractDataFrame, d::Int)
    sel = exp2[(exp2.geometry .== "gaussian") .& (exp2.method .== "hNNG-bulk") .& (exp2.d .== d), :]
    return nrow(sel) == 0 ? nothing : sel.frac_evaluated[1]
end
ratio_or_na(a, b, field, digits) =
    (a === nothing || b === nothing) ? "n/a" : fmt_fixed(getproperty(a, field) / getproperty(b, field), digits)

"Every number the manuscript prose quotes, as \\newcommand lines."
function macros_tex(exp1, exp1b, exp2, exp3)
    m = Pair{String,String}[]
    # Exp.2: ρ(d) on Gaussian data (TKDE lines 91-95)
    for (name, d) in (("rhoDtwo", 2), ("rhoDeight", 8), ("rhoDsixteen", 16), ("rhoDthirtytwo", 32))
        push!(m, name => or_na(x -> fmt_fixed(x, 3), rho_at(exp2, d)))
    end
    push!(m, "pruneDtwo" => or_na(x -> fmt_fixed(100 * (1 - x), 1), rho_at(exp2, 2)))
    # Exp.1: random-xs headline
    inc, bulk, brute = first_row(exp1, "hNNG-incremental"), first_row(exp1, "hNNG-bulk"), first_row(exp1, "brute-BLAS")
    push!(m, "hnngQpsRandomXs" => or_na(r -> fmt_qps(r.qps), inc))
    push!(m, "hnngBulkQpsRandomXs" => or_na(r -> fmt_qps(r.qps), bulk))
    push!(m, "bruteQpsRandomXs" => or_na(r -> fmt_qps(r.qps), brute))
    push!(m, "hnngOverBruteRandomXs" => ratio_or_na(inc, brute, :qps, 1))
    hn = exp1[exp1.method .== "HNSW", :]
    peak = nrow(hn) == 0 ? nothing : hn[argmax(hn.qps), :]
    ex = hn[hn.recall_ann .>= 1.0, :]
    exact = nrow(ex) == 0 ? nothing : ex[argmax(ex.qps), :]
    push!(m, "hnswPeakQpsRandomXs" => or_na(r -> fmt_qps(r.qps), peak))
    push!(m, "hnswPeakRecallRandomXs" => or_na(r -> fmt_fixed(r.recall_ann, 2), peak))
    push!(m, "hnswExactQpsRandomXs" => or_na(r -> fmt_qps(r.qps), exact))
    # Exp.1b: Fashion-MNIST
    finc, fbulk, fbrute = first_row(exp1b, "hNNG-incremental"), first_row(exp1b, "hNNG-bulk"), first_row(exp1b, "brute-BLAS")
    push!(m, "hnngQpsFashion" => or_na(r -> fmt_qps(r.qps), finc))
    push!(m, "hnngBulkQpsFashion" => or_na(r -> fmt_qps(r.qps), fbulk))
    push!(m, "bruteQpsFashion" => or_na(r -> fmt_qps(r.qps), fbrute))
    picks = hnsw_picks(exp1b)
    push!(m, "hnswMaxRecallFashion" => or_na(p -> fmt_fixed(p.max.recall_ann, 5), picks))
    push!(m, "hnswMaxRecallQpsFashion" => or_na(p -> fmt_qps(p.max.qps), picks))
    push!(m, "hnswSweetRecallFashion" => or_na(p -> fmt_fixed(p.sweet.recall_ann, 3), picks))
    push!(m, "hnswSweetQpsFashion" => or_na(p -> fmt_qps(p.sweet.qps), picks))
    push!(m, "hnngIncBuildFashion" => or_na(r -> fmt_sig3(r.build_s), finc))
    push!(m, "hnngBulkBuildFashion" => or_na(r -> fmt_sig3(r.build_s), fbulk))
    push!(m, "bulkBuildSpeedupFashion" => ratio_or_na(finc, fbulk, :build_s, 1))
    # Exp.3: rewires over every hNNG insert and delete of the table configurations
    ops = exp3[(exp3.config .== "table") .& (exp3.method .== "hNNG") .& in.(exp3.op, Ref(("insert", "delete"))), :]
    rw = collect(skipmissing(ops.rewired))
    push!(m, "rewireMean" => isempty(rw) ? "n/a" : fmt_fixed(mean(rw), 2))
    push!(m, "rewireMax" => isempty(rw) ? "n/a" : string(maximum(rw)))
    s = ops_summary(exp3)
    largest = nrow(s) == 0 ? nothing : s[end, :]
    push!(m, "insertMsLargest" => or_na(r -> fmt_fixed(1000 * r.insert_s, 2), largest))
    push!(m, "deleteMsLargest" => or_na(r -> fmt_fixed(1000 * r.delete_s, 2), largest))
    push!(m, "rebuildSLargest" => or_na(r -> fmt_sig3(r.rebuild_s), largest))
    push!(m, "bulkRebuildSLargest" => or_na(r -> fmt_sig3(r.rebuild_bulk_s), largest))
    push!(m, "speedupLargest" => or_na(r -> fmt_int(r.rebuild_s / r.insert_s), largest))
    push!(m, "speedupBulkLargest" => or_na(r -> fmt_int(r.rebuild_bulk_s / r.insert_s), largest))
    return join((macro_line(k, v) for (k, v) in m), "\n") * "\n"
end

# ---- files ------------------------------------------------------------------------------
function read_result(resdir::AbstractString, file::AbstractString)
    p = joinpath(resdir, file)
    isfile(p) || error("$p is missing; run the experiment that writes it first")
    return CSV.read(p, DataFrame)
end

function write_tables(resdir::AbstractString = results_dir())
    exp1, exp1b, exp2, exp3 = (read_result(resdir, f) for f in ("exp1.csv", "exp1b.csv", "exp2.csv", "exp3.csv"))
    texdir = joinpath(resdir, "tex")
    mkpath(texdir)
    header = "% generated by bench/tables.jl from the CSV files next to this directory; do not edit\n"
    write(joinpath(texdir, "tab_annbench.tex"), header * method_table_tex(exp1; recall_digits = 3))
    write(joinpath(texdir, "tab_fashion.tex"), header * method_table_tex(exp1b; recall_digits = 5))
    write(joinpath(texdir, "tab_ops.tex"), header * ops_tex(exp3))
    write(joinpath(texdir, "macros.tex"), header * macros_tex(exp1, exp1b, exp2, exp3))
    return texdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    write_tables()
end
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_tables.jl`
Expected: `Test Summary:` for `tables` with every test passing.

- [ ] **Step 5: Commit**

```bash
git add bench/tables.jl bench/test/test_tables.jl bench/test/runtests.jl
git commit -m "bench: TKDE-layout LaTeX tables and manuscript macros from the CSVs" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: figures (`plots.jl`)

**Files:**
- Create: `bench/plots.jl`, `bench/test/test_plots.jl`
- Modify: `bench/test/runtests.jl` (one include line)

**Interfaces:**
- Consumes: `op_latency`, `empty_exp3`, `results_dir` (Tasks 1, 8); CSV columns of `SCHEMA_EXP1/2/3`.
- Produces: `plot_recall_qps(exp1, path)`, `plot_dist_eval(exp2, path)`, `plot_qps_dim(exp2, path)`, `plot_ops(exp3, path)`, each returning `path`; constants `FIG_SINGLE`, `FIG_DOUBLE`, `FIG_FONT`; `write_plots(resdir = results_dir()) -> figdir`, writing `fig/recall_qps.pdf` (from exp1.csv, random-xs), `fig/dist_eval_vs_dim.pdf`, `fig/qps_vs_dim.pdf` (from exp2.csv), `fig/ops_comparison.pdf` (from every hNNG row of exp3.csv); helper `n_label(n) -> String` (`8000 -> "8k"`).

Sizes: every figure is designed at its final print size and saved with `save(path, fig; pt_per_unit = 1)`, so `Figure(size = ...)` is in pt (assumption A24): `FIG_SINGLE = (215, 165)` for the three single-column figures and `FIG_DOUBLE = (450, 150)` for `ops_comparison` (a double-column `figure*` in the manuscript), all with `fontsize = 8`.

Encoding: one colour and marker per method in a fixed order, the same in every figure. The palette was checked with the dataviz validator on a light surface: all checks pass; three hues sit below 3:1 contrast, so every series also has its own marker, and every figure has a legend. `ops_comparison` follows the TKDE layout: three panels side by side, one per `d` (titles `d = 16`, `d = 64`, `d = 128`), x = n on a log axis with ticks 8k, 16k, 32k, y = milliseconds per operation on a log axis, three lines (full rebuild per op, incremental insert, incremental delete) that differ in colour, marker and line style, a text label `rebuild/insert ≈ N×` at the largest n in each panel, and one shared legend below. One y-axis per panel (linked), no dual axes.

- [ ] **Step 1: Write the failing test**

`bench/test/test_plots.jl`:

```julia
include(joinpath(@__DIR__, "setup.jl"))
using DataFrames
isdefined(Main, :write_plots) || include(joinpath(BENCH_DIR, "plots.jl"))

ispdf(p) = isfile(p) && filesize(p) > 1000 && open(io -> String(read(io, 4)), p) == "%PDF"

@testset "plots" begin
    exp1 = DataFrame(method = ["hNNG-incremental", "hNNG-bulk", "brute-BLAS", "KD-tree", "ball-tree", "HNSW", "HNSW"],
                     recall_ann = [1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.99],
                     qps = [5e4, 4.9e4, 9e3, 2e4, 1.5e4, 9e4, 5e4])
    methods2 = ["hNNG-bulk", "KD-tree", "ball-tree", "brute-BLAS"]
    exp2 = DataFrame(geometry = repeat(["gaussian", "blobs"]; inner = 8),
                     n = repeat([5000, 10000]; inner = 8),
                     d = repeat([2, 2, 2, 2, 8, 8, 8, 8], 2),
                     method = repeat(methods2, 4),
                     qps = collect(range(10.0, 1e4; length = 16)),
                     frac_evaluated = [m == "hNNG-bulk" ? 0.3 : missing for m in repeat(methods2, 4)])
    exp3 = empty_exp3()
    series = (("hNNG", "insert", 1e-3), ("hNNG", "delete", 2e-3), ("hNNG", "rebuild", 0.5),
              ("hNNG", "rebuild-bulk", 0.1), ("HNSW", "insert", 5e-4), ("KD-tree", "rebuild", 0.01))
    for n in (8000, 16000, 32000), d in (16, 64, 128), run in 1:3, (meth, op, t) in series
        cfg = (n, d) in ((8000, 16), (32000, 16), (32000, 128)) ? "table" : "grid"
        push!(exp3, (config = cfg, n = n, d = d, run = run, method = meth, op = op, seq = 1,
                     latency_s = t * (n / 8000) * (d / 16), nevals = missing, rewired = missing,
                     top_level = missing, note = ""))
    end
    @test n_label(8000) == "8k" && n_label(32000) == "32k" && n_label(300) == "300"
    @test speedup_label(0.46, 0.0011) == "rebuild/insert ≈ 418×"
    @test FIG_SINGLE == (215, 165) && FIG_DOUBLE == (450, 150) && FIG_FONT == 8
    # the page size in pt, as poppler's pdfinfo reports it: 1 Makie unit = 1 pt (pt_per_unit = 1)
    page_pts(p) = (m = match(r"Page size:\s+([\d.]+) x ([\d.]+)", read(`pdfinfo $p`, String));
                   (parse(Float64, m[1]), parse(Float64, m[2])))
    mktempdir() do dir
        made = ("recall_qps" => plot_recall_qps(exp1, joinpath(dir, "recall_qps.pdf")),
                "dist_eval_vs_dim" => plot_dist_eval(exp2, joinpath(dir, "dist_eval_vs_dim.pdf")),
                "qps_vs_dim" => plot_qps_dim(exp2, joinpath(dir, "qps_vs_dim.pdf")),
                "ops_comparison" => plot_ops(exp3, joinpath(dir, "ops_comparison.pdf")))
        for (_, p) in made
            @test ispdf(p)
        end
        if Sys.which("pdfinfo") === nothing
            @info "pdfinfo not found; PDF page sizes not checked"
        else
            for (name, p) in made
                want = name == "ops_comparison" ? FIG_DOUBLE : FIG_SINGLE
                w, h = page_pts(p)
                @test abs(w - want[1]) <= 1 && abs(h - want[2]) <= 1
            end
        end
    end
end
```

In `bench/test/runtests.jl`, add `    include("test_plots.jl")` immediately before the final `end`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_plots.jl`
Expected: FAIL with `SystemError: opening file ".../bench/plots.jl": No such file or directory`.

- [ ] **Step 3: Write `bench/plots.jl`**

```julia
# The four TKDE benchmark figures as PDFs (CairoMakie), spec §11.
# Run: julia -t 1 --project=bench bench/plots.jl       (reads results_dir())
isdefined(Main, :BenchCore) || include(joinpath(@__DIR__, "src", "BenchCore.jl"))
using .BenchCore
using CairoMakie, CSV, DataFrames

# Fixed order, one colour + marker per method, identical in every figure.
const METHOD_ORDER = ["hNNG-incremental", "hNNG-bulk", "brute-BLAS", "KD-tree", "ball-tree",
                      "HNSW", "SS-SearchGraph", "SS-Exhaustive"]
const METHOD_STYLE = Dict(
    "hNNG-incremental" => (color = "#2a78d6", marker = :circle,    label = "hNNG (incremental build)"),
    "hNNG-bulk"        => (color = "#eb6834", marker = :rect,      label = "hNNG (bulk build)"),
    "brute-BLAS"       => (color = "#1baf7a", marker = :diamond,   label = "brute force (BLAS)"),
    "KD-tree"          => (color = "#eda100", marker = :utriangle, label = "KD-tree"),
    "ball-tree"        => (color = "#e87ba4", marker = :dtriangle, label = "ball tree"),
    "HNSW"             => (color = "#008300", marker = :xcross,    label = "HNSW (ef sweep)"),
    "SS-SearchGraph"   => (color = "#6250d6", marker = :star5,     label = "SearchGraph"),
    "SS-Exhaustive"    => (color = "#e34948", marker = :hexagon,   label = "SimilaritySearch exhaustive"),
)
const GEOMETRY_STYLE = Dict(
    "gaussian" => (color = "#2a78d6", marker = :circle, label = "Gaussian"),
    "blobs"    => (color = "#eb6834", marker = :rect,   label = "100 blobs"),
)
# TKDE ops_comparison lines (hNNG only), fixed order: (op, label, colour, marker, line style).
# The three operations are the entities here, so they take categorical slots 1-3.
const OPS_LINES = [
    ("rebuild", "full rebuild per op", "#2a78d6", :rect,    :dot),
    ("insert",  "incremental insert",  "#eb6834", :circle,  :solid),
    ("delete",  "incremental delete",  "#1baf7a", :diamond, :dash),
]

# Final print sizes in pt: with `pt_per_unit = 1` one Makie unit is one PDF point, so the text
# is 8 pt on the page. sn-jnl `iicol`: column 215 pt (76 mm), double column 453 pt (160 mm).
const FIG_SINGLE = (215, 165)    # one column
const FIG_DOUBLE = (450, 150)    # ops_comparison: figure* across both columns
const FIG_FONT = 8
const MS = 5                     # marker size (pt)
const LW = 1                     # line width (pt)
# Compact legend for a 215 pt column.
const LEGEND_STYLE = (framevisible = false, labelsize = 7, patchsize = (14, 8), padding = (2, 2, 2, 2),
                rowgap = 0, colgap = 6)

function save_pdf(fig, path::AbstractString)
    mkpath(dirname(path))
    save(path, fig; pt_per_unit = 1)
    return path
end
"Ticks at the dimensions `ds`, optionally only every `step`-th one (a half-column panel fits four labels)."
dim_ticks(ds; step::Int = 1) = (ds[1:step:end], string.(ds[1:step:end]))

"Recall vs QPS on random-xs: HNSW traces a frontier, exact methods are points at recall 1."
function plot_recall_qps(exp1::AbstractDataFrame, path::AbstractString)
    fig = Figure(size = FIG_SINGLE, fontsize = FIG_FONT)
    ax = Axis(fig[1, 1]; xlabel = "recall (ann-benchmarks rule, k = 10)", ylabel = "queries per second",
              yscale = log10)
    for m in METHOD_ORDER
        r = sort(exp1[exp1.method .== m, :], :recall_ann)
        nrow(r) == 0 && continue
        st = METHOD_STYLE[m]
        x, y = Float64.(r.recall_ann), Float64.(r.qps)
        if nrow(r) > 1
            scatterlines!(ax, x, y; color = st.color, marker = st.marker, markersize = MS, linewidth = LW, label = st.label)
        else
            scatter!(ax, x, y; color = st.color, marker = st.marker, markersize = MS + 2, label = st.label)
        end
    end
    axislegend(ax; position = :lb, LEGEND_STYLE...)
    return save_pdf(fig, path)
end

"ρ(d): fraction of points hNNG evaluates per query, both Exp.2 geometries."
function plot_dist_eval(exp2::AbstractDataFrame, path::AbstractString)
    ds = sort(unique(Int.(exp2.d)))
    fig = Figure(size = FIG_SINGLE, fontsize = FIG_FONT)
    ax = Axis(fig[1, 1]; xlabel = "dimension d", ylabel = "fraction of points evaluated ρ",
              xscale = log2, xticks = dim_ticks(ds), limits = (nothing, (0, 1.05)))
    for g in ("gaussian", "blobs")
        r = sort(exp2[(exp2.geometry .== g) .& (exp2.method .== "hNNG-bulk"), :], :d)
        nrow(r) == 0 && continue
        st = GEOMETRY_STYLE[g]
        scatterlines!(ax, Int.(r.d), Float64.(r.frac_evaluated); color = st.color, marker = st.marker,
                      markersize = MS, linewidth = LW, label = "$(st.label), n = $(r.n[1])")
    end
    axislegend(ax; position = :rb, LEGEND_STYLE...)
    return save_pdf(fig, path)
end

"QPS vs d for the exact methods; one panel per geometry (half a column each), shared legend below."
function plot_qps_dim(exp2::AbstractDataFrame, path::AbstractString)
    ds = sort(unique(Int.(exp2.d)))
    fig = Figure(size = FIG_SINGLE, fontsize = FIG_FONT)
    panels = Axis[]
    for g in ("gaussian", "blobs")
        sub = exp2[exp2.geometry .== g, :]
        nrow(sub) == 0 && continue
        ax = Axis(fig[1, length(panels) + 1]; title = "$(GEOMETRY_STYLE[g].label), n = $(sub.n[1])",
                  xlabel = "dimension d", ylabel = isempty(panels) ? "queries per second" : "",
                  xscale = log2, yscale = log10, xticks = dim_ticks(ds; step = 2))
        push!(panels, ax)
        for m in METHOD_ORDER
            r = sort(sub[sub.method .== m, :], :d)
            nrow(r) == 0 && continue
            st = METHOD_STYLE[m]
            scatterlines!(ax, Int.(r.d), Float64.(r.qps); color = st.color, marker = st.marker,
                          markersize = MS, linewidth = LW, label = st.label)
        end
    end
    isempty(panels) && error("exp2 has no rows to plot")
    length(panels) > 1 && linkyaxes!(panels...)
    Legend(fig[2, 1:length(panels)], panels[1]; orientation = :horizontal, nbanks = 2, LEGEND_STYLE...)
    return save_pdf(fig, path)
end

"Tick label for an index size: 8000 -> \"8k\", 300 -> \"300\"."
n_label(n::Integer) = n % 1000 == 0 ? "$(n ÷ 1000)k" : string(n)

"Panel annotation: rebuild time over insert time at the largest n."
speedup_label(rebuild_s::Real, insert_s::Real) = "rebuild/insert ≈ $(round(Int, rebuild_s / insert_s))×"

"TKDE ops_comparison: one panel per d, x = n (log), y = ms per operation (log), shared legend."
function plot_ops(exp3::AbstractDataFrame, path::AbstractString)
    h = exp3[exp3.method .== "hNNG", :]
    ds = sort(unique(Int.(h.d)))
    isempty(ds) && error("exp3 has no hNNG rows to plot")
    fig = Figure(size = FIG_DOUBLE, fontsize = FIG_FONT)
    panels = Axis[]
    for (col, d) in enumerate(ds)
        sub = h[h.d .== d, :]
        ns = sort(unique(Int.(sub.n)))
        ax = Axis(fig[1, col]; title = "d = $d", xlabel = "n",
                  ylabel = col == 1 ? "time per operation (ms)" : "",
                  xscale = log10, yscale = log10, xticks = (ns, n_label.(ns)))
        push!(panels, ax)
        lat = Dict(op => op_latency(sub, nothing, "hNNG", op) for (op, _, _, _, _) in OPS_LINES)
        for (op, label, color, marker, ls) in OPS_LINES
            s = lat[op]
            nrow(s) == 0 && continue
            scatterlines!(ax, Float64.(s.n), 1000 .* s.latency_s; color = color, marker = marker,
                          linestyle = ls, markersize = MS, linewidth = LW, label = label)
        end
        reb, ins = lat["rebuild"], lat["insert"]
        if nrow(reb) > 0 && nrow(ins) > 0
            text!(ax, Float64(reb.n[end]), 1000 * reb.latency_s[end];
                  text = speedup_label(reb.latency_s[end], ins.latency_s[end]),
                  align = (:right, :bottom), offset = (-3, 3), fontsize = 7)
        end
    end
    length(panels) > 1 && linkyaxes!(panels...)
    Legend(fig[2, 1:length(panels)], panels[1]; orientation = :horizontal, LEGEND_STYLE...)
    return save_pdf(fig, path)
end

function write_plots(resdir::AbstractString = results_dir())
    rd(f) = CSV.read(joinpath(resdir, f), DataFrame)
    figdir = joinpath(resdir, "fig")
    plot_recall_qps(rd("exp1.csv"), joinpath(figdir, "recall_qps.pdf"))
    exp2 = rd("exp2.csv")
    plot_dist_eval(exp2, joinpath(figdir, "dist_eval_vs_dim.pdf"))
    plot_qps_dim(exp2, joinpath(figdir, "qps_vs_dim.pdf"))
    plot_ops(rd("exp3.csv"), joinpath(figdir, "ops_comparison.pdf"))
    return figdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    write_plots()
end
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_plots.jl`
Expected: `Test Summary:` for `plots` with every test passing, including the four page-size checks when `pdfinfo` is installed (the first run includes CairoMakie compilation, about a minute). The visual check of the real figures happens in Task 11, Step 4, once `run_all.jl` exists.

- [ ] **Step 5: Commit**

```bash
git add bench/plots.jl bench/test/test_plots.jl bench/test/runtests.jl
git commit -m "bench: CairoMakie figures recall_qps, dist_eval_vs_dim, qps_vs_dim, ops_comparison" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: `run_all.jl`, `run.sh`, README benchmark section

**Files:**
- Create: `bench/run_all.jl`, `bench/run.sh`, `bench/test/test_run_all.jl`
- Modify: `bench/test/runtests.jl` (one include line), `README.md` (append a section; create the file with a title line if the index plan has not)

**Interfaces:**
- Consumes: `run_exp1`, `run_exp1b`, `run_exp2`, `run_exp3`, `write_tables`, `write_plots`, `write_env`, `check_protocol`, `is_smoke`, `results_dir` (Tasks 1–10).
- Produces: `EXPERIMENTS` (name, CSV file, runner; in run order exp2, exp3, exp1, exp1b), `bench_force() -> Bool` (`BENCH_FORCE=1`), `bench_only() -> Union{Nothing,Symbol}` (`BENCH_ONLY`), `run_experiments(; smoke, outdir, force, select, experiments = EXPERIMENTS) -> Vector{Symbol}` (the experiments that actually ran), `run_all(; smoke::Bool = is_smoke(), outdir = results_dir(smoke), force = bench_force(), select = bench_only()) -> outdir`; the executable `bench/run.sh`.

Run order and resumability (assumption A23): exp2, exp3, exp1 (random-xs only), exp1b (Fashion-MNIST, hours) last. `run_experiments` skips an experiment whose CSV exists in `outdir` unless `force` (`BENCH_FORCE=1`); smoke runs always rerun. `select` (`BENCH_ONLY=exp2|exp3|exp1|exp1b`) restricts the run to one experiment; skip-existing still applies to it, so rerunning one finished experiment needs `BENCH_FORCE=1` too. `run_all` writes the tables and figures only when all four CSVs exist, and warns about the missing ones otherwise.

- [ ] **Step 1: Write the failing test**

`bench/test/test_run_all.jl`:

```julia
include(joinpath(@__DIR__, "setup.jl"))
isdefined(Main, :run_all) || include(joinpath(BENCH_DIR, "run_all.jl"))

# A fake experiment that records its call and writes its CSV, to observe order and skipping.
fake_experiment(name, file, calls) = (name = name, file = file,
    run = function (args...; kw...)
        push!(calls, name)
        write(joinpath(kw[:outdir], file), "x\n")
        return nothing
    end)

@testset "run_experiments: order, skip-existing, BENCH_ONLY, BENCH_FORCE" begin
    @test [e.name for e in EXPERIMENTS] == [:exp2, :exp3, :exp1, :exp1b]       # Fashion-MNIST last
    @test [e.file for e in EXPERIMENTS] == ["exp2.csv", "exp3.csv", "exp1.csv", "exp1b.csv"]
    mktempdir() do out
        calls = Symbol[]
        fakes = [fake_experiment(:exp2, "exp2.csv", calls), fake_experiment(:exp3, "exp3.csv", calls),
                 fake_experiment(:exp1, "exp1.csv", calls), fake_experiment(:exp1b, "exp1b.csv", calls)]
        go(; smoke = false, force = false, select = nothing) =
            run_experiments(; smoke, outdir = out, force, select, experiments = fakes)
        @test go() == [:exp2, :exp3, :exp1, :exp1b] && calls == [:exp2, :exp3, :exp1, :exp1b]
        empty!(calls)
        @test isempty(go()) && isempty(calls)                       # every CSV exists: nothing is called
        rm(joinpath(out, "exp3.csv"))                               # a crashed experiment left no CSV
        @test go() == [:exp3] && calls == [:exp3]
        empty!(calls)
        @test go(; force = true) == [:exp2, :exp3, :exp1, :exp1b]   # BENCH_FORCE=1 reruns everything
        empty!(calls)
        @test isempty(go(; select = :exp1))                         # BENCH_ONLY alone still skips a finished CSV
        @test go(; select = :exp1, force = true) == [:exp1] && calls == [:exp1]
        empty!(calls)
        @test go(; smoke = true) == [:exp2, :exp3, :exp1, :exp1b]   # smoke runs always rerun
    end
    withenv("BENCH_FORCE" => "1", "BENCH_ONLY" => "exp1b") do
        @test bench_force() && bench_only() == :exp1b
    end
    withenv("BENCH_FORCE" => nothing, "BENCH_ONLY" => nothing) do
        @test !bench_force() && bench_only() === nothing
    end
    withenv("BENCH_ONLY" => "exp9") do
        @test_throws ErrorException bench_only()
    end
end

@testset "run_all: tables and plots only when all four CSVs exist" begin
    mktempdir() do out
        run_all(; outdir = out, force = true, select = :exp2)
        @test isfile(joinpath(out, "exp2.csv")) && isfile(joinpath(out, "env.txt"))
        @test !isfile(joinpath(out, "exp1.csv"))
        @test !isdir(joinpath(out, "tex")) && !isdir(joinpath(out, "fig"))
    end
end

@testset "run_all smoke: every output file" begin
    mktempdir() do out
        @test run_all(; outdir = out) == out
        for f in ("exp1.csv", "exp1b.csv", "exp2.csv", "exp3.csv", "env.txt",
                  "tex/tab_annbench.tex", "tex/tab_fashion.tex", "tex/tab_ops.tex", "tex/macros.tex",
                  "fig/recall_qps.pdf", "fig/dist_eval_vs_dim.pdf", "fig/qps_vs_dim.pdf", "fig/ops_comparison.pdf")
            @test isfile(joinpath(out, f))
        end
        ops = readlines(joinpath(out, "tex", "tab_ops.tex"))
        for row in ("200 (\$2\$) & ", "300 (\$2\$) & ", "300 (\$8\$) & ")
            @test count(l -> startswith(l, row), ops) == 1
        end
        annbench = read(joinpath(out, "tex", "tab_annbench.tex"), String)
        @test occursin("hNNG (incremental build) & 1.000 & ", annbench)
        @test occursin("HNSW (sweet spot) & ", annbench) == HAS_HNSW
        macros = read(joinpath(out, "tex", "macros.tex"), String)
        @test occursin("\\newcommand{\\rhoDtwo}{", macros) && occursin("\\newcommand{\\rhoDsixteen}{n/a}", macros)
        @test occursin("threads=1", read(joinpath(out, "env.txt"), String))
    end
end
```

In `bench/test/runtests.jl`, add `    include("test_run_all.jl")` immediately before the final `end`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_run_all.jl`
Expected: FAIL with `SystemError: opening file ".../bench/run_all.jl": No such file or directory`.

- [ ] **Step 3: Write `bench/run_all.jl` and `bench/run.sh`**

`bench/run_all.jl`:

```julia
# Every experiment, then the tables and the figures (spec §11).
# Pinned full run: bench/run.sh. Direct: julia -t 1 --project=bench bench/run_all.jl
isdefined(Main, :BenchCore) || include(joinpath(@__DIR__, "src", "BenchCore.jl"))
using .BenchCore
for (file, entry) in (("exp1.jl", :run_exp1), ("exp2.jl", :run_exp2), ("exp3.jl", :run_exp3),
                      ("tables.jl", :write_tables), ("plots.jl", :write_plots))
    isdefined(Main, entry) || include(joinpath(@__DIR__, file))
end

# Run order: exp2 and exp3 first (exp3 at n = 32000, d = 128 has never run at full size before
# the real run), then exp1 (random-xs), and the multi-hour Fashion-MNIST experiment last, so a
# late failure costs the least and the finished CSVs survive it.
const EXPERIMENTS = [
    (name = :exp2,  file = "exp2.csv",  run = run_exp2),
    (name = :exp3,  file = "exp3.csv",  run = run_exp3),
    (name = :exp1,  file = "exp1.csv",  run = run_exp1),
    (name = :exp1b, file = "exp1b.csv", run = run_exp1b),
]
const RESULT_CSVS = ["exp1.csv", "exp1b.csv", "exp2.csv", "exp3.csv"]

"BENCH_FORCE=1: rerun experiments whose CSV already exists."
bench_force() = get(ENV, "BENCH_FORCE", "0") == "1"

"BENCH_ONLY=exp1|exp1b|exp2|exp3: run only that experiment (`nothing` when unset)."
function bench_only()
    s = strip(get(ENV, "BENCH_ONLY", ""))
    isempty(s) && return nothing
    Symbol(s) in (e.name for e in EXPERIMENTS) ||
        error("BENCH_ONLY must be one of exp1, exp1b, exp2, exp3; got $(repr(s))")
    return Symbol(s)
end

"""
    run_experiments(; smoke, outdir, force, select, experiments) -> Vector{Symbol}

Runs `experiments` in order and returns the names of those that ran. An experiment whose CSV
is already in `outdir` is skipped unless `force`; smoke runs always rerun, since their
directory is scratch. `select` (a name) restricts the run to that experiment.
"""
function run_experiments(; smoke::Bool = is_smoke(), outdir::AbstractString = results_dir(smoke),
                         force::Bool = bench_force(), select::Union{Nothing,Symbol} = bench_only(),
                         experiments = EXPERIMENTS)
    ran = Symbol[]
    for e in experiments
        select === nothing || e.name == select || continue
        if !(force || smoke) && isfile(joinpath(outdir, e.file))
            @info "$(e.name): $(e.file) exists, skipped (BENCH_FORCE=1 reruns it)"
            continue
        end
        @info "running $(e.name)"
        e.run(; smoke, outdir)
        push!(ran, e.name)
    end
    return ran
end

function run_all(; smoke::Bool = is_smoke(), outdir::AbstractString = results_dir(smoke),
                 force::Bool = bench_force(), select::Union{Nothing,Symbol} = bench_only())
    check_protocol()
    mkpath(outdir)
    write_env(joinpath(outdir, "env.txt"))
    @info "benchmark results go to $outdir" smoke force select
    run_experiments(; smoke, outdir, force, select)
    absent = [f for f in RESULT_CSVS if !isfile(joinpath(outdir, f))]
    if isempty(absent)
        write_tables(outdir)
        write_plots(outdir)
    else
        @warn "tables and figures not written: missing $(join(absent, ", ")); rerun to continue" outdir
    end
    return outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
```

`bench/run.sh`:

```bash
#!/usr/bin/env bash
# Pinned, single-threaded benchmark run (spec §11). From the repository root:
#   bench/run.sh                  full run   -> bench/results/
#   BENCH_SMOKE=1 bench/run.sh    tiny sizes -> bench/results-smoke/
# BENCH_CPU picks the core (default 2), BENCH_RESULTS the output directory,
# BENCH_DATA the directory holding the ann-benchmarks HDF5 files.
# Resumable: experiments run in the order exp2, exp3, exp1, exp1b and one whose CSV already
# exists is skipped, so after a crash simply run the same command again.
#   BENCH_FORCE=1             rerun experiments even if their CSV exists
#   BENCH_ONLY=exp3           run only one of exp1, exp1b, exp2, exp3
# Tables and figures are written only when all four CSV files exist.
set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"     # juliaup's PATH entry lives in the login profile
case "${BENCH_ONLY:-}" in
    ""|exp1|exp1b|exp2|exp3) ;;
    *) echo "run.sh: BENCH_ONLY must be one of exp1, exp1b, exp2, exp3 (got '${BENCH_ONLY}')" >&2; exit 1 ;;
esac
cd "$(dirname "$0")/.."
command -v taskset >/dev/null || { echo "run.sh: taskset not found (util-linux)" >&2; exit 1; }
julia --project=bench -e 'using Pkg; Pkg.instantiate()'
exec taskset -c "${BENCH_CPU:-2}" julia -t 1 --project=bench bench/run_all.jl
```

```bash
chmod +x bench/run.sh
```

- [ ] **Step 4: Run the tests and the wrapper, and look at the figures**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/test_run_all.jl`
Expected: `Test Summary:` for `run_all smoke: every output file` with every test passing.

Run: `BENCH_SMOKE=1 bench/run.sh && grep -E '^(threads|cpus_allowed)=' bench/results-smoke/env.txt && ls bench/results-smoke/tex bench/results-smoke/fig`
Expected:

```
threads=1
cpus_allowed=2
macros.tex  tab_annbench.tex  tab_fashion.tex  tab_ops.tex
dist_eval_vs_dim.pdf  ops_comparison.pdf  qps_vs_dim.pdf  recall_qps.pdf
```

Check the page sizes. The figures are designed at their print size (assumption A24), so the PDFs must be 215 × 165 pt (three single-column figures) and 450 × 150 pt (`ops_comparison`):

```bash
for f in recall_qps dist_eval_vs_dim qps_vs_dim ops_comparison; do
    printf '%s: ' "$f"; pdfinfo "bench/results-smoke/fig/$f.pdf" | grep 'Page size'
done
```

Expected: `recall_qps`, `dist_eval_vs_dim` and `qps_vs_dim` report `Page size: 215 x 165 pts` and `ops_comparison` reports `Page size: 450 x 150 pts` (within a point; `mutool info` shows the same MediaBox if `pdfinfo` is missing).

Then open the four PDFs in `bench/results-smoke/fig/` at 100% zoom, so one PDF point is one screen point and the text is as large as it will be in the paper, and check by eye: the 8 pt text is legible, legends do not cover data, axis labels and ticks do not collide, every series named in the legend is visible. If a legend covers data, change only its `position` keyword in `bench/plots.jl` and rerun; if tick labels collide, reduce the number of ticks (`dim_ticks(...; step = 2)`) rather than the font size.

- [ ] **Step 5: Append the README benchmark section**

````bash
test -f README.md || printf '# HNNG.jl\n' > README.md
cat >> README.md <<'EOF'

## Benchmarks

The harness in `bench/` produces every benchmark number, table and figure the manuscript
uses. It has its own Julia environment (`bench/Project.toml`, with `HNNG` as a dev
dependency) and runs single-threaded, pinned to one core:

```bash
bench/run.sh                    # full run, several hours, results in bench/results/
BENCH_SMOKE=1 bench/run.sh      # every experiment on tiny inputs, results in bench/results-smoke/
```

The experiments run in the order Exp.2, Exp.3, Exp.1 (random-xs), Exp.1b (Fashion-MNIST, the
longest). An experiment whose CSV already exists is skipped, so after a crash the same command
continues where it stopped; `BENCH_FORCE=1` reruns everything and `BENCH_ONLY=exp3` (one of `exp1`,
`exp1b`, `exp2`, `exp3`) restricts the run to one experiment (combine with `BENCH_FORCE=1` to rerun a
finished one). Tables and figures are written only when all four CSV files exist. Smoke runs always
rerun everything.

`BENCH_CPU` picks the core (default 2), `BENCH_DATA` the directory with the ann-benchmarks
HDF5 files (default `/home/coral/working/benchmarks/data`), `BENCH_RESULTS` the output
directory. One step at a time: `julia -t 1 --project=bench bench/exp1.jl` (likewise
`exp2.jl`, `exp3.jl`, `tables.jl`, `plots.jl`). Tests: `julia -t 1 --project=bench bench/test/runtests.jl`.

| Output (in `bench/results/`) | Content |
|---|---|
| `exp1.csv`, `exp1b.csv` | random-xs-20 and fashion-mnist-784: build time, QPS, both recalls, fraction evaluated, hNNG levels and cluster counts, one HNSW row per `ef` |
| `exp2.csv` | Gaussian (n = 5,000) and 100 blobs (n = 10,000), d = 2 to 128: ρ, QPS, build time, levels |
| `exp3.csv` | Gaussian grid n ∈ {8,000, 16,000, 32,000} × d ∈ {16, 64, 128}, every insert and delete: latency, distance evaluations, level-0 rewires, highest level touched; streaming and bulk rebuilds and the baselines |
| `env.txt` | Julia version, Manifest hash, CPU model, governor, allowed CPUs, threads, commit and `dirty` flag, `smoke` flag, seeds, package versions, kernel, date |
| `tex/` | `tab_annbench.tex`, `tab_fashion.tex`, `tab_ops.tex` (tabular bodies) and `macros.tex` |
| `fig/` | `recall_qps.pdf`, `dist_eval_vs_dim.pdf`, `qps_vs_dim.pdf`, `ops_comparison.pdf` |

| Protocol | Setting |
|---|---|
| Threads | `julia -t 1` and `BLAS.set_num_threads(1)`, checked before every experiment |
| CPU | process pinned with `taskset -c 2` |
| Queries | one at a time, `k = 10`; QPS from the best of three timed passes after one warm-up query |
| Build time | one timed build after an untimed warm-up build on 200 points (not best of three) |
| Recall | ann-benchmarks rule (returned distance ≤ true k-th distance + 1e-3) and strict id recall, distances recomputed in Float64 from the returned ids |
| Fraction evaluated ρ | distance evaluations per query (`HNNG.knn_counted`) divided by live elements |
| Exp.3 | 200 inserts, then 200 deletes of random live ids; median over 3 runs of the mean latency per operation (a `GC.gc(false)` before each timed operation, outside the timing); 20 queries checked against brute force after every operation |
| Baselines | brute force (BLAS gemv, exact Float64 re-rank), KD-tree and ball tree (NearestNeighbors.jl, leaf size 25), HNSW.jl (M = 16, efConstruction = 200, ef from 10 to 640), SimilaritySearch.jl when it loads |
| Exactness | the exact methods (hNNG, brute force, KD-tree, ball tree) must return the true k nearest distances (relative tolerance 1e-5) for every query of Exp.1, Exp.1b and Exp.2; the run fails otherwise |

SimilaritySearch.jl method notes: its rows use `Dist.L2()` (Euclidean, like every other method) and
`SearchGraphContext(; neighborhood = Neighborhood(; neardup = 0f0))` with `expand!` after each search.
`neardup = 0f0` folds exact duplicate points into cluster members, so a plain search returns one
representative per duplicate group and would under-count recall on data with repeated points
(Fashion-MNIST); `expand!` returns the raw k nearest neighbours, duplicates included, as for every
other method.

The numbers are Julia measurements; parity with the C++ reference implementation is not a goal.
EOF
grep -c '^## Benchmarks$' README.md
````

Expected: `1`.

- [ ] **Step 6: Run the whole bench test suite**

Run: `BENCH_SMOKE=1 julia -t 1 --project=bench bench/test/runtests.jl`
Expected: `Test Summary:` for `bench` with no `Fail` or `Error` (`Broken` entries only for an absent HNSW.jl or SimilaritySearch).

- [ ] **Step 7: Commit**

```bash
git add bench/run_all.jl bench/run.sh bench/test/test_run_all.jl bench/test/runtests.jl README.md
git commit -m "bench: run_all pipeline, pinned run.sh wrapper, README benchmark section" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: full benchmark run and committed results

**Files:**
- Create: `bench/results/` (every output of spec §11, written by the run)

**Interfaces:**
- Consumes: `bench/run.sh` (Task 11).
- Produces: the committed files the manuscript `\input`s and `\includegraphics`es.

This run takes hours (Fashion-MNIST streaming build and 10,000 queries per method, the nine Exp.3 grid points up to n = 32,000, d = 128). Run it on the machine the manuscript names, with nothing else heavy running, from a clean working tree (`git status` shows nothing outside `bench/results*`), and with `BENCH_SMOKE`, `BENCH_ONLY`, `BENCH_FORCE` and `BENCH_RESULTS` unset. The runner is resumable (assumption A23): the order is Exp.2, Exp.3, Exp.1, Exp.1b, a finished experiment's CSV is kept, and running the same command again skips it.

- [ ] **Step 1: Start the pinned full run in the background**

Run: `bench/run.sh >> bench/run.log 2>&1` (in the background; wait for it to exit)
Expected: exit status 0 and no `Error` line in `bench/run.log`. The log shows one `running exp2`, `running exp3`, `running exp1`, `running exp1b` line each (or `... exists, skipped` for an experiment finished by an earlier invocation), and `bench/results/` holds the four CSV files, `env.txt`, `tex/` and `fig/`. If the run dies, read the error, fix the cause, and run the same command again: the experiments whose CSV exists are skipped. To redo one experiment, use `BENCH_ONLY=exp3 BENCH_FORCE=1 bench/run.sh >> bench/run.log 2>&1`. Tables and figures appear only once all four CSVs exist.

- [ ] **Step 2: Check the run against the protocol, the sizes and the exactness contract**

Run:

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
grep -E '^(threads|blas_threads|cpus_allowed|governor|smoke|dirty|kernel|pkg_versions)=' bench/results/env.txt
julia -t 1 --project=bench -e 'using CSV, DataFrames
res = "bench/results"
env = Dict(String(a) => String(b) for (a, b) in (split(l, "="; limit = 2) for l in eachline(joinpath(res, "env.txt"))))
env["smoke"] == "false" || error("env.txt says smoke=$(env["smoke"]): these are smoke numbers")
e1, e1b, e2, e3 = (CSV.read(joinpath(res, f), DataFrame) for f in ("exp1.csv", "exp1b.csv", "exp2.csv", "exp3.csv"))
all(e1.n .== 9000) && all(e1.nq .== 1000) || error("exp1.csv is not the full random-xs run")
all(e1b.n .== 60000) && all(e1b.nq .== 10000) || error("exp1b.csv is not the full Fashion-MNIST run")
nrow(e2) == 2 * 7 * 4 || error("exp2.csv has $(nrow(e2)) rows, expected 56")
nrow(unique(e3[:, [:n, :d]])) == 9 || error("exp3.csv does not cover the nine grid points")
println("sizes ok: exp1 n=", e1.n[1], " nq=", e1.nq[1], "; exp1b n=", e1b.n[1], " nq=", e1b.nq[1], "; exp2 rows=", nrow(e2), "; exp3 grid points=9")
for (f, df) in (("exp1.csv", e1), ("exp1b.csv", e1b))
    h = df[startswith.(df.method, "hNNG") .| (df.method .== "brute-BLAS"), :]
    println(f, " exact rows recall_ann = ", unique(h.recall_ann))
end'
grep -o 'min_match=[0-9.e-]*' bench/run.log | sort | uniq -c
ls bench/results/tex bench/results/fig
```

Expected: `smoke=false`, `threads=1`, `blas_threads=1`, `cpus_allowed=2`, the governor, `dirty=false`, the kernel and the `pkg_versions` lines; `sizes ok: exp1 n=9000 nq=1000; exp1b n=60000 nq=10000; exp2 rows=56; exp3 grid points=9`; `exp1.csv exact rows recall_ann = [1.0]` and `exp1b.csv exact rows recall_ann = [1.0]`; the `min_match` summary is a single line `<count> min_match=1.0` (the minimum over all queries of the per-query distance-multiset match, A25; any other value means an exact method returned a wrong neighbour that the recall rule missed); the four `.tex` files and the four `.pdf` files. Any other recall or `min_match` for an exact row is a bug: stop and report it. `dirty=true` means the results do not correspond to the recorded commit: commit or discard the pending changes and rerun with `BENCH_FORCE=1`.

- [ ] **Step 3: Commit the results**

```bash
git add bench/results
git commit -m "bench: full benchmark results (CSV, env.txt, tex, figures)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

**Release checklist (author).** These are the author's actions, not an agent's, done once the results commit above exists; there is no code:

- [ ] Tag `v0.1.0` on the commit that holds the frozen `bench/results`.
- [ ] Publish the GitHub release for that tag.
- [ ] Archive the release on Zenodo.
- [ ] Paste the Zenodo DOI and the repository URL into the manuscript's `%% AUTHOR DECISION` markers.
