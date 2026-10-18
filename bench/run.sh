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
# Julia is pinned to 1.13.2, the version bench/Manifest.toml was resolved on (once: juliaup add 1.13.2).
set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"     # juliaup's PATH entry lives in the login profile
case "${BENCH_ONLY:-}" in
    ""|exp1|exp1b|exp2|exp3) ;;
    *) echo "run.sh: BENCH_ONLY must be one of exp1, exp1b, exp2, exp3 (got '${BENCH_ONLY}')" >&2; exit 1 ;;
esac
cd "$(dirname "$0")/.."
command -v taskset >/dev/null || { echo "run.sh: taskset not found (util-linux)" >&2; exit 1; }
julia +1.13.2 --project=bench -e 'using Pkg; Pkg.instantiate()'
exec taskset -c "${BENCH_CPU:-2}" julia +1.13.2 -t 1 --project=bench bench/run_all.jl
