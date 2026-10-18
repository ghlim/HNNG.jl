# Timing protocol (spec §11) and the env.txt record.
export check_protocol, best_of, time_queries, timed_queries, write_env

"""
    check_protocol()

Refuses to run unless Julia was started with `-t 1` (one thread in total, no interactive pool), then sets
`BLAS.set_num_threads(1)`. Every experiment calls it first.
"""
function check_protocol()
    # maxthreadid, not nthreads: plain `julia` (1.12+) has 1 default + 1 interactive thread,
    # so nthreads() is 1 but maxthreadid() is 2.
    Threads.maxthreadid() == 1 ||
        error("the benchmark protocol needs `julia -t 1`; this process has " *
              "$(Threads.nthreads()) threads and $(Threads.nthreads(:interactive)) interactive threads")
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

"""
    timed_queries(h, qs, k) -> (results, seconds)

`time_queries(q -> query(h, q, k), qs)` behind a function barrier: the experiments hold the
handle as a non-inferred value (build_time over `m::BenchMethod`), so a closure built at the
call site would dispatch `query` dynamically on every query; here `h` has its concrete type.
"""
timed_queries(h, qs::AbstractVector, k::Int) = time_queries(q -> query(h, q, k), qs)

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
                  "HNSW=$SEED_HNSW", "SS=$SEED_SS"), ",")
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
        "interactive_threads=$(Threads.nthreads(:interactive))",
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
