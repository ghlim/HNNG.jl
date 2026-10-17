# Datasets: ann-benchmarks HDF5 loader, synthetic generators, exact ground truth.
export Dataset, data_dir, load_hdf5, smoke_subset, columns, dist64, ground_truth,
       gaussian, blob_centres, blobs, synthetic, draw_index,
       SEED_GAUSS_TRAIN, SEED_GAUSS_QUERY, SEED_BLOBS_CENTRES, SEED_BLOBS_TRAIN,
       SEED_BLOBS_QUERY, SEED_EXP3_POINTS, SEED_EXP3_DELETE, SEED_EXP3_CHECK, SEED_HNSW, SEED_SS

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
const SEED_SS = 9500  # Random.seed! before every SimilaritySearch build

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
