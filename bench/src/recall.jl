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
