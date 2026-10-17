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
            d = _checked(idx, _rawdist(idx, p, _point(idx.store, pop[j])), pop[i], pop[j])
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

bulk_add!(idx::HNNGIndex{<:VectorStore}, X::AbstractMatrix{<:Real}) = _locked(() -> _bulk_add!(idx, X), idx)

function _bulk_add!(idx::HNNGIndex{<:VectorStore}, X::AbstractMatrix{<:Real})
    _ntotal(idx) == 0 ||
        throw(ArgumentError("bulk_add! needs an empty index; this index has already assigned $(_ntotal(idx)) ids"))
    _check_dim(idx, X)
    cols = [_prepare(idx, view(X, :, j)) for j in axes(X, 2)]
    isempty(cols) && return 1:0
    try
        _bulk_build!(idx, cols)
    catch
        _reset_empty!(idx)                   # the index was empty: nothing else to restore
        rethrow()
    end
end

"""Return an index that never held an element to that state (after a failed `bulk_add!`)."""
function _reset_empty!(idx::HNNGIndex{VectorStore{T}}) where {T}
    disable_sigint() do
        idx.store.data = Matrix{T}(undef, dim(idx.store), 0)
        idx.store.n = 0
        empty!(idx.alive)
        idx.nlive = 0
        empty!(idx.levels)
        _flush!(idx)
    end
    idx
end

function _bulk_build!(idx::HNNGIndex{<:VectorStore}, cols)
    for v in cols
        _append!(idx.store, v)
        push!(idx.alive, true)
        idx.nlive += 1
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

_allpairs_nn(idx::HNNGIndex{<:VectorStore,<:EuclidFamily}, pop::Vector{Int}) = _allpairs_nn_blas(idx, pop)
