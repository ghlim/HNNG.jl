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
    idx.nlive += 1
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
