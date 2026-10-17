# verify(idx): the invariants of spec §6.1, recomputed from scratch.

_fail(msg::AbstractString) = error("HNNG.verify: " * msg)

"""
    verify(idx) -> true

Check every invariant of spec §6.1 from scratch: populations; exact 1-NN outlinks
with `(distance, id)` ties (brute force over each level's population); inlinks as
the sorted reverse index; clusters as connected components; the representative
rule; `repdist`, `radius` and `maxout` recomputed bottom-up; one cluster at the
top; the live counter `nlive` behind `length`. Throws
`ErrorException("HNNG.verify: ...")` naming the first violation.
Distance evaluations made here are not counted. Assumes a symmetric distance.
"""
function verify(idx::HNNGIndex)
    n = _ntotal(idx)
    length(idx.alive) == n || _fail("alive has length $(length(idx.alive)) but the store holds $n")
    if isempty(idx.levels)
        any(idx.alive) && _fail("the index has live elements but no levels")
        return _verify_nlive(idx)
    end
    for l in 0:nlevels(idx)-1
        _verify_level(idx, l)
    end
    _verify_nlive(idx)
    _verify_top(idx)
end

"""The O(1) live counter behind `length(idx)` equals the number of present level-0 elements."""
function _verify_nlive(idx::HNNGIndex)
    p = isempty(idx.levels) ? 0 : count(idx.levels[1].present)
    idx.nlive == p || _fail("live counter is $(idx.nlive) but level 0 holds $p present elements")
    true
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
            seen[m] && _fail("level $l: element $m is in two clusters")
            lv.cluster_of[m] == ci || _fail("level $l: cluster_of[$m] is $(lv.cluster_of[m]), expected $ci")
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
    freeseen = Set{Int}()
    for ci in lv.free
        1 <= ci <= length(lv.clusters) || _fail("level $l: free slot $ci is out of range")
        ci in freeseen && _fail("level $l: free slot $ci is listed twice")
        push!(freeseen, ci)
        lv.clusters[ci].rep == 0 || _fail("level $l: free slot $ci holds a live cluster")
    end
    true
end
