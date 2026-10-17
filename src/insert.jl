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
