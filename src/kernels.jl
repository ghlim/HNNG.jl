# Distance kernels (spec §5): Float32 data accumulates in Float32, the result is widened.

"""
    _euclid(a, b) -> Float64

Euclidean distance accumulated in the element type, widened to `Float64`, square
root taken in `Float64`. The summation order is fixed: eight independent
accumulators `s1..s8` take the squared differences in lane order over blocks of 8
(`s_j` gets coordinates `j, j+8, j+16, …`), are combined as
`((s1+s2)+(s3+s4))+((s5+s6)+(s7+s8))`, and the remaining `n mod 8` terms are added
left to right. No `@simd`, `@fastmath` or `fma` is used, so the bits are the same
at every call site, for every argument type, and under every compilation mode
(inlining, `--check-bounds`, Julia version). Swapping `a` and `b` gives the same
bits (each term is the square of an exact negation).
"""
@inline function _euclid(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(a)
    length(b) == n || throw(DimensionMismatch("lengths $n and $(length(b)) differ"))
    oa, ob = firstindex(a) - 1, firstindex(b) - 1
    z = zero(T)
    s1 = s2 = s3 = s4 = s5 = s6 = s7 = s8 = z
    i = 1
    @inbounds while i <= n - 7
        t1 = a[oa+i]   - b[ob+i];   s1 += t1 * t1
        t2 = a[oa+i+1] - b[ob+i+1]; s2 += t2 * t2
        t3 = a[oa+i+2] - b[ob+i+2]; s3 += t3 * t3
        t4 = a[oa+i+3] - b[ob+i+3]; s4 += t4 * t4
        t5 = a[oa+i+4] - b[ob+i+4]; s5 += t5 * t5
        t6 = a[oa+i+5] - b[ob+i+5]; s6 += t6 * t6
        t7 = a[oa+i+6] - b[ob+i+6]; s7 += t7 * t7
        t8 = a[oa+i+7] - b[ob+i+7]; s8 += t8 * t8
        i += 8
    end
    s = ((s1 + s2) + (s3 + s4)) + ((s5 + s6) + (s7 + s8))
    @inbounds while i <= n
        t = a[oa+i] - b[ob+i]
        s += t * t
        i += 1
    end
    sqrt(Float64(s))
end
