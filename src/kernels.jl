# Distance kernels (spec §5): Float32 data accumulates in Float32, the result is widened.

"""
    _euclid(a, b) -> Float64

Euclidean distance accumulated in the element type, widened to `Float64`.
Swapping `a` and `b` gives the same bits (each term is the square of an exact negation).
"""
@inline function _euclid(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    s = zero(T)
    @inbounds @simd for i in eachindex(a, b)
        t = a[i] - b[i]
        s += t * t
    end
    sqrt(Float64(s))
end
