# Distance kernels (spec §5): Float32 data accumulates in Float32, the result is widened.
#
# Explicit SIMD vectors built from `NTuple{W,VecElement{T}}` with `Base.llvmcall` for
# fsub/fmul/fadd (Core/Base only). The summation order is spelled out in `_euclid`'s
# docstring and does not depend on how LLVM vectorises anything.

const _V8f = NTuple{8,VecElement{Float32}}
const _V4d = NTuple{4,VecElement{Float64}}

# Lane-wise IEEE ops on <W x T>. Plain fsub/fmul/fadd (no fast-math flags, no fma),
# so lane l of the result is bit-identical to the scalar op on lane l.
for (V, lt, W) in ((:_V8f, "float", 8), (:_V4d, "double", 4))
    vt = "<$W x $lt>"
    @eval begin
        @inline _vsub(x::$V, y::$V) = Base.llvmcall($"%3 = fsub $vt %0, %1\nret $vt %3", $V, Tuple{$V,$V}, x, y)
        @inline _vmul(x::$V, y::$V) = Base.llvmcall($"%3 = fmul $vt %0, %1\nret $vt %3", $V, Tuple{$V,$V}, x, y)
        @inline _vadd(x::$V, y::$V) = Base.llvmcall($"%3 = fadd $vt %0, %1\nret $vt %3", $V, Tuple{$V,$V}, x, y)
    end
end
_vtype(::Type{Float32}) = _V8f
_vtype(::Type{Float64}) = _V4d

# True when `x` is a strided vector with unit stride, so `pointer(x)` addresses
# `x[firstindex(x)]` and the next elements follow contiguously (Vector, contiguous
# column views, reshapes, …). Public API only: `StridedVector`, `stride`, `pointer`.
@inline _unitstride(x::StridedVector) = stride(x, 1) == 1
@inline _unitstride(::AbstractVector) = false

# Load W lanes holding coordinates k+1 … k+W (k is a 0-based offset).
# Pointer path: one unaligned (align 1) vector load. Generic path: W getindex calls.
# Both produce the same lane values, so everything downstream is identical.
@inline _vload(::Type{V}, p::Ptr{T}, k::Int) where {V,T} = unsafe_load(Ptr{V}(p + k * sizeof(T)))
@inline _vload(::Type{NTuple{W,VecElement{T}}}, a::AbstractVector{T}, k::Int) where {W,T} =
    ntuple(l -> VecElement(@inbounds a[firstindex(a) + k + l - 1]), Val(W))

@inline function _vsq(acc::V, x::V, y::V) where {V}
    t = _vsub(x, y)
    _vadd(acc, _vmul(t, t))
end

# Halving tree: v[l] + v[l + W/2] for l = 1 … W/2, repeated until one lane remains.
@inline function _vhsum(v::NTuple{W,VecElement{T}}) where {W,T}
    W == 1 && return v[1].value
    h = W ÷ 2
    _vhsum(ntuple(l -> VecElement(v[l].value + v[l + h].value), Val(h)))
end

# `pa`, `pb` are pointers (dense path) or the arrays themselves (generic path);
# `a`, `b` are always the arrays (used for the scalar tail).
@inline function _euclid_core(::Type{V}, pa, pb, a, b, n::Int) where {W,T,V<:NTuple{W,VecElement{T}}}
    z = ntuple(_ -> VecElement(zero(T)), Val(W))
    c1 = c2 = c3 = c4 = z
    i = 0                                   # coordinates consumed so far
    while i + 4W <= n
        c1 = _vsq(c1, _vload(V, pa, i),      _vload(V, pb, i))
        c2 = _vsq(c2, _vload(V, pa, i + W),  _vload(V, pb, i + W))
        c3 = _vsq(c3, _vload(V, pa, i + 2W), _vload(V, pb, i + 2W))
        c4 = _vsq(c4, _vload(V, pa, i + 3W), _vload(V, pb, i + 3W))
        i += 4W
    end
    while i + W <= n
        c1 = _vsq(c1, _vload(V, pa, i), _vload(V, pb, i))
        i += W
    end
    s = _vhsum(_vadd(_vadd(c1, c2), _vadd(c3, c4)))
    oa, ob = firstindex(a) - 1, firstindex(b) - 1
    while i < n
        i += 1
        t = @inbounds(a[oa + i]) - @inbounds(b[ob + i])
        s += t * t
    end
    s
end

# Same order for any other float type (Float16, BigFloat, …), in plain scalar code, W = 8.
function _euclid_scalar(a::AbstractVector{T}, b::AbstractVector{T}, W::Int) where {T}
    n = length(a)
    oa, ob = firstindex(a) - 1, firstindex(b) - 1
    acc = [zero(T) for _ in 1:W, _ in 1:4]
    i = 0
    while i + 4W <= n
        for c in 1:4, l in 1:W
            k = i + (c - 1) * W + l
            t = a[oa + k] - b[ob + k]
            acc[l, c] = acc[l, c] + t * t
        end
        i += 4W
    end
    while i + W <= n
        for l in 1:W
            t = a[oa + i + l] - b[ob + i + l]
            acc[l, 1] = acc[l, 1] + t * t
        end
        i += W
    end
    v = [(acc[l, 1] + acc[l, 2]) + (acc[l, 3] + acc[l, 4]) for l in 1:W]
    while length(v) > 1
        h = length(v) ÷ 2
        v = [v[l] + v[l + h] for l in 1:h]
    end
    s = v[1]
    while i < n
        i += 1
        t = a[oa + i] - b[ob + i]
        s = s + t * t
    end
    s
end

"""
    _euclid(a, b) -> Float64

Euclidean distance accumulated in the element type `T`, widened to `Float64`, square
root taken in `Float64`. The summation order is fixed by the code (no `@simd`,
`@fastmath`, `fma`/`muladd`), so the bits do not depend on inlining, `--check-bounds`,
argument types (Vector, contiguous or strided view), Julia version or CPU.

Order, with `W = 8` lanes for `Float32` (and any other `T` except `Float64`), `W = 4`
for `Float64`, and coordinates numbered `1 … n` (`n = length(a)`). Every term is
`t = a[k] - b[k]` then `t * t`, each rounded separately, added to its accumulator:

1. Four accumulators `c1 … c4` of `W` lanes each, all `+0`. Blocks of `4W`
   coordinates starting at `i+1`: lane `l` of `c_j` gets coordinate `i + (j-1)W + l`.
   Repeat while a full block of `4W` remains.
2. Then, while at least `W` coordinates remain, blocks of `W` go into `c1`:
   lane `l` gets coordinate `i + l`.
3. Lane-wise combine `v = (c1 + c2) + (c3 + c4)`.
4. Horizontal halving: `v[l] + v[l + W/2]` for `l = 1 … W/2`, repeated to one lane
   (`W = 8`: `((v1+v5)+(v3+v7)) + ((v2+v6)+(v4+v8))`; `W = 4`: `(v1+v3) + (v2+v4)`).
5. The remaining `n mod W` coordinates are added to that scalar left to right.

Swapping `a` and `b` gives the same bits (each term is the square of an exact
negation). When both inputs are `StridedVector`s with `stride(x, 1) == 1` (`Vector`,
contiguous views, reshapes) the lanes are filled by unaligned vector loads through
`pointer` (inside `GC.@preserve`); any other `AbstractVector` builds the same lanes
with `getindex`, and gets the same bits. Throws `DimensionMismatch` if the lengths
differ; any length `≥ 0` and any `firstindex` is accepted.
"""
@inline function _euclid(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(a)
    length(b) == n || throw(DimensionMismatch("lengths $n and $(length(b)) differ"))
    s = if T === Float32 || T === Float64
        V = _vtype(T)
        if _unitstride(a) && _unitstride(b)
            GC.@preserve a b _euclid_core(V, pointer(a), pointer(b), a, b, n)
        else
            _euclid_core(V, a, b, a, b, n)
        end
    else
        _euclid_scalar(a, b, 8)
    end
    sqrt(Float64(s))
end
