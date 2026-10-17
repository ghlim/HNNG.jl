using Test, HNNG

@testset "HNNG.jl" begin
    include("oracle.jl")
    include("metrics.jl")
    include("invariants.jl")
    include("search.jl")
    include("updates.jl")
    include("objects.jl")
    include("order.jl")
end
