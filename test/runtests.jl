using Test, HNNG

@testset "HNNG.jl" begin
    include("oracle.jl")
    include("metrics.jl")
    include("invariants.jl")
    include("search.jl")
    include("knn_stored.jl")
    include("pf_filter.jl")
    include("updates.jl")
    include("objects.jl")
    include("order.jl")
    include("navigate.jl")
    include("io.jl")
    include("parity_cpp.jl")
    include("errors.jl")
    include("safety.jl")
    include("recipes.jl")
    if get(ENV, "HNNG_SLOW", "0") == "1"
        include("slow.jl")
    end
end
