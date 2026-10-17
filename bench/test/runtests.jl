include(joinpath(@__DIR__, "setup.jl"))

@testset "bench" begin
    include("test_datasets.jl")
end
