isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))
using Graphs

# The README cannot rot: every ```julia block of README.md except the Install block
# (the one with `Pkg.add`) runs, in order, in one fresh module. The recipes the
# testset below checks are the functions those blocks define.
const README_BLOCKS = let text = read(joinpath(@__DIR__, "..", "README.md"), String)
    [String(m.captures[1]) for m in eachmatch(r"^```julia\n(.*?)^```$"ms, text)
     if !occursin("Pkg.add", m.captures[1])]
end

const README_MODULE = Module(:READMERecipes)

@testset "README code blocks run" begin
    @test length(README_BLOCKS) == 5
    for (i, code) in enumerate(README_BLOCKS)
        @test (include_string(README_MODULE, code, "README.md block $i"); true)
    end
    @test verify(README_MODULE.idx)
    @test verify(README_MODULE.views)
end

@testset "README recipes" begin
    recognise = README_MODULE.recognise
    single_label_subtrees = README_MODULE.single_label_subtrees
    rng = StableRNG(25)
    centres = [-6.0 0.0 6.0; 0.0 5.0 0.0]
    X = reduce(hcat, [centres[:, c] .+ randn(rng, 2) for c in repeat(1:3, 40)])
    labels = repeat(["cup", "bowl", "plate"], 40)
    idx = HNNGIndex(Euclidean(), 2; T=Float64)
    add!(idx, X)
    @test verify(idx)
    @test recognise(idx, labels, [-6.0, 0.2]) == "cup"
    n0 = idx.nevals
    parts = single_label_subtrees(idx, labels, "bowl")
    @test idx.nevals == n0                                   # no distance evaluated
    @test sort!(reduce(vcat, parts)) == findall(==("bowl"), labels)
    @test all(p -> all(i -> labels[i] == "bowl", p), parts)
    # nearest-neighbour chain and shortest path over stored distances
    chain = [1]
    while outlink(idx, outlink(idx, chain[end])) != chain[end]
        push!(chain, outlink(idx, chain[end]))
    end
    g, W, ids = level_graph(idx, 0)
    st = dijkstra_shortest_paths(g, 1, W)
    @test st.dists[chain[end]] ≈ sum(outlink_distance(idx, x) for x in chain[1:end-1]; init=0.0)
end
