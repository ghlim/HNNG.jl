isdefined(@__MODULE__, :brute_knn) || include(joinpath(@__DIR__, "oracle.jl"))
using JSON3, Base64

@testset "C++ parity: random-xs-20, n = 2000, k = 10" begin
    fx = JSON3.read(read(joinpath(@__DIR__, "fixtures", "cpp_parity_random_xs_20.json"), String))
    d, ntr, nte, k = fx.dim, fx.n_train, fx.n_test, fx.k
    tr = reshape(ltoh.(reinterpret(Float32, base64decode(fx.train_f32_b64))), d, ntr)
    te = reshape(ltoh.(reinterpret(Float32, base64decode(fx.test_f32_b64))), d, nte)
    idx = HNNGIndex(Euclidean(), d)
    add!(idx, tr)
    @test verify(idx)
    s = stats(idx)
    @info "levels and clusters (printed, not gated: representative election differs, spec §12)" julia_levels = s.nlevels julia_clusters = s.nclusters cpp_levels = fx.cpp_nlevels cpp_clusters = collect(fx.cpp_nclusters)
    for j in 1:nte
        ids, ds = knn(idx, view(te, :, j), k)
        cpp = sort(sqrt.(Float64.(fx.cpp_sqdists[j])))
        @test all(isapprox.(sort(ds), cpp; rtol=1e-5))
        bf = Float64.(fx.bf_dists[j])
        if bf[k+1] - bf[k] > 1e-6 * bf[k]
            @test sort(ids) == sort(Int.(fx.cpp_ids[j]) .+ 1)
        end
    end
end
