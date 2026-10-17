include(joinpath(@__DIR__, "setup.jl"))

@testset "baselines (approximate and optional)" begin
    X, Q = synthetic("blobs", 300, 20, 8)
    nb, gd = ground_truth(X, Q, 10)
    qs = columns(Q)

    if HAS_HNSW
        @testset "HNSW.jl adapter" begin
            m = HNSWjl()
            @test (m.M, m.efConstruction, m.ef) == (16, 200, 10)
            @test method_name(m) == "HNSW"
            h, t = build_time(m, X)
            @test t > 0
            # standard HNSW at M = 16: 16 links per upper layer, 32 on layer 0, m_L = 1/ln 16
            @test h.g.lgraph.M == 16 && h.g.lgraph.M0 == 32 && h.g.lgraph.m_L ≈ 1 / log(16)
            set_ef!(h, 200)
            res = [query(h, q, 10) for q in qs]
            @test all(r[1] isa Vector{Int} && issorted(collect(zip(r[2], r[1]))) for r in res)
            ra, _ = mean_recalls(X, Q, nb, gd, res, 10)
            @test ra >= 0.95
        end

        @testset "HNSW.jl online insert of columns already in its data vector" begin
            h = build_hnsw(HNSWjl(ef = 50), X; nbuild = 250)
            q300 = X[:, 300]
            @test 300 ∉ query(h, q300, 5)[1]
            for i in 251:300
                hnsw_insert!(h, i)
            end
            ids, ds = query(h, q300, 5)
            @test ids[1] == 300 && ds[1] == 0.0
            # a driver that re-inserts an id fails loudly instead of being skipped
            @test_throws ArgumentError hnsw_insert!(h, 300)
            @test_throws ArgumentError hnsw_insert!(h, 1)
        end
    else
        @info "HNSW.jl not loaded; HNSW rows are skipped"
        @test_skip HAS_HNSW
    end

    @testset "SimilaritySearch rows (optional)" begin
        @test (length(optional_methods()) == 2) == HAS_SS
        if HAS_SS
            @test method_name.(optional_methods()) == ["SS-SearchGraph", "SS-Exhaustive"]
            for (m, minrecall) in ((SSExhaustive(), 1.0), (SSSearchGraph(), 0.9))
                h, _ = build_time(m, X)
                res = [query(h, q, 10) for q in qs]
                # result convention, checked: only filled slots come back (no padding id),
                # every id is a column of X, and distances are Euclidean, not squared
                @test all(r -> length(r[1]) == 10 && all(i -> 1 <= i <= size(X, 2), r[1]), res)
                @test all(j -> all(isapprox.(res[j][2], [dist64(view(X, :, i), qs[j]) for i in res[j][1]];
                                             rtol = 1e-4, atol = 1e-4)), eachindex(res))
                ra, _ = mean_recalls(X, Q, nb, gd, res, 10)
                @test ra >= minrecall
            end
            # exact duplicates come back as separate neighbours (neardup = 0f0 folds them, expand! restores them)
            Xd = hcat(X[:, 1:100], X[:, 1:100])                  # every point twice
            hd = build(SSSearchGraph(), Xd)
            ids, ds = query(hd, X[:, 1], 2)
            @test sort(ids) == [1, 101] && all(<(1e-4), ds)
        else
            @info "SimilaritySearch not loaded; SS rows are skipped"
            @test_skip HAS_SS
        end
    end
end
