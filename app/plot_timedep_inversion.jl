using Glaide, CairoMakie, JLD2, Printf

γ_reg      = 1.6e-9
resolution = 50
setup      = "synthetic"
# setup      = "aletsch"
ωᵥ         = 1
ωₕ         = 1

# outdir = "../output/time_dependent_$(setup)_$(resolution)m"
# outdir = "../output/time_dependent_$(setup)_$(resolution)m_1_10"
# outdir = "../output/time_dependent_$(setup)_$(resolution)m_reg_1.0e-9"
# outdir = "../output/time_dependent_$(setup)_$(resolution)m_reg_1.2e-8"
# outdir = "../output/time_dependent_$(setup)_$(resolution)m_reg_2.3e-8"
# outdir = "../output/time_dependent_$(setup)_$(resolution)m_reg_3.4e-8"
# outdir = "../output/time_dependent_$(setup)_$(resolution)m_reg_$(γ_reg)"
outdir = "../output/time_dependent_$(setup)_$(resolution)m_$(ωᵥ)_$(ωₕ)"
# outdir = "../output/time_dependent_$(setup)_$(resolution)m_test"

let fig = Figure(; size=(850, 600))
    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="H"),
           Axis(fig[1, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="V"),
           Axis(fig[1, 3][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="As"),
           Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="H_obs"),
           Axis(fig[2, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="V_obs"),
           Axis(fig[2, 3][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="As_syn"))

    istep = 0
    fname = @sprintf("step_%04d.jld2", istep)
    (X, V, H) = load(joinpath(outdir, fname), "X", "V", "H")
    (fields, scalars, numerics) = load("../datasets/$(setup)_$(resolution)m.jld2", "fields", "scalars", "numerics")

    (; xc, yc) = numerics

    dV = (V .- fields.V) ./ fields.V .* 100
    dV[fields.H.<0.01] .= NaN

    dH = (H .- fields.H) ./ fields.H .* 100

    hms = (heatmap!(axs[1], xc, yc, H .* 100),
           heatmap!(axs[2], xc, yc, V),
           heatmap!(axs[3], xc, yc, X .* log10(ℯ)),
           heatmap!(axs[4], xc, yc, dH),
           #    heatmap!(axs[5], xc, yc, fields.V),
           heatmap!(axs[5], xc, yc, dV),
           heatmap!(axs[6], xc, yc, fields.V))

    hms[1].colorrange = (0, 900)
    hms[2].colorrange = (0, 3)
    hms[3].colorrange = (-1, 4)
    hms[6].colorrange = (0, 3)
    hms[4].colorrange = (-25, 25)
    hms[5].colorrange = (-65, 65)

    hms[1].colormap = Reverse(:ice)
    hms[4].colormap = :diverging_bwg_20_95_c41_n256

    hms[2].colormap = :matter
    # hms[5].colormap = :matter
    hms[5].colormap = :BrBG_9

    hms[3].colormap = Reverse(:roma)
    hms[6].colormap = :matter

    cbs = (Colorbar(fig[1, 1][1, 2], hms[1]),
           Colorbar(fig[1, 2][1, 2], hms[2]),
           Colorbar(fig[1, 3][1, 2], hms[3]),
           Colorbar(fig[2, 1][1, 2], hms[4]),
           Colorbar(fig[2, 2][1, 2], hms[5]),
           Colorbar(fig[2, 3][1, 2], hms[6]))

    #     display(fig)

    for istep in 100:100:100
        fname = @sprintf("step_%04d.jld2", istep)
        (X, V, H) = load(joinpath(outdir, fname), "X", "V", "H")

        mask = fields.H .<= 0.0 .|| fields.V .<= 0.0

        dV = (V .- fields.V) ./ fields.V .* 100
        dV[mask] .= NaN

        dH = (H .- fields.H) ./ fields.H .* 100
        dH[mask] .= NaN
        dH[mask] .= NaN

        X[mask] .= NaN
        X[mask] .= NaN

        hms[1][3] = H .* 100
        hms[2][3] = V
        hms[3][3] = X .* log10(ℯ)
        hms[4][3] = dH
        hms[5][3] = dV
        display(fig)
    end
end
