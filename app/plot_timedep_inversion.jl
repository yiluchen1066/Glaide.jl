using CairoMakie, JLD2

outdir = "../output/time_dependent_synthetic_50m_0_1"

let fig = Figure(; size=(800, 450))
    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="H"),
           Axis(fig[1, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="V"),
           Axis(fig[1, 3][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="As"),
           Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="H_obs"),
           Axis(fig[2, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="V_obs"),
           Axis(fig[2, 3][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="As_syn"))

    istep = 0
    fname = @sprintf("step_%04d.jld2", istep)
    (X, V, H) = load(joinpath(outdir, fname), "X", "V", "H")
    (fields, scalars, numerics) = load("../datasets/synthetic_50m.jld2", "fields", "scalars", "numerics")

    (; xc, yc) = numerics

    hms = (heatmap!(axs[1], xc, yc, H .* 100),
           heatmap!(axs[2], xc, yc, V),
           heatmap!(axs[3], xc, yc, X .* log10(ℯ)),
           heatmap!(axs[4], xc, yc, fields.H .* 100),
           heatmap!(axs[5], xc, yc, fields.V),
           heatmap!(axs[6], xc, yc, log10.(fields.ρgnAs)))

    hms[1].colorrange = (0, 150)
    hms[4].colorrange = (0, 150)

    hms[1].colormap = Reverse(:ice)
    hms[4].colormap = Reverse(:ice)

    hms[2].colormap = :matter
    hms[5].colormap = :matter

    hms[3].colormap = Reverse(:roma)
    hms[6].colormap = Reverse(:roma)

    cbs = (Colorbar(fig[1, 1][1, 2], hms[1]),
           Colorbar(fig[1, 2][1, 2], hms[2]),
           Colorbar(fig[1, 3][1, 2], hms[3]),
           Colorbar(fig[2, 1][1, 2], hms[4]),
           Colorbar(fig[2, 2][1, 2], hms[5]),
           Colorbar(fig[2, 3][1, 2], hms[6]))

    display(fig)

    for istep in 0:10:50
        fname = @sprintf("step_%04d.jld2", istep)
        (X, V, H) = load(joinpath(outdir, fname), "X", "V", "H")
        hms[1][3] = H .* 100
        hms[2][3] = V
        hms[3][3] = X .* log10(ℯ)
        display(fig)
    end
end
