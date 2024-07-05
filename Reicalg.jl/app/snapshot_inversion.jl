using Reicalg
using CUDA
using CairoMakie
using Printf

function snapshot_inversion(filepath; As_init, E, β_reg, γ, niter, momentum)
    # model = TimeDependentSIA(filepath)
    # fill!(model.fields.As, As_init)
    # model.scalars.A *= E
    # model.scalars.dt = 1.0 * SECONDS_IN_YEAR

    # solve!(model)

    # H_spinup = copy(model.fields.H)

    model = SnapshotSIA(filepath)

    (; dx, dy, xc, yc) = model.numerics
    (; As, V)          = model.fields

    # copy!(model.fields.H, H_spinup)

    As0 = As

    model.scalars.A *= E

    V_obs = copy(V)
    ωᵥ    = inv(sum(V_obs .^ 2))

    fill!(model.fields.As, As_init)
    solve!(model)

    objective = SnapshotObjective(ωᵥ, V_obs)
    regularisation = Regularisation(β_reg, dx, dy, As)

    xc_km = xc ./ 1e3
    yc_km = yc ./ 1e3

    fig = Figure(; size=(650, 450))

    #! format:off
    ax = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="log10(As)"),
          Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="Ās"),
          Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="V_obs"),
          Axis(fig[2, 2][1, 1]; aspect=DataAspect(), title="V"))

    hm = (heatmap!(ax[1], xc_km, yc_km, Array(log10.(As0)); colormap=:turbo),
          heatmap!(ax[2], xc_km, yc_km, Array(log10.(As0)); colormap=:turbo),
          heatmap!(ax[3], xc_km, yc_km, Array(V_obs)      ; colormap=:turbo, colorrange=(0, 1e-5)),
          heatmap!(ax[4], xc_km, yc_km, Array(V)          ; colormap=:turbo, colorrange=(0, 1e-5)))

    cb = (Colorbar(fig[1, 1][1, 2], hm[1]),
          Colorbar(fig[1, 2][1, 2], hm[2]),
          Colorbar(fig[2, 1][1, 2], hm[3]),
          Colorbar(fig[2, 2][1, 2], hm[4]))
    #! format:on

    function callback(iter, γ, J1, As, Ās)
        if iter % 100 == 0
            @printf("  iter = %d, J = %1.3e\n", iter, J1)

            hm[1][3] = Array(log10.(As))
            hm[2][3] = Array(Ās)
            hm[4][3] = Array(V)
            autolimits!(ax[1])
            autolimits!(ax[2])
            autolimits!(ax[4])
            display(fig)
        end
    end

    gradient_descent(model, objective, As0, γ, niter; regularisation, callback, momentum)

    return
end

snapshot_inversion("datasets/synthetic/synthetic_setup.jld2";
                   As_init=1e-22, E=1.0, β_reg=1.0e-3, γ=2e3, niter=2000, momentum=0.8)

snapshot_inversion("datasets/aletsch/aletsch_setup.jld2";
                   As_init=1e-22, E=1e-2, β_reg=1.0e-3, γ=1e3, niter=2000, momentum=0.8)
