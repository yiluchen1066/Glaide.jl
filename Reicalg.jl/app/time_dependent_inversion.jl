using Reicalg
using CUDA
using CairoMakie
using Printf

function time_dependent_inversion(filepath; As_init, E, β_reg, γ, niter, momentum, α)
    model = TimeDependentSIA(filepath)

    (; dx, dy, xc, yc) = model.numerics
    (; As, V, H)       = model.fields

    As0 = As

    model.scalars.A *= E

    V_obs = copy(V)
    H_obs = copy(H)

    ωᵥ = α * inv(sum(V_obs .^ 2))
    ωₕ = √(1 - α^2) * inv(sum(H_obs .^ 2))

    fill!(As0, As_init)
    solve!(model)

    objective      = TimeDependentObjective(ωᵥ, ωₕ, V_obs, H_obs)
    regularisation = Regularisation(β_reg, dx, dy, As)

    xc_km = xc / 1e3
    yc_km = yc / 1e3

    fig = Figure(; size=(650, 750))

    #! format:off
    ax = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="log10(As)"),
          Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="Ās"),
          Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="V_obs"),
          Axis(fig[2, 2][1, 1]; aspect=DataAspect(), title="V"),
          Axis(fig[3, 1][1, 1]; aspect=DataAspect(), title="H_obs"),
          Axis(fig[3, 2][1, 1]; aspect=DataAspect(), title="H"))

    hm = (heatmap!(ax[1], xc_km, yc_km, Array(log10.(As0)); colormap=:turbo),
          heatmap!(ax[2], xc_km, yc_km, Array(log10.(As0)); colormap=:turbo),
          heatmap!(ax[3], xc_km, yc_km, Array(V_obs)      ; colormap=:turbo, colorrange=(0, 1e-5)),
          heatmap!(ax[4], xc_km, yc_km, Array(V)          ; colormap=:turbo, colorrange=(0, 1e-5)),
          heatmap!(ax[5], xc_km, yc_km, Array(H_obs)      ; colormap=:turbo, colorrange=(0, 800)),
          heatmap!(ax[6], xc_km, yc_km, Array(H)          ; colormap=:turbo, colorrange=(0, 800)))

    cb = (Colorbar(fig[1, 1][1, 2], hm[1]),
          Colorbar(fig[1, 2][1, 2], hm[2]),
          Colorbar(fig[2, 1][1, 2], hm[3]),
          Colorbar(fig[2, 2][1, 2], hm[4]),
          Colorbar(fig[3, 1][1, 2], hm[5]),
          Colorbar(fig[3, 2][1, 2], hm[6]))
    #! format:on

    function callback(iter, γ, J1, As, Ās)
        if iter % 10 == 0
            @printf("  iter = %d, J = %1.3e\n", iter, J1)

            hm[1][3] = Array(log10.(As))
            hm[2][3] = Array(Ās)
            hm[4][3] = Array(V)
            hm[6][3] = Array(H)
            autolimits!(ax[1])
            autolimits!(ax[2])
            display(fig)
        end
    end

    gradient_descent(model, objective, As0, γ, niter; regularisation, callback, momentum)

    return
end

time_dependent_inversion("datasets/synthetic/synthetic_setup.jld2";
                         As_init=1e-20, E=1.0, β_reg=1.0e-2, γ=1e3, niter=2000, momentum=0.5, α=0.5√2)

time_dependent_inversion("datasets/aletsch/aletsch_setup.jld2";
                         As_init=1e-20, E=2.0e-1, β_reg=1.0e-3, γ=2e3, niter=2000, momentum=0.5, α=0.5√2)
