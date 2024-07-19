using Reicalg
using CUDA
using CairoMakie
using Printf
using JLD2

Base.@kwdef struct SnapshotInversionParams{T,I,LS}
    # initial sliding parameter value
    As_init::T = 1e-22
    # ice flow enhancement factor
    E::T = 1.0
    # regularisation parameter
    β_reg::T = 1.0e-3
    # maximum number of iterations in the optimisation algorithm
    maxiter::I = 1000
    # line search algorithm
    line_search::LS
end

function snapshot_inversion(filepath, params::SnapshotInversionParams)
    # unpack params
    (; As_init, E, β_reg, maxiter, line_search) = params

    model = SnapshotSIA(filepath)

    (; xc, yc) = model.numerics
    (; As, V)  = model.fields

    As0 = As

    model.scalars.A *= E

    V_obs = copy(V)
    ωᵥ    = inv(sum(V_obs .^ 2))

    fill!(model.fields.As, As_init)
    solve!(model)

    objective = SnapshotObjective(ωᵥ, V_obs, β_reg)

    xc_km = xc ./ 1e3
    yc_km = yc ./ 1e3

    fig = Figure(; size=(650, 450))

    #! format:off
    ax = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="log10(As)"),
          Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="dJ/d(logAs)"),
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

    function callback(iter, α, J1, logAs, logĀs)
        if iter % 100 == 0
            @printf("  iter = %-4d, J = %1.3e, α = %1.3e\n", iter, J1, α)
            hm[1][3] = Array(logAs .* log10(ℯ))
            hm[2][3] = Array(logĀs ./ log10(ℯ))
            hm[4][3] = Array(V)
            autolimits!(ax[1])
            autolimits!(ax[2])
            autolimits!(ax[4])
            display(fig)
        end
    end

    options = OptimisationOptions(line_search, callback, maxiter)

    logAs = optimise(model, objective, log.(As0), options)

    return
end

synthetic_params = SnapshotInversionParams(; line_search=BacktrackingLineSearch(; α_min=1e0, α_max=1e5))

snapshot_inversion("datasets/synthetic_setup.jld2", synthetic_params)

aletsch_params = SnapshotInversionParams(;
                                         E           = 2.5e-1,
                                         line_search = BacktrackingLineSearch(; α_min=1e0, α_max=1e5))

snapshot_inversion("datasets/aletsch_setup.jld2", aletsch_params)
