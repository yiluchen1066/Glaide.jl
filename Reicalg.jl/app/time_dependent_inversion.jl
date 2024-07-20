using Reicalg
using CUDA
using CairoMakie
using Printf
using JLD2

Base.@kwdef struct TimeDependentInversionParams{T,I,LS}
    # initial sliding parameter value
    As_init::T = 1e-22
    # ice flow enhancement factor
    E::T = 1.0
    # regularisation parameter
    β_reg::T = 1.0e-3
    # weight for velocity matching
    ωᵥ::T = 1.0
    # weight for ice thickness matching
    ωₕ::T = 1.0
    # maximum number of iterations in the optimisation algorithm
    maxiter::I = 2000
    # line search algorithm
    line_search::LS
end

function time_dependent_inversion(filepath, params::TimeDependentInversionParams)
    # unpack params
    (; As_init, E, β_reg, ωᵥ, ωₕ, maxiter, line_search) = params

    model = TimeDependentSIA(filepath)

    (; xc, yc) = model.numerics
    (; As, V, H) = model.fields

    As0 = As

    model.scalars.A *= E

    V_obs = copy(V)
    H_obs = copy(H)

    ωn = sqrt(ωᵥ^2 + ωₕ^2)

    ωᵥ *= inv(ωn * sum(V_obs .^ 2))
    ωₕ *= inv(ωn * sum(H_obs .^ 2))

    fill!(As0, As_init)
    solve!(model)

    objective = TimeDependentObjective(ωᵥ, ωₕ, V_obs, H_obs, β_reg)

    xc_km = xc / 1e3
    yc_km = yc / 1e3

    fig = Figure(; size=(650, 800))

    #! format:off
    ax = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="log10(As)"),
          Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="dJ/d(log10(As))"),
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

    conv_ax = Axis(fig[4, :]; title="Convergence", xlabel="Iteration", ylabel="J", yscale=log10)
    conv    = lines!(conv_ax, Point2{Float64}[])
    #! format:on

    j_hist = Point2{Float64}[]
    function callback(state::OptmisationState)
        push!(j_hist, Point2(state.iter, state.j_value))

        if state.iter % 25 == 0
            @printf("  iter = %-4d, J = %1.3e, ΔJ/J = %1.3e, ΔX/X = %1.3e, α = %1.3e\n",
                    state.iter,
                    state.j_value,
                    state.j_change,
                    state.x_change,
                    state.α)

            hm[1][3] = Array(state.X .* log10(ℯ))
            hm[2][3] = Array(state.X̄ ./ log10(ℯ))
            hm[4][3] = Array(V)
            hm[6][3] = Array(H)
            conv[1]  = j_hist
            autolimits!(ax[1])
            autolimits!(ax[2])
            autolimits!(conv_ax)
            display(fig)
        end
    end

    options = OptimisationOptions(; line_search, callback, maxiter)

    optimise(model, objective, log.(As0), options)

    return
end

synthetic_params = TimeDependentInversionParams(;
                                                line_search=BacktrackingLineSearch(; α_min=1e2, α_max=5e4))

time_dependent_inversion("datasets/synthetic_setup.jld2", synthetic_params)

aletsch_params = TimeDependentInversionParams(;
                                              E           = 2.5e-1,
                                              line_search = BacktrackingLineSearch(; α_min=1e2, α_max=1.5e4))

time_dependent_inversion("datasets/aletsch_setup.jld2", aletsch_params)
