using CUDA, Glaide, CairoMakie, Printf, JLD2, Unitful

Base.@kwdef struct InversionScenarioTimeDependent{LS,AT}
    input_file::String
    output_dir::String
    # initial sliding parameter value
    As_init::AT = 1e-22u"Pa^-3*s^-1*m"
    # ice flow enhancement factor
    E::Float64 = 1.0
    # regularisation parameter
    γ_reg::Float64 = 1.0e-6
    # weight for velocity matching
    ωᵥ::Float64 = 1.0
    # weight for ice thickness matching
    ωₕ::Float64 = 1.0
    # maximum number of iterations in the optimisation algorithm
    maxiter::Int = 500
    # line search algorithm
    line_search::LS
end

function main(scenario::InversionScenarioTimeDependent)
    # unpack params
    (; input_file, output_dir, As_init, E, γ_reg, ωᵥ, ωₕ, maxiter, line_search) = scenario

    # if ispath(output_dir)
    #     @warn "path \"$(output_dir)\" already exists, skipping simulation; delete the directory to re-run"
    #     return
    # end

    mkpath(output_dir)

    reg = 5e-6u"s^-1" * T_REF |> NoUnits

    model = TimeDependentSIA(input_file; report=false, debug_vis=false, reg)

    (; ρgnAs, V, H, H_old) = model.fields
    (; xc, yc)             = model.numerics
    n                      = model.scalars.n

    ρgnAs0 = ρgnAs

    model.scalars.ρgnA *= E

    V_obs = copy(V)
    H_obs = copy(H)

    ωn = sqrt(ωᵥ^2 + ωₕ^2)

    ωᵥ *= inv(ωn * sum(V_obs .^ 2))
    ωₕ *= inv(ωn * sum(H_obs .^ 2))

    @show As_init

    ρgnAs_init = RHOG^n * As_init * (L_REF^(n - 1) * T_REF) |> NoUnits

    fill!(ρgnAs0, ρgnAs_init)
    copy!(H, H_old)

    @show extrema(V_obs)
    @show model.numerics.dx
    @show model.numerics.dy
    @show model.scalars.ρgnA
    @show model.scalars.dt
    @show model.scalars.b
    @show model.scalars.mb_max
    @show model.scalars.ela
    @show model.scalars.n

    # return

    solve!(model)

    @show extrema(model.fields.V)
    @show extrema(model.fields.H)
    @show extrema(model.fields.B)
    @show extrema(V_obs)
    @show extrema(H_obs)

    # return

    objective = TimeDependentObjective(ωᵥ, ωₕ, V_obs, H_obs, γ_reg)

    j_hist = Point2{Float64}[]
    step   = 0

    fig = Figure(size=(800, 500))
    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="H"),
           Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="V"),
           Axis(fig[1, 3][1, 1]; aspect=DataAspect(), title="As"),
           Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="H_obs"),
           Axis(fig[2, 2][1, 1]; aspect=DataAspect(), title="V_obs"))

    hms = (heatmap!(axs[1], xc, yc, Array(model.fields.H); colormap=:ice),
           heatmap!(axs[2], xc, yc, Array(model.fields.V); colormap=:turbo),
           heatmap!(axs[3], xc, yc, Array(log10.(model.fields.ρgnAs)); colormap=Reverse(:roma)),
           heatmap!(axs[4], xc, yc, Array(H_obs); colormap=:ice),
           heatmap!(axs[5], xc, yc, Array(V_obs); colormap=:turbo))

    hms[1].colorrange = (0, 6)
    hms[4].colorrange = (0, 6)
    hms[2].colorrange = (0, 2)
    hms[5].colorrange = (0, 2)

    cbs = (Colorbar(fig[1, 1][1, 2], hms[1]),
           Colorbar(fig[1, 2][1, 2], hms[2]),
           Colorbar(fig[1, 3][1, 2], hms[3]),
           Colorbar(fig[2, 1][1, 2], hms[4]),
           Colorbar(fig[2, 2][1, 2], hms[5]))

    function callback(state::OptmisationState)
        push!(j_hist, Point2(state.iter, state.j_value))

        # if state.iter % 10 != 0
        #     return
        # end

        @info @sprintf("iter = %-4d, J = %1.3e, ΔJ/J = %1.3e, ΔX/X = %1.3e, α = %1.3e\n",
                       state.iter,
                       state.j_value,
                       state.j_change,
                       state.x_change,
                       state.α)

        hms[1][3] = Array(model.fields.H)
        hms[2][3] = Array(model.fields.V)
        hms[3][3] = Array(log10.(model.fields.ρgnAs))

        display(fig)

        output_path = joinpath(output_dir, @sprintf("step_%04d.jld2", step))

        jldsave(output_path;
                X=Array(state.X),
                X̄=Array(state.X̄),
                V=Array(V),
                H=Array(H),
                iter=state.iter,
                j_hist,)

        step += 1

        return
    end

    options = OptimisationOptions(; line_search, callback, maxiter)

    optimise(model, objective, log.(ρgnAs0), options)

    return
end

aletsch_params = InversionScenarioTimeDependent(;
                                                input_file  = "../datasets/aletsch_100m.jld2",
                                                output_dir  = "../output/test_inversion",
                                                E           = 0.25,
                                                line_search = BacktrackingLineSearch(; α_min=1e2, α_max=1e6, c=0.2))
main(aletsch_params)
