using Glaide, CUDA, Printf, JLD2, Unitful, Logging, CairoMakie

Base.@kwdef struct InversionScenarioTimeDependent{LS,AT}
    input_file::String
    output_dir::String
    # initial sliding parameter value
    As_init::AT = 1e-22u"Pa^-3*s^-1*m"
    # ice flow enhancement factor
    E::Float64 = 1.0
    # regularisation parameter
    γ_reg::Float64 = 3e-8
    # weight for velocity matching
    ωᵥ::Float64 = 1.0
    # weight for ice thickness matching
    ωₕ::Float64 = 1.0
    # maximum number of iterations in the optimisation algorithm
    maxiter::Int = 1000
    # tolerance for the damping acceleration
    dmptol::Float64 = 1e-4
    # line search algorithm
    line_search::LS
end

function time_dependent_inversion(scenario::InversionScenarioTimeDependent)
    # unpack params
    (; input_file, output_dir, As_init, E, γ_reg, ωᵥ, ωₕ, maxiter, dmptol, line_search) = scenario

    if ispath(output_dir)
        @warn "path \"$(output_dir)\" already exists, skipping simulation; delete the directory to re-run"
        return
    end

    mkpath(output_dir)

    reg = 5e-8u"s^-1" * T_REF |> NoUnits

    model = TimeDependentSIA(input_file;
                             forward_overrides=(; reg, maxf=100.0, dmptol),
                             adjoint_overrides=(; maxf=100.0, dmptol))

    (; ρgnAs, V, H, H_old) = model.fields
    n = model.scalars.n

    ρgnAs0 = ρgnAs

    model.scalars.ρgnA *= E

    V_obs = copy(V)
    H_obs = copy(H)

    ωn = sqrt(ωᵥ^2 + ωₕ^2)

    ωᵥ *= inv(ωn * sum(V_obs .^ 2))
    ωₕ *= inv(ωn * sum(H_obs .^ 2))

    ρgnAs_init = RHOG^n * As_init * (L_REF^(n - 1) * T_REF) |> NoUnits

    fill!(ρgnAs0, ρgnAs_init)
    copy!(H, H_old)

    solve!(model)

    objective = TimeDependentObjective(ωᵥ, ωₕ, V_obs, H_obs, γ_reg)

    j_hist = Point2{Float64}[]
    step   = 0

    function callback(state::OptmisationState)
        push!(j_hist, Point2(state.iter, state.j_value))

        if state.iter % 10 != 0
            return
        end

        @info @sprintf("iter = %-4d, J = %1.3e, ΔJ/J = %1.3e, ΔX/X = %1.3e, α = %1.3e\n",
                       state.iter,
                       state.j_value,
                       state.j_change,
                       state.x_change,
                       state.α)

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

function runme()
    # systematic study of the relative weights of the velocity and ice thickness matching terms (synthetic)
    # for (ωᵥ, ωₕ) in ((0, 1), (1e-3, 1), (1e-2, 1), (1e-1, 1), (1, 1), (1, 1e-1), (1, 1e-2), (1, 1e-3), (1, 0))
    #     @info "time-dependent synthetic inversion for ωᵥ = $(ωᵥ) and ωₕ = $(ωₕ)"
    #     synthetic_params = InversionScenarioTimeDependent(;
    #                                                       input_file="../datasets/synthetic_50m.jld2",
    #                                                       output_dir="../output/time_dependent_synthetic_50m_$(ωᵥ)_$(ωₕ)",
    #                                                       ωᵥ=float(ωᵥ),
    #                                                       ωₕ=float(ωₕ),
    #                                                       γ_reg=1e-6,
    #                                                       line_search=BacktrackingLineSearch(; α_min=1e1, α_max=5e5, c=0.1, τ=0.8))
    #     time_dependent_inversion(synthetic_params)
    # end

    # systematic study of the regularisation parameter (L-curve analysis)
    # for γ_reg in 5 .* 10 .^ LinRange(-9, -7, 20)
    #     @info "time-dependent Aletsch inversion for γ_reg = $(round(γ_reg; sigdigits=2))"
    #     aletsch_params = InversionScenarioTimeDependent(;
    #                                                     input_file  = "../datasets/aletsch_50m.jld2",
    #                                                     output_dir  = "../output/time_dependent_aletsch_50m_reg_$(round(γ_reg; sigdigits=2))",
    #                                                     As_init     = 1e-22u"Pa^-3*s^-1*m",
    #                                                     E           = 0.25,
    #                                                     γ_reg       = γ_reg,
    #                                                     ωᵥ          = 0.01,
    #                                                     ωₕ          = 1.0,
    #                                                     dmptol      = 1e-7,
    #                                                     line_search = BacktrackingLineSearch(; α_min=1e1, α_max=5e5, c=0.1, τ=0.8))
    #     time_dependent_inversion(aletsch_params)
    # end

    # systematic study of the mesh convergence
    # for res in (200, 100, 50, 25)
    #     @info "time-dependent Aletsch inversion for resolution $(res)m"
    #     aletsch_params = InversionScenarioTimeDependent(;
    #                                                     input_file  = "../datasets/aletsch_$(res)m.jld2",
    #                                                     output_dir  = "../output/time_dependent_aletsch_$(res)m",
    #                                                     As_init     = 1e-22u"Pa^-3*s^-1*m",
    #                                                     ωᵥ          = 0.01,
    #                                                     ωₕ          = 1.0,
    #                                                     E           = 0.25,
    #                                                     dmptol      = 0.0,
    #                                                     line_search = BacktrackingLineSearch(; α_min=1e1, α_max=5e5, c=0.1, τ=0.8))
    #     time_dependent_inversion(aletsch_params)
    # end

    # # systematic study of the weight for the velocity matching term (Aletsch)
    # for ωᵥ in 5 .* 10 .^ LinRange(-4, -1, 10)
    #     @info "time-dependent Aletsch inversion for ωᵥ = $(round(ωᵥ; sigdigits=2))"
    #     aletsch_params = InversionScenarioTimeDependent(;
    #                                                     input_file  = "../datasets/aletsch_50m.jld2",
    #                                                     output_dir  = "../output/time_dependent_aletsch_50m_$(round(ωᵥ; sigdigits=2))_1",
    #                                                     As_init     = 1e-22u"Pa^-3*s^-1*m",
    #                                                     E           = 0.25,
    #                                                     ωᵥ          = ωᵥ,
    #                                                     ωₕ          = 1.0,
    #                                                     dmptol      = 1e-4,
    #                                                     line_search = BacktrackingLineSearch(; α_min=1e1, α_max=5e5, c=0.1, τ=0.8))
    #     time_dependent_inversion(aletsch_params)
    # end

    return
end

runme()
