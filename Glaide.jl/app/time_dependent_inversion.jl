using Glaide
using CUDA
using CairoMakie
using Printf
using JLD2

Base.@kwdef struct InversionScenario{LS}
    input_file::String
    output_dir::String
    # initial sliding parameter value
    As_init::Float64 = 1e-22
    # ice flow enhancement factor
    E::Float64 = 1.0
    # regularisation parameter
    β_reg::Float64 = 1.0e-6
    # weight for velocity matching
    ωᵥ::Float64 = 1.0
    # weight for ice thickness matching
    ωₕ::Float64 = 1.0
    # maximum number of iterations in the optimisation algorithm
    maxiter::Int = 1000
    # line search algorithm
    line_search::LS
end

function time_dependent_inversion(scenario::InversionScenario)
    # unpack params
    (; input_file, output_dir, As_init, E, β_reg, ωᵥ, ωₕ, maxiter, line_search) = scenario

    if ispath(output_dir)
        @warn "path \"$(output_dir)\" already exists, skipping simulation; delete the directory to re-run"
        return
    end

    mkpath(output_dir)

    model = TimeDependentSIA(input_file)
    model.report = true

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

    j_hist = Point2{Float64}[]
    step   = 0

    function callback(state::OptmisationState)
        push!(j_hist, Point2(state.iter, state.j_value))

        if state.iter % 10 != 0
            return
        end

        @printf("  iter = %-4d, J = %1.3e, ΔJ/J = %1.3e, ΔX/X = %1.3e, α = %1.3e\n",
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

    optimise(model, objective, log.(As0), options)

    return
end

synthetic_line_search = BacktrackingLineSearch(; α_min=1e3, α_max=1e6)

for (ωᵥ, ωₕ) in ((0, 1), (1, 0), (1, 1))
    synthetic_params = InversionScenario(;
                                         input_file="datasets/synthetic_25m.jld2",
                                         output_dir="output/time_dependent_synthetic_25m_$(ωᵥ)_$(ωₕ)",
                                         ωᵥ=float(ωᵥ),
                                         ωₕ=float(ωₕ),
                                         line_search=synthetic_line_search)
    time_dependent_inversion(synthetic_params)
end

for res in (200, 100, 50, 25)
    aletsch_params = InversionScenario(;
                                       input_file  = "datasets/aletsch_$(res)m.jld2",
                                       output_dir  = "output/time_dependent_aletsch_$(res)m",
                                       E           = 2.5e-1,
                                       line_search = BacktrackingLineSearch(; α_min=1e1, α_max=1e6))
    time_dependent_inversion(aletsch_params)
end
