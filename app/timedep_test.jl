using CUDA, Glaide, CairoMakie, Printf, JLD2

Base.@kwdef struct InversionScenarioTimeDependent{LS}
    input_file::String
    output_dir::String
    # initial sliding parameter value
    As_init::Float64 = 1e-22
    # ice flow enhancement factor
    E::Float64 = 1.0
    # regularisation parameter
    γ_reg::Float64 = 1.0e-6
    # weight for velocity matching
    ωᵥ::Float64 = 1.0
    # weight for ice thickness matching
    ωₕ::Float64 = 1.0
    # maximum number of iterations in the optimisation algorithm
    maxiter::Int = 1000
    # line search algorithm
    line_search::LS
end

function main(scenario::InversionScenarioTimeDependent)
    # unpack params
    (; input_file, output_dir, As_init, E, γ_reg, ωᵥ, ωₕ, maxiter, line_search) = scenario

    if ispath(output_dir)
        rm(output_dir; recursive=true)
    end

    mkpath(output_dir)

    model = TimeDependentSIA(input_file; report=true, debug_vis=false)

    (; As, V, H, H_old) = model.fields

    As0 = As

    model.scalars.A *= E

    V_obs = copy(V)
    H_obs = copy(H)

    ωn = sqrt(ωᵥ^2 + ωₕ^2)

    ωᵥ *= inv(ωn * sum(V_obs .^ 2))
    ωₕ *= inv(ωn * sum(H_obs .^ 2))

    fill!(As0, As_init)
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

    optimise(model, objective, log.(As0), options)

    return
end

synthetic_params = InversionScenarioTimeDependent(;
                                                  input_file="datasets/synthetic_50m.jld2",
                                                  output_dir="output/test_inversion",
                                                  ωᵥ=1.0,
                                                  ωₕ=0.0,
                                                  line_search=BacktrackingLineSearch(; α_min=1e1, α_max=1e6))
main(synthetic_params)
