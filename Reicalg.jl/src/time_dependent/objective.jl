struct TimeDependentObjective{T<:Real,A<:AbstractMatrix{T}}
    ωᵥ::T
    ωₕ::T
    V_obs::A
    H_obs::A
    β_reg::T
end

function J(logAs, objective::TimeDependentObjective, model::TimeDependentSIA)
    # unpack
    (; ωₕ, ωᵥ, H_obs, V_obs, β_reg) = objective
    (; H, V)                        = model.fields
    (; dx, dy)                      = model.numerics

    # copy the field to the model state
    @. model.fields.As = exp(logAs)

    # solve forward problem
    solve!(model)

    # Tikhonov regularisation term
    J_reg = sum(@. ((logAs[2:end, :] - logAs[1:end-1, :]) / dx)^2) +
            sum(@. ((logAs[:, 2:end] - logAs[:, 1:end-1]) / dy)^2)

    # normalise and weight misfit
    return ωₕ * 0.5 * sum((H .- H_obs) .^ 2) +
           ωᵥ * 0.5 * sum((V .- V_obs) .^ 2) + β_reg * 0.5 * J_reg
end

function ∇J!(logĀs, logAs, objective::TimeDependentObjective, model::TimeDependentSIA)
    # unpack
    (; ωₕ, ωᵥ, H_obs, V_obs, β_reg) = objective
    (; H, V)                        = model.fields
    (; H̄, V̄)                        = model.adjoint_fields
    (; dx, dy)                      = model.numerics

    # copy the field to the model state
    @. model.fields.As = exp(logAs)

    # solve forward problem
    solve!(model)

    # compute partial derivatives
    @. V̄ = ωᵥ * (V - V_obs)
    @. H̄ = ωₕ * (H - H_obs)

    solve_adjoint!(logĀs, model)

    # convert gradient to log-space
    @. logĀs *= model.fields.As

    tikhonov_regularisation!(logĀs, logAs, β_reg, dx, dy)

    return
end
