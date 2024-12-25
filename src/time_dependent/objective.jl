struct TimeDependentObjective{T<:Real,A<:AbstractMatrix{T}}
    ωᵥ::T
    ωₕ::T
    V_obs::A
    H_obs::A
    γ_reg::T
end

function J(log_ρgnAs, objective::TimeDependentObjective, model::TimeDependentSIA)
    # unpack
    (; ωₕ, ωᵥ, H_obs, V_obs, γ_reg) = objective
    (; H, V)                        = model.fields
    (; dx, dy)                      = model.numerics

    # copy the field to the model state
    @. model.fields.ρgnAs = exp(log_ρgnAs)

    # solve forward problem
    solve!(model)

    # Tikhonov regularisation term
    J_reg = dx * dy * (sum(@. ((log_ρgnAs[2:end, :] - log_ρgnAs[1:end-1, :]) / dx)^2) +
                       sum(@. ((log_ρgnAs[:, 2:end] - log_ρgnAs[:, 1:end-1]) / dy)^2))

    # normalise and weight misfit
    return ωₕ * 0.5 * sum((H .- H_obs) .^ 2) +
           ωᵥ * 0.5 * sum((V .- V_obs) .^ 2) + γ_reg * 0.5 * J_reg
end

function ∇J!(log_ρgnĀs, log_ρgnAs, objective::TimeDependentObjective, model::TimeDependentSIA)
    # unpack
    (; ωₕ, ωᵥ, H_obs, V_obs, γ_reg) = objective
    (; H, V)                        = model.fields
    (; H̄, V̄)                        = model.adjoint_fields
    (; dx, dy)                      = model.numerics

    # copy the field to the model state
    @. model.fields.ρgnAs = exp(log_ρgnAs)

    # solve forward problem
    solve!(model)

    # compute partial derivatives
    @. V̄ = ωᵥ * (V - V_obs)
    @. H̄ = ωₕ * (H - H_obs)

    solve_adjoint!(log_ρgnĀs, model)

    # convert gradient to log-space
    @. log_ρgnĀs *= model.fields.ρgnAs

    tikhonov_regularisation!(log_ρgnĀs, log_ρgnAs, dx * dy * γ_reg, dx, dy)

    return
end
