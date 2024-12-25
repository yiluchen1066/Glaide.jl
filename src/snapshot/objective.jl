struct SnapshotObjective{T<:Real,A<:AbstractMatrix{T}}
    ωᵥ::T
    V_obs::A
    γ_reg::T
end

function J(log_ρgnAs, objective::SnapshotObjective, model::SnapshotSIA)
    # unpack
    (; ωᵥ, V_obs, γ_reg) = objective
    (; V)                = model.fields
    (; dx, dy)           = model.numerics

    # set the parameter in the model state
    @. model.fields.As = exp(log_ρgnAs)

    # forward model run
    solve!(model)

    # Tikhonov regularisation term
    J_reg = dx * dy * (sum(@. ((log_ρgnAs[2:end, :] - log_ρgnAs[1:end-1, :]) / dx)^2) +
                       sum(@. ((log_ρgnAs[:, 2:end] - log_ρgnAs[:, 1:end-1]) / dy)^2))

    # normalise and weight misfit
    return ωᵥ * 0.5 * sum((V .- V_obs) .^ 2) + γ_reg * 0.5 * J_reg
end

function ∇J!(log_ρgnĀs, log_ρgnAs, objective::SnapshotObjective, model::SnapshotSIA)
    # unpack
    (; ωᵥ, V_obs, γ_reg) = objective
    (; V)                = model.fields
    (; V̄)               = model.adjoint_fields
    (; dx, dy)           = model.numerics

    # set the parameter in the model state
    @. model.fields.ρgnAs = exp(log_ρgnAs)

    # forward model run
    solve!(model)

    # partial derivative can be computed analytically for least-squares type objectives
    @. V̄ = ωᵥ * (V - V_obs)

    # Enzyme accumulates results in-place, initialise with zeros
    fill!(log_ρgnĀs, 0.0)

    solve_adjoint!(log_ρgnĀs, model)

    # convert gradient to log-space
    @. log_ρgnĀs *= model.fields.ρgnAs

    tikhonov_regularisation!(log_ρgnĀs, log_ρgnAs, dx * dy * γ_reg, dx, dy)

    return
end
