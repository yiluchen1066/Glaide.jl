struct SnapshotObjective{T<:Real,A<:AbstractMatrix{T}}
    ωᵥ::T
    V_obs::A
    γ_reg::T
end

function J(logAs, objective::SnapshotObjective, model::SnapshotSIA)
    # unpack
    (; ωᵥ, V_obs, γ_reg) = objective
    (; V)                = model.fields
    (; dx, dy)           = model.numerics

    # set the parameter in the model state
    @. model.fields.As = exp(logAs)

    # forward model run
    solve!(model)

    # Tikhonov regularisation term
    J_reg = dx * dy * (sum(@. ((logAs[2:end, :] - logAs[1:end-1, :]) / dx)^2) +
                       sum(@. ((logAs[:, 2:end] - logAs[:, 1:end-1]) / dy)^2))

    # normalise and weight misfit
    return ωᵥ * 0.5 * sum((V .- V_obs) .^ 2) + γ_reg * 0.5 * J_reg
end

function ∇J!(logĀs, logAs, objective::SnapshotObjective, model::SnapshotSIA)
    # unpack
    (; ωᵥ, V_obs, γ_reg) = objective
    (; V)                = model.fields
    (; V̄)                = model.adjoint_fields
    (; dx, dy)           = model.numerics

    # set the parameter in the model state
    @. model.fields.As = exp(logAs)

    # forward model run
    solve!(model)

    # partial derivative can be computed analytically for least-squares type objectives
    @. V̄ = ωᵥ * (V - V_obs)

    # Enzyme accumulates results in-place, initialise with zeros
    fill!(logĀs, 0.0)

    solve_adjoint!(logĀs, model)

    # convert gradient to log-space
    @. logĀs *= model.fields.As

    tikhonov_regularisation!(logĀs, logAs, dx * dy * γ_reg, dx, dy)

    return
end
