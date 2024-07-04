struct SnapshotObjective{T<:Real,A<:AbstractMatrix{T}}
    ωᵥ::T
    V_obs::A
end

function J(As, objective::SnapshotObjective, model::SnapshotSIA; kwargs...)
    # unpack
    (; ωᵥ, V_obs) = objective
    (; V)         = model.fields

    # set the parameter in the model state (not necessary for the snapshot inversion)
    if As !== model.fields.As
        copy!(model.fields.As, As)
    end

    # forward model run
    solve!(model)

    # normalise and weight misfit
    return ωᵥ * 0.5 * sum((V .- V_obs) .^ 2)
end

function ∇J!(Ās, As, objective::SnapshotObjective, model::SnapshotSIA; kwargs...)
    # unpack
    (; ωᵥ, V_obs) = objective
    (; V)         = model.fields
    (; V̄)        = model.adjoint_fields

    # set the parameter in the model state (not necessary for the snapshot inversion)
    if As !== model.fields.As
        copy!(model.fields.As, As)
    end

    # forward model run
    solve!(model)

    # partial derivative easily computed analytically for least-squares type objectives
    @. V̄ = ωᵥ * (V - V_obs)

    # Enzyme accumulates results in-place, initialise with zeros
    fill!(Ās, 0.0)

    solve_adjoint!(Ās, model)

    return
end
