struct TimeDependentObjective{T<:Real,A<:AbstractMatrix{T}}
    ωᵥ::T
    ωₕ::T
    V_obs::A
    H_obs::A
end

function J(As, objective::TimeDependentObjective, model::TimeDependentSIA; kwargs...)
    # unpack
    (; ωₕ, ωᵥ, H_obs, V_obs) = objective
    (; H, V)                 = model.fields

    # copy the field to the model state
    if As !== model.fields.As
        copy!(model.fields.As, As)
    end

    # solve forward problem
    solve!(model; kwargs...)

    # normalise and weight misfit
    return ωₕ * 0.5 * sum((H .- H_obs) .^ 2) +
           ωᵥ * 0.5 * sum((V .- V_obs) .^ 2)
end

function ∇J!(Ās, As, objective::TimeDependentObjective, model::TimeDependentSIA; kwargs...)
    # unpack
    (; ωₕ, ωᵥ, H_obs, V_obs) = objective
    (; H, V)                 = model.fields
    (; H̄, V̄)               = model.adjoint_fields

    # copy the field to the model state
    if As !== model.fields.As
        copy!(model.fields.As, As)
    end

    # solve forward problem
    solve!(model; kwargs...)

    # compute partial derivatives
    @. V̄ = -ωᵥ * (V - V_obs) # I have no idea why there is a minus sign here
    @. H̄ = ωₕ * (H - H_obs)

    solve_adjoint!(Ās, model; kwargs...)

    return
end
