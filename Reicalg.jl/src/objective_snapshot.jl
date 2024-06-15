function objective_time_dependent!(fwd_params, obj_params; kwargs...)
    # unpack objective params
    (; ωᵥ, V_obs) = obj_params

    # unpack forward problem solution
    (; H, B, As, V)  = fwd_params.fields
    (; A, ρgn, npow) = fwd_params.scalars
    (; dx, dy)       = fwd_params.numerics

    # forward model run
    surface_velocity!(V, H, B, As, A, ρgn, npow, dx, dy)

    # normalise and weight misfit
    return ωᵥ * 0.5 * sum((V .- V_obs) .^ 2)
end

function grad_objective_time_dependent!(∇J, fwd_params, adj_params, obj_params; kwargs...)
    # unpack objective params
    (; ωᵥ, V_obs) = obj_params

    # unpack forward problem solution
    (; H, B, As, V)  = fwd_params.fields
    (; A, ρgn, npow) = fwd_params.scalars
    (; dx, dy)       = fwd_params.numerics

    # forward model run
    surface_velocity!(V, H, B, As, A, ρgn, npow, dx, dy)

    # unpack adjoint params
    (; V̄) = adj_params.fields

    # velocity is a function of the ice thickness, propagate derivatives with AD
    @. V̄ = ωᵥ * (V - V_obs)

    ∇J .= 0.0

    ∇surface_velocity!(DupNN(V, V̄), Const(H),
                       Const(B), DupNN(As, ∇J), Const(A),
                       Const(ρgn), Const(npow),
                       Const(dx), Const(dy))

    return
end
