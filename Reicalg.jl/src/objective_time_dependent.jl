function objective_time_dependent!(fwd_params, obj_params; kwargs...)
    # unpack objective params
    (; H_obs, V_obs) = obj_params

    # weights for ice thickness and velocity
    (; ωₕ, ωᵥ) = obj_params

    # unpack forward problem solution
    (; H, V) = fwd_params.fields

    # solve forward problem
    solve_sia!(fwd_params; kwargs...)

    # normalise and weight misfit
    return ωₕ * 0.5 * sum((H .- H_obs) .^ 2) +
           ωᵥ * 0.5 * sum((V .- V_obs) .^ 2)
end

function grad_objective_time_dependent!(∇J, fwd_params, adj_params, obj_params; kwargs...)
    # unpack objective params
    (; H_obs, V_obs) = obj_params

    # weights for ice thickness and velocity
    (; ωₕ, ωᵥ) = obj_params

    # unpack forward problem solution
    (; B, H, H_old, D, As, V, ELA, mb_mask, r_H) = fwd_params.fields
    (; A, ρgn, npow, β, b_max, dt)               = fwd_params.scalars
    (; dx, dy)                                   = fwd_params.numerics

    # solve forward problem
    solve_sia!(fwd_params; kwargs...)

    # unpack adjoint params
    (; ∂J_∂H, V̄, D̄, ψ) = adj_params.fields

    # velocity is a function of the ice thickness, propagate derivatives with AD
    @. V̄ = -ωᵥ * (V - V_obs)
    @. ∂J_∂H = 0.0

    ∇surface_velocity!(DupNN(V, V̄), DupNN(H, ∂J_∂H),
                       Const(B), Const(As), Const(A),
                       Const(ρgn), Const(npow),
                       Const(dx), Const(dy))

    @. ∂J_∂H += ωₕ * (H - H_obs)

    # solve adjoint problem
    adjoint_sia!(fwd_params, adj_params; kwargs...)

    # propagate derivatives to compute gradient without regularisation term
    ∇residual!(DupNN(r_H, copy(ψ)),
               Const(B), Const(H), Const(H_old),
               DupNN(D, D̄),
               Const(β), Const(ELA), Const(b_max), Const(mb_mask),
               Const(dt), Const(dx), Const(dy))

    # Enzyme accumulates results in-place, initialise with zeros
    ∇J .= 0.0

    ∇diffusivity!(DupNN(D, D̄),
                  Const(H), Const(B),
                  DupNN(As, ∇J),
                  Const(A), Const(ρgn), Const(npow),
                  Const(dx), Const(dy))

    return
end
