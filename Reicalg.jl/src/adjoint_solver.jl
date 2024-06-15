function adjoint_sia!(fwd_params, adj_params; debug_vis=false, report=true)
    # unpack forward parameters
    (; B, H, H_old, D, As, ELA, mb_mask, r_H, d_H, dH_dτ) = fwd_params.fields
    (; ρgn, A, npow, β, b_max, dt) = fwd_params.scalars

    # unpack adjoint state and shadows
    (; ψ, r̄_H, H̄, D̄, ∂J_∂H) = adj_params.fields

    # unpack numerical parameters
    (; nx, ny, dx, dy, cfl, maxiter, ncheck, ϵtol) = adj_params.numerics

    # create debug visualisation
    if debug_vis
        vis = create_adjoint_debug_visualisation(adj_params)
    end

    # init
    ψ .= 0.0

    # we reuse memory for some of the fields to save memory
    d_H   .= 0.0
    dH_dτ .= 0.0

    d_ψ   = d_H
    dψ_dτ = @view(dH_dτ[2:end-1, 2:end-1])

    # pseudo-time step (constant beteween iterations, depends only on D)
    dτ = compute_pt_time_step(cfl, D, β, dt, dx, dy)

    # iterative loop
    iter = 1
    stop_iterations = false
    while !stop_iterations
        # save adjoint state to check relative change
        (iter % ncheck == 0) && copyto!(d_ψ, ψ)

        # initialize shadow variables (Enzyme accumulates derivatives in-place)
        r̄_H .= ψ
        H̄   .= ∂J_∂H
        D̄   .= 0.0

        ∇residual!(DupNN(r_H, r̄_H),
                   Const(B),
                   DupNN(H, H̄),
                   Const(H_old),
                   DupNN(D, D̄),
                   Const(β), Const(ELA), Const(b_max), Const(mb_mask),
                   Const(dt), Const(dx), Const(dy))

        ∇diffusivity!(DupNN(D, D̄),
                      DupNN(H, H̄),
                      Const(B), Const(As), Const(A),
                      Const(ρgn), Const(npow),
                      Const(dx), Const(dy))

        dmp = 0.9
        update_adjoint_state!(ψ, dψ_dτ, H̄, H, dτ, dmp)

        if iter % ncheck == 0
            # difference in adjoint state between iterations
            d_ψ .-= ψ

            # compute L∞ norm of adjoint state increment
            err_rel = maximum(abs.(d_ψ)) / (maximum(abs.(ψ)) + eps())

            # print convergence status
            report && @printf("    iter = %.2f × nx, error: [rel = %1.3e]\n", iter / nx, err_rel)

            # check if simulation has failed
            if !isfinite(err_rel)
                error("simulation failed: detected NaNs")
            end

            debug_vis && update_adjoint_debug_visualisation!(vis, adj_params, iter / nx, (; err_rel))

            stop_iterations = (err_rel < ϵtol)
        end

        # check if too many iterations
        (iter < maxiter) || error("simulation failed: iter > maxiter")

        iter += 1
    end

    report && println("adjoint solver converged")

    return
end
