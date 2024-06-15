function solve_sia!(params; debug_vis=false, report=true)
    # unpack SIA parameters
    (; B, H, H_old, V, D, As, r_H, d_H, dH_dτ, ELA, mb_mask) = params.fields
    (; ρgn, A, npow, β, b_max, dt) = params.scalars

    # unpack numerical params
    (; nx, ny, dx, dy, cfl, maxiter, ncheck, ϵtol) = params.numerics

    # create debug visualisation
    if debug_vis
        vis = create_debug_visualisation(params)
    end

    # initialise ice thickness
    copy!(H, H_old)

    # iterative loop
    iter = 1
    stop_iterations = false
    while !stop_iterations
        # save ice thickness to check relative change
        (iter % ncheck == 0) && copyto!(d_H, @view(H[2:end-1, 2:end-1]))

        # update flux and residual
        diffusivity!(D, H, B, As, A, ρgn, npow, dx, dy)
        residual!(r_H, B, H, H_old, D, β, ELA, b_max, mb_mask, dt, dx, dy)
        # compute pseudo-time step
        dτ = compute_pt_time_step(cfl, D, β, dt, dx, dy)

        # empirically calibrated damping coefficient to accelerate convergence
        dmp = iter > 5max(nx, ny) ? 0.7 : 0.5
        update_ice_thickness!(H, dH_dτ, r_H, dτ, dmp)

        # apply Neumann boundary conditions
        bc!(H, B)

        if iter % ncheck == 0
            # difference in thickness between iterations
            d_H .-= @view(H[2:end-1, 2:end-1])

            lsc = maximum(H)
            vsc = 2 / (npow + 2) * ρgn * (A * lsc^(npow + 1) + minimum(As) * lsc^(npow - 1))

            # compute absolute and relative errors
            err_abs = maximum(abs.(r_H)) / vsc
            err_rel = maximum(abs.(d_H)) / lsc

            # print convergence status
            report && @printf("    iter = %.2f × nx, error: [abs = %1.3e, rel = %1.3e]\n", iter / nx, err_abs, err_rel)

            # check if simulation has failed
            if !isfinite(err_abs) || !isfinite(err_rel)
                error("simulation failed: detected NaNs")
            end

            debug_vis && update_debug_visualisation!(vis, params, iter / nx, (; err_abs, err_rel))

            stop_iterations = (err_rel < ϵtol)
        end

        # check if too many iterations
        ((iter < maxiter) || stop_iterations) || error("simulation failed: iter > maxiter")

        iter += 1
    end

    report && println("forward solver converged")

    # compute surface velocity
    surface_velocity!(V, H, B, As, A, ρgn, npow, dx, dy)

    return
end

function compute_pt_time_step(cfl, D, β, dt, dx, dy)
    return inv(maximum(D) / min(dx, dy)^2 / cfl + β + inv(dt))
end
