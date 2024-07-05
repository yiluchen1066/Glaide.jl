"""
    struct TimeDependentSIA{F,S,N}

A struct representing a time-dependent forward solver state for the Shallow Ice Approximation (SIA) method.
"""
struct TimeDependentSIA{F<:TimeDependentFields,
                        S<:TimeDependentScalars,
                        N<:TimeDependentNumerics,
                        AF<:TimeDependentAdjointFields,
                        AN<:TimeDependentAdjointNumerics}
    fields::F
    scalars::S
    numerics::N
    adjoint_fields::AF
    adjoint_numerics::AN
end

"""
    TimeDependentSIA(scalars, numerics)

Constructs a time-dependent forward solver object.
"""
function TimeDependentSIA(scalars, numerics, adjoint_numerics=nothing)
    if isnothing(adjoint_numerics)
        adjoint_numerics = TimeDependentAdjointNumerics(numerics.xc, numerics.yc)
    end

    fields         = TimeDependentFields(numerics.nx, numerics.ny)
    adjoint_fields = TimeDependentAdjointFields(numerics.nx, numerics.ny)

    return TimeDependentSIA(fields, scalars, numerics, adjoint_fields, adjoint_numerics)
end

function TimeDependentSIA(path::AbstractString, adjoint_numerics=nothing; numeric_overrides...)
    data = load(path)

    dfields = data["fields"]

    (; nx, ny, xc, yc)                          = data["numerics"]
    (; lx, ly, npow, A, ρgn, β, b_max, ela, dt) = data["scalars"]

    fields   = TimeDependentFields(nx, ny)
    scalars  = TimeDependentScalars(lx, ly, npow, A, ρgn, β, b_max, ela, dt)
    numerics = TimeDependentNumerics(xc, yc; numeric_overrides...)

    adjoint_fields = TimeDependentAdjointFields(nx, ny)

    if isnothing(adjoint_numerics)
        adjoint_numerics = TimeDependentAdjointNumerics(xc, yc)
    end

    copy!(fields.H, dfields.H)
    copy!(fields.B, dfields.B)
    copy!(fields.V, dfields.V)
    copy!(fields.H_old, dfields.H_old)
    copy!(fields.mb_mask, dfields.mb_mask)

    return TimeDependentSIA(fields, scalars, numerics, adjoint_fields, adjoint_numerics)
end

"""
    solve!(solver; debug_vis=false, report=true)

Iteratively solves the time-dependent forward problem using the SIA (Shallow Ice Approximation) method.
The surface velocity is also computed.

## Keyword arguments
- `debug_vis`: Whether to create a debug visualization.
- `report`: Whether to print convergence status.
"""
function solve!(model::TimeDependentSIA; debug_vis=false, report=false)
    # unpack SIA parameters
    (; B, H, H_old, V, D, As, r_H, d_H, dH_dτ, mb_mask) = model.fields
    (; ρgn, A, npow, β, b_max, ela, dt)                 = model.scalars

    # unpack numerical parameters
    (; nx, ny, dx, dy, cfl, dmp1, dmp2, dmpswitch, maxiter, ncheck, εtol) = model.numerics

    N = max(nx, ny)

    # create debug visualisation
    if debug_vis
        vis = create_debug_visualisation(model)
    end

    # initialise ice thickness
    fill!(dH_dτ, 0.0)

    # iterative loop
    iter            = 1
    stop_iterations = false
    converged       = false
    while !stop_iterations
        # save ice thickness to check relative change
        (iter % ncheck == 0) && copyto!(d_H, @view(H[2:end-1, 2:end-1]))

        # apply homogeneous Neumann boundary conditions before
        # computing diffusivity to ensure correct interpolation
        bc!(H)
        diffusivity!(D, H, B, As, A, ρgn, npow, dx, dy)

        # compute pseudo-time step
        dτ = compute_pt_time_step(cfl, D, β, dt, dx, dy)

        # apply Neumann boundary conditions before computing residual
        # to set zero flux on boundaries
        bc!(H, B)
        residual!(r_H, B, H, H_old, D, β, ela, b_max, mb_mask, dt, dx, dy)

        # empirically calibrated damping coefficient to accelerate convergence
        dmp = iter < dmpswitch ? dmp1 : dmp2
        update_ice_thickness!(H, dH_dτ, r_H, dτ, dmp)

        if iter % ncheck == 0
            # difference in thickness between iterations
            d_H .-= @view(H[2:end-1, 2:end-1])

            # characteristic length and velocity scales
            lsc = maximum(H)
            vsc = 2 / (npow + 2) * ρgn * (A * lsc^(npow + 1) + maximum(As) * lsc^(npow - 1))

            # compute absolute and relative errors
            err_abs = maximum(abs.(r_H)) / vsc
            err_rel = maximum(abs.(d_H)) / lsc

            # print convergence status
            report && @printf("    iter = %.2f × N, error: [abs = %1.3e, rel = %1.3e]\n", iter / N, err_abs, err_rel)

            # check if simulation has failed
            if !isfinite(err_abs) || !isfinite(err_rel)
                error("forward solver failed: detected NaNs")
            end

            debug_vis && update_debug_visualisation!(vis, model, iter / N, (; err_abs, err_rel))

            converged = (err_rel < εtol)
        end

        stop_iterations = converged || (iter > maxiter - 1)

        iter += 1
    end

    if converged
        report && println("forward solver converged")
    else
        @warn("forward solver not converged: iter > maxiter")
    end

    # apply boundary conditions for consistency
    bc!(H, B)

    # compute surface velocity
    surface_velocity!(V, H, B, As, A, ρgn, npow, dx, dy)

    return
end

function solve_adjoint!(Ās, model::TimeDependentSIA; debug_vis=false, report=false)
    # unpack forward parameters
    (; B, H, H_old, D, V, As, mb_mask, r_H, d_H, dH_dτ) = model.fields
    (; ρgn, A, npow, β, b_max, ela, dt)                 = model.scalars

    # unpack adjoint state and shadows
    (; ψ, r̄_H, H̄, D̄, V̄, ∂J_∂H) = model.adjoint_fields

    # unpack numerical parameters
    (; nx, ny, dx, dy)                  = model.numerics
    (; cfl, dmp, maxiter, ncheck, εtol) = model.adjoint_numerics

    N = max(nx, ny)

    # create debug visualisation
    if debug_vis
        vis = create_adjoint_debug_visualisation(model)
    end

    # reuse memory for some of the fields to save memory
    fill!(d_H, 0.0)
    fill!(dH_dτ, 0.0)

    d_ψ   = d_H
    dψ_dτ = @view(dH_dτ[2:end-1, 2:end-1])

    # pseudo-time step (constant beteween iterations, depends only on D)
    dτ = compute_pt_time_step(cfl, D, β, dt, dx, dy)

    # first propagate partial velocity derivatives
    ∇surface_velocity!(DupNN(V, V̄), DupNN(H, H̄),
                       Const(B), Const(As), Const(A),
                       Const(ρgn), Const(npow),
                       Const(dx), Const(dy))

    # Enzyme overwrites memory, save array
    copy!(∂J_∂H, H̄)

    # iterative loop
    iter            = 1
    stop_iterations = false
    converged       = false
    while !stop_iterations
        # save adjoint state to check relative change
        (iter % ncheck == 0) && copy!(d_ψ, ψ)

        # initialize shadow variables (Enzyme accumulates derivatives in-place)
        copy!(r̄_H, ψ)
        copy!(H̄, ∂J_∂H)
        fill!(D̄, 0.0)

        ∇residual!(DupNN(r_H, r̄_H),
                   Const(B),
                   DupNN(H, H̄),
                   Const(H_old),
                   DupNN(D, D̄),
                   Const(β), Const(ela), Const(b_max), Const(mb_mask),
                   Const(dt), Const(dx), Const(dy))

        ∇diffusivity!(DupNN(D, D̄),
                      DupNN(H, H̄),
                      Const(B), Const(As), Const(A),
                      Const(ρgn), Const(npow),
                      Const(dx), Const(dy))

        update_adjoint_state!(ψ, dψ_dτ, H̄, H, dτ, dmp)

        if iter % ncheck == 0
            # difference in adjoint state between iterations
            d_ψ .-= ψ

            # compute L∞ norm of adjoint state increment
            err_rel = maximum(abs.(d_ψ)) / (maximum(abs.(ψ)) + eps())

            # print convergence status
            report && @printf("    iter = %.2f × N, error: [rel = %1.3e]\n", iter / N, err_rel)

            # check if simulation has failed
            if !isfinite(err_rel)
                error("adjoint solver failed: detected NaNs")
            end

            debug_vis && update_adjoint_debug_visualisation!(vis, model, iter / N, (; err_rel))

            converged = (err_rel < εtol)
        end

        stop_iterations = converged || (iter > maxiter - 1)

        iter += 1
    end

    if converged
        report && println("adjoint solver converged")
    else
        @warn("adjoint solver not converged: iter > maxiter")
    end

    # Enzyme accumulates results in-place, initialise with zeros
    fill!(Ās, 0.0)
    fill!(D̄, 0.0)

    # propagate derivatives w.r.t. sliding parameter
    ∇residual!(DupNN(r_H, copy(ψ)),
               Const(B), Const(H), Const(H_old),
               DupNN(D, D̄),
               Const(β), Const(ela), Const(b_max), Const(mb_mask),
               Const(dt), Const(dx), Const(dy))

    ∇diffusivity!(DupNN(D, D̄),
                  Const(H), Const(B),
                  DupNN(As, Ās),
                  Const(A), Const(ρgn), Const(npow),
                  Const(dx), Const(dy))

    return
end

# compute admissible pseudo-time step based on the von Neumann stability criterion
function compute_pt_time_step(cfl, D, β, dt, dx, dy)
    return inv(maximum(D) / min(dx, dy)^2 / cfl + β + inv(dt))
end
