"""
    TimeDependentSIA

A struct representing a time-dependent forward solver state for the Shallow Ice Approximation (SIA) method.
"""
mutable struct TimeDependentSIA{F<:TimeDependentFields,
                                S<:TimeDependentScalars,
                                N<:TimeDependentNumerics,
                                AF<:TimeDependentAdjointFields,
                                AN<:TimeDependentAdjointNumerics}
    fields::F
    scalars::S
    numerics::N
    adjoint_fields::AF
    adjoint_numerics::AN
    debug_vis::Bool
end

"""
    TimeDependentSIA(scalars, numerics)

Constructs a time-dependent forward solver object.
"""
function TimeDependentSIA(scalars, numerics, adjoint_numerics=nothing; debug_vis=false)
    if isnothing(adjoint_numerics)
        adjoint_numerics = TimeDependentAdjointNumerics()
    end

    fields         = TimeDependentFields(numerics.nx, numerics.ny)
    adjoint_fields = TimeDependentAdjointFields(numerics.nx, numerics.ny)

    return TimeDependentSIA(fields, scalars, numerics, adjoint_fields, adjoint_numerics, debug_vis)
end

function TimeDependentSIA(path::AbstractString; debug_vis=false, forward_overrides=(), adjoint_overrides=())
    data = load(path)

    dfields = data["fields"]

    (; nx, ny)                              = data["numerics"]
    (; lx, ly, n, ρgnA, b, mb_max, ela, dt) = data["scalars"]

    fields   = TimeDependentFields(nx, ny)
    scalars  = TimeDependentScalars(lx, ly, n, ρgnA, b, mb_max, ela, dt)
    numerics = TimeDependentNumerics(nx, ny; forward_overrides...)

    adjoint_fields   = TimeDependentAdjointFields(nx, ny)
    adjoint_numerics = TimeDependentAdjointNumerics(; adjoint_overrides...)

    copy!(fields.H, dfields.H)
    copy!(fields.B, dfields.B)
    copy!(fields.V, dfields.V)
    copy!(fields.H_old, dfields.H_old)
    copy!(fields.mb_mask, dfields.mb_mask)

    return TimeDependentSIA(fields, scalars, numerics, adjoint_fields, adjoint_numerics, debug_vis)
end

"""
    solve!(solver)

Iteratively solves the time-dependent forward problem using the SIA (Shallow Ice Approximation) method.
The surface velocity is also computed.
"""
function solve!(model::TimeDependentSIA)
    # unpack SIA parameters
    (; B, H, H_old, V, mb_mask, ρgnAs, r, r0, z, p, d) = model.fields
    (; lx, ly, ρgnA, n, b, mb_max, ela, dt)            = model.scalars

    # unpack numerical parameters
    (; nx, ny, α, reg, maxf, checkf, εtol, dmptol) = model.numerics

    # preprocessing
    N       = max(nx, ny)
    dx, dy  = lx / nx, ly / ny
    ncheck  = ceil(Int, checkf * N)
    maxiter = ceil(Int, maxf * N)

    # create debug visualisation
    if model.debug_vis
        vis = create_debug_visualisation(model)
    end

    # initialise search direction
    residual!(r, z, B, H, H_old, ρgnAs, mb_mask, ρgnA, n, b, mb_max, ela, dt, dx, dy, reg, ComputePreconditionedResidual())
    copy!(p, z)

    # iterative loop
    iter               = 1
    stop_iterations    = false
    converged          = false
    β                  = 0.0
    accelerate_damping = false
    while !stop_iterations
        # save ice thickness to check relative change
        (iter % ncheck == 0) && copy!(d, H)

        update_ice_thickness!(H, p, α)

        # save previous residual
        if accelerate_damping
            copy!(r0, r)
        end

        residual!(r, z, B, H, H_old, ρgnAs, mb_mask, ρgnA, n, b, mb_max, ela, dt, dx, dy, reg, ComputePreconditionedResidual())

        if accelerate_damping
            dkyk, yy = mapreduce((_r, _r0, _p) -> (_p * (_r - _r0), (_r - _r0)^2), (x, y) -> x .+ y, r, r0, p; init=(0.0, 0.0))
            β₀       = mapreduce((_r, _r0, _p, _z) -> (_r - _r0 + (2yy / dkyk) * _p) * _z, +, r, r0, p, z) / dkyk
            β        = clamp(β₀, 0, 1)
        else
            β = 0.0
        end

        @. p = p * β + z

        if iter % ncheck == 0
            # difference in thickness between iterations
            d .-= H

            # characteristic length and velocity scales
            lsc = maximum(H)
            vsc = 2 / (n + 2) * ρgnA * lsc^(n + 1) + maximum(ρgnAs) * lsc^n

            # compute absolute and relative errors
            err_abs = maximum(abs, r) / vsc
            err_rel = maximum(abs, d) / (lsc + (lsc == 0))

            # print convergence status
            @debug @sprintf("    iter = %.2f × N, error: [abs = %1.3e, rel = %1.3e]\n", iter / N, err_abs, err_rel)

            # check if simulation has failed
            if !isfinite(err_abs) || !isfinite(err_rel)
                error("forward solver failed: detected NaNs at iter #$iter")
            end

            model.debug_vis && update_debug_visualisation!(vis, model, iter / N, (; err_abs, err_rel))

            converged = (err_rel < εtol)

            accelerate_damping = (err_rel < dmptol)
        end

        stop_iterations = converged || (iter > maxiter - 1)

        iter += 1
    end

    if converged
        @debug "forward solver converged"
    else
        @warn "forward solver not converged: iter > maxiter (= $maxiter)"
    end

    # compute surface velocity
    surface_velocity!(V, H, B, ρgnAs, ρgnA, n, dx, dy)

    return
end

function solve_adjoint!(ρgnĀs, model::TimeDependentSIA)
    # unpack forward parameters
    (; B, H, H_old, V, mb_mask, ρgnAs, r, r0, z, p, d) = model.fields
    (; lx, ly, ρgnA, n, b, mb_max, ela, dt)            = model.scalars

    # unpack adjoint state and shadows
    (; ψ, r̄, z̄, H̄, V̄, ∂J_∂H) = model.adjoint_fields

    # unpack numerical parameters
    (; nx, ny, reg)                   = model.numerics
    (; α, maxf, checkf, εtol, dmptol) = model.adjoint_numerics

    N       = max(nx, ny)
    dx, dy  = lx / nx, ly / ny
    ncheck  = ceil(Int, checkf * N)
    maxiter = ceil(Int, maxf * N)

    # create debug visualisation
    if model.debug_vis
        vis = create_adjoint_debug_visualisation(model)
    end

    # reuse memory for some of the fields to save memory
    fill!(d, 0.0)
    fill!(p, 0.0)

    # Enzyme accumulates results in-place, initialise with zeros
    fill!(ρgnĀs, 0.0)

    # Enzyme overwrites memory, save array
    copy!(∂J_∂H, H̄)

    # first propagate partial velocity derivatives
    ∇surface_velocity!(DupNN(V, V̄), DupNN(H, H̄),
                       Const(B), DupNN(ρgnAs, ρgnĀs),
                       ρgnA, n, dx, dy)

    # compute preconditioner
    residual!(r, z, B, H, H_old, ρgnAs, mb_mask, ρgnA, n, b, mb_max, ela, dt, dx, dy, reg, ComputePreconditioner())

    # initialize shadow variables (Enzyme accumulates derivatives in-place)
    copy!(r̄, ψ)
    copy!(H̄, ∂J_∂H)

    ∇residual!(DupNN(r, r̄), Const(z),
               Const(B), DupNN(H, H̄), Const(H_old),
               Const(ρgnAs), Const(mb_mask),
               ρgnA, n, b, mb_max, ela, dt, dx, dy, reg,
               ComputeResidual())

    @. z̄ = z * H̄
    copy!(p, z̄)

    # iterative loop
    iter               = 1
    stop_iterations    = false
    converged          = false
    β                  = 0.0
    accelerate_damping = false
    while !stop_iterations
        # save adjoint state to check relative change
        (iter % ncheck == 0) && copy!(d, ψ)

        update_adjoint_state!(ψ, p, α)

        # residual
        if accelerate_damping
            copy!(r0, H̄)
        end

        # initialize shadow variables (Enzyme accumulates derivatives in-place)
        copy!(r̄, ψ)
        copy!(H̄, ∂J_∂H)

        ∇residual!(DupNN(r, r̄), Const(z),
                   Const(B), DupNN(H, H̄), Const(H_old),
                   Const(ρgnAs), Const(mb_mask),
                   ρgnA, n, b, mb_max, ela, dt, dx, dy, reg,
                   ComputeResidual())

        @. z̄ = z * H̄

        if accelerate_damping
            dkyk, yy = mapreduce((_r, _r0, _p) -> (_p * (_r - _r0), (_r - _r0)^2), (x, y) -> x .+ y, H̄, r0, p; init=(0.0, 0.0))
            β₀       = mapreduce((_r, _r0, _p, _z) -> (_r - _r0 + (2yy / dkyk) * _p) * _z, +, H̄, r0, p, z̄) / dkyk
            β        = clamp(β₀, 0, 1)
        else
            β = 0.0
        end

        @. p = p * β + z̄

        if iter % ncheck == 0
            # difference in the adjoint state between iterations
            d .-= ψ

            # compute L∞ norm of adjoint state increment
            err_abs = maximum(abs, H̄)
            err_rel = maximum(abs, d) / (maximum(abs, ψ) + eps())

            # print convergence status
            @debug @sprintf("    iter = %.2f × N, error: [abs = %1.3e, rel = %1.3e]\n", iter / N, err_abs, err_rel)

            # check if simulation has failed
            isfinite(err_rel) || error("adjoint solver failed: detected NaNs at iter #$iter")

            model.debug_vis && update_adjoint_debug_visualisation!(vis, model, iter / N, (; err_rel))

            converged = (err_rel < εtol)

            accelerate_damping = (err_rel < dmptol)
        end

        stop_iterations = converged || (iter > maxiter - 1)

        iter += 1
    end

    if converged
        @debug "adjoint solver converged"
    else
        @warn "adjoint solver not converged: iter > maxiter (= $maxiter)"
    end

    copy!(r̄, ψ)

    # propagate derivatives w.r.t. sliding parameter
    ∇residual!(DupNN(r, r̄), Const(z),
               Const(B), Const(H), Const(H_old),
               DupNN(ρgnAs, ρgnĀs), Const(mb_mask),
               ρgnA, n, b, mb_max, ela, dt, dx, dy, reg,
               ComputeResidual())

    return
end
