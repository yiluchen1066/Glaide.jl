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
    report::Bool
    debug_vis::Bool
end

"""
    TimeDependentSIA(scalars, numerics)

Constructs a time-dependent forward solver object.
"""
function TimeDependentSIA(scalars, numerics, adjoint_numerics=nothing; report=false, debug_vis=false)
    if isnothing(adjoint_numerics)
        adjoint_numerics = TimeDependentAdjointNumerics(numerics.xc, numerics.yc)
    end

    fields         = TimeDependentFields(numerics.nx, numerics.ny)
    adjoint_fields = TimeDependentAdjointFields(numerics.nx, numerics.ny)

    return TimeDependentSIA(fields, scalars, numerics, adjoint_fields, adjoint_numerics, report, debug_vis)
end

function TimeDependentSIA(path::AbstractString, adjoint_numerics=nothing; report=false, debug_vis=false, numeric_overrides...)
    data = load(path)

    dfields = data["fields"]

    (; nx, ny, xc, yc)                        = data["numerics"]
    (; lx, ly, n, A, ρgn, b, mb_max, ela, dt) = data["scalars"]

    fields   = TimeDependentFields(nx, ny)
    scalars  = TimeDependentScalars(lx, ly, n, A, ρgn, b, mb_max, ela, dt)
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

    return TimeDependentSIA(fields, scalars, numerics, adjoint_fields, adjoint_numerics, report, debug_vis)
end

"""
    solve!(solver)

Iteratively solves the time-dependent forward problem using the SIA (Shallow Ice Approximation) method.
The surface velocity is also computed.
"""
function solve!(model::TimeDependentSIA)
    # unpack SIA parameters
    (; B, H, H_old, V, mb_mask, As, r, r0, z, p, d) = model.fields
    (; ρgn, A, n, b, mb_max, ela, dt)               = model.scalars

    # unpack numerical parameters
    (; nx, ny, dx, dy, α, dmpswitch, ndmp, maxiter, ncheck, εtol) = model.numerics

    N = max(nx, ny)

    # create debug visualisation
    if model.debug_vis
        vis = create_debug_visualisation(model)
    end

    # initialise search direction
    residual!(r, z, B, H, H_old, A, As, ρgn, n, b, ela, mb_max, mb_mask, dt, dx, dy, ComputePreconditionedResidual())
    copy!(p, z)

    # iterative loop
    iter            = 1
    stop_iterations = false
    converged       = false
    β               = 0.0
    while !stop_iterations
        # save ice thickness to check relative change
        (iter % ncheck == 0) && copy!(d, H)

        update_ice_thickness!(H, p, α)

        # residual
        if (iter > dmpswitch) && (iter % ndmp == 0)
            copy!(r0, r)
        end
        residual!(r, z, B, H, H_old, A, As, ρgn, n, b, ela, mb_max, mb_mask, dt, dx, dy, ComputePreconditionedResidual())

        if (iter > dmpswitch) && (iter % ndmp == 0)
            dkyk, yy = mapreduce((_r, _r0, _p) -> (_p * (_r - _r0), (_r - _r0)^2), (x, y) -> x .+ y, r, r0, p; init=(0.0, 0.0))
            β₀       = mapreduce((_r, _r0, _p, _z) -> (_r - _r0 + (2yy / dkyk) * _p) * _z, +, r, r0, p, z) / dkyk
            β        = clamp(β₀, 0, 1)
        end

        @. p = p * β + z

        if iter % ncheck == 0
            # difference in thickness between iterations
            d .-= H

            # characteristic length and velocity scales
            lsc = maximum(H)
            vsc = ρgn * (2 / (n + 2) * A * lsc^(n + 1) + maximum(As) * lsc^(n - 1))

            # compute absolute and relative errors
            err_abs = maximum(abs, r) / vsc
            err_rel = maximum(abs, d) / (lsc + (lsc == 0))

            # print convergence status
            model.report && @printf("    iter = %.2f × N, error: [abs = %1.3e, rel = %1.3e]\n", iter / N, err_abs, err_rel)

            # check if simulation has failed
            if !isfinite(err_abs) || !isfinite(err_rel)
                error("forward solver failed: detected NaNs at iter #$iter")
            end

            model.debug_vis && update_debug_visualisation!(vis, model, iter / N, (; err_abs, err_rel))

            converged = (err_rel < εtol)
        end

        stop_iterations = converged || (iter > maxiter - 1)

        iter += 1
    end

    if converged
        model.report && println("forward solver converged")
    else
        @warn("forward solver not converged: iter > maxiter")
    end

    # compute surface velocity
    surface_velocity!(V, H, B, As, A, ρgn, n, dx, dy)

    return
end

function solve_adjoint!(Ās, model::TimeDependentSIA)
    # unpack forward parameters
    (; B, H, H_old, V, mb_mask, As, r, r0, z, p, d) = model.fields
    (; ρgn, A, n, b, mb_max, ela, dt)               = model.scalars

    # unpack adjoint state and shadows
    (; ψ, r̄, z̄, H̄, V̄, ∂J_∂H) = model.adjoint_fields

    # unpack numerical parameters
    (; nx, ny, dx, dy)           = model.numerics
    (; α, maxiter, ncheck, εtol) = model.adjoint_numerics

    N = max(nx, ny)

    # create debug visualisation
    if model.debug_vis
        vis = create_adjoint_debug_visualisation(model)
    end

    # reuse memory for some of the fields to save memory
    fill!(d, 0.0)
    fill!(p, 0.0)

    # Enzyme accumulates results in-place, initialise with zeros
    fill!(Ās, 0.0)

    # first propagate partial velocity derivatives
    Enzyme.autodiff(Enzyme.Reverse, surface_velocity!,
                    DupNN(V, V̄), DupNN(H, H̄),
                    Const(B), DupNN(As, Ās), Const(A),
                    Const(ρgn), Const(n),
                    Const(dx), Const(dy))

    # Enzyme overwrites memory, save array
    copy!(∂J_∂H, H̄)

    # compute preconditioner
    residual!(r, z, B, H, H_old, A, As, ρgn, n, b, ela, mb_max, mb_mask, dt, dx, dy, ComputePreconditioner())

    # initialize shadow variables (Enzyme accumulates derivatives in-place)
    copy!(r̄, ψ)
    copy!(H̄, ∂J_∂H)

    Enzyme.autodiff(Enzyme.Reverse, residual!,
                    DupNN(r, r̄),
                    Const(z),
                    Const(B),
                    DupNN(H, H̄),
                    Const(H_old),
                    Const(A),
                    Const(As),
                    Const(ρgn), Const(n),
                    Const(b), Const(ela), Const(mb_max), Const(mb_mask),
                    Const(dt), Const(dx), Const(dy), Const(ComputeResidual()))

    @. z̄ = r̄ * z

    copy!(p, z̄)

    # iterative loop
    iter            = 1
    stop_iterations = false
    converged       = false
    β               = 0.0
    while !stop_iterations
        # save adjoint state to check relative change
        (iter % ncheck == 0) && copy!(d, ψ)

        update_adjoint_state!(ψ, p, α)

        # residual
        if (iter > dmpswitch) && (iter % ndmp == 0)
            copy!(r0, r̄)
        end

        # initialize shadow variables (Enzyme accumulates derivatives in-place)
        copy!(r̄, ψ)
        copy!(H̄, ∂J_∂H)

        Enzyme.autodiff(Enzyme.Reverse, residual!,
                        DupNN(r, r̄),
                        Const(z),
                        Const(B),
                        DupNN(H, H̄),
                        Const(H_old),
                        Const(A),
                        Const(As),
                        Const(ρgn), Const(n),
                        Const(b), Const(ela), Const(mb_max), Const(mb_mask),
                        Const(dt), Const(dx), Const(dy), Const(ComputeResidual()))

        @. z̄ = r̄ * z

        if (iter > dmpswitch) && (iter % ndmp == 0)
            dkyk, yy = mapreduce((_r, _r0, _p) -> (_p * (_r - _r0), (_r - _r0)^2), (x, y) -> x .+ y, r̄, r0, p; init=(0.0, 0.0))
            β₀       = mapreduce((_r, _r0, _p, _z) -> (_r - _r0 + (2yy / dkyk) * _p) * _z, +, r̄, r0, p, z̄) / dkyk
            β        = clamp(β₀, 0, 1)
        end

        @. p = p * β + z

        if iter % ncheck == 0
            # difference in the adjoint state between iterations
            d .-= ψ

            # compute L∞ norm of adjoint state increment
            err_abs = maximum(abs, H̄)
            err_rel = maximum(abs, d) / (maximum(abs, ψ) + eps())

            # print convergence status
            model.report && @printf("    iter = %.2f × N, error: [abs = %1.3e, rel = %1.3e]\n", iter / N, err_abs, err_rel)

            # check if simulation has failed
            if !isfinite(err_rel)
                error("adjoint solver failed: detected NaNs at iter #$iter")
            end

            model.debug_vis && update_adjoint_debug_visualisation!(vis, model, iter / N, (; err_rel))

            converged = (err_rel < εtol)
        end

        stop_iterations = converged || (iter > maxiter - 1)

        iter += 1
    end

    if converged
        model.report && println("adjoint solver converged")
    else
        @warn("adjoint solver not converged: iter > maxiter")
    end

    # propagate derivatives w.r.t. sliding parameter
    ∇residual!(DupNN(r, copy(ψ)),
               Const(B), Const(H), Const(H_old),
               DupNN(D, D̄),
               Const(β), Const(ela), Const(b_max), Const(mb_mask),
               Const(dt), Const(dx), Const(dy), Const(ComputeResidual()))

    return
end
