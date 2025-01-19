struct ComputeResidual end
struct ComputePreconditionedResidual end
struct ComputePreconditioner end

# CUDA kernels
Base.@propagate_inbounds function residual(B, H, H_old, ρgnAₛ, mb_mask, ρgnA2_n2, b, mb_max, ela, _dt, _dx, _dy, n, nm1, mode)
    # contract
    LLVM.Interop.assume(n > 0 && nm1 > 0)

    # surface elevation
    S = B .+ H

    # surface gradient
    ∇Sˣˣ = δˣₐ(S) .* _dx
    ∇Sʸʸ = δʸₐ(S) .* _dy

    ∇Sˣᵥ = avʸ(∇Sˣˣ)
    ∇Sʸᵥ = avˣ(∇Sʸʸ)

    ∇Sₙ = sqrt.(∇Sˣᵥ .^ 2 .+ ∇Sʸᵥ .^ 2) .^ nm1

    Hᵥ   = av4(H)
    Hᵥⁿ⁰ = Hᵥ .^ n # directly computing H^(n+1) results in a 2x performance penalty
    Hᵥⁿ¹ = Hᵥⁿ⁰ .* Hᵥ
    Hᵥⁿ² = Hᵥⁿ¹ .* Hᵥ

    Dᵥ = (ρgnA2_n2 .* Hᵥⁿ² .+ av4(ρgnAₛ) .* Hᵥⁿ¹) .* ∇Sₙ

    qˣ = .-avʸ(Dᵥ) .* δˣ(S) .* _dx
    qʸ = .-avˣ(Dᵥ) .* δʸ(S) .* _dy

    r = -(H[2, 2] - H_old) * _dt
    r += -(δˣ(qˣ) * _dx + δʸ(qʸ) * _dy)
    r += ela_mass_balance(S[2, 2], b, ela, mb_max) * mb_mask

    if (mode == ComputePreconditionedResidual()) && (H[2, 2] == 0) && (r < 0)
        r = zero(r)
    end

    return r
end

function _residual!(r, z, B, H, H_old, ρgnAₛ, mb_mask, ρgnA2_n2, b, mb_max, ela, _dt, _dx, _dy, n, nm1, reg, mode)
    @get_indices
    @inbounds if ix <= oftype(ix, size(r, 1)) && iy <= oftype(iy, size(r, 2))
        Hₗ  = st3x3(H, ix, iy)
        Bₗ  = st3x3(B, ix, iy)
        Aₛₗ = st3x3(ρgnAₛ, ix, iy)

        H_o  = H_old[ix, iy]
        mb_m = mb_mask[ix, iy]

        R(x) = residual(Bₗ, x, H_o, Aₛₗ, mb_m, ρgnA2_n2, b, mb_max, ela, _dt, _dx, _dy, n, nm1, mode)

        if mode == ComputeResidual()
            r[ix, iy] = residual(Bₗ, Hₗ, H_o, Aₛₗ, mb_m, ρgnA2_n2, b, mb_max, ela, _dt, _dx, _dy, n, nm1, mode)
        elseif mode == ComputePreconditionedResidual()
            r̄, r[ix, iy] = Enzyme.autodiff_deferred(Enzyme.ReverseWithPrimal, Const(R), Active, Active(Hₗ))
            q             = max(0.5sum(abs.(r̄[1])), reg)
            z[ix, iy]     = r[ix, iy] / q
        elseif mode == ComputePreconditioner()
            r̄        = Enzyme.autodiff_deferred(Enzyme.Reverse, Const(R), Active, Active(Hₗ))
            q         = max(0.5sum(abs.(r̄[1][1])), reg)
            z[ix, iy] = inv(q)
        end
    end
    return
end

# update ice thickness with constraint H > 0
function _update_ice_thickness!(H, p, α)
    @get_indices
    if ix <= size(H, 1) && iy <= size(H, 2)
        @inbounds H[ix, iy] = max(H[ix, iy] + α * p[ix, iy], zero(H[ix, iy]))
    end
    return
end

function _surface_velocity(H, B, ρgnAₛ, ρgnA2_n1, _dx, _dy, n, n1)
    # contract
    LLVM.Interop.assume(n > 0 && n1 > 0)

    # surface elevation
    S = B .+ H

    # surface gradient
    ∇Sˣ = δˣ(S) .* _dx
    ∇Sʸ = δʸ(S) .* _dy

    ∇Sₙ = sqrt(avˣ(∇Sˣ)^2 + avʸ(∇Sʸ)^2)^n

    return (ρgnA2_n1 * H[2, 2]^n1 + ρgnAₛ * H[2, 2]^n) * ∇Sₙ
end

# surface velocity magnitude
function _surface_velocity!(V, H, B, ρgnAs, ρgnA2_n1, _dx, _dy, n, n1)
    @get_indices
    @inbounds if ix <= size(V, 1) && iy <= size(V, 2)
        Hₗ = st3x3(H, ix, iy)
        Bₗ = st3x3(B, ix, iy)
        V[ix, iy] = _surface_velocity(Hₗ, Bₗ, ρgnAs[ix, iy], ρgnA2_n1, _dx, _dy, n, n1)
    end
    return
end

# wrappers
function residual!(r, z, B, H, H_old, ρgnAₛ, mb_mask, ρgnA, n, b, mb_max, ela, dt, dx, dy, reg, mode)
    nthreads, nblocks = launch_config(size(H))

    # precompute constants
    ρgnA2_n2 = ρgnA * 2 / (n + 2)
    _dt      = inv(dt)
    _dx      = inv(dx)
    _dy      = inv(dy)
    nm1      = n - oneunit(n)

    @cuda threads = nthreads blocks = nblocks _residual!(r, z, B, H, H_old, ρgnAₛ, mb_mask, ρgnA2_n2, b, mb_max, ela, _dt, _dx, _dy, n, nm1, reg, mode)
    return
end

function update_ice_thickness!(H, p, α)
    nthreads, nblocks = launch_config(size(H))
    @cuda threads = nthreads blocks = nblocks _update_ice_thickness!(H, p, α)
    return
end

function surface_velocity!(V, H, B, ρgnAs, ρgnA, n, dx, dy)
    nthreads, nblocks = launch_config(size(H))

    # precompute constants
    ρgnA2_n1 = ρgnA * 2 / (n + 1)
    n1       = n + oneunit(n)
    _dx      = inv(dx)
    _dy      = inv(dy)

    @cuda threads = nthreads blocks = nblocks _surface_velocity!(V, H, B, ρgnAs, ρgnA2_n1, _dx, _dy, n, n1)
end
