struct ComputeResidual end
struct ComputePreconditionedResidual end
struct ComputePreconditioner end

# CUDA kernels
Base.@propagate_inbounds function residual(B, H, H_old, Aₛ, mb_mask, A2_n2, ρgn, b, mb_max, ela, _dt, _dx, _dy, _n3, _n2, n, nm1, mode)
    # contract
    LLVM.Interop.assume(n > 0 && nm1 > 0)

    # surface elevation
    S = B .+ H


    # surface gradient
    ∇Sˣˣ = δˣₐ(S) .* _dx
    ∇Sʸʸ = δʸₐ(S) .* _dy

    # surface gradient magnitude
    ∇Sˣₙ = sqrt.(innʸ(∇Sˣˣ) .^ 2 .+ av4(∇Sʸʸ) .^ 2) .^ nm1
    ∇Sʸₙ = sqrt.(av4(∇Sˣˣ) .^ 2 .+ innˣ(∇Sʸʸ) .^ 2) .^ nm1

    Γˣ₁ = A2_n2 .* ∇Sˣₙ
    Γʸ₁ = A2_n2 .* ∇Sʸₙ

    Γˣ₂ = avˣ(Aₛ) .* ∇Sˣₙ
    Γʸ₂ = avʸ(Aₛ) .* ∇Sʸₙ

    # diffusivity of the deformational component
    Dˣ₁ = Γˣ₁ .* _n3
    Dʸ₁ = Γʸ₁ .* _n3

    # diffusivity of the sliding component
    Dˣ₂ = Γˣ₂ .* _n2
    Dʸ₂ = Γʸ₂ .* _n2

    # bedrock gradient
    ∇Bˣ = δˣ(B) .* _dx
    ∇Bʸ = δʸ(B) .* _dy

    # velocity of the deformational component
    Wˣ₁ = .-Γˣ₁ .* ∇Bˣ
    Wʸ₁ = .-Γʸ₁ .* ∇Bʸ

    # velocity of the sliding component
    Wˣ₂ = .-Γˣ₂ .* ∇Bˣ
    Wʸ₂ = .-Γʸ₂ .* ∇Bʸ

    Hⁿ⁰ = H .^ n # directly computing H^(n+1) results in a 2x performance penalty
    Hⁿ¹ = Hⁿ⁰ .* H
    Hⁿ² = Hⁿ¹ .* H
    Hⁿ³ = Hⁿ² .* H

    Wˣ₁⁺ = max.(Wˣ₁, 0)
    Wʸ₁⁺ = max.(Wʸ₁, 0)

    Wˣ₂⁺ = max.(Wˣ₂, 0)
    Wʸ₂⁺ = max.(Wʸ₂, 0)

    Wˣ₁⁻ = min.(Wˣ₁, 0)
    Wʸ₁⁻ = min.(Wʸ₁, 0)

    Wˣ₂⁻ = min.(Wˣ₂, 0)
    Wʸ₂⁻ = min.(Wʸ₂, 0)

    # fluxes
    qˣ = .-(Dˣ₁ .* δˣ(Hⁿ³) .* _dx .+ Dˣ₂ .* δˣ(Hⁿ²) .* _dx) .+
         Wˣ₁⁺ .* lˣ(Hⁿ²) .+ Wˣ₁⁻ .* rˣ(Hⁿ²) .+
         Wˣ₂⁺ .* lˣ(Hⁿ¹) .+ Wˣ₂⁻ .* rˣ(Hⁿ¹)

    qʸ = .-(Dʸ₁ .* δʸ(Hⁿ³) .* _dy .+ Dʸ₂ .* δʸ(Hⁿ²) .* _dy) .+
         Wʸ₁⁺ .* lʸ(Hⁿ²) .+ Wʸ₁⁻ .* rʸ(Hⁿ²) .+
         Wʸ₂⁺ .* lʸ(Hⁿ¹) .+ Wʸ₂⁻ .* rʸ(Hⁿ¹)

    r = -(H[2, 2] - H_old) * _dt
    r += -ρgn * (δˣ(qˣ) * _dx + δʸ(qʸ) * _dy)
    r += ela_mass_balance(S[2, 2], b, ela, mb_max) * mb_mask

    if (mode == ComputePreconditionedResidual()) && (H[2, 2] == 0) && (r < 0)
        r = zero(r)
    end

    return r
end

function _residual!(r, z, B, H, H_old, Aₛ, mb_mask, A2_n2, ρgn, b, mb_max, ela, _dt, _dx, _dy, _n3, _n2, n, nm1, mode)
    @get_indices
    @inbounds if ix <= oftype(ix, size(r, 1)) && iy <= oftype(iy, size(r, 2))
        Hₗ  = st3x3(H, ix, iy)
        Bₗ  = st3x3(B, ix, iy)
        Aₛₗ = st3x3(Aₛ, ix, iy)

        H_o  = H_old[ix, iy]
        mb_m = mb_mask[ix, iy]

        R(x) = residual(Bₗ, x, H_o, Aₛₗ, mb_m, A2_n2, ρgn, b, mb_max, ela, _dt, _dx, _dy, _n3, _n2, n, nm1, mode)

        if mode == ComputeResidual()
            r[ix, iy] = residual(Bₗ, Hₗ, H_o, Aₛₗ, mb_m, A2_n2, ρgn, b, mb_max, ela, _dt, _dx, _dy, _n3, _n2, n, nm1, mode)
        elseif mode == ComputePreconditionedResidual()
            r̄, r[ix, iy] = Enzyme.autodiff_deferred(Enzyme.ReverseWithPrimal, Const(R), Active, Active(Hₗ))
            q             = 0.5sum(abs.(r̄[1])) + 1e-8
            z[ix, iy]     = r[ix, iy] / q
        elseif mode == ComputePreconditioner()
            r̄        = Enzyme.autodiff_deferred(Enzyme.Reverse, Const(R), Active, Active(Hₗ))
            q         = 0.5sum(abs.(r̄[1][1])) + 1e-8
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

function _surface_velocity(H, B, Aₛ, A2_n1, ρgn, _dx, _dy, n, n1)
    # contract
    LLVM.Interop.assume(n > 0 && n1 > 0)

    # surface elevation
    S = B .+ H

    # surface gradient
    ∇Sˣ = δˣ(S) .* _dx
    ∇Sʸ = δʸ(S) .* _dy

    ∇Sₙ = sqrt(avˣ(∇Sˣ)^2 + avʸ(∇Sʸ)^2)^n

    return ρgn * (A2_n1 * H[2, 2]^n1 + Aₛ * H[2, 2]^n) * ∇Sₙ
end

# surface velocity magnitude
function _surface_velocity!(V, H, B, As, A2_n1, ρgn, _dx, _dy, n, n1)
    @get_indices
    @inbounds if ix <= size(V, 1) && iy <= size(V, 2)
        Hₗ = st3x3(H, ix, iy)
        Bₗ = st3x3(B, ix, iy)
        V[ix, iy] = _surface_velocity(Hₗ, Bₗ, As[ix, iy], A2_n1, ρgn, _dx, _dy, n, n1)
    end
    return
end

# wrappers
function residual!(r, z, B, H, H_old, Aₛ, mb_mask, A, ρgn, n, b, mb_max, ela, dt, dx, dy, mode)
    nthreads, nblocks = launch_config(size(H))

    # precompute constants
    A2_n2 = 2 / (n + 2) * A
    _n3   = inv(n + 3)
    _n2   = inv(n + 2)
    _dt   = inv(dt)
    _dx   = inv(dx)
    _dy   = inv(dy)
    nm1   = n - oneunit(n)

    @cuda threads = nthreads blocks = nblocks _residual!(r, z, B, H, H_old, Aₛ, mb_mask, A2_n2, ρgn, b, mb_max, ela, _dt, _dx, _dy, _n3, _n2, n, nm1, mode)
    return
end

function update_ice_thickness!(H, p, α)
    nthreads, nblocks = launch_config(size(H))
    @cuda threads = nthreads blocks = nblocks _update_ice_thickness!(H, p, α)
    return
end

function surface_velocity!(V, H, B, As, A, ρgn, n, dx, dy)
    nthreads, nblocks = launch_config(size(H))

    # precompute constants
    A2_n1 = 2 / (n + 1) * A
    n1    = n + oneunit(n)
    _dx   = inv(dx)
    _dy   = inv(dy)

    @cuda threads = nthreads blocks = nblocks _surface_velocity!(V, H, B, As, A2_n1, ρgn, _dx, _dy, n, n1)
end
