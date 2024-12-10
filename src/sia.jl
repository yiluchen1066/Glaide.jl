struct ComputeResidual end
struct ComputePreconditionedResidual end

# CUDA kernels
Base.@propagate_inbounds function residual(B, H, H_old, A, Aₛ, ρgn, n, b, ela, mb_max, mb_mask, _dt, _dx, _dy)
    LLVM.Interop.assume(n > 0)

    # average sliding coefficient
    Aₛˣ = avˣ(Aₛ)
    Aₛʸ = avʸ(Aₛ)

    # surface elevation
    S = B .+ H

    # bedrock gradient
    ∇Bˣ = δˣ(B) .* _dx
    ∇Bʸ = δʸ(B) .* _dy

    # surface gradient
    ∇Sˣˣ = δˣₐ(S) .* _dx
    ∇Sʸʸ = δʸₐ(S) .* _dy

    # surface gradient magnitude
    ∇Sˣₙ = sqrt.(innʸ(∇Sˣˣ) .^ 2 .+ av4(∇Sʸʸ) .^ 2) .^ (n - 1)
    ∇Sʸₙ = sqrt.(av4(∇Sˣˣ) .^ 2 .+ innˣ(∇Sʸʸ) .^ 2) .^ (n - 1)

    Γˣ₁ = (2 * inv(n + 2) * A) .* ∇Sˣₙ
    Γʸ₁ = (2 * inv(n + 2) * A) .* ∇Sʸₙ

    Γˣ₂ = Aₛˣ .* ∇Sˣₙ
    Γʸ₂ = Aₛʸ .* ∇Sʸₙ

    # diffusivity of the deformational component
    Dˣ₁ = Γˣ₁ .* inv(n + 3)
    Dʸ₁ = Γʸ₁ .* inv(n + 3)

    # diffusivity of the sliding component
    Dˣ₂ = Γˣ₂ .* inv(n + 2)
    Dʸ₂ = Γʸ₂ .* inv(n + 2)

    # velocity of the deformational component
    Wˣ₁ = .-Γˣ₁ .* ∇Bˣ
    Wʸ₁ = .-Γʸ₁ .* ∇Bʸ

    # velocity of the sliding component
    Wˣ₂ = .-Γˣ₂ .* ∇Bˣ
    Wʸ₂ = .-Γʸ₂ .* ∇Bʸ

    Hⁿ⁰ = H .^ n
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

    ∇q = ρgn * (δˣ(qˣ) * _dx + δʸ(qʸ) * _dy)

    mb = ela_mass_balance(S[2, 2], b, ela, mb_max) * mb_mask

    dH_dt = (H[2, 2] - H_old) * _dt

    r = mb - dH_dt - ∇q

    if (H[2, 2] == 0) && (r < 0)
        r = zero(r)
    end

    return r
end

function _residual!(r, z, B, H, H_old, A, Aₛ, ρgn, n, b, ela, mb_max, mb_mask, _dt, _dx, _dy, mode)
    @get_indices
    @inbounds begin
        Hₗ  = st3x3(H, ix, iy)
        Bₗ  = st3x3(B, ix, iy)
        Aₛₗ = st3x3(Aₛ, ix, iy)

        if mode == ComputeResidual()
            r[ix, iy] = residual(Bₗ, Hₗ, H_old[ix, iy], A, Aₛₗ, ρgn, n, b, ela, mb_max, mb_mask[ix, iy], _dt, _dx, _dy)
        elseif mode == ComputePreconditionedResidual()
            r̄, r[ix, iy] = Enzyme.autodiff_deferred(Enzyme.ReverseWithPrimal, Const(residual), Active,
                                                     Const(Bₗ),
                                                     Active(Hₗ),
                                                     Const(H_old[ix, iy]),
                                                     Const(A),
                                                     Const(Aₛₗ),
                                                     Const(ρgn),
                                                     Const(n),
                                                     Const(b),
                                                     Const(ela),
                                                     Const(mb_max),
                                                     Const(mb_mask[ix, iy]),
                                                     Const(_dt),
                                                     Const(_dx),
                                                     Const(_dy))
            q = 0.5sum(abs.(r̄[2])) + 1e-8
            z[ix, iy] = r[ix, iy] / q
        end
    end
    return
end

# update ice thickness with constraint H > 0
function _update_ice_thickness!(H, p, α)
    @get_indices
    # update ice thickness
    @inbounds H[ix, iy] = max(H[ix, iy] + α * p[ix, iy], zero(H[ix, iy]))
    return
end

function _surface_velocity(H, B, Aₛ, A, ρgn, n, _dx, _dy)
    # surface elevation
    S = B .+ H

    # surface gradient
    ∇Sˣ = δˣ(S) .* _dx
    ∇Sʸ = δʸ(S) .* _dy

    ∇Sₙ = sqrt(avˣ(∇Sˣ)^2 + avʸ(∇Sʸ)^2)^n

    return ρgn * (2 / (n + 1) * A * H[2, 2]^(n + 1) + Aₛ * H[2, 2]^n) * ∇Sₙ
end

# surface velocity magnitude
function _surface_velocity!(V, H, B, As, A, ρgn, n, dx, dy)
    @get_indices
    @inbounds begin
        Hₗ = st3x3(H, ix, iy)
        Bₗ = st3x3(B, ix, iy)
        V[ix, iy] = _surface_velocity(Hₗ, Bₗ, As[ix, iy], A, ρgn, n, inv(dx), inv(dy))
    end
    return
end

# wrappers
function residual!(r, z, B, H, H_old, A, Aₛ, ρgn, n, b, ela, mb_max, mb_mask, dt, dx, dy, mode)
    nthreads, nblocks = launch_config(size(H))
    @cuda threads = nthreads blocks = nblocks _residual!(r, z, B, H, H_old, A, Aₛ, ρgn, n, b, ela, mb_max, mb_mask, inv(dt), inv(dx), inv(dy), mode)
    return
end

function residual2!(r, z, B, H, H_old, A, Aₛ, ρgn, n, b, ela, mb_max, mb_mask, dt, dx, dy, mode)
    nthreads, nblocks = launch_config(size(H))
    nshmem = (prod(nthreads) * 4 + prod(nthreads .+ 2) * 3) * sizeof(Float64)
    @cuda threads = nthreads blocks = nblocks shmem = nshmem _residual2!(r, z, B, H, H_old, A, Aₛ, ρgn, n, b, ela, mb_max, mb_mask, inv(dt), inv(dx), inv(dy),
                                                                         mode)
    return
end

function update_ice_thickness!(H, p, α)
    nthreads, nblocks = launch_config(size(H))
    @cuda threads = nthreads blocks = nblocks _update_ice_thickness!(H, p, α)
    return
end

function surface_velocity!(V, H, B, As, A, ρgn, n, dx, dy)
    nthreads, nblocks = launch_config(size(H))
    @cuda threads = nthreads blocks = nblocks _surface_velocity!(V, H, B, As, A, ρgn, n, dx, dy)
end
