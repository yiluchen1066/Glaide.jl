struct ComputePreconditionedResidual end
struct ComputePreconditioner end
struct ComputeResidual end

# CUDA kernels
Base.@propagate_inbounds function residual(B, H, H_old, A, Aₛ, ρgn, n, b, ela, mb_max, mb_mask, _dt, _dx, _dy)
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
    ∇Sˣₙ = sqrt.(innʸ(∇Sˣˣ) .^ 2 + av4(∇Sʸʸ) .^ 2) .^ (n - 1)
    ∇Sʸₙ = sqrt.(av4(∇Sˣˣ) .^ 2 + innˣ(∇Sʸʸ) .^ 2) .^ (n - 1)

    Γˣ₁ = (2 / (n + 2) * A) .* ∇Sˣₙ
    Γʸ₁ = (2 / (n + 2) * A) .* ∇Sʸₙ

    Γˣ₂ = Aₛˣ .* ∇Sˣₙ
    Γʸ₂ = Aₛʸ .* ∇Sʸₙ

    # diffusivity of the deformational component
    Dˣ₁ = Γˣ₁ ./ (n + 3)
    Dʸ₁ = Γʸ₁ ./ (n + 3)

    # diffusivity of the sliding component
    Dˣ₂ = Γˣ₂ ./ (n + 2)
    Dʸ₂ = Γʸ₂ ./ (n + 2)

    # velocity of the deformational component
    Wˣ₁ = .-Γˣ₁ .* ∇Bˣ
    Wʸ₁ = .-Γʸ₁ .* ∇Bʸ

    # velocity of the sliding component
    Wˣ₂ = .-Γˣ₂ .* ∇Bˣ
    Wʸ₂ = .-Γʸ₂ .* ∇Bʸ

    Hⁿ¹ = H .^ (n + 1)
    Hⁿ² = Hⁿ¹ .* H
    Hⁿ³ = Hⁿ² .* H

    # fluxes
    qˣ = .-(Dˣ₁ .* δˣ(Hⁿ³) .* _dx .+ Dˣ₂ .* δˣ(Hⁿ²) .* _dx) .+
         max.(Wˣ₁, 0) .* lˣ(Hⁿ²) .+ min.(Wˣ₁, 0) .* rˣ(Hⁿ²) .+
         max.(Wˣ₂, 0) .* lˣ(Hⁿ¹) .+ min.(Wˣ₂, 0) .* rˣ(Hⁿ¹)

    qʸ = .-(Dʸ₁ .* δʸ(Hⁿ³) .* _dy .+ Dʸ₂ .* δʸ(Hⁿ²) .* _dy) .+
         max.(Wʸ₁, 0) .* lʸ(Hⁿ²) .+ min.(Wʸ₁, 0) .* rʸ(Hⁿ²) .+
         max.(Wʸ₂, 0) .* lʸ(Hⁿ¹) .+ min.(Wʸ₂, 0) .* rʸ(Hⁿ¹)

    ∇q = ρgn * (δˣ(qˣ) * _dx + δʸ(qʸ) * _dy)

    mb = ela_mass_balance(S[2, 2], b, ela, mb_max) * mb_mask

    dH_dt = (H[2, 2] - H_old) * _dt

    r = mb - dH_dt - ∇q

    if H[2, 2] == 0 && r < 0
        r = zero(r)
    end

    return r
end

function _residual!(r, z, B, H, H_old, A, Aₛ, ρgn, n, b, ela, mb_max, mb_mask, _dt, _dx, _dy, mode)
    @get_indices
    @inbounds if ix <= size(H, 1) && iy <= size(H, 2)
        Hₗ  = st3x3(H, ix, iy)
        Bₗ  = st3x3(B, ix, iy)
        Aₛₗ = st3x3(Aₛ, ix, iy)

        H_o = H_old[ix, iy]
        mbm = mb_mask[ix, iy]

        ℜ(x) = @inbounds residual(Bₗ, x, H_o, A, Aₛₗ, ρgn, n, b, ela, mb_max, mbm, _dt, _dx, _dy)

        if mode == ComputeResidual()
            r[ix, iy] = ℜ(Hₗ)
        elseif mode == ComputePreconditionedResidual()
            result    = ForwardDiff.gradient!(DiffResults.GradientResult(Hₗ), ℜ, Hₗ)
            r[ix, iy] = DiffResults.value(result)
            q         = 0.5sum(abs.(DiffResults.gradient(result))) + 1e-8
            z[ix, iy] = r[ix, iy] / q
        elseif mode == ComputePreconditioner()
            result    = ForwardDiff.gradient!(DiffResults.GradientResult(Hₗ), ℜ, Hₗ)
            q         = 0.5sum(abs.(DiffResults.gradient(result))) + 1e-8
            z[ix, iy] = inv(q)
        end
    end
    return
end

# update ice thickness with constraint H > 0
function _update_ice_thickness!(H, p, α)
    @get_indices
    @inbounds if ix <= size(H, 1) && iy <= size(H, 2)
        # update ice thickness
        H[ix, iy] = max(H[ix, iy] + α * p[ix, iy], zero(H[ix, iy]))
    end
    return
end

function _surface_velocity(H, B, Aₛ, A, ρgn, n, dx, dy)
    # surface elevation
    S = B .+ H

    # surface gradient
    ∇Sˣ = δˣ(S) ./ dx
    ∇Sʸ = δʸ(S) ./ dy

    ∇Sₙ = sqrt(avˣ(∇Sˣ)^2 + avʸ(∇Sʸ)^2)^n

    return ρgn * (2 / (n + 1) * A * H[2, 2]^(n + 1) + Aₛ * H[2, 2]^n) * ∇Sₙ
end

# surface velocity magnitude
function _surface_velocity!(V, H, B, As, A, ρgn, n, dx, dy)
    @get_indices
    @inbounds if ix <= size(V, 1) && iy <= size(V, 2)
        Hₗ = st3x3(H, ix, iy)
        Bₗ = st3x3(B, ix, iy)
        V[ix, iy] = _surface_velocity(Hₗ, Bₗ, As[ix, iy], A, ρgn, n, dx, dy)
    end
    return
end

# wrappers
function residual!(r, z, B, H, H_old, A, Aₛ, ρgn, n, b, ela, mb_max, mb_mask, dt, dx, dy, mode)
    nthreads, nblocks = launch_config(size(H))
    @cuda threads = nthreads blocks = nblocks _residual!(r, z, B, H, H_old, A, Aₛ, ρgn, n, b, ela, mb_max, mb_mask, inv(dt), inv(dx), inv(dy), mode)
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
