# CUDA kernels

# surface gradient
Base.@propagate_inbounds function ∇S(H, B, dx, dy, ix, iy)
    ∇Sx = 0.5 * (@d_xi(B) + @d_xa(B)) / dx +
          0.5 * (@d_xi(H) + @d_xa(H)) / dx

    ∇Sy = 0.5 * (@d_yi(B) + @d_ya(B)) / dy +
          0.5 * (@d_yi(H) + @d_ya(H)) / dy

    return sqrt(∇Sx^2 + ∇Sy^2)
end

# SIA fluxes
Base.@propagate_inbounds function qx(H, B, D, dx, dy, ix, iy)
    return -@av_ya(D) * (@d_xi(H) + @d_xi(B)) / dx
end

Base.@propagate_inbounds function qy(H, B, D, dx, dy, ix, iy)
    return -@av_xa(D) * (@d_yi(H) + @d_yi(B)) / dy
end

# SIA diffusivity
function _diffusivity!(D, H, B, As, A, ρgn, npow, dx, dy)
    @get_indices
    @inbounds if ix <= size(D, 1) && iy <= size(D, 2)
        # surface gradient
        gradS = ∇S(H, B, dx, dy, ix, iy)
        # diffusion coefficient
        H_av      = @av_xy(H)
        D[ix, iy] = ρgn * (2.0 / (npow + 2) * A * H_av^(npow + 2) + As[ix, iy] * H_av^(npow + 1)) * gradS^(npow - 1)
    end
    return
end

# mass conservation residual
function _residual!(r_H, B, H, H_old, D, β, ela, b_max, mb_mask, dt, dx, dy)
    @get_indices
    @inbounds if ix <= size(H, 1) - 2 && iy <= size(H, 2) - 2
        # observed geometry changes
        dH_dt = (H[ix+1, iy+1] - H_old[ix+1, iy+1]) / dt

        # interface fluxes
        qx_w = qx(H, B, D, dx, dy, ix, iy)
        qx_e = qx(H, B, D, dx, dy, ix + 1, iy)

        qy_s = qy(H, B, D, dx, dy, ix, iy)
        qy_n = qy(H, B, D, dx, dy, ix, iy + 1)

        # ice flow
        divQ = (qx_e - qx_w) / dx + (qy_n - qy_s) / dy

        # surface mass balance
        b = ela_mass_balance(B[ix+1, iy+1] + H[ix+1, iy+1], β, ela, b_max)

        # no accumulation if (b > 0) && (H == 0) at t == t0
        b *= mb_mask[ix, iy]

        # total mass conservation
        r_H[ix, iy] = b - dH_dt - divQ
    end
    return
end

# update ice thickness with constraint H > 0
function _update_ice_thickness!(H, dH_dτ, r_H, dτ, dmp)
    @get_indices
    @inbounds if ix <= size(H, 1) - 2 && iy <= size(H, 2) - 2
        # residual damping improves convergence
        dH_dτ[ix, iy] = dH_dτ[ix, iy] * dmp + r_H[ix, iy]

        # update ice thickness
        H[ix+1, iy+1] = max(0.0, H[ix+1, iy+1] + dτ * dH_dτ[ix, iy])

        # set residual to zero in ice-free cells to correctly detect convergence
        if H[ix+1, iy+1] == 0.0
            r_H[ix, iy] = 0.0
        end
    end
    return
end

# surface velocity magnitude
function _surface_velocity!(V, H, B, As, A, ρgn, npow, dx, dy)
    @get_indices
    @inbounds if ix <= size(V, 1) && iy <= size(V, 2)
        # surface gradient
        gradS = ∇S(H, B, dx, dy, ix, iy)
        # velocity magnitude
        H_av      = @av_xy(H)
        V[ix, iy] = ρgn * (2.0 / (npow + 1) * A * H_av^(npow + 1) + As[ix, iy] * H_av^npow) * gradS^npow
    end
    return
end

# wrappers
function diffusivity!(D, H, B, As, A, ρgn, npow, dx, dy)
    nthreads, nblocks = launch_config(size(H))
    @cuda threads = nthreads blocks = nblocks _diffusivity!(D, H, B, As, A, ρgn, npow, dx, dy)
    return
end

function residual!(r_H, B, H, H_old, D, β, ela, b_max, mb_mask, dt, dx, dy)
    nthreads, nblocks = launch_config(size(H))
    @cuda threads = nthreads blocks = nblocks _residual!(r_H, B, H, H_old, D, β, ela, b_max, mb_mask, dt, dx, dy)
    return
end

function update_ice_thickness!(H, dH_dτ, r_H, dτ, dmp)
    nthreads, nblocks = launch_config(size(H))
    @cuda threads = nthreads blocks = nblocks _update_ice_thickness!(H, dH_dτ, r_H, dτ, dmp)
    return
end

function surface_velocity!(V, H, B, As, A, ρgn, npow, dx, dy)
    nthreads, nblocks = launch_config(size(H))
    @cuda threads = nthreads blocks = nblocks _surface_velocity!(V, H, B, As, A, ρgn, npow, dx, dy)
end
