# CUDA kernels

# SIA diffusivity
function _diffusivity!(D, H, B, As, A, ρgn, npow, dx, dy)
    @get_indices
    @inbounds if ix <= size(D, 1) && iy <= size(D, 2)
        # surface graient
        ∇Sx = 0.5 * (@d_xi(B) / dx + @d_xa(B) / dx) +
              0.5 * (@d_xi(H) / dx + @d_xa(H) / dx)

        ∇Sy = 0.5 * (@d_yi(H) / dy + @d_ya(H) / dy) +
              0.5 * (@d_yi(B) / dy + @d_ya(B) / dy)

        # diffusion coefficient
        D[ix, iy] = 2.0 / (npow + 2.0) * ρgn * (A * @av_xy(H)^(npow + 2) + As[ix, iy] * @av_xy(H)^npow) * sqrt(∇Sx^2 + ∇Sy^2)^(npow - 1)
    end
    return
end

# surface flux q = ∫v dz
function _flux!(qx, qy, D, H, B, dx, dy)
    @get_indices
    @inbounds if ix <= size(qx, 1) && iy <= size(qx, 2)
        qx[ix, iy] = -@av_ya(D) * (@d_xi(H) + @d_xi(B)) / dx
    end
    @inbounds if ix <= size(qy, 1) && iy <= size(qy, 2)
        qy[ix, iy] = -@av_xa(D) * (@d_yi(H) + @d_yi(B)) / dy
    end
    return
end

# mass conservation residual
function _residual!(r_H, B, H, H_old, qx, qy, β, ELA, b_max, mb_mask, dt, dx, dy)
    @get_indices
    @inbounds if ix <= size(H, 1) - 2 && iy <= size(H, 2) - 2
        # observed geometry changes
        dH_dt = (H[ix+1, iy+1] - H_old[ix+1, iy+1]) / dt

        # ice flow
        divQ = @d_xa(qx) / dx + @d_ya(qy) / dy

        # surface mass balance
        b = ela_mass_balance(B[ix+1, iy+1] + H[ix+1, iy+1], β, ELA[ix, iy], b_max)

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
        dH_dτ[ix, iy] = dH_dτ[ix, iy] * dmp + r_H[ix, iy]
        H[ix+1, iy+1] = max(0.0, H[ix+1, iy+1] + dτ * dH_dτ[ix, iy])

        # set residual to zero in ice-free cells to correctly detect convergence
        if H[ix+1, iy+1] == 0.0
            r_H[ix, iy] = 0.0
        end
    end
    return
end

# boundary conditions
#! format: off
function _bc_x!(H, B)
    @get_index_1d
    @inbounds if i <= size(H, 2)
        H[1  , i] = H[2    , i] + (B[2    , i] - B[1  , i])
        H[end, i] = H[end-1, i] + (B[end-1, i] - B[end, i])
    end
    return
end

function _bc_y!(H, B)
    @get_index_1d
    @inbounds if i <= size(H, 1)
        H[i, 1  ] = H[i, 2    ] + (B[i, 2    ] - B[i, 1  ])
        H[i, end] = H[i, end-1] + (B[i, end-1] - B[i, end])
    end
    return
end
#! format: on

# surface velocity magnitude
function surface_velocity!(v, H, B, As, A, ρgn, npow, dx, dy)
    # TODO
    return
end

# wrappers
function diffusivity!(D, H, B, As, A, ρgn, npow, dx, dy)
    nthreads, nblocks = launch_config(size(H))
    @cuda threads = nthreads blocks = nblocks _diffusivity!(D, H, B, As, A, ρgn, npow, dx, dy)
    return
end

function flux!(qx, qy, D, H, B, dx, dy)
    nthreads, nblocks = launch_config(size(H))
    @cuda threads = nthreads blocks = nblocks _flux!(qx, qy, D, H, B, dx, dy)
    return
end

function residual!(r_H, B, H, H_old, qx, qy, β, ELA, b_max, mb_mask, dt, dx, dy)
    nthreads, nblocks = launch_config(size(H))
    @cuda threads = nthreads blocks = nblocks _residual!(r_H, B, H, H_old, qx, qy, β, ELA, b_max, mb_mask, dt, dx, dy)
    return
end

function update_ice_thickness!(H, dH_dτ, r_H, dτ, dmp)
    nthreads, nblocks = launch_config(size(H))
    @cuda threads = nthreads blocks = nblocks _update_ice_thickness!(H, dH_dτ, r_H, dτ, dmp)
    return
end

function bc!(H, B)
    # bc in x
    nthreads, nblocks = launch_config(size(H, 2))
    @cuda threads = nthreads blocks = nblocks _bc_x!(H, B)

    # bc in y
    nthreads, nblocks = launch_config(size(H, 1))
    @cuda threads = nthreads blocks = nblocks _bc_y!(H, B)

    return
end