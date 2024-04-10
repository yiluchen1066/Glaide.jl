using CUDA
using CairoMakie
using Printf

include("macros.jl")

function solve_sia_explicit!(logAs, fields, scalars, numerical_params, launch_config; visu=nothing)
    # extract variables from tuples
    (; H, H_old, B, β, ELA, ELA_2, D, qHx, qHy, As, RH, qmag, Err_rel, Err_abs) = fields
    (; aρgn0, b_max, npow)                              = scalars
    (; nx, ny, dx, dy, t_total, maxiter, ncheck, ϵtol)       = numerical_params
    (; nthreads, nblocks)                               = launch_config
    # initialize 
    err_abs0 = Inf
    RH .= 0
    Err_rel .= 0
    Err_abs .= 0
    As      .= exp10.(logAs)
    ELA     .= ELA_2
    CUDA.synchronize()
    # iterative loop 
    t  = 0.0
    it = 1
    while t < t_total
        @cuda threads = nthreads blocks = nblocks compute_D!(D, H, B, As, aρgn0, npow, dx, dy)
        @cuda threads = nthreads blocks = nblocks compute_q!(qHx, qHy, D, H, B, dx, dy)
        @. qmag = sqrt($avx(qHx)^2 + $avy(qHy)^2)
        dτ = min(1 / (8.1 * maximum(D) / dx^2 + 10 * maximum(β)), t_total - t)
        @cuda threads = nthreads blocks = nblocks residual!(RH, qHx, qHy, β, H, B, ELA, b_max, dx, dy)
        @cuda threads = nthreads blocks = nblocks update_H!(H, RH, dτ)
        @cuda threads = nthreads blocks = nblocks set_BC!(H)
        t += dτ
        it += 1
        if it % 1000 == 0
            @printf("it = %d, dτ = %1.3e, t = %1.5e/%1.5e\n", it, dτ, t, t_total)
        end
    end 
    return
end

# kernels

# compute diffusion coefficient
function compute_D!(D, H, B, As, aρgn0, npow, dx, dy)
    @get_indices
    @inbounds if ix <= size(D, 1) && iy <= size(D, 2)
        ∇Sx = 0.5 *
              ((B[ix + 1, iy + 1] - B[ix, iy + 1]) / dx +
               (H[ix + 1, iy + 1] - H[ix, iy + 1]) / dx +
               (B[ix + 1, iy] - B[ix, iy]) / dx +
               (H[ix + 1, iy] - H[ix, iy]) / dx)
        ∇Sy = 0.5 *
              ((B[ix + 1, iy + 1] - B[ix + 1, iy]) / dy +
               (H[ix + 1, iy + 1] - H[ix + 1, iy]) / dy +
               (B[ix, iy + 1] - B[ix, iy]) / dy +
               (H[ix, iy + 1] - H[ix, iy]) / dy)
        D[ix, iy] = (aρgn0 * @av_xy(H)^(npow + 2) + As[ix, iy] * @av_xy(H)^npow) * sqrt(∇Sx^2 + ∇Sy^2)^(npow - 1)
    end
    return
end

# compute ice flux
function compute_q!(qHx, qHy, D, H, B, dx, dy)
    @get_indices
    @inbounds if ix <= size(qHx, 1) && iy <= size(qHx, 2)
        qHx[ix, iy] = -@av_ya(D) * (@d_xi(H) + @d_xi(B)) / dx
    end
    @inbounds if ix <= size(qHy, 1) && iy <= size(qHy, 2)
        qHy[ix, iy] = -@av_xa(D) * (@d_yi(H) + @d_yi(B)) / dy
    end
    return
end

# compute ice flow residual
function residual!(RH, qHx, qHy, β, H, B, ELA, b_max, dx, dy)
    @get_indices
    if ix <= size(H, 1) - 2 && iy <= size(H, 2) - 2
        MB = min(β[ix + 1, iy + 1] * (H[ix + 1, iy + 1] + B[ix + 1, iy + 1] - ELA[ix + 1, iy + 1]), b_max)
        #H_diff = (H[ix+1, iy+1] - H_old[ix+1, iy+1])/dt
        @inbounds RH[ix + 1, iy + 1] = -(@d_xa(qHx) / dx + @d_ya(qHy) / dy) + MB
    end
    return
end

# update ice thickness
function update_H!(H, RH, dτ)
    @get_indices
    if ix <= size(H, 1) - 2 && iy <= size(H, 2) - 2
        @inbounds H[ix + 1, iy + 1] = max(0.0, H[ix + 1, iy + 1] + dτ * RH[ix + 1, iy + 1])
    end
    return
end

# set boundary conditions
function set_BC!(H)
    @get_indices
    if ix == 1 && iy <= size(H, 2)
        @inbounds H[ix, iy] = H[ix + 1, iy]
    end
    if ix == size(H, 1) && iy <= size(H, 2)
        @inbounds H[ix, iy] = H[ix - 1, iy]
    end
    if ix <= size(H, 1) && iy == 1
        @inbounds H[ix, iy] = H[ix, iy + 1]
    end
    if ix <= size(H, 1) && iy == size(H, 2)
        @inbounds H[ix, iy] = H[ix, iy - 1]
    end
    return
end

# compute absolute error
function compute_abs_error!(Err_abs, RH, H)
    @get_indices
    if ix <= size(H, 1) - 2 && iy <= size(H, 2) - 2
        @inbounds if H[ix + 1, iy + 1] ≈ 0.0
            Err_abs[ix + 1, iy + 1] = 0.0
        else
            Err_abs[ix + 1, iy + 1] = RH[ix + 1, iy + 1]
        end
    end
    return
end

function update_visualisation_new!(As, visu, fields, err_evo)
    (; fig, plts)     = visu
    (; H)            = fields
    plts.H[3]        = Array(H)
    plts.As[3]       = Array(log10.(As))
    # plts.err[1][1][] = Point2.(err_evo.iters, err_evo.abs)
    # plts.err[2][1][] = Point2.(err_evo.iters, err_evo.rel)
    display(fig)
    return
end