using CUDA
using CairoMakie
using Printf

include("macros.jl")

# integrate SIA equations to steady state
function solve_sia!(logAs, fields, scalars, numerical_params, launch_config; visu=nothing)
    # extract variables from tuples
    (; H, B, β, ELA, D, qHx, qHy, As, RH, Err_rel, Err_abs) = fields
    (; aρgn0, b_max, npow)                              = scalars
    (; nx, ny, dx, dy, maxiter, ncheck, ϵtol)           = numerical_params
    (; nthreads, nblocks)                               = launch_config
    # initialize 
    err_abs0 = Inf
    RH .= 0
    Err_rel .= 0
    Err_abs .= 0
    As .= exp10.(logAs)
    CUDA.synchronize()
    # iterative loop 
    err_evo = (iters = Float64[],
               abs   = Float64[],
               rel   = Float64[])
    for iter in 1:maxiter
        if iter == 1 || iter % ncheck == 0
            CUDA.@sync Err_rel .= H
        end
        @cuda threads = nthreads blocks = nblocks compute_D!(D, H, B, As, aρgn0, npow, dx, dy)
        @cuda threads = nthreads blocks = nblocks compute_q!(qHx, qHy, D, H, B, dx, dy)
        CUDA.synchronize()
        # compute stable time step
        dτ = 1 / (12.1 * maximum(D) / dx^2 + maximum(β))
        @cuda threads = nthreads blocks = nblocks residual!(RH, qHx, qHy, β, H, B, ELA, b_max, dx, dy)
        @cuda threads = nthreads blocks = nblocks update_H!(H, RH, dτ)
        @cuda threads = nthreads blocks = nblocks set_BC!(H)
        if iter == 1 || iter % ncheck == 0
            @cuda threads = nthreads blocks = nblocks compute_abs_error!(Err_abs, RH, H)
            Err_rel .-= H
            CUDA.synchronize()
            err_abs = maximum(abs.(Err_abs))
            err_rel = maximum(abs.(Err_rel)) / maximum(H)
            push!(err_evo.iters, iter / nx)
            push!(err_evo.abs, err_abs)
            push!(err_evo.rel, err_rel)
            @printf("  iter/nx^2=%.3e, err= [abs=%.3e, rel=%.3e] \n", iter / nx^2, err_abs, err_rel)
            isnothing(visu) || update_visualisation!(As, visu, fields, err_evo)
            if err_rel < ϵtol.rel
                break
            end
        end
    end
    return
end

# kernels

# compute diffusion coefficient
function compute_D!(D, H, B, As, aρgn0, n, dx, dy)
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
        D[ix, iy] = (aρgn0 * @av_xy(H)^(n + 2) + As[ix, iy] * @av_xy(H)^n) * sqrt(∇Sx^2 + ∇Sy^2)^(n - 1)
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

function update_visualisation!(As, visu, fields, err_evo)
    (; fig, plt)    = visu
    (; H)           = fields
    plt.H[3]        = Array(H)
    plt.As[3]       = Array(log10.(As))
    plt.Err[1][1][] = Point2.(err_evo.iters, err_evo.abs)
    plt.Err[2][1][] = Point2.(err_evo.iters, err_evo.rel)
    display(fig)
    return
end