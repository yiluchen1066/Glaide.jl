include("macros.jl")
include("sia_forward_flux_2D.jl")

using Enzyme

@inline ∇(fun, args...) = (Enzyme.autodiff_deferred(Enzyme.Reverse, fun, Const, args...); return)
const DupNN = DuplicatedNoNeed

function solve_adjoint_sia!(fwd_params, adj_params, loss_params)
    #unpack forward 
    (; H, B, β, ELA, D, qx, qy, As, RH, qmag) = fwd_params.fields
    (; aρgn0, b_max, npow)                = fwd_params.scalars
    (; nx, ny, dx, dy, maxiter)           = fwd_params.numerical_params
    (; nthreads, nblocks)                 = fwd_params.launch_config
    #unpack adjoint 
    (; R̄H, H̄, ψ_H, q̄Hx, q̄Hy, D̄) = adj_params.fields
    (; ϵtol_adj, ncheck_adj, H_cut) = adj_params.numerical_params
    (; H_obs,  ∂J_∂H, qobs_mag) = loss_params.fields

    dt = 1.0 / (8.1 * maximum(D) / min(dx, dy)^2 + maximum(β))
    ∂J_∂H .= H .- H_obs

    ∂J_∂qx_vec!(q̄Hx, qmag, qobs_mag, qx)
    ∂J_∂qy_vec!(q̄Hy, qmag, qobs_mag, qy)

    # ∂J_∂qx .= qx .- qx_obs
    # ∂J_∂qy .= qy .- qy_obs
    # q̄Hx .= ∂J_∂qx
    # q̄Hy .= ∂J_∂qy

    D̄ .= 0.0
    @cuda threads = nthreads blocks = nblocks ∇(compute_q!,
                                                         DupNN(qx, q̄Hx),
                                                         DupNN(qy, q̄Hy),
                                                         DupNN(D, D̄),
                                                         DupNN(H, ∂J_∂H),
                                                         Const(B), Const(dx), Const(dy))
    @cuda threads = nthreads blocks = nblocks ∇(compute_D!,
                                                         DupNN(D, D̄),
                                                         DupNN(H, ∂J_∂H),
                                                         Const(B), Const(As), Const(aρgn0), Const(npow), Const(dx),
                                                         Const(dy))

    merr = 2ϵtol_adj
    iter = 1
    #CUDA.synchronize()
    while merr >= ϵtol_adj && iter < maxiter
        #initialization 
        R̄H  .= ψ_H
        q̄Hx .= 0.0
        q̄Hy .= 0.0
        D̄   .= 0.0
        H̄   .= .-∂J_∂H
        #CUDA.synchronize()

        @cuda threads = nthreads blocks = nblocks ∇(residual!,
                                                    DupNN(RH, R̄H),
                                                    DupNN(qx, q̄Hx),
                                                    DupNN(qy, q̄Hy),
                                                    Const(β),
                                                    DupNN(H, H̄), # dR_H
                                                    Const(B), Const(ELA), Const(b_max), Const(dx), Const(dy))

        @cuda threads = nthreads blocks = nblocks ∇(compute_q!,
                                                    DupNN(qx, q̄Hx),
                                                    DupNN(qy, q̄Hy),
                                                    DupNN(D, D̄),
                                                    DupNN(H, H̄), # dq_H
                                                    Const(B), Const(dx), Const(dy))
        @cuda threads = nthreads blocks = nblocks ∇(compute_D!,
                                                    DupNN(D, D̄),
                                                    DupNN(H, H̄), # dD_H
                                                    Const(B), Const(As), Const(aρgn0), Const(npow), Const(dx),
                                                    Const(dy))
        @cuda threads = nthreads blocks = nblocks update_ψ!(H, H̄, ψ_H, H_cut, dt)

        # in the new code formulation, H̄ equals to dR/R 
        if iter % ncheck_adj == 0
            CUDA.synchronize()
            merr = maximum(abs.(dt .* H̄[2:(end - 1), 2:(end - 1)])) / maximum(abs.(ψ_H))
            @printf("error = %.1e\n", merr)
            # visualization for adjoint solver
            @printf("H̄ = %.1e\n", maximum(abs.(H̄)))
            (isfinite(merr) && merr > 0) || error("adjoint solve failed")
        end

        iter += 1
    end

    if iter == maxiter && merr >= ϵtol_adj
        error("adjoint not converged")
    end

    @printf("adjoint solve converged: #iter/nx = %.1f, err = %.1e\n", iter / nx, merr)
    return
end

function update_ψ!(H, H̄, ψ_H, H_cut, dt)
    @get_indices
    @inbounds if ix <= size(H, 1) && iy <= size(H, 2)
        if ix > 1 && ix < size(H, 1) && iy > 1 && iy < size(H, 2)
            if H[ix, iy] <= H_cut ||
               H[ix - 1, iy] <= H_cut ||
               H[ix + 1, iy] <= H_cut ||
               H[ix, iy - 1] <= H_cut ||
               H[ix, iy + 1] <= H_cut
                H̄[ix, iy] = 0.0
                ψ_H[ix, iy] = 0.0
            else
                ψ_H[ix, iy] = ψ_H[ix, iy] + dt * H̄[ix, iy]
            end
        end
        if ix == 1 || ix == size(H, 1)
            ψ_H[ix, iy] = 0.0
        end
        if iy == 1 || iy == size(H, 2)
            ψ_H[ix, iy] = 0.0
        end
    end
    return
end

function ∂J_∂qx_vec!(q̄Hx, qmag, qobs_mag, qx)
    q̄Hx                .= 0
    @. q̄Hx[1:end-1, :] += (qmag - qobs_mag) * $avx(qx) / (2 * qmag + (qmag == 0))
    @. q̄Hx[2:end, :]   += (qmag - qobs_mag) * $avx(qx) / (2 * qmag + (qmag == 0))
    return
end

function ∂J_∂qy_vec!(q̄Hy, qmag, qobs_mag, qy)
    q̄Hy                .= 0
    @. q̄Hy[:, 1:end-1] += (qmag - qobs_mag) * $avy(qy) / (2 * qmag + (qmag == 0))
    @. q̄Hy[:, 2:end]   += (qmag - qobs_mag) * $avy(qy) / (2 * qmag + (qmag == 0))
    return
end
