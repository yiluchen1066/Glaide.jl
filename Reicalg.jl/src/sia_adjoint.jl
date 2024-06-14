# update adjoint thickness, manually setting to zero where H == 0
function _update_adjoint_state!(ψ, dψ_dτ, H̄, H, dτ, dmp)
    @get_indices
    @inbounds if ix <= size(ψ, 1) && iy <= size(ψ, 2)
        # residual damping improves convergence
        dψ_dτ[ix, iy] = dψ_dτ[ix, iy] * dmp + H̄[ix+1, iy+1]

        if H[ix+1, iy+1] > 0.0
            ψ[ix, iy] += dτ * dψ_dτ[ix, iy]
        else
            ψ[ix, iy] = 0.0
            H̄[ix+1, iy+1] = 0.0
        end
    end
    return
end

function ∇diffusivity!(D, H, B, As, A, ρgn, npow, dx, dy)
    nthreads, nblocks = launch_config(size(H.val))
    @cuda threads = nthreads blocks = nblocks ∇(_diffusivity!, D, H, B, As, A, ρgn, npow, dx, dy)
    return
end

function ∇residual!(r_H, B, H, H_old, D, β, ELA, b_max, mb_mask, dt, dx, dy)
    nthreads, nblocks = launch_config(size(H.val))
    @cuda threads = nthreads blocks = nblocks ∇(_residual!, r_H, B, H, H_old, D, β, ELA, b_max, mb_mask, dt, dx, dy)
    return
end

function ∇surface_velocity!(v, H, B, As, A, ρgn, npow, dx, dy)
    nthreads, nblocks = launch_config(size(H.val))
    @cuda threads = nthreads blocks = nblocks ∇(_surface_velocity!, v, H, B, As, A, ρgn, npow, dx, dy)
    return
end

function update_adjoint_state!(ψ, dψ_dτ, H̄, H, dτ, dmp)
    nthreads, nblocks = launch_config(size(H))
    @cuda threads = nthreads blocks = nblocks _update_adjoint_state!(ψ, dψ_dτ, H̄, H, dτ, dmp)
    return
end
