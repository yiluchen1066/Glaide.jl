# update adjoint thickness, manually setting to zero where H == 0
function _update_adjoint_state!(ψ, dψ_dτ, H̄, D, dmp, β, dt, cfl, dx, dy)
    @get_indices
    @inbounds if ix <= size(ψ, 1) && iy <= size(ψ, 2)
        # residual damping improves convergence
        dψ_dτ[ix, iy] = dψ_dτ[ix, iy] * dmp + H̄[ix+1, iy+1]

        # local timestep (same as forward model)
        D_av = @av_xy(D)
        dτ   = inv((D_av + 1e-2) / min(dx, dy)^2 / cfl + β + inv(dt))

        # update adjoint state
        ψ[ix, iy] = ψ[ix, iy] + dτ * dψ_dτ[ix, iy]
    end
    return
end

function ∇diffusivity!(D, H, B, As, A, ρgn, npow, dx, dy)
    nthreads, nblocks = launch_config(size(H.val))
    @cuda threads = nthreads blocks = nblocks ∇(_diffusivity!, D, H, B, As, A, ρgn, npow, dx, dy)
    return
end

function ∇residual!(r_H, B, H, H_old, D, β, ela, b_max, mb_mask, dt, dx, dy)
    nthreads, nblocks = launch_config(size(H.val))
    @cuda threads = nthreads blocks = nblocks ∇(_residual!, r_H, B, H, H_old, D, β, ela, b_max, mb_mask, dt, dx, dy)
    return
end

function ∇surface_velocity!(v, H, B, As, A, ρgn, npow, dx, dy)
    nthreads, nblocks = launch_config(size(H.val))
    @cuda threads = nthreads blocks = nblocks ∇(_surface_velocity!, v, H, B, As, A, ρgn, npow, dx, dy)
    return
end

function update_adjoint_state!(ψ, dψ_dτ, H̄, D, dmp, β, dt, cfl, dx, dy)
    nthreads, nblocks = launch_config(size(ψ))
    @cuda threads = nthreads blocks = nblocks _update_adjoint_state!(ψ, dψ_dτ, H̄, D, dmp, β, dt, cfl, dx, dy)
    return
end
