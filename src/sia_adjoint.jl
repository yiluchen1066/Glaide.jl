# update adjoint thickness, manually setting to zero where H == 0
function _update_adjoint_state!(ψ, p, α)
    @get_indices
    @inbounds if ix <= size(ψ, 1) && iy <= size(ψ, 2)
        # update adjoint state
        ψ[ix, iy] += α * p[ix, iy]
    end
    return
end

function ∇residual!(r, z, B, H, H_old, A, Aₛ, ρgn, n, b, ela, mb_max, mb_mask, dt, dx, dy, mode)
    nthreads, nblocks = launch_config(size(H.val))
    @cuda threads = nthreads blocks = nblocks ∇(_residual!, r, z, B, H, H_old, A, Aₛ, ρgn, n, b, ela, mb_max, mb_mask, dt, dx, dy, mode)
    return
end

function ∇surface_velocity!(V, H, B, As, A, ρgn, n, dx, dy)
    nthreads, nblocks = launch_config(size(H.val))
    @cuda threads = nthreads blocks = nblocks ∇(_surface_velocity!, V, H, B, As, A, ρgn, n, dx, dy)
    return
end

function update_adjoint_state!(ψ, p, α)
    nthreads, nblocks = launch_config(size(ψ))
    @cuda threads = nthreads blocks = nblocks _update_adjoint_state!(ψ, p, α)
    return
end
