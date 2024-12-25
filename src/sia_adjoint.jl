# update adjoint thickness, manually setting to zero where H == 0
function _update_adjoint_state!(ψ, p, α)
    @get_indices
    @inbounds if ix <= size(ψ, 1) && iy <= size(ψ, 2)
        # update adjoint state
        ψ[ix, iy] += α * p[ix, iy]
    end
    return
end

function ∇residual!(r, z, B, H, H_old, ρgnAₛ, mb_mask, ρgnA, n, b, mb_max, ela, dt, dx, dy, mode)
    nthreads, nblocks = launch_config(size(H.val), 2)

    # precompute constants
    ρgnA2_n2 = 2 / (n + 2) * ρgnA
    _n3      = inv(n + 3)
    _n2      = inv(n + 2)
    _dt      = inv(dt)
    _dx      = inv(dx)
    _dy      = inv(dy)
    nm1      = n - oneunit(n)

    @cuda threads = nthreads blocks = nblocks ∇(_residual!, r, z, B, H, H_old, ρgnAₛ, mb_mask,
                                                Const.((ρgnA2_n2, b, mb_max, ela, _dt, _dx, _dy, _n3, _n2, n, nm1, mode))...)

    return
end

function ∇surface_velocity!(V, H, B, ρgnAs, ρgnA, n, dx, dy)
    nthreads, nblocks = launch_config(size(H.val))

    # precompute constants
    ρgnA2_n1 = 2 / (n + 1) * ρgnA
    n1       = n + oneunit(n)
    _dx      = inv(dx)
    _dy      = inv(dy)

    @cuda threads = nthreads blocks = nblocks ∇(_surface_velocity!, V, H, B, ρgnAs, Const.((ρgnA2_n1, _dx, _dy, n, n1))...)
    return
end

function update_adjoint_state!(ψ, p, α)
    nthreads, nblocks = launch_config(size(ψ))
    @cuda threads = nthreads blocks = nblocks _update_adjoint_state!(ψ, p, α)
    return
end
