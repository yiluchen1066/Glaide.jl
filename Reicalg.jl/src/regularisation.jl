# laplacian smoothing
function _smooth!(Ãs, As, dτβ, dx, dy)
    @get_indices
    @inbounds if ix <= size(As, 1) - 2 && iy <= size(As, 2) - 2
        ΔAs = (As[ix, iy+1] - 2.0 * As[ix+1, iy+1] + As[ix+2, iy+1]) / dx^2 +
              (As[ix+1, iy] - 2.0 * As[ix+1, iy+1] + As[ix+1, iy+2]) / dy^2

        Ãs[ix+1, iy+1] = As[ix+1, iy+1] + dτβ * ΔAs
    end
    return
end

# boundary conditions
#! format: off
function _bc_x!(As)
    @get_index_1d
    @inbounds if i <= size(As, 2)
        As[1  , i] = As[2    , i]
        As[end, i] = As[end-1, i]
    end
    return
end

function _bc_y!(As)
    @get_index_1d
    @inbounds if i <= size(As, 1)
        As[i, 1  ] = As[i, 2    ]
        As[i, end] = As[i, end-1]
    end
    return
end
#! format: on

# Tikhonov regularisation with regularity parameter β and step size γ
function regularise!(As, Ãs, γ, β, dx, dy)
    dτ     = min(dx, dy)^2 / β / 4.1
    nsteps = ceil(Int, γ / dτ)
    dτβ    = γ / nsteps * β

    # smoothing kernel launch config
    nth, nbl = launch_config(size(As))

    # bc kernels launch config
    nth_x, nbl_x = launch_config(size(As, 2))
    nth_y, nbl_y = launch_config(size(As, 1))

    for _ in 1:nsteps
        CUDA.@sync @cuda threads = nth blocks = nbl _smooth!(Ãs, As, dτβ, dx, dy)
        CUDA.@sync @cuda threads = nth_x blocks = nbl_x _bc_x!(Ãs)
        CUDA.@sync @cuda threads = nth_y blocks = nbl_y _bc_y!(Ãs)
        As, Ãs = Ãs, As
    end

    return As, Ãs
end
