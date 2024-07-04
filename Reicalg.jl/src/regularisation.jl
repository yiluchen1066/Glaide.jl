struct Regularisation{T<:Real,A<:AbstractMatrix{T}}
    β::T
    dx::T
    dy::T
    X̃::A
end
function Regularisation(β::Real, dx::Real, dy::Real, X::AbstractMatrix)
    return Regularisation(promote(β, dx, dy)..., similar(X))
end

# laplacian smoothing
function _smooth!(X̃, X, dτβ, dx, dy)
    @get_indices
    @inbounds if ix <= size(X, 1) - 2 && iy <= size(X, 2) - 2
        ΔX = (X[ix, iy+1] - 2.0 * X[ix+1, iy+1] + X[ix+2, iy+1]) / dx^2 +
             (X[ix+1, iy] - 2.0 * X[ix+1, iy+1] + X[ix+1, iy+2]) / dy^2

        X̃[ix+1, iy+1] = X[ix+1, iy+1] + dτβ * ΔX
    end
    return
end

function smooth!(X̃, X, dτβ, dx, dy)
    nthreads, nblocks = launch_config(size(X))
    @cuda threads = nthreads blocks = nblocks _smooth!(X̃, X, dτβ, dx, dy)
    return
end

# fallback for no regularisation
regularise!(::Any, ::Any, ::Nothing) = nothing

# Tikhonov regularisation with regularity parameter β and step size γ
function regularise!(X, γ, regularisation::Regularisation)
    (; β, dx, dy, X̃) = regularisation

    # maximum step size for Laplacian smoothing from von Neumann stability analysis
    dτ = min(dx, dy)^2 / β / 4

    # number of steps for Laplacian smoothing
    nsteps = ceil(Int, γ / dτ)

    # total step size for Laplacian smoothing should sum up to γ
    dτβ = γ / nsteps * β

    for _ in 1:nsteps
        # smooth As and apply homogeneous Neumann boundary conditions
        smooth!(X̃, X, dτβ, dx, dy)
        bc!(X̃)
        # swap As and Ãs
        X, X̃ = X̃, X
    end

    # copy back to the original array if number of steps is odd
    # to make sure X always contains the regularised field
    if nsteps % 2 == 1
        copy!(X̃, X)
    end

    return
end
