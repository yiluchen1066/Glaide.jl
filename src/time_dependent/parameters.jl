struct TimeDependentFields{A,B}
    B::A
    H::A
    H_old::A
    ρgnAs::A
    V::A
    mb_mask::B
    r::A
    r0::A
    z::A
    p::A
    d::A
end
function TimeDependentFields(nx, ny, T=Float64)
    return TimeDependentFields(CUDA.zeros(T, nx, ny),
                               CUDA.zeros(T, nx, ny),
                               CUDA.zeros(T, nx, ny),
                               CUDA.zeros(T, nx, ny),
                               CUDA.zeros(T, nx, ny),
                               CUDA.zeros(T, nx, ny),
                               CUDA.zeros(T, nx, ny),
                               CUDA.zeros(T, nx, ny),
                               CUDA.zeros(T, nx, ny),
                               CUDA.zeros(T, nx, ny),
                               CUDA.zeros(T, nx, ny))
end

struct TimeDependentAdjointFields{A<:AbstractMatrix{<:Real}}
    ∂J_∂H::A
    H̄::A
    V̄::A
    ψ::A
    r̄::A
    z̄::A
end
function TimeDependentAdjointFields(nx, ny, T=Float64)
    return TimeDependentAdjointFields(CUDA.zeros(T, nx, ny),
                                      CUDA.zeros(T, nx, ny),
                                      CUDA.zeros(T, nx, ny),
                                      CUDA.zeros(T, nx, ny),
                                      CUDA.zeros(T, nx, ny),
                                      CUDA.zeros(T, nx, ny))
end

# scalars
mutable struct TimeDependentScalars{T<:Real,NP<:Real}
    lx::T
    ly::T
    n::NP
    ρgnA::T
    b::T
    mb_max::T
    ela::T
    dt::T
end
function TimeDependentScalars(; lx, ly, n=GLEN_N, ρgnA0=RHOGNA, E=1.0, b, mb_max, ela, dt)
    return TimeDependentScalars(lx, ly, n, ρgnA0 * E, b, mb_max, ela, dt)
end

# numerics
mutable struct TimeDependentNumerics{T<:Real,I<:Integer,R}
    nx::I
    ny::I
    dx::T
    dy::T
    xc::R
    yc::R
    εtol::T
    α::T
    dmpswitch::I
    ndmp::I
    maxiter::I
    ncheck::I
end
function TimeDependentNumerics(xc, yc;
                               εtol      = 1e-8,
                               α         = 1.0,
                               dmpswitch = ceil(Int, 1max(length(xc), length(yc))),
                               ndmp      = 1,
                               maxiter   = 20max(length(xc), length(yc)),
                               ncheck    = ceil(Int, 0.25max(length(xc), length(yc))))
    nx, ny = length(xc), length(yc)
    dx, dy = step(xc), step(yc)
    return TimeDependentNumerics(nx, ny, dx, dy, xc, yc, εtol, α, dmpswitch, ndmp, maxiter, ncheck)
end

mutable struct TimeDependentAdjointNumerics{T<:Real,I<:Integer}
    εtol::T
    α::T
    maxiter::I
    ncheck::I
end
function TimeDependentAdjointNumerics(xc, yc;
                                      εtol    = 1e-8,
                                      α       = 1.0,
                                      maxiter = 20max(length(xc), length(yc)),
                                      ncheck  = ceil(Int, 0.25max(length(xc), length(yc))))
    return TimeDependentAdjointNumerics(εtol, α, maxiter, ncheck)
end
