struct TimeDependentFields{A<:AbstractMatrix{<:Real}}
    B::A
    H::A
    H_old::A
    As::A
    V::A
    D::A
    mb_mask::A
    r_H::A
    d_H::A
    dH_dτ::A
end
function TimeDependentFields(nx, ny, T=Float64)
    return TimeDependentFields(CUDA.zeros(T, nx, ny),
                               CUDA.zeros(T, nx, ny),
                               CUDA.zeros(T, nx, ny),
                               CUDA.zeros(T, nx - 1, ny - 1),
                               CUDA.zeros(T, nx - 1, ny - 1),
                               CUDA.zeros(T, nx - 1, ny - 1),
                               CUDA.zeros(T, nx - 2, ny - 2),
                               CUDA.zeros(T, nx - 2, ny - 2),
                               CUDA.zeros(T, nx - 2, ny - 2),
                               CUDA.zeros(T, nx - 2, ny - 2))
end

struct TimeDependentAdjointFields{A<:AbstractMatrix{<:Real}}
    ∂J_∂H::A
    H̄::A
    D̄::A
    V̄::A
    ψ::A
    r̄_H::A
end
function TimeDependentAdjointFields(nx, ny, T=Float64)
    return TimeDependentAdjointFields(CUDA.zeros(T, nx, ny),
                                      CUDA.zeros(T, nx, ny),
                                      CUDA.zeros(T, nx - 1, ny - 1),
                                      CUDA.zeros(T, nx - 1, ny - 1),
                                      CUDA.zeros(T, nx - 2, ny - 2),
                                      CUDA.zeros(T, nx - 2, ny - 2))
end

# scalars
mutable struct TimeDependentScalars{T<:Real,NP<:Real}
    lx::T
    ly::T
    npow::NP
    A::T
    ρgn::T
    β::T
    b_max::T
    ela::T
    dt::T
end
function TimeDependentScalars(; lx, ly, npow=GLEN_N, A0=GLEN_A, E=1.0, ρgn=RHOG_N, β, b_max, ela, dt)
    return TimeDependentScalars(lx, ly, npow, A0 * E, ρgn, β, b_max, ela, dt)
end

# numerics
mutable struct TimeDependentNumerics{T<:Real,I<:Integer,R}
    nx::I
    ny::I
    dx::T
    dy::T
    xc::R
    yc::R
    cfl::T
    εtol::T
    dmp1::T
    dmp2::T
    dmpswitch::I
    maxiter::I
    ncheck::I
end
function TimeDependentNumerics(xc, yc;
                               cfl       = 1 / 5.1,
                               εtol      = 1e-6,
                               dmp1      = 0.6,
                               dmp2      = 0.8,
                               dmpswitch = 5max(length(xc), length(yc)),
                               maxiter   = 100max(length(xc), length(yc)),
                               ncheck    = 1max(length(xc), length(yc)))
    nx, ny = length(xc), length(yc)
    dx, dy = step(xc), step(yc)
    return TimeDependentNumerics(nx, ny, dx, dy, xc, yc, cfl, εtol, dmp1, dmp2, dmpswitch, maxiter, ncheck)
end

mutable struct TimeDependentAdjointNumerics{T<:Real,I<:Integer}
    cfl::T
    εtol::T
    dmp::T
    maxiter::I
    ncheck::I
end
function TimeDependentAdjointNumerics(xc, yc;
                                      cfl     = 1 / 4.1,
                                      εtol    = 1e-6,
                                      dmp     = 0.8,
                                      maxiter = 50max(length(xc), length(yc)),
                                      ncheck  = ceil(Int, 0.5 * max(length(xc), length(yc))))
    return TimeDependentAdjointNumerics(cfl, εtol, dmp, maxiter, ncheck)
end
