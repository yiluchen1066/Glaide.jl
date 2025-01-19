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
@kwdef mutable struct TimeDependentScalars{T<:Real,NP<:Real}
    lx::T
    ly::T
    n::NP
    ρgnA::T
    b::T
    mb_max::T
    ela::T
    dt::T
end

# numerics
mutable struct TimeDependentNumerics{T<:Real,I<:Integer}
    nx::I
    ny::I
    reg::T
    εtol::T
    dmptol::T
    α::T
    maxf::T
    checkf::T
end
function TimeDependentNumerics(nx, ny;
                               reg,
                               εtol   = 1e-8,
                               dmptol = 1e-4,
                               α      = 1.0,
                               maxf   = 100.0,
                               checkf = 0.25)
    return TimeDependentNumerics(nx, ny, reg, εtol, dmptol, α, maxf, checkf)
end

mutable struct TimeDependentAdjointNumerics{T<:Real}
    εtol::T
    dmptol::T
    α::T
    maxf::T
    checkf::T
end
function TimeDependentAdjointNumerics(;
                                      εtol   = 1e-8,
                                      dmptol = 1e-4,
                                      α      = 1.0,
                                      maxf   = 100.0,
                                      checkf = 0.25)
    return TimeDependentAdjointNumerics(εtol, dmptol, α, maxf, checkf)
end
