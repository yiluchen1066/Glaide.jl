struct SnapshotFields{A<:AbstractArray{<:Real}}
    B::A
    H::A
    As::A
    V::A
end
function SnapshotFields(nx, ny, T=Float64)
    SnapshotFields(CUDA.zeros(T, nx, ny),
                   CUDA.zeros(T, nx, ny),
                   CUDA.zeros(T, nx, ny),
                   CUDA.zeros(T, nx, ny))
end

struct SnapshotAdjointFields{A<:AbstractArray{<:Real}}
    V̄::A
end
SnapshotAdjointFields(nx, ny, T=Float64) = SnapshotAdjointFields(CUDA.zeros(T, nx, ny))

mutable struct SnapshotScalars{T<:Real,NP<:Real}
    lx::T
    ly::T
    n::NP
    A::T
    ρgn::T
end
function SnapshotScalars(; lx, ly, n=GLEN_N, A0=GLEN_A, E=1.0, ρgn=RHOG_N)
    return SnapshotScalars(lx, ly, n, A0 * E, ρgn)
end

struct SnapshotNumerics{T<:Real,I<:Integer,R}
    nx::I
    ny::I
    dx::T
    dy::T
    xc::R
    yc::R
end
function SnapshotNumerics(xc, yc)
    nx, ny = length(xc), length(yc)
    dx, dy = step(xc), step(yc)
    return SnapshotNumerics(nx, ny, dx, dy, xc, yc)
end
