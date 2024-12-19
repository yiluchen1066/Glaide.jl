using Enzyme, CUDA, StaticArrays

# helper functions for static arrays
δˣₐ(S::StaticMatrix{3,<:Any}) = S[SVector(2, 3), :] .- S[SVector(1, 2), :]
δʸₐ(S::StaticMatrix{<:Any,3}) = S[:, SVector(2, 3)] .- S[:, SVector(1, 2)]

δˣ(S::StaticMatrix{3,3}) = S[SVector(2, 3), SVector(2)] .- S[SVector(1, 2), SVector(2)]
δʸ(S::StaticMatrix{3,3}) = S[SVector(2), SVector(2, 3)] .- S[SVector(2), SVector(1, 2)]

δˣ(S::StaticMatrix{2,1}) = S[2] - S[1]
δʸ(S::StaticMatrix{1,2}) = S[2] - S[1]

function av4(S::StaticMatrix{2,3})
    0.25 .* (S[SVector(1), SVector(1, 2)] .+ S[SVector(2), SVector(1, 2)] .+
             S[SVector(1), SVector(2, 3)] .+ S[SVector(2), SVector(2, 3)])
end

function av4(S::StaticMatrix{3,2})
    0.25 .* (S[SVector(1, 2), SVector(1)] .+ S[SVector(1, 2), SVector(2)] .+
             S[SVector(2, 3), SVector(1)] .+ S[SVector(2, 3), SVector(2)])
end

innˣ(S::StaticMatrix{3,<:Any}) = S[SVector(2), :]
innʸ(S::StaticMatrix{<:Any,3}) = S[:, SVector(2)]

# extract 3x3 stencil
function st3x3(M, ix, iy)
    nx, ny = oftype.((ix, iy), size(M))
    # neighbor indices
    di = oneunit(ix)
    dj = oneunit(iy)
    iW = max(ix - di, di)
    iE = min(ix + di, nx)
    iS = max(iy - dj, dj)
    iN = min(iy + dj, ny)
    return SMatrix{3,3}(M[iW, iS], M[ix, iS], M[iE, iS],
                        M[iW, iy], M[ix, iy], M[iE, iy],
                        M[iW, iN], M[ix, iN], M[iE, iN])
end

# Enzyme utils
∇(fun, args::Vararg{Any,N}) where {N} = (Enzyme.autodiff_deferred(Enzyme.Reverse, Const(fun), Const, args...); return)
const DupNN = DuplicatedNoNeed

function residual(H, n, d)
    # surface gradient
    ∇Hˣ = δˣₐ(H)
    ∇Hʸ = δʸₐ(H)

    # surface gradient magnitude
    ∇Sˣ = sqrt.(innʸ(∇Hˣ) .^ 2 .+ av4(∇Hʸ) .^ 2) .^ (n - 1)
    ∇Sʸ = sqrt.(av4(∇Hˣ) .^ 2 .+ innˣ(∇Hʸ) .^ 2) .^ (n - 1)

    qˣ = ∇Sˣ .* δˣ(H .^ (n + 3))
    qʸ = ∇Sʸ .* δʸ(H .^ (n + 3))

    r = d * (δˣ(qˣ) + δʸ(qʸ)) + H[2, 2]

    return r
end

function gpu_residual!(r, H, n, d)
    ix = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    Hₗ        = st3x3(H, ix, iy)
    r[ix, iy] = residual(Hₗ, n, d)

    return
end

function cpu_residual!(r, H, n, d)
    nx, ny = size(H)
    for ix in 1:nx, iy in 1:ny
        Hₗ        = st3x3(H, ix, iy)
        r[ix, iy] = residual(Hₗ, n, d)
    end
    return
end

function gpu_runme()
    nthreads = 32, 4
    nx, ny   = nthreads
    # fake data
    n = 3
    d = 1e0
    # arrays
    H1 = [Float64(i/nx + j/ny) for i in 1:nx, j in 1:ny]
    r1 = zeros(Float64, nx, ny)
    H2 = CuArray(H1)
    r2 = CuArray(r1)
    # shadows
    r̄1 = ones(Float64, nx, ny)
    H̄1 = zeros(Float64, nx, ny)
    r̄2 = CUDA.ones(Float64, nx, ny)
    H̄2 = CUDA.zeros(Float64, nx, ny)

    cpu_residual!(r1, H1, n, d)
    Enzyme.autodiff(Enzyme.Reverse, Const(cpu_residual!), DupNN(r1, r̄1), DupNN(H1, H̄1), Const(n), Const(d))

    H̄1 = CuArray(H̄1)

    for i in 1:1000
        r̄2 .= 1.0
        H̄2 .= 0.0
        @cuda threads = nthreads gpu_residual!(r2, H2, n, d)
        @cuda threads = nthreads ∇(gpu_residual!, DupNN(r2, r̄2), DupNN(H2, H̄2), Const(n), Const(d))

        if H̄1 != H̄2
            println("r1:")
            display(r1)
            println("r2:")
            display(Array(r2))
            println("r1 - r2:")
            display(r1 .- Array(r2))
            println("H1:")
            display(H̄1)
            println("H2:")
            display(H̄2)
            println("H1 - H2:")
            display(H̄1 .- H̄2)
            error("CUDA: non-deterministic results at iteration $i")
        end
    end

    println("CUDA: no errors")

    return
end

gpu_runme()
