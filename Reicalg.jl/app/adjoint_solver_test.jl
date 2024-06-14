using Reicalg
using CUDA
using CairoMakie
using JLD2

function create_sia_fields(nx, ny)
    return (B       = CUDA.zeros(Float64, nx, ny),
            H       = CUDA.zeros(Float64, nx, ny),
            H_old   = CUDA.zeros(Float64, nx, ny),
            D       = CUDA.zeros(Float64, nx - 1, ny - 1),
            As      = CUDA.zeros(Float64, nx - 1, ny - 1),
            mb_mask = CUDA.zeros(Float64, nx - 2, ny - 2),
            r_H     = CUDA.zeros(Float64, nx - 2, ny - 2),
            d_H     = CUDA.zeros(Float64, nx - 2, ny - 2),
            dH_dτ   = CUDA.zeros(Float64, nx - 2, ny - 2),
            ELA     = CUDA.zeros(Float64, nx - 2, ny - 2))
end

function load_from_file(path)
    scalars, numerics = load(path, "scalars", "numerics")
    fields = create_sia_fields(numerics.nx, numerics.ny)

    host_fields = load(path, "fields")

    for name in intersect(keys(fields), keys(host_fields))
        copy!(fields[name], host_fields[name])
    end

    return fields, scalars, numerics
end

function adjoint_solver_test()
    fields, scalars, numerics = load_from_file("datasets/synthetic/synthetic_setup.jld2")

    ψ     = similar(fields.r_H)
    r̄_H  = similar(fields.r_H)
    H̄    = similar(fields.H)
    D̄    = similar(fields.D)
    ∂J_∂H = similar(fields.H)

    ψ     .= 0.0
    r̄_H  .= 0.0
    H̄    .= 0.0
    D̄    .= 0.0
    ∂J_∂H .= 1.0

    (; nx, ny, dx, dy) = numerics

    cfl     = 1 / 6.1
    maxiter = 20max(nx, ny)
    ncheck  = 1max(nx, ny)
    ϵtol    = 1e-8

    fwd_params = (; fields, scalars, numerics)

    adj_fields   = (; ψ, r̄_H, H̄, D̄, ∂J_∂H)
    adj_numerics = (; nx, ny, dx, dy, cfl, maxiter, ncheck, ϵtol)

    adj_params = (fields=adj_fields, numerics=adj_numerics)

    (; D, H, B, As) = fields
    (; A, ρgn, npow) = scalars

    diffusivity!(D, H, B, As, A, ρgn, npow, dx, dy)
    adjoint_sia!(fwd_params, adj_params; debug_vis=true)

    return
end

adjoint_solver_test()
