using Reicalg
using CUDA
using CairoMakie
using JLD2

function load_from_file(path)
    scalars, numerics = load(path, "scalars", "numerics")
    fields = SIA_fields(numerics.nx, numerics.ny)

    host_fields = load(path, "fields")

    for name in intersect(keys(fields), keys(host_fields))
        copy!(fields[name], host_fields[name])
    end

    return fields, scalars, numerics
end

function adjoint_solver_test()
    fields, scalars, numerics = load_from_file("datasets/synthetic/synthetic_setup.jld2")

    (; H, H_old, As, V) = fields
    (; nx, ny, dx, dy)  = numerics

    adj_fields = SIA_adjoint_fields(nx, ny)

    cfl     = 1 / 6.1
    maxiter = 20max(nx, ny)
    ncheck  = 1max(nx, ny)
    ϵtol    = 1e-8

    numerics   = merge(numerics, (; cfl, maxiter, ncheck, ϵtol))
    fwd_params = (; fields, scalars, numerics)

    adj_numerics = (; nx, ny, dx, dy, cfl, maxiter, ncheck, ϵtol)
    adj_params   = (fields=adj_fields, numerics=adj_numerics)

    obj_params = (ωₕ    = √2 / 2 * inv(sum(H_old .^ 2)),
                  ωᵥ    = √2 / 2 * 1.0,
                  H_obs = 1.1 .* H,
                  V_obs = 1.1 .* V)


    @show objective_time_dependent!(fwd_params, obj_params)

    dJ_dAs = CUDA.zeros(Float64, size(As))

    grad_objective_time_dependent!(dJ_dAs, fwd_params, adj_params, obj_params)

    # adjoint_sia!(fwd_params, adj_params; debug_vis=true)

    return
end

adjoint_solver_test()
