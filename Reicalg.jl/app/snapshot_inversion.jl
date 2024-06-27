using Reicalg
using CUDA
using CairoMakie
using JLD2
using Printf

function load_from_file(path)
    scalars, numerics = load(path, "scalars", "numerics")
    fields = SIA_fields(numerics.nx, numerics.ny)

    host_fields = load(path, "fields")

    for name in intersect(keys(fields), keys(host_fields))
        copy!(fields[name], host_fields[name])
    end

    return fields, scalars, numerics
end

function snapshot_inversion()
    fields, scalars, numerics = load_from_file("datasets/aletsch/aletsch_setup.jld2")
    # fields, scalars, numerics = load_from_file("datasets/synthetic/synthetic_setup.jld2")

    (; nx, ny, dx, dy, xc, yc) = numerics

    E = 1e-1
    scalars = merge(scalars, (; A=scalars.A * E))

    xc = xc ./ 1e3
    yc = yc ./ 1e3

    V_obs = copy(fields.V)
    ωᵥ    = inv(sum(V_obs .^ 2))

    V̄ = CUDA.zeros(Float64, nx - 1, ny - 1)

    As0 = fields.As
    fill!(As0, 1e-20)

    obj_params = (; V_obs, ωᵥ)
    adj_params = (; fields=(; V̄))
    reg_params = (; β=1e-3, dx, dy)

    function J(As)
        fields     = merge(fields, (; As))
        fwd_params = (; fields, scalars, numerics)
        objective_snapshot!(fwd_params, obj_params)
    end

    function ∇J!(Ās, As)
        fields     = merge(fields, (; As))
        fwd_params = (; fields, scalars, numerics)
        grad_objective_snapshot!(Ās, fwd_params, adj_params, obj_params)
    end

    fig = Figure(; size=(650, 450))

    ax = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="log10(As)"),
          Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="Ās"),
          Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="V_obs"),
          Axis(fig[2, 2][1, 1]; aspect=DataAspect(), title="V"))

    hm = (heatmap!(ax[1], xc, yc, Array(log10.(As0)); colormap=:turbo),
          heatmap!(ax[2], xc, yc, Array(log10.(As0)); colormap=:turbo),
          heatmap!(ax[3], xc, yc, Array(V_obs); colormap=:turbo, colorrange=(0, 1e-5)),
          heatmap!(ax[4], xc, yc, Array(fields.V); colormap=:turbo, colorrange=(0, 1e-5)))

    cb = (Colorbar(fig[1, 1][1, 2], hm[1]),
          Colorbar(fig[1, 2][1, 2], hm[2]),
          Colorbar(fig[2, 1][1, 2], hm[3]),
          Colorbar(fig[2, 2][1, 2], hm[4]))

    function callback(iter, γ, J1, As, Ās)
        if iter % 100 == 0
            @printf("  iter = %d, J = %1.3e\n", iter, J1)

            hm[1][3] = Array(log10.(As))
            hm[2][3] = Array(Ās)
            hm[4][3] = Array(fields.V)
            autolimits!(ax[1])
            autolimits!(ax[2])
            autolimits!(ax[4])
            display(fig)
        end
    end

    # gradient_descent(J, ∇J!, As0, 4e3, 2000; callback, reg_params)
    gradient_descent(J, ∇J!, As0, 5e3, 2000; callback, reg_params)

    display(fig)

    return
end

snapshot_inversion()
