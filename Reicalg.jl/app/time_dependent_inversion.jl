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

function time_dependent_inversion()
    fields, scalars, numerics = load_from_file("datasets/synthetic/synthetic_setup.jld2")

    (; nx, ny, dx, dy, xc, yc) = numerics

    fwd_numerics = merge(numerics, (; cfl=1 / 6.1,
                                    maxiter=100 * max(nx, ny),
                                    ncheck=2nx,
                                    ϵtol=1e-6))

    adj_numerics = merge(numerics, (; cfl=1 / 4.1,
                                    maxiter=50 * max(nx, ny),
                                    ncheck=1nx,
                                    ϵtol=1e-6))

    xc = xc ./ 1e3
    yc = yc ./ 1e3

    V_obs = copy(fields.V)
    H_obs = copy(fields.H)

    α = 0.5√2

    ωᵥ = α * inv(sum(V_obs .^ 2))
    ωₕ = √(1 - α^2) * inv(sum(H_obs .^ 2))

    adj_fields = SIA_adjoint_fields(nx, ny)

    As0 = fields.As
    fill!(As0, 1e-20)
    As0 .= exp.(log.(As0) .+ 1.0 .* (2.0 .* CUDA.rand(Float64, size(As0)) .- 1.0))

    obj_params = (; V_obs, H_obs, ωᵥ, ωₕ)
    adj_params = (; fields=adj_fields, numerics=adj_numerics)
    reg_params = (; β=10.0, dx, dy)

    function J(As)
        fields     = merge(fields, (; As))
        fwd_params = (; fields, scalars, numerics=fwd_numerics)
        objective_time_dependent!(fwd_params, obj_params)
    end

    function ∇J!(Ās, As)
        fields     = merge(fields, (; As))
        fwd_params = (; fields, scalars, numerics=fwd_numerics)
        grad_objective_time_dependent!(Ās, fwd_params, adj_params, obj_params)
    end

    fig = Figure(; size=(650, 450))

    ax = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="log10(As)"),
          Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="Ās"),
          Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="V_obs"),
          Axis(fig[2, 2][1, 1]; aspect=DataAspect(), title="V"))

    hm = (heatmap!(ax[1], xc, yc, Array(log10.(As0)); colormap=:turbo),
          heatmap!(ax[2], xc, yc, Array(log10.(As0)); colormap=:turbo),
          heatmap!(ax[3], xc, yc, Array(V_obs); colormap=:turbo),
          heatmap!(ax[4], xc, yc, Array(fields.V); colormap=:turbo))

    cb = (Colorbar(fig[1, 1][1, 2], hm[1]),
          Colorbar(fig[1, 2][1, 2], hm[2]),
          Colorbar(fig[2, 1][1, 2], hm[3]),
          Colorbar(fig[2, 2][1, 2], hm[4]))

    function callback(iter, γ, J1, As, Ās)
        if iter % 200 == 0
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

    gradient_descent(J, ∇J!, As0, 2e3, 2000; callback, reg_params)

    display(fig)

    return
end

time_dependent_inversion()
