### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 1e1881e8-4b55-11ef-292e-8536405b7a19
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(Base.current_project())
    Pkg.instantiate()

    using Glaide
    using CairoMakie
    using Printf
    using JLD2
    using CUDA

    using PlutoUI
    TableOfContents()
end

# ╔═╡ ed930fe1-2f84-4445-9bce-420f7039304b
md"""
# Time-dependent inversion
"""

# ╔═╡ 4c90e341-2156-4978-a075-815730f8d8fe
md"""
## Configuration
"""

# ╔═╡ 33ae25bb-3efa-4cb8-9591-ed676a568f00
Base.@kwdef struct InversionScenario
    input_file::String
    E::Float64  = 1.0
    ωₕ::Float64 = 1.0
    ωᵥ::Float64 = 1.0
end;

# ╔═╡ 45a039e1-aaad-4275-8bb3-3d070f613dfd
As_init = 1e-22;

# ╔═╡ 6a8c2160-e759-406a-86fd-43e8c7ac00ab
β_reg = 1e-3;

# ╔═╡ e096dcf8-41df-47a3-b399-aed370ec1ab6
maxiter = 5000;

# ╔═╡ 3ec533a0-3f7b-4add-be81-2aebb1d7c14a
line_search = BacktrackingLineSearch(; α_min=1e0, α_max=1e4);

# ╔═╡ 939d13a9-a08e-4614-a939-49440a024f3b
md"""
## Synthetic glacier
"""

# ╔═╡ 8a30e84f-0c2a-4dab-8311-266bd1b65320
synthetic_scenario = InversionScenario(; input_file="../datasets/synthetic_25m.jld2");

# ╔═╡ 13bd69ce-057b-407d-a193-cc412c02f97a
begin
    mutable struct Callback{M,JH}
        model::M
        j_hist::JH
        fig::Figure
        axs
        hms
        lns
        cbs
        video_stream

        function Callback(model, V_obs)
            j_hist = Point2{Float64}[]

            fig = Figure(; size=(800, 850))

            axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect()),
                   Axis(fig[1, 2][1, 1]; aspect=DataAspect()),
                   Axis(fig[2, 1][1, 1]; aspect=DataAspect()),
                   Axis(fig[2, 2][1, 1]; aspect=DataAspect()),
                   Axis(fig[3, :]; yscale=log10))

            axs[1].title = "log10(As)"
            axs[2].title = "dJ/d(logAs)"
            axs[3].title = "V_obs"
            axs[4].title = "V"
            axs[5].title = "Convergence"

            axs[5].xlabel = "Iteration";
            axs[5].ylabel = "J"

            xc_km, yc_km = model.numerics.xc ./ 1e3, model.numerics.yc ./ 1e3;

            hms = (heatmap!(axs[1], xc_km, yc_km, Array(log10.(model.fields.As))),
                   heatmap!(axs[2], xc_km, yc_km, Array(log10.(model.fields.As))),
                   heatmap!(axs[3], xc_km, yc_km, Array(V_obs)),
                   heatmap!(axs[4], xc_km, yc_km, Array(model.fields.V)))

            hms[1].colormap = Reverse(:roma)
            hms[2].colormap = Reverse(:roma)
            hms[3].colormap = :turbo
            hms[4].colormap = :turbo

            hms[1].colorrange = (-24, -20)
            hms[2].colorrange = (-1e-8, 1e-8)
            hms[3].colorrange = (0, 1e-5)
            hms[4].colorrange = (0, 1e-5)

            lns = (lines!(axs[5], Point2{Float64}[]), )

            cbs = (Colorbar(fig[1, 1][1, 2], hms[1]),
                   Colorbar(fig[1, 2][1, 2], hms[2]),
                   Colorbar(fig[2, 1][1, 2], hms[3]),
                   Colorbar(fig[2, 2][1, 2], hms[4]))

            new{typeof(model), typeof(j_hist)}(model,
                                               j_hist,
                                               fig,
                                               axs,
                                               hms,
                                               lns,
                                               cbs,
                                               nothing)
        end
    end

    function (cb::Callback)(state::OptmisationState)
        if state.iter == 0
            empty!(cb.j_hist)
            cb.video_stream = VideoStream(cb.fig; framerate=10)
        end

        push!(cb.j_hist, Point2(state.iter, state.j_value))

        if state.iter % 20 == 0
            @info @sprintf("iter #%-4d, J = %1.3e, ΔJ/J = %1.3e, ΔX/X = %1.3e, α = %1.3e\n", state.iter,
                        state.j_value,
                        state.j_change,
                        state.x_change,
                        state.α)

            cb.hms[1][3] = Array(state.X .* log10(ℯ))
            cb.hms[2][3] = Array(state.X̄ ./ log10(ℯ))
            cb.hms[4][3] = Array(cb.model.fields.V)
            cb.lns[1][1] = cb.j_hist
            autolimits!(cb.axs[5])
            recordframe!(cb.video_stream)
        end
        return
    end

    md"""
    ⬅️ Show the contents of this cell to see the definition of `Callback` and the visualisation code.
    """
end

# ╔═╡ 20b21077-02a5-4840-853d-3dc2692e7acf
function run_inversion(scenario::InversionScenario)
    model = TimeDependentSIA(scenario.input_file)

    model.scalars.A *= scenario.E

    V_obs = copy(model.fields.V)

    V_obs = copy(model.fields.V)
    H_obs = copy(model.fields.H)

    ωn = sqrt(scenario.ωᵥ^2 + scenario.ωₕ^2)

    ωᵥ = scenario.ωᵥ * inv(ωn * sum(V_obs .^ 2))
    ωₕ = scenario.ωₕ * inv(ωn * sum(H_obs .^ 2))

    objective = TimeDependentObjective(ωᵥ, ωₕ, V_obs, H_obs, β_reg)

    # see cells below for details on how the callback is implemented
    callback = Callback(model, V_obs)
    options  = OptimisationOptions(; line_search, callback, maxiter)

    # initial guess
    logAs0 = CUDA.fill(log(As_init), size(V_obs))

    # inversion
    optimise(model, objective, logAs0, options)

    # show animation
    return callback.video_stream
end;

# ╔═╡ 3210e218-e632-4f37-ae24-b46aff3a7325
run_inversion(synthetic_scenario)

# ╔═╡ Cell order:
# ╟─1e1881e8-4b55-11ef-292e-8536405b7a19
# ╟─ed930fe1-2f84-4445-9bce-420f7039304b
# ╟─4c90e341-2156-4978-a075-815730f8d8fe
# ╠═33ae25bb-3efa-4cb8-9591-ed676a568f00
# ╠═45a039e1-aaad-4275-8bb3-3d070f613dfd
# ╠═6a8c2160-e759-406a-86fd-43e8c7ac00ab
# ╠═e096dcf8-41df-47a3-b399-aed370ec1ab6
# ╠═3ec533a0-3f7b-4add-be81-2aebb1d7c14a
# ╠═20b21077-02a5-4840-853d-3dc2692e7acf
# ╠═939d13a9-a08e-4614-a939-49440a024f3b
# ╠═8a30e84f-0c2a-4dab-8311-266bd1b65320
# ╠═3210e218-e632-4f37-ae24-b46aff3a7325
# ╟─13bd69ce-057b-407d-a193-cc412c02f97a
