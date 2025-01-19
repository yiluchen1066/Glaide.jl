### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 815419fa-b7cc-11ef-225d-1be9a77d974c
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(Base.current_project())
    Pkg.instantiate()

    using Glaide, CairoMakie, Printf, JLD2, CUDA, Unitful
	
    using PlutoUI; TableOfContents()
end

# ╔═╡ 48c28533-fbc6-40bd-a18f-35324b1ff32c
md"""
# Snapshot inversion

In this notebook, we will use inverse modelling routine implemented in Glaide.jl to reconstruct spatially variable sliding parameter $A_\mathrm{s}$. The inverse modelling problem is defined as a minimisation problem with the following objective functional:

```math
J(A_\mathrm{s}) = \frac{\omega_V}{2}\sum_i\left(V_i(A_\mathrm{s}) - V^\mathrm{obs}_i\right)^2 + \frac{\gamma}{2}\sum_i\left(\nabla A_{\mathrm{s}_i}\right)^2~,
```
where $\omega_V$ is the normalisation factor and $\gamma$ is the Tikhonov regularisation parameter.

The normalisation constant is defined as the inverse of the $L_2$-norm of the observed velocity field:

```math
\omega_V = \left[\sum_i\left(V^\mathrm{obs}_i\right)^2\right]^{-1}
```
"""

# ╔═╡ 9afa1d2e-e9fd-4d38-9ecb-1dfc0ec0d75e
md"""
## Configuration

Define the type encapsulating the properties of the inversion that might be different between cases. In this study, these are the path to the input file and the enhancement factor $E$, which is needed to reduce the influence of the ice deformation in the Aletsch case:
"""

# ╔═╡ 485febf2-6338-415d-8f4c-77568af5ab57
Base.@kwdef struct InversionScenarioSnapshot
    input_file::String
    output_dir::String
    E::Float64 = 1.0
end;

# ╔═╡ bebfc476-c778-42bd-a807-59adfea64788
md"""
!!! warning "Prerequisites"
    Before running this notebook, make sure input files exist on your filesystem.
	- To generate the input files for the synthetic setup, run the notebook [`generate_synthetic_setup.jl`](./open?path=app/generate_synthetic_setup.jl).
	- To generate the input files for the Aletsch setup, run the notebook [`generate_aletsch_setup.jl`](./open?path=app/generate_synthetic_setup.jl).
"""

# ╔═╡ c69c5ce6-a8af-49e4-8100-398d8df0b0b3
md"""
Define the initial guess for the sliding parameter:
"""

# ╔═╡ a19c5990-1d62-4349-84c1-4cf2522b0eb4
As_init = 1e-22u"Pa^-3*s^-1*m"

# ╔═╡ 5206fc60-6bcd-444a-82bb-f8c2f336c393
md"""
Define the regularisation parameter:
"""

# ╔═╡ 9f08540e-deac-4999-b6cc-c685d5ec679d
γ_reg = 1e-6;

# ╔═╡ c6f6e76c-9f06-463e-a7b4-0d2aee421275
md"""
Define the maximum number of iterations in the optimisation algorithm:
"""

# ╔═╡ 5594683b-17ed-48b6-9804-9286b0f7ea40
maxiter = 1000;

# ╔═╡ 78e44885-07e5-43e8-ba3b-b6c94e6e40d8
md"""
Define the parameters of the line search. Here, we only configure the minimal and maximal step size. In Glaide, the two-way backtracking line search based on Armijo-Goldstein condition is implemented. If in the line search loop the step size decreases below $\alpha_\min$, the optimisation stops with an error. If the step size increases above $\alpha_\max$, line search accepts $\alpha_\max$ as the step size. Increasing $\alpha_\max$ might improve convergence rate in some cases, but can also lead to instabilities and convergence issues in the forward solver.
"""

# ╔═╡ 494028ea-ad86-476c-b52a-d5c2102d55e9
line_search = BacktrackingLineSearch(; α_min=1e2, α_max=1e6);

# ╔═╡ f13d57d1-a793-4566-b325-aa416a53e0ca
md"""
## Inversion function

Here, we create a function that executes the inversion scenario:
"""

# ╔═╡ c910ebda-7bb9-418f-a572-989ee04cb4f3
md"""
Some comments on the above code:

- We initialise the observed velocity field `V_obs` with `model.fields.V`. This is because the synthetic velocity is stored in the model's field `V`, as implemented in the notebook [`generate_synthetic_setup.jl`](./open?path=Glaide.jl/app/generate_synthetic_setup.jl);
- We create the callback object `callback = Callback(model, obs)`. The definition of `Callback` is a bit convoluted, but in short, it handles the debug visualisation, keeping track of the convergence history, and saving the intermediate results of the optimisation. For implementation details, see the __Extras__ section at the end of the notebook.
"""

# ╔═╡ 16574a8b-5dc9-47ba-b924-5545f7d5e893
md"""
## Synthetic inversion

First, we define the inversion scenario for the synthetic glacier. Since the input data was generated using the standard physical parameters, we leave the enhancement factor to the default value $E = 1$:
"""

# ╔═╡ 2dfdbd36-ab24-4a1e-aa57-89c272bd2866
synthetic_scenario = InversionScenarioSnapshot(; input_file="../datasets/synthetic_50m.jld2",
output_dir="../output/snapshot_synthetic_50m");

# ╔═╡ 0ed5551b-5584-4792-9c27-aacf0f8919f6
md"""
Run the inversion:
"""

# ╔═╡ b93039b7-9413-4625-b84c-d17e3d003b15
md"""
## Aletsch inversion

Then, we create the inversion scenario to reconstruct the distribution of the sliding parameter at the base of the Aletsch glacier. There, using the standard parameters for the flow parameter $A = 2.5\times10^{-24}\,\text{Pa}\,\text{s}^{-3}$ result in the surface velocity values much higher than the observed ones even without any sliding. This is likely due to using the SIA model that doesn't account for longitudinal stresses and non-hydrostatic pressure variations. We correct this by introducing a flow enhancement factor $E = 0.25$:
"""

# ╔═╡ 4f4bccc4-e7b6-45b8-8500-d985352f1cbb
aletsch_scenario = InversionScenarioSnapshot(; input_file="../datasets/aletsch_25m.jld2", output_dir="../output/snapshot_aletsch_25m", E=0.25);

# ╔═╡ 4e7b2d0f-9b3c-4df8-a4e5-3518dae0127a
md"""
Run the inversion:
"""

# ╔═╡ 4e3d8f5c-974b-442a-9267-67ff53121872
md"""
## Extras
"""

# ╔═╡ 7dd6f707-c723-4666-977b-efec92b2a5f0
begin
    mutable struct Callback{M,S,JH}
        model::M
        scenario::S
        j_hist::JH
        fig::Figure
        axs
        hms
        lns
        cbs
        video_stream
        step::Int

        function Callback(model, scenario, V_obs)
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

            xc_km = ustrip.(u"km", model.numerics.xc .* L_REF)
			yc_km = ustrip.(u"km", model.numerics.yc .* L_REF)

			n     = model.scalars.n
			ρgnAs = model.fields.ρgnAs
			V     = model.fields.V

			As_v    = ustrip.(u"Pa^-3*s^-1*m", ρgnAs .* (L_REF^(1-n) * T_REF^(-1) / RHOG^n))
			V_obs_v = ustrip.(u"m/yr", V_obs .* (L_REF / T_REF))
			V_v     = ustrip.(u"m/yr", V     .* (L_REF / T_REF))

            hms = (heatmap!(axs[1], xc_km, yc_km, Array(log10.(As_v))),
                   heatmap!(axs[2], xc_km, yc_km, Array(log10.(As_v))),
                   heatmap!(axs[3], xc_km, yc_km, Array(V_obs_v)),
                   heatmap!(axs[4], xc_km, yc_km, Array(V_v)))

            hms[1].colormap = Reverse(:roma)
            hms[2].colormap = Reverse(:roma)
            hms[3].colormap = :turbo
            hms[4].colormap = :turbo

            hms[1].colorrange = (-24, -20)
            hms[2].colorrange = (-1e-8, 1e-8)
            hms[3].colorrange = (0, 300)
            hms[4].colorrange = (0, 300)

            lns = (lines!(axs[5], Point2{Float64}[]), )

            cbs = (Colorbar(fig[1, 1][1, 2], hms[1]),
                   Colorbar(fig[1, 2][1, 2], hms[2]),
                   Colorbar(fig[2, 1][1, 2], hms[3]),
                   Colorbar(fig[2, 2][1, 2], hms[4]))

            new{typeof(model), typeof(scenario), typeof(j_hist)}(model,
                                                                 scenario,
                                                                 j_hist,
                                                                 fig,
                                                                 axs,
                                                                 hms,
                                                                 lns,
                                                                 cbs,
                                                                 nothing, 0)
        end
    end

    function (cb::Callback)(state::OptmisationState)
        if state.iter == 0
            empty!(cb.j_hist)
            cb.video_stream = VideoStream(cb.fig; framerate=10)
            cb.step = 0

            mkpath(cb.scenario.output_dir)
        end

        push!(cb.j_hist, Point2(state.iter, state.j_value))

        if state.iter % 10 == 0
            @info @sprintf("iter #%-4d, J = %1.3e, ΔJ/J = %1.3e, ΔX/X = %1.3e, α = %1.3e\n", state.iter,
                      state.j_value,
                      state.j_change,
                      state.x_change,
                      state.α)

			n     = cb.model.scalars.n
			ρgnAs = cb.model.fields.ρgnAs
			V     = cb.model.fields.V

			coef = (L_REF^(1-n) * T_REF^(-1) / RHOG^n)

		    As_v = ustrip.(u"Pa^-3*s^-1*m", ρgnAs .* coef)
			V_v  = ustrip.(u"m/yr", V .* (L_REF / T_REF))
			

            cb.hms[1][3] = Array(log10.(As_v))
            cb.hms[2][3] = Array(state.X̄ ./ log10(ℯ))
            cb.hms[4][3] = Array(V_v)
            cb.lns[1][1] = cb.j_hist
            autolimits!(cb.axs[5])
            recordframe!(cb.video_stream)

            output_path = joinpath(cb.scenario.output_dir, @sprintf("step_%04d.jld2", cb.step))

            jldsave(output_path;
                    X=Array(state.X),
                    X̄=Array(state.X̄),
                    V=Array(cb.model.fields.V),
                    H=Array(cb.model.fields.H),
                    iter=state.iter,
                    j_hist=cb.j_hist,)

            cb.step += 1
        end
        return
    end

    md"""
    ⬅️ Show the contents of this cell to see the definition of `Callback` and the visualisation code.
    """
end

# ╔═╡ 0acabae5-199d-4447-9b02-11a86ea5e805
function run_inversion(scenario::InversionScenarioSnapshot)
    model = SnapshotSIA(scenario.input_file)

    model.scalars.ρgnA *= scenario.E

    V_obs = copy(model.fields.V)

    ωᵥ = inv(sum(V_obs .^ 2))

    objective = SnapshotObjective(ωᵥ, V_obs, γ_reg)

    # see cells below for details on how the callback is implemented
    callback = Callback(model, scenario, V_obs)
    options  = OptimisationOptions(; line_search, callback, maxiter)

	n = model.scalars.n
	
    # initial guess
	ρgnAs_init = RHOG^n * As_init * (L_REF^(n-1) * T_REF) |> NoUnits
    log_ρgnAs0 = CUDA.fill(log(ρgnAs_init), size(V_obs))

    # inversion
    optimise(model, objective, log_ρgnAs0, options)

    # show animation
    return callback.video_stream
end;

# ╔═╡ ddcfe1a4-5d0d-47d0-af19-ab3bfeb8206c
run_inversion(synthetic_scenario)

# ╔═╡ d514e2e7-9a11-4645-939a-d34d3d13d99f
run_inversion(aletsch_scenario)

# ╔═╡ a417d5ae-98f2-4cc4-bfe2-e7d95323916f
run_inversion(InversionScenarioSnapshot(; input_file="../datasets/aletsch_200m.jld2", output_dir="../output/snapshot_aletsch_200m", E=0.25))

# ╔═╡ 61a9f3ae-a6f6-4f12-a7d7-fdd8c7b6c29d
run_inversion(InversionScenarioSnapshot(; input_file="../datasets/aletsch_100m.jld2", output_dir="../output/snapshot_aletsch_100m", E=0.25))

# ╔═╡ 756ccb69-dd09-4f43-a8a5-fd7a9c2ec500
run_inversion(InversionScenarioSnapshot(; input_file="../datasets/aletsch_50m.jld2", output_dir="../output/snapshot_aletsch_50m", E=0.25))

# ╔═╡ Cell order:
# ╟─815419fa-b7cc-11ef-225d-1be9a77d974c
# ╟─48c28533-fbc6-40bd-a18f-35324b1ff32c
# ╟─9afa1d2e-e9fd-4d38-9ecb-1dfc0ec0d75e
# ╠═485febf2-6338-415d-8f4c-77568af5ab57
# ╟─bebfc476-c778-42bd-a807-59adfea64788
# ╟─c69c5ce6-a8af-49e4-8100-398d8df0b0b3
# ╠═a19c5990-1d62-4349-84c1-4cf2522b0eb4
# ╟─5206fc60-6bcd-444a-82bb-f8c2f336c393
# ╠═9f08540e-deac-4999-b6cc-c685d5ec679d
# ╟─c6f6e76c-9f06-463e-a7b4-0d2aee421275
# ╠═5594683b-17ed-48b6-9804-9286b0f7ea40
# ╟─78e44885-07e5-43e8-ba3b-b6c94e6e40d8
# ╠═494028ea-ad86-476c-b52a-d5c2102d55e9
# ╟─f13d57d1-a793-4566-b325-aa416a53e0ca
# ╠═0acabae5-199d-4447-9b02-11a86ea5e805
# ╟─c910ebda-7bb9-418f-a572-989ee04cb4f3
# ╟─16574a8b-5dc9-47ba-b924-5545f7d5e893
# ╠═2dfdbd36-ab24-4a1e-aa57-89c272bd2866
# ╟─0ed5551b-5584-4792-9c27-aacf0f8919f6
# ╠═ddcfe1a4-5d0d-47d0-af19-ab3bfeb8206c
# ╟─b93039b7-9413-4625-b84c-d17e3d003b15
# ╠═4f4bccc4-e7b6-45b8-8500-d985352f1cbb
# ╟─4e7b2d0f-9b3c-4df8-a4e5-3518dae0127a
# ╠═d514e2e7-9a11-4645-939a-d34d3d13d99f
# ╠═a417d5ae-98f2-4cc4-bfe2-e7d95323916f
# ╠═61a9f3ae-a6f6-4f12-a7d7-fdd8c7b6c29d
# ╠═756ccb69-dd09-4f43-a8a5-fd7a9c2ec500
# ╟─4e3d8f5c-974b-442a-9267-67ff53121872
# ╟─7dd6f707-c723-4666-977b-efec92b2a5f0
