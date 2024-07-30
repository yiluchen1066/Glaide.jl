### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 40661bea-47ac-11ef-1a58-f5deede4bf68
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.activate(Base.current_project())
    Pkg.instantiate()

	using Reicalg
	using CairoMakie
	using Printf
	using JLD2
	using CUDA

	using PlutoUI
    TableOfContents()
end

# ╔═╡ add4b2c2-3c27-44be-a300-922b27606cf2
md"""
# Snapshot inversion

In this notebook, we will use inverse modelling routine implemented in Reicalg.jl to reconstruct spatially variable sliding parameter $A_\mathrm{s}$. The inverse modelling problem is defined as a minimisation problem with the following objective functional:

```math
J(A_\mathrm{s}) = \frac{\omega_V}{2}\sum_i\left(V_i(A_\mathrm{s}) - V^\mathrm{obs}_i\right)^2 + \frac{\beta}{2}\sum_i\left(\nabla A_{\mathrm{s}_i}\right)^2~,
```
where $\omega_V$ is the normalisation factor and $\beta$ is the Tikhonov regularisation parameter.

The normalisation constant is defined as the inverse of the $L_2$-norm of the observed velocity field:

```math
\omega_V = \left[\sum_i\left(V^\mathrm{obs}_i\right)^2\right]^{-1}
```
"""

# ╔═╡ 5dfb85d7-54ca-4322-ab0a-c64af54dc436
md"""
## Configuration

Define the type encapsulating the properties of the inversion that might be different between cases. In this study, these are the path to the input file and the enhancement factor $E$, which is needed to reduce the influence of the ice deformation in the Aletsch case:
"""

# ╔═╡ 4a40c5c0-4e5e-4c7c-a4c7-2d8e67f5e60e
Base.@kwdef struct InversionScenario
	input_file::String
	output_dir::String
	E::Float64 = 1.0
end;

# ╔═╡ 812269a3-dbc7-429c-9869-d96d15be34e9
md"""
!!! warning "Prerequisites"
	Before running this notebook, make sure input files exist on your filesystem. To generate the input files for the synthetic setup, run the notebook [`generate_synthetic_setup.jl`](./open?path=Reicalg.jl/app/generate_synthetic_setup.jl).
"""

# ╔═╡ 383e178a-1053-48a3-b6ab-ab60ce0df19b
md"""
Define the initial guess for the sliding parameter:
"""

# ╔═╡ 416951d7-4dae-4bda-8b4d-590f61e3200b
As_init = 1e-22;

# ╔═╡ 5885cdb4-aa0d-40b3-b01b-070f83aec0c7
md"""
Define the regularisation parameter:
"""

# ╔═╡ 703d7708-4668-466b-b733-70da37f7feb6
β_reg = 1e-6;

# ╔═╡ a9568707-3936-47f8-ac48-9a3ade80917f
md"""
Define the maximum number of iterations in the optimisation algorithm:
"""

# ╔═╡ 4ba99ac3-6e91-4d03-95a9-45507d0939a5
maxiter = 1000;

# ╔═╡ 725a0a00-e07d-44b1-9a4d-ed0feb16b9aa
md"""
Define the parameters of the line search. Here, we only configure the minimal and maximal step size. In Reicalg, the two-way backtracking line search based on Armijo-Goldstein condition is implemented. If in the line search loop the step size decreases below $\alpha_\min$, the optimisation stops with an error. If the step size increases above $\alpha_\max$, line search accepts $\alpha_\max$ as the step size. Increasing $\alpha_\max$ might improve convergence rate in some cases, but can also lead to instabilities and convergence issues in the forward solver.
"""

# ╔═╡ 70056cde-df6f-4ca0-bd45-eb574acfa21f
line_search = BacktrackingLineSearch(; α_min=1e2, α_max=1e6);

# ╔═╡ fa564469-61b7-475a-9276-d0bb8be3104c
md"""
## Inversion function

Here, we create a funciton that executes the inversion scenario:
"""

# ╔═╡ 97bcae72-f6a8-4cc7-99d4-e383069085e1
md"""
Some comments on the above code:

- We initialise the observed velocity field `V_obs` with `model.fields.V`. This is because the synthetic velocity is stored in the model's field `V`, as implemented in the notebook [`generate_synthetic_setup.jl`](./open?path=Reicalg.jl/app/generate_synthetic_setup.jl);
- We create the callback object `callback = Callback(model, obs)`. The definition of `Callback` is a bit convoluted, but in short, it handles the debug visualisation, keeping track of the convergence history, and saving the intermediate results of the optimisation. For implementation details, see the __Extras__ section at the end of the notebook.
"""

# ╔═╡ 2dcf4323-3bdf-4c26-ae38-30cd96734572
md"""
## Synthetic inversion

First, we define the inversion scenario for the synthetic glacier. Since the input data was generated using the standard physical parameters, we leave the enhancement factor to the default value $E = 1$:
"""

# ╔═╡ bcff8e89-e6c0-4a7a-a6b6-3617000bc721
synthetic_scenario = InversionScenario(; input_file="../datasets/synthetic_25m.jld2",
output_dir="../output/snapshot_synthetic_25m");

# ╔═╡ 13ee7e08-64cb-47cb-a3bc-614396cef169
md"""
Run the inversion:
"""

# ╔═╡ dd93c5e2-1278-4503-8654-89eac2cd518c
md"""
### Aletsch inversion

Then, we create the inversion scenarion to reconstruct sliding parameter at the base of the Aletsch glacier. There, using the standard parameters for the flow parameter $A = 2.5\times10^{-24}\,\text{Pa}\,\text{s}^{-3}$ result in the surface velocity values much higher than the observed ones even without any sliding. This is likely due to using the SIA model that doesn't account for longitudinal stresses and non-hydrostatic pressure variations. We correct this by introducing a flow enhancement factor $E = 0.25$:
"""

# ╔═╡ f95355fd-86f2-4320-bac3-a3301fb73394
aletsch_scenario = InversionScenario(; input_file="../datasets/aletsch_25m.jld2", output_dir="../output/snapshot_aletsch_25m", E=0.25);

# ╔═╡ 506d30a1-588e-496b-a4da-50888b21c393
md"""
Run the inversion:
"""

# ╔═╡ c38d1717-c4ee-49da-98b6-f31257a9a4b4
md"""
## Extras
"""

# ╔═╡ 70286306-748d-4362-9f64-bcd102392c3e
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

			cb.hms[1][3] = Array(state.X .* log10(ℯ))
			cb.hms[2][3] = Array(state.X̄ ./ log10(ℯ))
			cb.hms[4][3] = Array(cb.model.fields.V)
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

# ╔═╡ 700749e5-5028-4ec3-ab64-92f80a283745
function run_inversion(scenario::InversionScenario)
	model = SnapshotSIA(scenario.input_file)

	model.scalars.A *= scenario.E

	V_obs = copy(model.fields.V)

	ωᵥ = inv(sum(V_obs .^ 2))

	objective = SnapshotObjective(ωᵥ, V_obs, β_reg)

	# see cells below for details on how the callback is implemented
	callback = Callback(model, scenario, V_obs)
	options  = OptimisationOptions(; line_search, callback, maxiter)

	# initial guess
	logAs0 = CUDA.fill(log(As_init), size(V_obs))

	# inversion
	optimise(model, objective, logAs0, options)

	# show animation
	return callback.video_stream
end;

# ╔═╡ 776fccde-8268-4b4c-bb98-f73b0bc02eb4
# ╠═╡ show_logs = false
run_inversion(synthetic_scenario)

# ╔═╡ 6726a19f-99fa-4a05-a90b-b10079e152f9
# ╠═╡ show_logs = false
run_inversion(aletsch_scenario)

# ╔═╡ 45dacb6d-de79-476e-84fc-c394644c9e00
# ╠═╡ show_logs = false
run_inversion(InversionScenario(; input_file="../datasets/aletsch_200m.jld2", output_dir="../output/snapshot_aletsch_200m", E=0.25))

# ╔═╡ 2a6367e7-6521-4188-8a40-11748cf11ffc
# ╠═╡ show_logs = false
run_inversion(InversionScenario(; input_file="../datasets/aletsch_100m.jld2", output_dir="../output/snapshot_aletsch_100m", E=0.25))

# ╔═╡ 6c6d30ae-63b0-421a-8061-c4bb73d95f91
# ╠═╡ show_logs = false
run_inversion(InversionScenario(; input_file="../datasets/aletsch_50m.jld2", output_dir="../output/snapshot_aletsch_50m", E=0.25))

# ╔═╡ Cell order:
# ╟─40661bea-47ac-11ef-1a58-f5deede4bf68
# ╟─add4b2c2-3c27-44be-a300-922b27606cf2
# ╟─5dfb85d7-54ca-4322-ab0a-c64af54dc436
# ╠═4a40c5c0-4e5e-4c7c-a4c7-2d8e67f5e60e
# ╟─812269a3-dbc7-429c-9869-d96d15be34e9
# ╟─383e178a-1053-48a3-b6ab-ab60ce0df19b
# ╠═416951d7-4dae-4bda-8b4d-590f61e3200b
# ╟─5885cdb4-aa0d-40b3-b01b-070f83aec0c7
# ╠═703d7708-4668-466b-b733-70da37f7feb6
# ╟─a9568707-3936-47f8-ac48-9a3ade80917f
# ╠═4ba99ac3-6e91-4d03-95a9-45507d0939a5
# ╟─725a0a00-e07d-44b1-9a4d-ed0feb16b9aa
# ╠═70056cde-df6f-4ca0-bd45-eb574acfa21f
# ╟─fa564469-61b7-475a-9276-d0bb8be3104c
# ╠═700749e5-5028-4ec3-ab64-92f80a283745
# ╟─97bcae72-f6a8-4cc7-99d4-e383069085e1
# ╟─2dcf4323-3bdf-4c26-ae38-30cd96734572
# ╠═bcff8e89-e6c0-4a7a-a6b6-3617000bc721
# ╟─13ee7e08-64cb-47cb-a3bc-614396cef169
# ╠═776fccde-8268-4b4c-bb98-f73b0bc02eb4
# ╟─dd93c5e2-1278-4503-8654-89eac2cd518c
# ╠═f95355fd-86f2-4320-bac3-a3301fb73394
# ╟─506d30a1-588e-496b-a4da-50888b21c393
# ╠═6726a19f-99fa-4a05-a90b-b10079e152f9
# ╠═45dacb6d-de79-476e-84fc-c394644c9e00
# ╠═2a6367e7-6521-4188-8a40-11748cf11ffc
# ╠═6c6d30ae-63b0-421a-8061-c4bb73d95f91
# ╟─c38d1717-c4ee-49da-98b6-f31257a9a4b4
# ╟─70286306-748d-4362-9f64-bcd102392c3e
