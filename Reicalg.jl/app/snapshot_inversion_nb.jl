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

# ╔═╡ f3ba256d-7048-4776-a5ce-083ac1e875cb
E = 1.0;

# ╔═╡ 703d7708-4668-466b-b733-70da37f7feb6
β_reg = 1e-3;

# ╔═╡ 416951d7-4dae-4bda-8b4d-590f61e3200b
As_init = 1e-22;

# ╔═╡ a9bc7adb-bd57-45ab-8fe1-becdc967379f
model = SnapshotSIA("../../datasets/synthetic_setup.jld2");

# ╔═╡ 1991b5fb-994d-43ad-b81b-19730b5aad01
A_ini = model.scalars.A;

# ╔═╡ 524e2615-8f7d-4a06-bad0-74c473851c58
V_obs = copy(model.fields.V);

# ╔═╡ 0feaedc3-4d6b-4409-97a7-5510bce75cf9
ωᵥ = inv(sum(V_obs .^ 2));

# ╔═╡ 2fb25317-925a-4698-b77b-a3da69433cfd
objective = SnapshotObjective(ωᵥ, V_obs, β_reg);

# ╔═╡ 55b128b5-c1ed-4315-a68f-98fe32da2ce2
(; nx, ny, xc, yc) = model.numerics;

# ╔═╡ 84d5ce11-9d2a-4533-8c16-8d588698c87c
xc_km, yc_km = xc ./ 1e3, yc ./ 1e3;

# ╔═╡ 70056cde-df6f-4ca0-bd45-eb574acfa21f
line_search = BacktrackingLineSearch(; α_min=1e0, α_max=1e6);

# ╔═╡ 4ba99ac3-6e91-4d03-95a9-45507d0939a5
maxiter = 2000;

# ╔═╡ 70286306-748d-4362-9f64-bcd102392c3e
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
	
		    hms = (heatmap!(axs[1], xc_km, yc_km, Array(log10.(model.fields.As))),
		           heatmap!(axs[2], xc_km, yc_km, Array(log10.(model.fields.As))),
		           heatmap!(axs[3], xc_km, yc_km, Array(V_obs)),
		           heatmap!(axs[4], xc_km, yc_km, Array(model.fields.V)))
	
			hms[1].colormap = :roma
			hms[2].colormap = :roma
			hms[3].colormap = :turbo
			hms[4].colormap = :turbo
		
			hms[1].colorrange = (-24, -20)
			hms[2].colorrange = (-1e-8, 1e-8)
			hms[3].colorramge = (0, 1e-5)
			hms[4].colorramge = (0, 1e-5)
	
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
			cb.video_stream = VideoStream(cb.fig; framerate=20)
		end

		push!(cb.j_hist, Point2(state.iter, state.j_value))
	
		if state.iter % 10 == 0
			@info @sprintf("  iter = %-4d, J = %1.3e, ΔJ/J = %1.3e, ΔX/X = %1.3e, α = %1.3e\n", state.iter,
						state.j_value,
					    state.j_change,
				        state.x_change,
				        state.α)
	
			cb.hms[1][3] = Array(state.X .* log10(ℯ))
			cb.hms[2][3] = Array(state.X̄ ./ log10(ℯ))
			cb.hms[4][3] = Array(model.fields.V)
			cb.lns[1][1] = cb.j_hist
			autolimits!(cb.axs[5])
			recordframe!(cb.video_stream)
		end
		return
	end
end;

# ╔═╡ 588456ad-3c3d-4740-9863-8ec8df28627d
callback = Callback(model, V_obs);

# ╔═╡ 68c86cd0-17c3-43b3-8b02-dfff111ba720
options = OptimisationOptions(; line_search, callback, maxiter);

# ╔═╡ cdfca633-8f6b-42c9-a865-df36469c3d11
optimise(model, objective, CUDA.fill(log(As_init), nx-1, ny-1), options);

# ╔═╡ cd64b35a-d881-4ce3-a79a-cd4ca4e450fe
callback.video_stream

# ╔═╡ Cell order:
# ╟─40661bea-47ac-11ef-1a58-f5deede4bf68
# ╠═f3ba256d-7048-4776-a5ce-083ac1e875cb
# ╠═703d7708-4668-466b-b733-70da37f7feb6
# ╠═416951d7-4dae-4bda-8b4d-590f61e3200b
# ╠═a9bc7adb-bd57-45ab-8fe1-becdc967379f
# ╠═1991b5fb-994d-43ad-b81b-19730b5aad01
# ╠═524e2615-8f7d-4a06-bad0-74c473851c58
# ╠═0feaedc3-4d6b-4409-97a7-5510bce75cf9
# ╠═2fb25317-925a-4698-b77b-a3da69433cfd
# ╠═55b128b5-c1ed-4315-a68f-98fe32da2ce2
# ╠═84d5ce11-9d2a-4533-8c16-8d588698c87c
# ╠═70056cde-df6f-4ca0-bd45-eb574acfa21f
# ╠═4ba99ac3-6e91-4d03-95a9-45507d0939a5
# ╠═588456ad-3c3d-4740-9863-8ec8df28627d
# ╠═68c86cd0-17c3-43b3-8b02-dfff111ba720
# ╠═cdfca633-8f6b-42c9-a865-df36469c3d11
# ╟─cd64b35a-d881-4ce3-a79a-cd4ca4e450fe
# ╠═70286306-748d-4362-9f64-bcd102392c3e
