### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 7f0bf442-4e59-11ef-129d-811ff0175c06
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

    CairoMakie.activate!(type="svg", pt_per_unit=1)

    using PlutoUI
    TableOfContents()
end

# ╔═╡ 3df59bed-a8e0-4670-83c5-56d5e87b8f55
md"""
# Running the forward model

In this notebook, we run the SIA model using the reconstructed distribution of the sliding parameter $A_s$. We do this to compare the snapshot and time-dependent inversion products.
"""

# ╔═╡ b06089a6-4433-4dcb-bb24-45fbfba0bd65
input_file = "../datasets/aletsch_25m.jld2";

# ╔═╡ 9c08b16c-c809-4919-9afb-04387eb9abd7
inversion_files = ["../output/snapshot_aletsch_25m/step_0100.jld2",
                   "../output/time_dependent_aletsch_25m/step_0100.jld2"];

# ╔═╡ efefd1dd-7439-436a-ad5d-9054307cb368
E = 0.25;

# ╔═╡ b70f7a61-c958-47a2-925a-75efe46f8aaa
nts = Int[1,2,11,50];

# ╔═╡ 9ebf02c2-7234-4f09-8108-b671c99263cc
models = let
    mkpath("../output/forward_aletsch/")
    models = Matrix{Any}(undef,length(inversion_files),length(nts))
    for (flag, inversion_file) in enumerate(inversion_files)
        for (i, nt) in enumerate(nts)
            model = TimeDependentSIA(input_file; report=true)
            model.scalars.A *= E

            X = load(inversion_file, "X")
            copy!(model.fields.As, exp.(X))

            for it in 1:nt
                solve!(model)
                model.fields.H_old .= model.fields.H
            end

            # save the results
            V = Array(model.fields.V)
            H = Array(model.fields.H)

            if flag==1
                jldsave("../output/forward_aletsch/forward_aletsch_25m_snap_$(nt)yrs.jld2"; V, H)
            else
                jldsave("../output/forward_aletsch/forward_aletsch_25m_time_dep_$(nt)yrs.jld2"; V, H)
            end
            models[flag,i] = model
        end
    end
    models
end;

# ╔═╡ a2a8b755-7466-4fe5-aa7a-f2043d96d04b
with_theme(theme_latexfonts()) do
    model = models[1,1]

    fig = Figure(size=(700, 280))

    axs = (Axis(fig[1,1][1,1]; aspect=DataAspect()),
           Axis(fig[1,2][1,1]; aspect=DataAspect()),
           Axis(fig[1,3][1,1]; aspect=DataAspect()))

    for ax in axs
        limits!(ax, -7, 7, -10, 10)
    end

    hideydecorations!(axs[2])
    hideydecorations!(axs[3])

    axs[1].title = L"A_s\ \mathrm{[Pa^{-3}\,m\,s^{-1}]}"
    axs[2].title = L"V\ \mathrm{[m\ a^{-1}]}"
    axs[3].title = L"H\ \mathrm{[m]}"

    axs[1].xlabel = L"x\ \mathrm{[km]}"
    axs[2].xlabel = L"x\ \mathrm{[km]}"
    axs[3].xlabel = L"x\ \mathrm{[km]}"

    axs[1].ylabel = L"y\ \mathrm{[km]}"

    x, y = model.numerics.xc ./ 1e3, model.numerics.yc ./ 1e3

    H  = Array(model.fields.H)
    HH = copy(H)
    V  = Array(model.fields.V)
    As = Array(model.fields.As)

    im = H      .< 1.0
    iv = av4(H) .< 1.0

    H[im]  .= NaN
    V[iv]  .= NaN
    As[iv] .= NaN

    hms = (heatmap!(axs[1], x, y, As; colorscale=log10),
           heatmap!(axs[2], x, y, V .* SECONDS_IN_YEAR),
           heatmap!(axs[3], x, y, H))

    [contour!(axs[i], x, y, HH; levels=1:1, linewidth=0.5, color=:black) for i=1:3]

    hms[1].colormap = Reverse(:roma)
    hms[2].colormap = :matter
    hms[3].colormap = Reverse(:ice)

    hms[1].colorrange = (1e-24, 1e-20)
    hms[2].colorrange = (0, 300)
    hms[3].colorrange = (0, 900)

    for hm in hms
        hm.interpolate = true
        hm.rasterize   = 4
    end

    cbs = (Colorbar(fig[1,1][1,2], hms[1]),
           Colorbar(fig[1,2][1,2], hms[2]),
           Colorbar(fig[1,3][1,2], hms[3]))

    fig
end

# ╔═╡ Cell order:
# ╟─7f0bf442-4e59-11ef-129d-811ff0175c06
# ╟─3df59bed-a8e0-4670-83c5-56d5e87b8f55
# ╠═b06089a6-4433-4dcb-bb24-45fbfba0bd65
# ╠═9c08b16c-c809-4919-9afb-04387eb9abd7
# ╠═efefd1dd-7439-436a-ad5d-9054307cb368
# ╠═b70f7a61-c958-47a2-925a-75efe46f8aaa
# ╠═9ebf02c2-7234-4f09-8108-b671c99263cc
# ╠═a2a8b755-7466-4fe5-aa7a-f2043d96d04b
