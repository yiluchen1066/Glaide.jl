### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 7f0bf442-4e59-11ef-129d-811ff0175c06
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(Base.current_project())
    Pkg.instantiate()

    using Glaide, CairoMakie, Printf, JLD2, CUDA, Unitful, Logging

    CairoMakie.activate!(; type="svg", pt_per_unit=1)

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
inversion_files = ("../output/snapshot_aletsch_25m/step_0100.jld2",
                   "../output/time_dependent_aletsch_25m/step_0100.jld2");

# ╔═╡ 590ab964-3074-4167-8ec1-77aca3586471
out_prefixes = ("../output/forward_aletsch/forward_aletsch_25m_snap",
                "../output/forward_aletsch/forward_aletsch_25m_time_dep");

# ╔═╡ efefd1dd-7439-436a-ad5d-9054307cb368
E = 0.25;

# ╔═╡ b70f7a61-c958-47a2-925a-75efe46f8aaa
nts = Int[1, 2, 11, 50];

# ╔═╡ 9ebf02c2-7234-4f09-8108-b671c99263cc
models = let
    mkpath("../output/forward_aletsch/")
    models = Matrix{Any}(undef, length(inversion_files), length(nts))
    for (isim, inversion_file) in enumerate(inversion_files)
        @info "solve for" isim
        for (i, nt) in enumerate(nts)
            @info "solve for" nt

            reg = 5e-8u"s^-1" * T_REF |> NoUnits
            model = TimeDependentSIA(input_file; forward_overrides=(; reg, dmptol=0.0, checkf=1.0, α=0.5))
            model.scalars.ρgnA *= E

            X = load(inversion_file, "X")
            copy!(model.fields.ρgnAs, exp.(X))

            for it in 1:nt
                model.fields.H_old .= model.fields.H
                with_logger(ConsoleLogger(Debug)) do
                    solve!(model)
                end
            end

            # save the results
            V = Array(model.fields.V)
            H = Array(model.fields.H)

            out_path = "$(out_prefixes[isim])_$(nt)yrs.jld2"
            jldsave(out_path; V, H)

            models[isim, i] = model
        end
    end
    models
end;

# ╔═╡ a2a8b755-7466-4fe5-aa7a-f2043d96d04b
with_theme(theme_latexfonts()) do
    model = models[2, 4]

    fig = Figure(; size=(700, 280))

    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect()),
           Axis(fig[1, 2][1, 1]; aspect=DataAspect()),
           Axis(fig[1, 3][1, 1]; aspect=DataAspect()))

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

    (; nx, ny) = model.numerics
    (; lx, ly) = model.scalars

    # preprocessing
    dx, dy = lx / nx, ly / ny
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)

    # convert to km for plotting
    xc_km = xc .* ustrip(u"km", L_REF)
    yc_km = yc .* ustrip(u"km", L_REF)

    H  = Array(model.fields.H) .* ustrip(u"m", L_REF)
    V  = Array(model.fields.V) .* ustrip(u"m/yr", L_REF / T_REF)
    As = Array(model.fields.ρgnAs) .* ustrip.(u"Pa^-3*s^-1*m",
    (L_REF^(-2) * T_REF^(-1) / RHOG^GLEN_N))

    H_c = copy(H)

    im = H .< 1.0

    H[im]  .= NaN
    V[im]  .= NaN
    As[im] .= NaN

    hms = (heatmap!(axs[1], xc_km, yc_km, As; colorscale=log10),
           heatmap!(axs[2], xc_km, yc_km, V),
           heatmap!(axs[3], xc_km, yc_km, H))

    [contour!(axs[i], xc_km, yc_km, H_c;
              levels    = 1:1,
              linewidth = 0.5,
              color     = :black) for i in 1:3]

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

    cbs = (Colorbar(fig[1, 1][1, 2], hms[1]),
           Colorbar(fig[1, 2][1, 2], hms[2]),
           Colorbar(fig[1, 3][1, 2], hms[3]))

    fig
end

# ╔═╡ Cell order:
# ╟─7f0bf442-4e59-11ef-129d-811ff0175c06
# ╟─3df59bed-a8e0-4670-83c5-56d5e87b8f55
# ╠═b06089a6-4433-4dcb-bb24-45fbfba0bd65
# ╠═9c08b16c-c809-4919-9afb-04387eb9abd7
# ╠═590ab964-3074-4167-8ec1-77aca3586471
# ╠═efefd1dd-7439-436a-ad5d-9054307cb368
# ╠═b70f7a61-c958-47a2-925a-75efe46f8aaa
# ╠═9ebf02c2-7234-4f09-8108-b671c99263cc
# ╠═a2a8b755-7466-4fe5-aa7a-f2043d96d04b
