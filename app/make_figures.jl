### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ cc163f70-4dac-11ef-3e26-7366ab19d20e
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(Base.current_project())
    Pkg.instantiate()

    using Glaide
    using CairoMakie
    using Printf
    using JLD2

    CairoMakie.activate!(type="svg", pt_per_unit=2)

    using PlutoUI
    TableOfContents()
end

# ╔═╡ 63512265-bbc4-49ba-b460-2e69c98f1228
begin
    two_column_cm   = 17.8
    one_column_cm   = 8.6
    font_size_pt    = 10
    label_font_size = 12
end;

# ╔═╡ 5d6f0d91-a968-413f-90a5-87e5d982daaa
pt_in_cm = 28.3465;

# ╔═╡ a37259a8-6b04-47db-b1d1-841c1021e9c8
two_column_pt = two_column_cm * pt_in_cm;

# ╔═╡ 53e7d652-864e-4afd-9192-9c714504821c
one_column_pt = one_column_cm * pt_in_cm;

# ╔═╡ 049a38f9-abe4-466d-9d99-a42a88461c4f
dpi = 600;

# ╔═╡ 8182d000-8783-4b5f-99c3-d09104414e2b
cm_in_inch = 2.54;

# ╔═╡ 31c83c57-2fc7-41ae-8169-d18432e94297
px_per_unit = two_column_cm / cm_in_inch * dpi / two_column_pt;

# ╔═╡ 00042d22-7004-4c84-ac57-3826420d2d57
pt_per_unit = 1;

# ╔═╡ a1682eff-3d75-4483-91ed-2b4490807e33
makie_theme = merge(
    theme_latexfonts(),
    Theme(fontsize=font_size_pt,
          Axis=(spinewidth=0.5,
                xtickwidth=0.5,
                ytickwidth=0.5,
                xticksize=3,
                yticksize=3,),
          Colorbar=(spinewidth=0.5, tickwidth=0.5, ticksize=3, size=7),
          Label=(fontsize = label_font_size, font=:bold),
          Legend=(rowgap=-8, labelsize=8, framewidth=0.25, padding=(2, 2, 2, 2), margin = (4, 4, 4, 4)))
);

# ╔═╡ c097b1fc-28d7-49c7-8734-9b63b94ba8cd
md"""
# Input data
"""

# ╔═╡ b92a481f-4977-4f51-a68f-f210864388b0
md"""
## Synthetic setup
"""

# ╔═╡ 9902e887-1496-4686-9c27-04b73d751ef6
with_theme(makie_theme) do
    vis_path = "../datasets/synthetic_25m.jld2"
    
    fields, scalars, numerics = load(vis_path, "fields", "scalars", "numerics")

    H     = fields.H
    H_old = fields.H_old
    As    = fields.As
    B     = fields.B

    V_old = fields.V_old
    V     = fields.V
    
    ice_mask_old = H_old .< 1.0
    ice_mask     = H     .< 1.0
    
    ice_mask_old_v = av4(H_old) .< 1.0

    
    ice_mask_v = av4(H) .< 1.0

    H_old_v = copy(H_old)
    H_v     = copy(H)
    As_v    = copy(As)
    # convert to m/a
    V_old_v = copy(V_old) .* SECONDS_IN_YEAR
    V_v     = copy(V)     .* SECONDS_IN_YEAR

    # mask out ice-free pixels
    H_old_v[ice_mask_old]   .= NaN
    V_old_v[ice_mask_old_v] .= NaN

    H_v[ice_mask]    .= NaN
    As_v[ice_mask_v] .= NaN
    V_v[ice_mask_v]  .= NaN

    fig = Figure(; size=(two_column_pt, 280))

    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect()),
           Axis(fig[2, 1][1, 1]; aspect=DataAspect()),
           Axis(fig[1, 2][1, 1]; aspect=DataAspect()),
           Axis(fig[2, 2][1, 1]; aspect=DataAspect()),
           Axis(fig[1, 3][1, 1]; aspect=DataAspect()),
           Axis(fig[2, 3][1, 1]; aspect=DataAspect()))

    hidexdecorations!.((axs[1], axs[3], axs[5]))
    hideydecorations!.((axs[3], axs[4], axs[5], axs[6]))

    for ax in axs
        ax.xgridvisible=true
        ax.ygridvisible=true
        limits!(ax, -10, 10, -10, 10)
    end

    axs[1].ylabel = L"y~\mathrm{[km]}"
    axs[2].ylabel = L"y~\mathrm{[km]}"

    axs[2].xlabel = L"x~\mathrm{[km]}"
    axs[4].xlabel = L"x~\mathrm{[km]}"
    axs[6].xlabel = L"x~\mathrm{[km]}"

    axs[1].title = L"B~\mathrm{[m]}"
    axs[2].title = L"\log_{10}(As)"
    axs[3].title = L"H_\mathrm{old}~\mathrm{[m]}"
    axs[4].title = L"H~\mathrm{[m]}"
    axs[5].title = L"V_\mathrm{old}~\mathrm{[m/a]}"
    axs[6].title = L"V~\mathrm{[m/a]}"
    
    # convert to km for plotting
    xc_km, yc_km = numerics.xc / 1e3, numerics.yc / 1e3

    hms = (heatmap!(axs[1], xc_km, yc_km, B),
           heatmap!(axs[2], xc_km, yc_km, log10.(As_v)),
           heatmap!(axs[3], xc_km, yc_km, H_old_v),
           heatmap!(axs[4], xc_km, yc_km, H_v),
           heatmap!(axs[5], xc_km, yc_km, V_old_v),
           heatmap!(axs[6], xc_km, yc_km, V_v))

    contour!(axs[1], xc_km, yc_km, H; levels=1:1, linewidth=0.5, color=:black)
    
    # enable interpolation for smoother picture
    foreach(hms) do h
        h.interpolate = true
        h.rasterize   = px_per_unit
    end
    
    hms[1].colormap = :terrain
    hms[2].colormap = Reverse(:roma)
    hms[3].colormap = :vik
    hms[4].colormap = :vik
    hms[5].colormap = :turbo
    hms[6].colormap = :turbo

    hms[1].colorrange = (1000, 4000)
    hms[2].colorrange = (-24, -20)
    hms[3].colorrange = (0, 150)
    hms[4].colorrange = (0, 150)
    hms[5].colorrange = (0, 300)
    hms[6].colorrange = (0, 300)

    cbs = (Colorbar(fig[1, 1][1, 2], hms[1]),
           Colorbar(fig[2, 1][1, 2], hms[2]),
           Colorbar(fig[1, 2][1, 2], hms[3]),
           Colorbar(fig[2, 2][1, 2], hms[4]),
           Colorbar(fig[1, 3][1, 2], hms[5]),
           Colorbar(fig[2, 3][1, 2], hms[6]))

    for (label, idx) in zip('a':'f',
                            ((1,1), (1,2), (1,3), (2,1), (2,2), (2,3)))
        Label(fig[idx..., TopLeft()], string(label); padding=(0, 10, 0, 0))
    end

    for l in fig.layout.content[1:6]
        colgap!(l.content, 1, Fixed(10))
    end

    save("../figures/synthetic_setup.pdf", fig; pt_per_unit, px_per_unit)
    save("../figures/synthetic_setup.png", fig; pt_per_unit, px_per_unit)

    fig
end

# ╔═╡ a785cc48-5386-49b5-901c-780568d7301b
md"""
## Aletsch setup
"""

# ╔═╡ b0844542-8f1f-4691-9777-23ce34675e19
with_theme(makie_theme) do
    vis_path = "../datasets/aletsch_25m.jld2"
    
    fields, scalars, numerics, eb, mb = load(vis_path, "fields",
                                                        "scalars",
                                                        "numerics",
                                                        "eb", 
                                                        "mb")

    fig = Figure(; size=(two_column_pt, 310))

    # convert to km
    x_km = numerics.xc ./ 1e3
    y_km = numerics.yc ./ 1e3
    
    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect()),
           Axis(fig[1, 2][1, 1]; aspect=DataAspect()),
           Axis(fig[2, 1][1, 1]; aspect=DataAspect()),
           Axis(fig[2, 2][1, 1]; aspect=DataAspect()),
           Axis(fig[1, 3][1, 1]; aspect=DataAspect()),
           Axis(fig[2, 3]      ; title=L"\text{SMB model}"))

    for ax in axs
        limits!(ax, -7, 7, -10, 10)
    end

    colgap!(fig.layout, 1, Relative(0.08))

    axs[1].title = L"B~\mathrm{[m]}"
    axs[2].title = L"V~\mathrm{[m\,a^{-1}]}"
    axs[3].title = L"H_{2016}~\mathrm{[m]}"
    axs[4].title = L"H_{2017} - H_{2016}~\mathrm{[m]}"
    axs[5].title = L"\text{Mass balance mask}"

    axs[3].xlabel = L"x~\mathrm{[km]}"
    axs[4].xlabel = L"x~\mathrm{[km]}"

    axs[1].ylabel = L"y~\mathrm{[km]}"
    axs[3].ylabel = L"y~\mathrm{[km]}"

    axs[6].xlabel = L"z~\mathrm{[km]}"
    axs[6].ylabel = L"\dot{b}~\mathrm{[m/a]}"

    hidexdecorations!.((axs[1], axs[2], axs[5]))
    hideydecorations!.((axs[2], axs[4], axs[5]))

    axs[2].xgridvisible=true
    axs[2].ygridvisible=true
    axs[4].ygridvisible=true

    # cut off everything where the ice thickness is less than 1m
    ice_mask   = fields.H 	   .< 1.0
    ice_mask_v = av4(fields.H) .< 1.0

    fields.V .*= SECONDS_IN_YEAR

    H_c = copy(fields.H)

    fields.V[ice_mask_v]   .= NaN
    fields.H_old[ice_mask] .= NaN
    fields.H[ice_mask]     .= NaN

    hms = (heatmap!(axs[1], x_km, y_km, fields.B),
           heatmap!(axs[2], x_km, y_km, fields.V),
           heatmap!(axs[3], x_km, y_km, fields.H_old),
           heatmap!(axs[4], x_km, y_km, fields.H .- fields.H_old ),
           heatmap!(axs[5], x_km, y_km, fields.mb_mask))

    contour!(axs[1], x_km, y_km, H_c; levels=1:1, linewidth=0.5, color=:black)

    foreach(hms) do h
        h.interpolate = true
        h.rasterize   = px_per_unit
    end
    
    hms[1].colormap = :terrain
    hms[2].colormap = :turbo
    hms[3].colormap = :vik
    hms[4].colormap = :vik
    hms[5].colormap = :grays

    hms[1].colorrange = (1000, 4000)
    hms[2].colorrange = (0, 300)
    hms[3].colorrange = (0, 900)
    hms[4].colorrange = (-10, 0)
    
    z = LinRange(1900, 4150, 1000)
    b = @. min(scalars.β * (z - scalars.ela), scalars.b_max)

    # observational mass balance data
    scatter!(axs[6], eb ./ 1e3,
                     mb .* SECONDS_IN_YEAR; markersize=3,
                                            color=:red,
                                            label="data")

    # parametrised model
    lin = lines!(axs[6], z ./ 1e3, b .* SECONDS_IN_YEAR; linewidth=1, label="model")
    
    sc = scatter!(axs[6], scalars.ela / 1e3, 0; strokecolor = :black,
                                                strokewidth = 0.5,
                                                color       = :transparent,
                                                markersize  = 3,
                                                marker      = :diamond,
                                                label       = "ELA")
    
    lg = Legend(fig[2,3], axs[6])
    lg.halign = :right
    lg.valign = :bottom
    lg.tellwidth=false
    lg.height=45

    cb = (Colorbar(fig[1, 1][1, 2], hms[1]),
          Colorbar(fig[1, 2][1, 2], hms[2]),
          Colorbar(fig[2, 1][1, 2], hms[3]),
          Colorbar(fig[2, 2][1, 2], hms[4]),
          Colorbar(fig[1, 3][1, 2], hms[5]))

    for (label, idx) in zip('a':'f',
                            ((1,1), (1,2), (1,3), (2,1), (2,2), (2,3)))
        Label(fig[idx..., TopLeft()], string(label); padding  = (0, 10, 0, 0))
    end

    for l in fig.layout.content[1:5]
        colgap!(l.content, 1, Fixed(10))
    end

    save("../figures/aletsch_setup.pdf", fig; pt_per_unit, px_per_unit)
    save("../figures/aletsch_setup.png", fig; pt_per_unit, px_per_unit)

    fig
end

# ╔═╡ e3c678e8-d43a-4a3c-a26e-ccbc15d89501
md"""
# Results
"""

# ╔═╡ 2f8c5e86-c904-4710-96dc-5375864d3166
last_step = 100;

# ╔═╡ a35b1524-48d3-404e-b727-2ebef5c8b2c2
md"""
## Different weights in time-dependent inversions
"""

# ╔═╡ a2be1579-5cf4-4763-8529-abf3beb08b76
synthetic_output_dirs = ("../output/time_dependent_synthetic_25m_1_1",
                         "../output/time_dependent_synthetic_25m_1_0",
                         "../output/time_dependent_synthetic_25m_0_1");

# ╔═╡ 84737c35-b4f4-4097-a4a4-9861b2a109f4
synthetic_input_file = "../datasets/synthetic_25m.jld2";

# ╔═╡ d6defbeb-74f2-4139-8918-1a9336b736f6
with_theme(makie_theme) do
    fields, scalars, numerics = load(synthetic_input_file, "fields",
                                                           "scalars",
                                                           "numerics")

    x, y = numerics.xc ./ 1e3, numerics.yc ./ 1e3

    ice_mask   = fields.H .< 1.0;
    ice_mask_v = av4(fields.H) .< 1.0

    max_V = maximum(fields.V);
    max_H = maximum(fields.H);

    V_obs = copy(fields.V)
    V_obs[ice_mask_v] .= NaN

    H_obs = copy(fields.H)
    H_obs[ice_mask] .= NaN

    As_synth = copy(fields.As)
    As_synth[ice_mask_v] .= NaN

    fig = Figure(; size=(two_column_pt, 500))

    axs = [Axis(fig[row, col][1, 1]) for row in 1:4, col in 1:3]

    axs[1, 1].title = L"A^\mathrm{synth}_s\ \mathrm{[Pa^{-3}\,m\,s^{-1}]}"
    axs[1, 2].title = L"V_\mathrm{obs}\ \mathrm{[m\,a^{-1}]}"
    axs[1, 3].title = L"H_\mathrm{obs}\ \mathrm{[m]}"

    for ax in axs[2:end, 1]
        ax.title = L"A_s\ \mathrm{[Pa^{-3}\,m\,s^{-1}]}"
    end

    for ax in axs[2:end, 2]
        ax.title = L"\Delta V"
    end

    for ax in axs[2:end, 3]
        ax.title = L"\Delta H"
    end

    for ax in axs[:, 2:end]
        hideydecorations!(ax)
    end

    for ax in axs[1:end-1, :]
        hidexdecorations!(ax)
    end

    for ax in axs[:, 1]
        ax.ylabel = L"y\ [\mathrm{km}]"
    end

    for ax in axs[end, :]
        ax.xlabel = L"x\ [\mathrm{km}]"
    end

    for ax in axs
        limits!(ax, -10, 10, -10, 10)
        ax.aspect 	    = DataAspect()
        ax.xgridvisible = true
        ax.ygridvisible = true
    end

    arrays = (As_synth, V_obs .* SECONDS_IN_YEAR, H_obs)

    hms = [heatmap!(axs[row, col], x, y, data) for row in 1:4,
                                         (col, data) in enumerate(arrays)]

    for hm in hms
        hm.interpolate = true
        hm.rasterize   = px_per_unit
    end

    for hm in hms[:, 1]
        hm.colorscale = log10
        hm.colormap   = Reverse(:roma)
        hm.colorrange = (1e-24, 1e-20)
    end

    hms[1, 2].colormap   = :turbo
    hms[1, 2].colorrange = (0, 300)

    hms[1, 3].colormap   = :vik
    hms[1, 3].colorrange = (0, 120)

    for hm in hms[2:end, 2]
        hm.colorscale = log10
        hm.colormap   = :turbo
        hm.colorrange = (1e-4, 1e0)
    end

    for hm in hms[2:end, 3]
        hm.colorscale = log10
        hm.colormap   = :vik
        hm.colorrange = (1e-4, 1e0)
    end

    cbs = [Colorbar(fig[row, col][1, 2], hms[row, col]) for row in 1:4, col in 1:3]
    
    for (label, idx) in zip('a':'l', [(row, col) for col in 1:3, row in 1:4])
        Label(fig[idx..., TopLeft()], string(label))
    end

    for l in fig.layout.content[1:12]
        colgap!(l.content, 1, Fixed(0))
    end

    for (i, dir) in enumerate(synthetic_output_dirs)
        output_path = joinpath(dir, @sprintf("step_%04d.jld2", last_step))
        X, V, H 	= load(output_path, "X", "V", "H")

        As = exp.(X)

        ice_mask   = H .< 1.0;
        ice_mask_v = av4(H) .< 1.0

        As[ice_mask_v] .= NaN
        V[ice_mask_v]  .= NaN
        H[ice_mask]    .= NaN

        hms[i+1, 1][3] = As
        hms[i+1, 2][3] = abs.(V .- V_obs) ./ max_V
        hms[i+1, 3][3] = abs.(H .- H_obs) ./ max_H
    end

    save("../figures/synthetic_td_inversion.pdf", fig; pt_per_unit, px_per_unit)
    save("../figures/synthetic_td_inversion.png", fig; pt_per_unit, px_per_unit)
    
    fig
end

# ╔═╡ 5dc29e4b-f572-4b46-8ccf-f65cd94b0433
md"""
## Mesh convergence of inversions for Aletsch
"""

# ╔═╡ db859254-b076-4b88-bae8-cffc82f863f9
aletsch_resolutions = (200, 100, 50, 25);

# ╔═╡ eb90e0e4-95b3-4e1a-8239-fa891517a1c4
with_theme(makie_theme) do
    fig = Figure(size=(two_column_pt, 330))
    axs = [Axis(fig[row, col]) for row in 1:2, col in eachindex(aletsch_resolutions)]

    axs[1, 1].ylabel = L"y\,[\mathrm{km}]"
    axs[2, 1].ylabel = L"y\,[\mathrm{km}]"

    for ax in axs[1, :]
        hidexdecorations!(ax)
    end
    
    for ax in axs[:, 2:end]
        hideydecorations!(ax)
    end

    for (i, ax) in enumerate(axs[1, :])
        ax.title  = L"%$(aletsch_resolutions[i])\,\mathrm{m}"
    end

    for ax in axs[end, :]
        ax.xlabel = L"x\,[\mathrm{km}]"
    end

    for ax in axs
        limits!(ax, -7, 7, -10, 10)

        ax.aspect = DataAspect()
        ax.xgridvisible = true
        ax.ygridvisible = true
    end

    hms = Heatmap[]

    for (i, res) in enumerate(aletsch_resolutions)
        input_file = "../datasets/aletsch_$(res)m.jld2"
        numerics   = load(input_file, "numerics")
        
        x, y = numerics.xc ./ 1e3, numerics.yc ./ 1e3

        out_files = (
            @sprintf("../output/time_dependent_aletsch_%dm/step_%04d.jld2", res, last_step),
            @sprintf("../output/snapshot_aletsch_%dm/step_%04d.jld2"      , res, last_step),
        )

        for (row, out_file) in enumerate(out_files)
            X, H = load(out_file, "X", "H")

            As = exp.(X)

            ice_mask = H .< 1.0
            ice_mask_v = av4(H) .< 1.0

            As[ice_mask_v] .= NaN

            hm = heatmap!(axs[row, i], x, y, As)

            hm.colorscale = log10
            hm.colormap   = Reverse(:roma)
            hm.colorrange = (1e-24, 1e-20)
            hm.rasterize  = px_per_unit

            push!(hms, hm)
        end
    end

    for (label, idx) in zip('a':'h', [(row, col) for row in 1:2, col in 1:4])
        Label(fig[idx..., TopLeft()], string(label); padding=(0, 5, 0, 0))
    end

    cb = Colorbar(fig[:, length(aletsch_resolutions) + 1], hms[end])
    cb.label = L"A_s\ \mathrm{[Pa^{-3}\,m\,s^{-1}]}"

    colgap!(fig.layout, Fixed(10))
    
    save("../figures/aletsch_td_inversion_resolution.pdf", fig; pt_per_unit, px_per_unit)
    save("../figures/aletsch_td_inversion_resolution.png", fig; pt_per_unit, px_per_unit)

    fig
end

# ╔═╡ 3d2bea48-6f61-433d-97a8-8ac76137ff75
md"""
## Inversion summary for Aletsch inversions
"""

# ╔═╡ 62c22e3d-fbc1-4925-ad36-6225dbbe78e5
md"""
### Velocity maps
"""

# ╔═╡ 9515f730-94a4-4262-8bc4-f3511aa797ad
with_theme(makie_theme) do
    fig = Figure(size=(two_column_pt, 185))
    
    axs = [Axis(fig[1, col]) for col in 1:4]

    axs[1].title = L"\mathrm{Observed}"
    axs[2].title = L"\mathrm{Snapshot}"
    axs[3].title = L"\mathrm{Snapshot+}"
    axs[4].title = L"\mathrm{Time-dependent}"

    axs[1].ylabel = L"y\ \mathrm{[km]}"
    
    for ax in axs[2:end]
        hideydecorations!(ax)
    end
    
    for ax in axs
        limits!(ax, -7, 7, -10, 10)
        ax.aspect = DataAspect()
        ax.ygridvisible = true
        ax.xlabel = L"x\ \mathrm{[km]}"
    end

    fields, numerics = load("../datasets/aletsch_25m.jld2", "fields", "numerics")
    
    x, y = numerics.xc ./ 1e3, numerics.yc ./ 1e3
    
    H_obs  = fields.H
    V_obs  = fields.V

    im = H_obs      .< 1
    iv = av4(H_obs) .< 1

    H_obs[im] .= NaN
    V_obs[iv] .= NaN
    
    V_s1, H_s1 = load("../output/snapshot_aletsch_25m/step_0100.jld2", "V", "H")
    H_s1[im] .= NaN
    V_s1[iv] .= NaN
    
    V_s2, H_s2 = load("../output/snapshot_forward_aletsch_25m.jld2", "V" ,"H")
    H_s2[im] .= NaN
    V_s2[iv] .= NaN
    
    V_td, H_td = load("../output/time_dependent_aletsch_25m/step_0100.jld2", "V", "H")
    H_td[im] .= NaN
    V_td[iv] .= NaN

    arrays = (V_obs, V_s1, V_s2, V_td)

    hms = [heatmap!(ax, x, y, data .* SECONDS_IN_YEAR) for (ax, data) in zip(axs, arrays)]

    for hm in hms
        hm.colormap    = :turbo
        hm.colorrange  = (0, 300)
        hm.interpolate = true
        hm.rasterise   = px_per_unit
    end

    cb = Colorbar(fig[1,end+1], hms[end])
    cb.label = L"V\ \mathrm{[m\,a^{-1}]}"

    for (label, idx) in zip('a':'d', [(1, col) for col in 1:4])
        Label(fig[idx..., TopLeft()], string(label); padding=(0, 5, 0, 0))
    end

    colgap!(fig.layout, Fixed(10))
    
    save("../figures/aletsch_inversion_velocities.pdf", fig; pt_per_unit, px_per_unit)
    save("../figures/aletsch_inversion_velocities.png", fig; pt_per_unit, px_per_unit)
    
    fig
end

# ╔═╡ 4917b271-6dff-4f80-a51b-83eead3e6719
md"""
### Geometry changes
"""

# ╔═╡ c38a1355-61eb-4b44-bca5-5c7c93bbf5a8
with_theme(makie_theme) do
    fig = Figure(size=(one_column_pt, 260))
    
    axs = [Axis(fig[row, col]) for row in 1:2, col in 1:2]

    axs[1,1].title = L"\Delta V"
    axs[1,2].title = L"\Delta V"
    axs[2,1].title = L"\Delta H"
    axs[2,2].title = L"\Delta H"
    
    for ax in axs[1, :]
        hidexdecorations!(ax)
    end
    
    for ax in axs[:, 2:end]
        hideydecorations!(ax)
    end
    
    for ax in axs
        limits!(ax, -7, 7, -10, 10)
        ax.aspect = DataAspect()
        ax.xgridvisible = true
        ax.ygridvisible = true
    end

    for ax in axs[end, :]
        ax.xlabel = L"x\ \mathrm{[km]}"
    end

    for ax in axs[:, 1]
        ax.ylabel = L"y\ \mathrm{[km]}"
    end

    fields, numerics = load("../datasets/aletsch_25m.jld2", "fields", "numerics")
    
    x, y = numerics.xc ./ 1e3, numerics.yc ./ 1e3
    
    H_obs  = fields.H
    V_obs  = fields.V

    H_max = maximum(H_obs)
    V_max = maximum(abs.(V_obs))

    im = H_obs      .< 1
    iv = av4(H_obs) .< 1

    H_obs[im] .= NaN
    V_obs[iv] .= NaN
    
    V_s, H_s = load("../output/snapshot_forward_aletsch_25m.jld2", "V" ,"H")
    H_s[im] .= NaN
    V_s[iv] .= NaN
    
    V_td, H_td = load("../output/time_dependent_aletsch_25m/step_0100.jld2", "V", "H")
    H_td[im] .= NaN
    V_td[iv] .= NaN

    V_s[V_s .== V_obs] .= NaN
    V_td[V_td .== V_obs] .= NaN
    
    H_s[H_s .== H_obs] .= NaN
    H_td[H_td .== H_obs] .= NaN

    hms = (heatmap!(axs[1,1], x, y, abs.(V_s  .- V_obs)./V_max),
           heatmap!(axs[1,2], x, y, abs.(V_td .- V_obs)./V_max),
           heatmap!(axs[2,1], x, y, abs.(H_s  .- H_obs)./H_max),
           heatmap!(axs[2,2], x, y, abs.(H_td .- H_obs)./H_max))

    for hm in hms
        hm.interpolate = true
        hm.rasterise   = px_per_unit
        hm.colorscale  = log10
    end

    hms[1].colorrange = (1e-2, 1e0)
    hms[2].colorrange = (1e-2, 1e0)
    hms[3].colorrange = (1e-3, 1e-1)
    hms[4].colorrange = (1e-3, 1e-1)
    
    hms[1].colormap    = :turbo
    hms[2].colormap    = :turbo
    hms[3].colormap    = :vik
    hms[4].colormap    = :vik

    cbs = (Colorbar(fig[1,3], hms[2]),
           Colorbar(fig[2,3], hms[4]))

    for (label, idx) in zip('a':'d', [(col, row) for row in 1:2, col in 1:2])
        Label(fig[idx..., TopLeft()], string(label); padding=(0, 0, 0, 0))
    end

    colgap!(fig.layout, Fixed(10))
    
    save("../figures/aletsch_inversion_deltas.pdf", fig; pt_per_unit, px_per_unit)
    save("../figures/aletsch_inversion_deltas.png", fig; pt_per_unit, px_per_unit)
    
    fig
end

# ╔═╡ Cell order:
# ╟─cc163f70-4dac-11ef-3e26-7366ab19d20e
# ╠═63512265-bbc4-49ba-b460-2e69c98f1228
# ╠═5d6f0d91-a968-413f-90a5-87e5d982daaa
# ╠═a37259a8-6b04-47db-b1d1-841c1021e9c8
# ╠═53e7d652-864e-4afd-9192-9c714504821c
# ╠═049a38f9-abe4-466d-9d99-a42a88461c4f
# ╠═8182d000-8783-4b5f-99c3-d09104414e2b
# ╠═31c83c57-2fc7-41ae-8169-d18432e94297
# ╠═00042d22-7004-4c84-ac57-3826420d2d57
# ╠═a1682eff-3d75-4483-91ed-2b4490807e33
# ╟─c097b1fc-28d7-49c7-8734-9b63b94ba8cd
# ╟─b92a481f-4977-4f51-a68f-f210864388b0
# ╟─9902e887-1496-4686-9c27-04b73d751ef6
# ╟─a785cc48-5386-49b5-901c-780568d7301b
# ╟─b0844542-8f1f-4691-9777-23ce34675e19
# ╟─e3c678e8-d43a-4a3c-a26e-ccbc15d89501
# ╠═2f8c5e86-c904-4710-96dc-5375864d3166
# ╟─a35b1524-48d3-404e-b727-2ebef5c8b2c2
# ╠═a2be1579-5cf4-4763-8529-abf3beb08b76
# ╠═84737c35-b4f4-4097-a4a4-9861b2a109f4
# ╟─d6defbeb-74f2-4139-8918-1a9336b736f6
# ╟─5dc29e4b-f572-4b46-8ccf-f65cd94b0433
# ╠═db859254-b076-4b88-bae8-cffc82f863f9
# ╠═eb90e0e4-95b3-4e1a-8239-fa891517a1c4
# ╟─3d2bea48-6f61-433d-97a8-8ac76137ff75
# ╟─62c22e3d-fbc1-4925-ad36-6225dbbe78e5
# ╟─9515f730-94a4-4262-8bc4-f3511aa797ad
# ╟─4917b271-6dff-4f80-a51b-83eead3e6719
# ╠═c38a1355-61eb-4b44-bca5-5c7c93bbf5a8
