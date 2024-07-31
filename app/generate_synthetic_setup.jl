### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 3b975e48-4794-11ef-3fc0-6deaeef86908
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(Base.current_project())
    Pkg.instantiate()

    using Rasters, ArchGDAL, NCDatasets, Extents
    using CairoMakie
    using DelimitedFiles
    using JLD2
    using Glaide

    using PlutoUI
    TableOfContents()
end

# ╔═╡ f0f3e0fa-d870-4ba8-84d8-bbe7f05035aa
md"""
# Generating the input for the synthetic glacier

In this notebook, we will generate synthetic data to test how well the inversion procedure described in the paper and implemented in Glaide.jl can reconstruct known distribution of the sliding coefficient. We will generate the "observed" ice thickness and velocity distributions by running the forward SIA model.

## Configuration

Define the path where the resulting input file will be saved, relative to the location of the notebook:
"""

# ╔═╡ 0969205b-2d11-4ba2-b43a-f10fcd0dd85d
datasets_dir = "../datasets"; mkpath(datasets_dir);

# ╔═╡ cef6480e-d715-42a8-93d7-fa676fad09c1
md"""
Define the extents of the computational domain in meters:
"""

# ╔═╡ c4ebd113-cc41-4f66-87db-2b10383c03b9
Lx, Ly = 20e3, 20e3;

# ╔═╡ 3fa6bc48-1d64-4358-bc6b-086adab81805
md"""
Define the resolution of the computational grid in meters, note that the smaller the number is, the longer the simulation will take:
"""

# ╔═╡ 62605691-13c4-4ed1-ba76-933f2aa8b9ba
resolution = 25.0;

# ╔═╡ ea677e88-1ab2-47b4-89dc-bb9049658931
md"""
!!! note "Changing the resolution"
    The forward solver is based on the pseudo-transient method. While very efficient on GPUs, for strongly non-linear problems like SIA it might require resolution-dependent tuning of the iteration parameters. For much lower resolutions than 50m it is recommended reducing the damping parameters `dmp1` and `dmp2` to values < 0.1. These values can be passed to the `TimeDependentNumerics` constructor as keyword arguments.
"""

# ╔═╡ e05e846a-58c2-40c7-aa0b-c25209296136
md"""
## Sliding parameter distribution

We will create a synthetic distribution of the sliding parameter:

```math
\log_{10} A_\mathrm{s} = \log_{10} A_{\mathrm{s}_0} +
                        A_{\mathrm{s}_\mathrm{a}}
                            \cos\left(\omega\frac{X}{L_x}\right)
                            \sin\left(\omega\frac{Y}{L_y}\right)~,
```
where $A_{\mathrm{s}_0}$ is the background value, $A_{\mathrm{s}_\mathrm{a}}$ is the perturbations amplitude, and $\omega$ is the perturbations wavelength:
"""

# ╔═╡ 226790d3-127a-410c-9a7a-09ba01d85cf8
begin
    As_0 = 1e-22
    As_a = 2.0
    ω    = 3π
end;

# ╔═╡ f50ce1aa-27f1-4fb3-8a27-e6550ddec3f4
md"""
## Bed elevation

The synthetic mountain elevation profile is generated by adding two gaussian-shaped functions, similar to the test scenario from [Višnjević et al. (2018)](https://doi.org/10.1017/jog.2018.82):

```math
B = B_0 + \frac{B_\mathrm{A}}{2}
    \left\{ \exp\left[-\left(\frac{X}{W_1}\right)^2 -
                       \left(\frac{Y}{W_2}\right)^2\right] +
            \exp\left[-\left(\frac{X}{W_2}\right)^2 -
                       \left(\frac{Y}{W_1}\right)^2\right]
    \right\}~,
```

where $B_0$ is the background elevation, $B_\mathrm{A}$ is the mountain height, $W_1$ and $W_2$ are characteristic widths:
"""

# ╔═╡ f564c689-a6af-4941-b250-ac83727907f7
begin
    B_0 = 1000.0
    B_a = 3000.0
    W_1 = 1e4
    W_2 = 3e3
end;

# ╔═╡ a309ad80-5f10-49d8-bbc9-de40f85048fa
md"""
## Surface mass balance

In Glaide.jl, the SMB model is based on the simple altitude-dependent parametrisation:

```math
\dot{b}(z) = \min\left\{\beta (z - \mathrm{ELA}),\ \dot{b}_\mathrm{max}\right\}~,
```

where $z$ is the altitude, $\beta$ is the rate of change of mass balance, $\mathrm{ELA}$ is the equilibrium line altitude where $\dot{b}=0$, and $\dot{b}_\mathrm{max}$ is the maximum accumulation rate.
"""

# ╔═╡ f9dba7bb-a2a9-4b30-9fd2-5db2af34ef24
begin
    β     = 0.01 / SECONDS_IN_YEAR
    b_max = 2.5  / SECONDS_IN_YEAR
    ela   = 1800.0
end;

# ╔═╡ bcbf9797-6a71-4001-8ce9-ba045648308a
md"""
## Preprocessing

In this section, we define the variables describing the computational grid, i.e. number of grid cells in x and y direction, the grid spacing (equal to user-specified resolution), and coordinates of grid cells and nodes:
"""

# ╔═╡ 463ddc92-690f-49d2-8be8-45d9e2ddb84f
begin
    dx, dy = resolution, resolution
    nx, ny = ceil(Int, Lx / dx), ceil(Int, Ly / dy)

    # if the resolution is fixed, domain extents need to be corrected
    lx, ly = nx * dx, ny * dy

    # grid cell center coordinates
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)

    # grid node coordinates
    xv, yv = av1(xc), av1(yc)
end;

# ╔═╡ dab8104a-9d68-4526-84f2-18fb4ce3d870
md"""
## Generating the synthetic input data

We generate the inputs for Glaide.jl with the following processing seqence:

1. Create the time-dependent forward model with the default parameters and the timestep $\Delta t = \infty$. This corresponds to running the simulation to a steady state;
2. Setup the synthetic mountain by setting the bed elevation in the model;
3. Set the sliding parameter $A_\mathrm{s}$ to the background value;
4. Set the mass balance mask to $1$ to allow accumulation everywhere in the computational domain;
5. Solve the model by integrating the SIA equations to a steady state;
6. Make a step change in the distribution of the sliding parameter $A_\mathrm{s}$ by introducing the spatial variation around the background value;
7. Make a step change in the surface mass balance by increasing the ELA by 20\% compared to its initial value;
8. Set the time step $\Delta t = 15~\mathrm{y}$;
9. Solve the model again, integrating the SIA equations for a period of 15 y.
"""

# ╔═╡ 052648a6-6197-4cd5-b7bb-bbfaa2958103
model, V_old = let
    # default scalar parameters for a steady state (dt = ∞)
    scalars = TimeDependentScalars(; lx, ly, dt=Inf, β, b_max, ela)

    # default solver parameters
    numerics = TimeDependentNumerics(xc, yc)

    model = TimeDependentSIA(scalars, numerics)

    # set the bed elevation
    copy!(model.fields.B, @. B_0 + 0.5 * B_a * (exp(-(xc / W_1)^2 - (yc' / W_2)^2) +
                                                exp(-(xc / W_2)^2 - (yc' / W_1)^2)));

    # set As to the background value
    fill!(model.fields.As, As_0)

    # accumulation allowed everywhere
    fill!(model.fields.mb_mask, 1.0)

    # solve the model to steady state
    solve!(model)

    # save geometry and surface velocity
    model.fields.H_old .= model.fields.H
    V_old 				= Array(model.fields.V)

    # sliding parameter perturbation
    As_synthetic = @. 10^(log10(As_0) + As_a * cos(ω * xv  / lx) * sin(ω * yv' / ly))

    copy!(model.fields.As, As_synthetic)

    # step change in mass balance (ELA +20%)
    model.scalars.ela = ela * 1.2

    # finite time step (15y)
    model.scalars.dt  = 15 * SECONDS_IN_YEAR

    # solve again
    solve!(model)

    model, V_old
end;

# ╔═╡ 2a0cc261-ea6d-4f64-a4bb-c2b980b9396f
md"""
## Saving data to disk

After generating the synthetic input data, we copy the arrays from GPU to host memory:
"""

# ╔═╡ f2a4541d-b524-41c1-afcd-6a233853961d
begin
    B       = Array(model.fields.B)
    H       = Array(model.fields.H)
    H_old   = Array(model.fields.H_old)
    V       = Array(model.fields.V)
    As      = Array(model.fields.As)
    mb_mask = Array(model.fields.mb_mask)
end;

# ╔═╡ 6b358a59-26c4-434a-9a05-cc14c35fe2bb
md"""
Then, we save the fields, scalars, and numerical parameters as a JLD2 file in a format recognisable by Glaide.jl:
"""

# ╔═╡ a1cdae82-f33c-48a8-a10f-02bb55e2d93f
let
    fields = (; B, H, H_old, V, V_old, As, mb_mask)

    scalars = let
        (; lx, ly, β, b_max, ela, dt, npow, A, ρgn) = model.scalars
    end

    numerics = (; nx, ny, dx, dy, xc, yc)

    output_path = joinpath(datasets_dir, "synthetic_$(Int(resolution))m.jld2")

    jldsave(output_path; fields, scalars, numerics)
end

# ╔═╡ 263ab006-a16a-446f-827b-e59bf9ee5622
md"""
## Visualisation

Finally, we visualise the input data:
"""

# ╔═╡ 6365ffea-405f-4941-9894-a566a1d0aa08
with_theme(theme_latexfonts()) do
    ice_mask_old = H_old .== 0
    ice_mask     = H     .== 0

    ice_mask_old_v = H_old[1:end-1, 1:end-1] .== 0 .||
                     H_old[2:end  , 1:end-1] .== 0 .||
                     H_old[1:end-1, 2:end  ] .== 0 .||
                     H_old[2:end  , 2:end  ] .== 0


    ice_mask_v = H[1:end-1, 1:end-1] .== 0 .||
                 H[2:end  , 1:end-1] .== 0 .||
                 H[1:end-1, 2:end  ] .== 0 .||
                 H[2:end  , 2:end  ] .== 0

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

    fig = Figure(; size=(800, 450), fontsize=16)

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
    xc_km, yc_km = xc / 1e3, yc / 1e3

    hms = (heatmap!(axs[1], xc_km, yc_km, B),
           heatmap!(axs[2], xc_km, yc_km, log10.(As_v)),
           heatmap!(axs[3], xc_km, yc_km, H_old_v),
           heatmap!(axs[4], xc_km, yc_km, H_v),
           heatmap!(axs[5], xc_km, yc_km, V_old_v),
           heatmap!(axs[6], xc_km, yc_km, V_v))

    # enable interpolation for smoother picture
    foreach(hms) do h
        h.interpolate = true
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

    fig
end

# ╔═╡ Cell order:
# ╟─3b975e48-4794-11ef-3fc0-6deaeef86908
# ╟─f0f3e0fa-d870-4ba8-84d8-bbe7f05035aa
# ╠═0969205b-2d11-4ba2-b43a-f10fcd0dd85d
# ╟─cef6480e-d715-42a8-93d7-fa676fad09c1
# ╠═c4ebd113-cc41-4f66-87db-2b10383c03b9
# ╟─3fa6bc48-1d64-4358-bc6b-086adab81805
# ╠═62605691-13c4-4ed1-ba76-933f2aa8b9ba
# ╟─ea677e88-1ab2-47b4-89dc-bb9049658931
# ╟─e05e846a-58c2-40c7-aa0b-c25209296136
# ╠═226790d3-127a-410c-9a7a-09ba01d85cf8
# ╟─f50ce1aa-27f1-4fb3-8a27-e6550ddec3f4
# ╠═f564c689-a6af-4941-b250-ac83727907f7
# ╟─a309ad80-5f10-49d8-bbc9-de40f85048fa
# ╠═f9dba7bb-a2a9-4b30-9fd2-5db2af34ef24
# ╟─bcbf9797-6a71-4001-8ce9-ba045648308a
# ╠═463ddc92-690f-49d2-8be8-45d9e2ddb84f
# ╟─dab8104a-9d68-4526-84f2-18fb4ce3d870
# ╠═052648a6-6197-4cd5-b7bb-bbfaa2958103
# ╟─2a0cc261-ea6d-4f64-a4bb-c2b980b9396f
# ╠═f2a4541d-b524-41c1-afcd-6a233853961d
# ╟─6b358a59-26c4-434a-9a05-cc14c35fe2bb
# ╠═a1cdae82-f33c-48a8-a10f-02bb55e2d93f
# ╟─263ab006-a16a-446f-827b-e59bf9ee5622
# ╟─6365ffea-405f-4941-9894-a566a1d0aa08
