### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ be4c2496-b7be-11ef-37b6-9d1804244625
# ╠═╡ show_logs = false
begin
	import Pkg
    Pkg.activate(Base.current_project())
    Pkg.instantiate()

	using Glaide, JLD2, CairoMakie

    using PlutoUI; TableOfContents()
end

# ╔═╡ 9a39fbd3-752a-4e8c-91c2-e51333015325
md"""
# Generating the input for the synthetic glacier

In this notebook, we will generate synthetic data to test how well the inversion procedure described in the paper and implemented in Glaide.jl can reconstruct known distribution of the sliding coefficient. We will generate the "observed" ice thickness and velocity distributions by running the forward SIA model.

## Configuration

Define the path where the resulting input file will be saved, relative to the location of the notebook:
"""

# ╔═╡ 3a5bf3aa-12c4-40d6-9235-ff530037e6da
datasets_dir = "../datasets"; mkpath(datasets_dir);

# ╔═╡ 66eb63b7-41f5-481f-88b7-8c71e9f148b2
md"""
Define the extents of the computational domain in meters:
"""

# ╔═╡ 0d5f918b-383e-464f-a965-df84a73251ba
Lx, Ly = 20e3, 20e3;

# ╔═╡ 0bd8bf4f-983c-4659-813e-750cbec30e25
md"""
Define the resolution of the computational grid in meters, note that the smaller the number is, the longer the simulation will take:
"""

# ╔═╡ c0ed33ff-6279-443f-8300-b3ed3f95576e
resolution = 50.0;

# ╔═╡ bd264944-6101-4935-9023-608f44ce267a
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

# ╔═╡ b226f26c-2929-4a27-a465-62ab966806d4
begin
    As_0 = 1e-22
    As_a = 2.0
    ω    = 3π
end;

# ╔═╡ 80895b27-5f32-46ff-bb4e-fcc5436172c8
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

# ╔═╡ d2b31e07-282d-470e-b96f-676e09a0b70d
begin
    B_0 = 1000.0
    B_a = 3000.0
    W_1 = 1e4
    W_2 = 3e3
end;

# ╔═╡ 9012c142-0570-4794-8ba8-2601ad2bd876
md"""
## Surface mass balance

In Glaide.jl, the SMB model is based on the simple altitude-dependent parametrisation:

```math
\dot{b}(z) = \min\left\{\beta (z - \mathrm{ELA}),\ \dot{b}_\mathrm{max}\right\}~,
```

where $z$ is the altitude, $\beta$ is the rate of change of mass balance, $\mathrm{ELA}$ is the equilibrium line altitude where $\dot{b}=0$, and $\dot{b}_\mathrm{max}$ is the maximum accumulation rate.
"""

# ╔═╡ 0a6ac215-6a0f-47ce-b2f6-93ba0302b1bc
begin
    b      = 0.01 / SECONDS_IN_YEAR
    mb_max = 2.5  / SECONDS_IN_YEAR
    ela    = 1800.0
end;

# ╔═╡ 0b42a811-ac66-426a-9054-1ed18ce45707
md"""
## Preprocessing

In this section, we define the variables describing the computational grid, i.e. number of grid cells in x and y direction, the grid spacing (equal to user-specified resolution), and coordinates of grid cells and nodes:
"""

# ╔═╡ 473a8bf6-0b40-4d6f-a418-9245f575c9d7
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

# ╔═╡ 064941af-0da5-49fc-b896-a64ad124d627
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

# ╔═╡ f9ab79f0-e826-402d-a61e-2e96b07be806
model, V_old = let
    # default scalar parameters for a steady state (dt = ∞)
    scalars = TimeDependentScalars(; lx, ly, dt=Inf, b, mb_max, ela)

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

    # solve the model to the steady state
    solve!(model)

    # save geometry and surface velocity
    model.fields.H_old .= model.fields.H
    V_old 				= Array(model.fields.V)

    # sliding parameter perturbation
    As_synthetic = @. 10^(log10(As_0) + As_a * cos(ω * xc  / lx) * sin(ω * yc' / ly))

    copy!(model.fields.As, As_synthetic)

    # step change in mass balance (ELA +20%)
    model.scalars.ela = ela * 1.2

    # finite time step (15y)
    model.scalars.dt  = 15 * SECONDS_IN_YEAR

    # solve again
    solve!(model)

    model, V_old
end;

# ╔═╡ c7fd5e5a-4a35-40e3-b98b-fb4487aea699
md"""
## Saving data to disk

After generating the synthetic input data, we copy the arrays from GPU to host memory:
"""

# ╔═╡ f4e01694-5005-4c86-8cc0-5473903ab82d
begin
    B       = Array(model.fields.B)
    H       = Array(model.fields.H)
    H_old   = Array(model.fields.H_old)
    V       = Array(model.fields.V)
    As      = Array(model.fields.As)
    mb_mask = Array(model.fields.mb_mask)
end;

# ╔═╡ d5ea40bf-c1e2-4689-a980-d0dc1cac2203
md"""
Then, we save the fields, scalars, and numerical parameters as a JLD2 file in a format recognisable by Glaide.jl:
"""

# ╔═╡ f4b05b52-1ea1-48be-93f1-cce7f476319f
let
    fields = (; B, H, H_old, V, V_old, As, mb_mask)

    scalars = let
        (; lx, ly, b, mb_max, ela, dt, n, A, ρgn) = model.scalars
    end

    numerics = (; nx, ny, dx, dy, xc, yc)

    output_path = joinpath(datasets_dir, "synthetic_$(Int(resolution))m.jld2")

    jldsave(output_path; fields, scalars, numerics)
end

# ╔═╡ 27360bfc-9862-478e-803c-6e104f77d318
md"""
## Visualisation

Finally, we visualise the input data:
"""

# ╔═╡ f12ad11c-538a-4236-860e-e7d0b3528918
with_theme(theme_latexfonts()) do
    ice_mask_old = H_old .== 0
    ice_mask     = H     .== 0

    H_old_v = copy(H_old)
    H_v     = copy(H)
    As_v    = copy(As)
    # convert to m/a
    V_old_v = copy(V_old) .* SECONDS_IN_YEAR
    V_v     = copy(V)     .* SECONDS_IN_YEAR

    # mask out ice-free pixels
    H_old_v[ice_mask_old]   .= NaN
    V_old_v[ice_mask_old] .= NaN

    H_v[ice_mask]  .= NaN
    As_v[ice_mask] .= NaN
    V_v[ice_mask]  .= NaN

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

	[contour!(axs[i], xc_km, yc_km, H_old; levels=0:0, linewidth=0.5, color=:black) for i=(1,3,5)]
    [contour!(axs[i], xc_km, yc_km, H; levels=0:0, linewidth=0.5, color=:black) for i=(2,4,6)]

    # enable interpolation for smoother picture
    foreach(hms) do h
        h.interpolate = true
    end

    hms[1].colormap = :terrain
    hms[2].colormap = Reverse(:roma)
    hms[3].colormap = :matter
    hms[4].colormap = :matter
    hms[5].colormap = Reverse(:ice)
    hms[6].colormap = Reverse(:ice)

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
# ╟─be4c2496-b7be-11ef-37b6-9d1804244625
# ╟─9a39fbd3-752a-4e8c-91c2-e51333015325
# ╠═3a5bf3aa-12c4-40d6-9235-ff530037e6da
# ╟─66eb63b7-41f5-481f-88b7-8c71e9f148b2
# ╠═0d5f918b-383e-464f-a965-df84a73251ba
# ╟─0bd8bf4f-983c-4659-813e-750cbec30e25
# ╠═c0ed33ff-6279-443f-8300-b3ed3f95576e
# ╟─bd264944-6101-4935-9023-608f44ce267a
# ╠═b226f26c-2929-4a27-a465-62ab966806d4
# ╟─80895b27-5f32-46ff-bb4e-fcc5436172c8
# ╠═d2b31e07-282d-470e-b96f-676e09a0b70d
# ╟─9012c142-0570-4794-8ba8-2601ad2bd876
# ╠═0a6ac215-6a0f-47ce-b2f6-93ba0302b1bc
# ╟─0b42a811-ac66-426a-9054-1ed18ce45707
# ╠═473a8bf6-0b40-4d6f-a418-9245f575c9d7
# ╟─064941af-0da5-49fc-b896-a64ad124d627
# ╠═f9ab79f0-e826-402d-a61e-2e96b07be806
# ╟─c7fd5e5a-4a35-40e3-b98b-fb4487aea699
# ╠═f4e01694-5005-4c86-8cc0-5473903ab82d
# ╟─d5ea40bf-c1e2-4689-a980-d0dc1cac2203
# ╠═f4b05b52-1ea1-48be-93f1-cce7f476319f
# ╟─27360bfc-9862-478e-803c-6e104f77d318
# ╟─f12ad11c-538a-4236-860e-e7d0b3528918
