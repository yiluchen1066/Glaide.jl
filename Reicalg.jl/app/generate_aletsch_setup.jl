### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ d80af3c8-36ec-4d09-9b83-46b45bb25374
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(Base.current_project())
    Pkg.instantiate()

    using Rasters, ArchGDAL, NCDatasets, Extents
    using CairoMakie
    using DelimitedFiles
    using JLD2
    using Reicalg

    using PlutoUI
    TableOfContents()
end

# ╔═╡ 764a33ef-e288-45ec-960c-caf2eb56e3d3
md"""
# Generating the input data for the Aletsch glacier

In this notebook, we will prepare the inputs for the Reicalg.jl model.

We will perform the following steps:

1. Download the datasets containing the elevation, velocity, and mass balance for the Aletsch glacier in years 2016--2017;
2. Crop and resample the raster data to match the specified extent and resolution
3. Create the best fit curve for the measured mass balance;
4. Create the mass balance mask to remove ice accumulation in the areas where there is no ice by the end of the modelled period.
"""

# ╔═╡ 8bc9c1af-6f5e-40f7-9afb-a9134c56412a
md"""
## Configuration

In this section we define all the parameters that can be tuned in the setup, such as filesystem paths and data extents.

First, define the path where the datasets will be stored. The path is relative to the location of the notebook:
"""

# ╔═╡ 0f9575cb-b4b6-4d3a-97d4-dc5db7c23c12
datasets_dir = "../datasets";

# ╔═╡ 72db902c-10f3-4446-99d5-4eb0f810662b
md"""
Define the list of resolutions in meters for which the input files should be generated:
"""

# ╔═╡ b6ccaa3e-d948-4485-9d4a-b18fce33cb1a
resolutions = [25, 50, 100, 200];

# ╔═╡ a5871e66-3688-499d-b251-99b0f23e6726
md"""
Define the extents covering the Aletsch glacier in the Swiss coordinate system LV95:
"""

# ╔═╡ 6f1088a6-93b3-44e2-9a50-611660edaa84
glacier_extent = Extent(; X=(2.637e6, 2.651e6), Y=(1.1385e6, 1.158e6));

# ╔═╡ d4ef10ab-fde9-43cc-96fc-6f31785ab85b
md"""
!!! note "What if I want to change the extents?"
	By changing the extents, you can generate input files for other Swiss glaciers than Aletsch. However, only the bedrock and velocity datasets covers entire Swiss Alps. You will need to provide elevation data and mass balance data separately.
"""

# ╔═╡ ee1916bf-d3b4-41b8-b507-57b57a324bce
md"""
Small isolated ice patches might hinder convergence of the numerical model, so we remove them. We define the threshold area (in $m^2$) below which the connected ice-covered area is treated as small ice patch:
"""

# ╔═╡ 055bd14d-a02f-4610-8212-3ea52ce4d7c0
small_patch_area = 200.0^2;

# ╔═╡ 375feb17-9d84-42c8-b5b0-c7da73a17462
md"""
The source data might be noisy, and we apply a low-pass filter to the data to remove this noise. Define the (resolution-independent) amount of smoothing:
"""

# ╔═╡ 8d1acf5a-4ada-43ed-a081-c7a4766c0219
smooth_amount = 1e3;

# ╔═╡ 5b0af720-f416-438d-ab8d-42c2a4c4ca65
md"""
## Prerequisites

Bed and some surface elevation data can be downloaded automatically, but other data needs to be downloaded manually and saved in the correct places in the filesystem to reproduce the results. The source files will be places into the `datasets/_sources` directory:
"""

# ╔═╡ 37042783-bafe-444f-85ff-bc9f17d4ff53
sources_dir = joinpath(datasets_dir, "_sources"); mkpath(sources_dir);

# ╔═╡ 4af27194-4ebe-41a9-a716-142b3f4817e0
 Markdown.parse("""
Unfortunately, it is impossible to automatically download the dataset for ice velocities from [Rabatel et al. (2023)](https://doi.org/10.3390/data8040066) so it needs to be downloaded manually using [this link](https://entrepot.recherche.data.gouv.fr/file.xhtml?persistentId=doi:10.57745/VJYARH&version=1.1) 
and saved to the directory `"$sources_dir"`. The file path should be the same as this variable:
""")

# ╔═╡ 24930db6-f893-4cc6-ad30-58e8bac0ee1c
velocity_path = joinpath(sources_dir, "ALPES_wFLAG_wKT_ANNUALv2016-2021.nc");

# ╔═╡ b0969b25-551f-4e72-b08c-63881e05f359
md"""
The elevation and mass balance data are distributed as the datasets accompanying the paper, they need to be downloaded manually and also stored at the sources directory:
"""

# ╔═╡ 77e4a6bc-8502-4328-a08f-c28796b3d55d
begin
    surface_2009_path = joinpath(sources_dir, "aletsch2009.asc")
    surface_2017_path = joinpath(sources_dir, "aletsch2017.asc")
    mass_balance_path = joinpath(sources_dir, "aletsch_fix.dat")
end;

# ╔═╡ 20b667e7-2adf-428a-b7b2-fb684fb401b1
md"""
## Processing the raster data

### Downloading the elevation data
"""

# ╔═╡ f370b133-c07d-431b-a7ba-2252d16ded53
md"""
First, we define the URLs for downloading bed and surface elevation from [Grab et al. (2020)](https://doi.org/10.1017/jog.2021.55):
"""

# ╔═╡ 9f5a6312-04c6-4881-81e0-4003f865828d
begin
    bed_url = "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/434697/07_GlacierBed_SwissAlps.zip"
    surface_url = "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/434697/08_SurfaceElevation_SwissAlps.zip"
end;

# ╔═╡ 0faf7da5-464b-4943-821b-5259714d8f87
md"""
Then we define which files we need from the datasets, and download these files using the `download_raster` function provided by Reicalg.jl. If the files already exist in the filesystem, they won't be downloaded.
"""

# ╔═╡ 6c93121b-90e8-4a3b-adcd-d4ed118466f6
begin
    # paths to the target GeoTIFF files in the ZIP archives:
    bed_zip_path     = "07_GlacierBed_SwissAlps/GlacierBed.tif"
    surface_zip_path = "08_SurfaceElevation_SwissAlps/SwissALTI3D_r2019.tif"

    # paths to the GeoTIFF files in the filesystem
    bed_tif_path     = joinpath(sources_dir, bed_zip_path)
    surface_tif_path = joinpath(sources_dir, surface_zip_path)

    # download datasets from ETH research collection
    download_raster(bed_url, bed_zip_path, bed_tif_path)
    download_raster(surface_url, surface_zip_path, surface_tif_path)
end;

# ╔═╡ b55c88fd-ef10-42a4-92ad-83ca671bd96a
md"""
### Processing the elevation data
"""

# ╔═╡ 4b6fa9b4-4528-4e28-aaff-e13825a26037
md"""
Bed elevation data from [Grab et al. (2020)](https://doi.org/10.1017/jog.2021.55) is only defined at ice-covered regions. In our model, we need the bedrock to be defined everywhere in the computational domain. To achieve this, we combine the bedrock dataset with the surface elevation model at the ice-free regions.

We crop the bed and the surface data to match the provided extent. Then we replace the missing data points in the bedrock dataset with the corresponding values in the surface elevation dataset from [swissALTI3D](https://www.swisstopo.admin.ch/en/height-model-swissalti3d). Finally, we resample the bedrock raster to target spatial resolution using cubic spline interpolation:
"""

# ╔═╡ 9c0af9c4-cb6a-46e8-b222-9e4a75eeb4c7
function create_bedrock(bed_path, surface_path, extent, resolution)
    bedrock = crop(Raster(bed_path); to=extent)
    surface = crop(Raster(surface_path); to=extent)
    
    # create mosaic to replace missing points with surface elevation from ALTI3D
    bedrock = mosaic(first, bedrock, surface)
    bedrock = resample(bedrock; to     = extent,
                                res    = resolution,
                                method = :cubicspline)
	return bedrock
end;

# ╔═╡ 0b99796a-6edb-4a53-a875-bd5d329643d7
Markdown.parse("""
We define a helper function to create ice thickness data raster from known surface and bed elevation. In this function, we perform the following steps:

1. Resample the surface data to match the extent and resolution of the bed elevation model using cubic spline interpolation;
2. Clamp the values to prevent negative ice thickness;
3. Replace missing values with zeros, as the numerical model expects zero ice thickness in ice-free regions;
4. Remove the small patches of ice by setting the ice thickness of all connected regions covering area smaller then `small_patch_area = $small_patch_area m²`.
""")

# ╔═╡ aca29b01-114a-492b-9598-84be41452183
function create_ice_thickness(surface, bedrock, resolution)
    # resample surface raster to match the bedrock extent and resolution
    surface = resample(surface; to=bedrock, method=:cubicspline)

    # clamp to prevent negative values
    thickness = max.(surface .- bedrock, 0.0)

    # where elevation data is missing, we assume that there is no ice: thickness is 0
    thickness = replace_missing(thickness, 0.0)

	# number of cells covering the area
	min_length = ceil(Int, small_patch_area / resolution^2)
	
    # remove small ice patches
    remove_components!(thickness; min_length)

    return thickness
end;

# ╔═╡ 1262ffa4-1233-451a-82b0-f4c666e36bac
md"""
### Processing the velocity data

In this section, we load and process the velocity data from the dataset from [Rabatel et al. (2023)](https://doi.org/10.3390/data8040066). The processing includes the following steps:

1. Load the data, georeferenced using the WGS 84 corrdinate system;
2. Flip the data array for consistency with the elevation data;
3. Replace missing values with zeros to run the numerical code;
4. Resample the data to match the bedrock extent and resolution using cubic spline interpolation;
5. The data is provided in m/a, convert to m/s;
6. Mask the velocity with the ice mask. This is the simplest way to ensure consistency between velocity and ice thickness datasets.
"""

# ╔═╡ 578dbf49-b433-4cb3-8e94-6388beb3f68a
function create_velocity(velocity_path, bedrock, thickness)
    # load velocity data from Rabatel et al.
    velocity = Raster(velocity_path; name=:v2016_2017, crs=EPSG(32632))

    # the data in this dataset is stored in reversed order
    velocity = reverse(velocity; dims=2)

    # reproject the raster to the same crs
    # (seems to be a bug in Rasters.jl, we get segfaults without this step)
    velocity = Raster(velocity; crs=crs(velocity), dims=coords_as_ranges(velocity))

    # need to replace missing values with something that is supported by GDAL
    velocity = replace_missing(velocity, 0.0)

    # resample using cubic spline interpolation
    velocity = resample(velocity; to=bedrock, method=:cubicspline)

    # convert units to m/s (original data is in m/a)
    velocity ./= SECONDS_IN_YEAR

    # mask velocity to exclude areas where ice thickness is 0
    velocity = mask(velocity; with=thickness, missingval=0.0)

	return velocity
end;

# ╔═╡ 137a9311-f5ed-4ce0-8493-c212ed161434
md"""
### Smoothing the raster data

As a final pre-processing step, we apply several steps of Laplacian smoothing to all the rasters that constitute the input data, i.e. the bed elevation, the ice thickness, and the velocity. This is needed to reduce the noise in the data. Note that the Laplacian filter preserves monotonicity, no new local extrema can be created in the data.

We use the function `laplacian_smoothing`, provided by Reicalg.jl. Note that we cannot update the rasters in-place or re-use the variable names due to constraints posed by reactivity of Pluto.jl notebooks.
"""

# ╔═╡ 926569d4-fdc7-491e-8759-437c0f6c588e
function smooth_rasters!(amount, resolution, rasters...)
	for raster in rasters
		laplacian_smoothing!(raster, amount, resolution, resolution)
	end
	return
end;

# ╔═╡ 6c73898b-b480-4efe-840b-c078120a5703
md"""
## Processing the mass balance data

The surface mass balance (SMB) data is provided in the form of annual mass balance per elevation band. In Reicalg.jl, the SMB model is based on the simple altitude-dependent parametrisation:

```math
\dot{b}(z) = \min\left\{\beta (z - \mathrm{ELA}),\ \dot{b}_\mathrm{max}\right\}~,
```

where $z$ is the altitude, $\beta$ is the rate of change of mass balance, $\mathrm{ELA}$ is the equilibrium line altitude where $\dot{b}=0$, and $\dot{b}_\mathrm{max}$ is the maximum accumulation rate.


The data set covers time period from 1914 to 2022. In this section, we extract the SMB data for 2016-2017 hydrological year, and fit the parameters of the simple model:
"""

# ╔═╡ be59f0e5-8a65-4e6e-87ac-b683670ee169
function create_mass_balance_model(mass_balance_path)
    data = readdlm(mass_balance_path; skipstart=4)
    
    # extract data for 2016-2017 hydrological year
    data = vec(data[102, :])
    
    elevation_bands = LinRange(data[11], data[12], 26)
    ela             = data[8]

	mb_rng    = (13+26):(13+2*26-1)
	rho_w_i   = 1000 / 910

	# convert from mm w.e./a to m/s
    mass_balance = data[mb_rng] * 1e-3 * rho_w_i / SECONDS_IN_YEAR
    
    # remove NaNs from the mass balance data
    nan_mask = .!isnan.(mass_balance)
    
    mass_balance    = mass_balance[nan_mask]
    elevation_bands = elevation_bands[nan_mask]
    
    # find indices of minimal and maximal mass balance
	# we'll fit the data between these bounds
    imin = argmin(mass_balance)
    imax = argmax(mass_balance)
    
    # skip 5 points to only take fit the linear part
    nskip   = 5
    fit_rng = imin:imax-nskip
    
    # fit the slope of the line crossing the 0 mass balance at ela
    β = lsq_fit(elevation_bands[fit_rng] .- ela, mass_balance[fit_rng])
    
    # fit the high altitude 
    mass_balance_flat = mass_balance[(imax-nskip+1):end]
    b_max = sum(mass_balance_flat) / length(mass_balance_flat)

	return β, ela, b_max, elevation_bands, mass_balance
end;

# ╔═╡ 442e8496-850f-45b9-80ab-addf317f9e27
md"""
## Creating the input files

In this section, we define the function that combines all the previous developments to convert the pre-processed data into a format recognisable by Reicalg.jl. We pack data arrays, scalar physics parameters, and numerical parameters into named tuples, then save these tuples as JLD2:
"""

# ╔═╡ 68a2f01e-88d1-4655-8125-baab7d7c025f
function create_input_file(resolution)
	bedrock = create_bedrock(bed_tif_path,
					         surface_tif_path,
							 glacier_extent,
						     resolution)

	# The surface elevation data is georeferenced using the LV03 coordinate system
	surface_2009 = Raster(surface_2009_path; crs=EPSG(21781))
	surface_2017 = Raster(surface_2017_path; crs=EPSG(21781))

	# create ice thickness from surface and bedrock models
	thickness_2009 = create_ice_thickness(surface_2009, bedrock, resolution)
	thickness_2017 = create_ice_thickness(surface_2017, bedrock, resolution)

	t = (2016 - 2009) / (2017 - 2009)
	thickness_2016 = lerp.(thickness_2009, thickness_2017, t)
	
	# masking velocity with ice thickness at the end of the modelled period
	velocity = create_velocity(velocity_path, bedrock, thickness_2017)

	# fit the mass balance model
	β, ela, b_max, eb, mb = create_mass_balance_model(mass_balance_path)

	smooth_rasters!(smooth_amount, resolution, bedrock,
									           thickness_2016,
									           thickness_2017,
										       velocity)

	B     = Array{Float64}(bedrock)
	H_old = Array{Float64}(thickness_2016)
	H     = Array{Float64}(thickness_2017)

	# av4 needed as velocity in Reicalg.jl is defined on grid nodes
	V = Array{Float64}(velocity) |> av4;

	S_old = B .+ H_old
	
    # the mask excludes areas where ice thickness is 0 and mass balance is positive
    bool_mask = @. !((H_old[2:end-1, 2:end-1] == 0) &&
                     (S_old[2:end-1, 2:end-1] - ela > 0))

	# convert to Float64
    mb_mask = convert(Matrix{Float64}, bool_mask)

	# create fields (reverse the data to undo reversing done by GDAL)
	fields = map(x -> reverse(x, dims=2), (; H_old, H, B, V, mb_mask))
	
    nx, ny = size(H)
    dx, dy = float(resolution), float(resolution)
    lx, ly = nx * dx, ny * dy

	# coordinates of cell centers
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)
	
	numerics = (; nx, ny, dx, dy, xc, yc)

	scalars = (ρgn  = RHOG_N,
               A    = GLEN_A,
               npow = GLEN_N,
               dt   = 1.0 * SECONDS_IN_YEAR,
               lx,
               ly,
               β,
           	   b_max,
               ela)

	output_path = joinpath(datasets_dir, "aletsch_$(resolution)m.jld2")

	# save elevation bands and mass balance data for visualisation
	jldsave(output_path; fields, scalars, numerics, eb, mb)

	return
end;

# ╔═╡ ac8a9b9c-b0a5-42c3-96e0-de5b68abf66e
md"""
!!! note "Some comments on the code"
	- Ice velocity data from [Rabatel et al. (2023)](https://doi.org/10.3390/data8040066) is given on an annual basis. We create the intermediate ice thickness to match the velocity in years 2016--2017 by assuming linear variation between snapshots. Since the linear interpolation preserves monotonicity, no clamping is needed after this step.
	- The altitude-dependent SMB model that we use doesn't account for lateral variations of the mass balance. Therefore, in the numerical simulation the forward model might create non-zero ice thickness in the regions where the observed surface is ice-free. This is inconsistent with both the observed surface changes and velocity data that is used for the inversion. Here, we use the simple distributed correction for the mass balance: we introduce a mass balance mask, which remove ice accumulation in the regions where the observed ice thickness is zero.
	- We pack the input data into named tuples. The raster data is stored in the flipped order in the underlying array. We fix this by reversing all the arrays along the second dimension before saving as JLD2.

With all preparations done, let's generate the input files:
"""

# ╔═╡ b5624ea5-cd19-4b87-9634-8ab3682e2912
for resolution in resolutions
	@info "generating input files for resolution $resolution m"
	create_input_file(resolution)
end;

# ╔═╡ fa29f186-68c4-41f1-9a4a-de0f895b41ca
md"""
## Visualisation

Finally, we visualise the input data for the highest resolution (which is $(minimum(resolutions)) m) and save a nice figure for the publication:
"""

# ╔═╡ 4c80c4b5-0a66-48c2-8617-20e33501f5ec
with_theme(theme_latexfonts()) do
	# visualise the highest resolution
	vis_resolution = minimum(resolutions)
	
	vis_path = joinpath(datasets_dir, "aletsch_$(vis_resolution)m.jld2")
	
	fields, scalars, numerics, eb, mb = load(vis_path, "fields",
															 "scalars",
															 "numerics",
															 "eb", 
															 "mb")

	
    fig = Figure(; size=(850, 550), fontsize=16)

    # convert to km
    x_km = numerics.xc ./ 1e3
    y_km = numerics.yc ./ 1e3
    
    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect()),
           Axis(fig[1, 2][1, 1]; aspect=DataAspect()),
           Axis(fig[2, 1][1, 1]; aspect=DataAspect()),
           Axis(fig[2, 2][1, 1]; aspect=DataAspect()),
           Axis(fig[1, 3][1, 1]; aspect=DataAspect()),
           Axis(fig[2, 3]      ; title=L"\text{SMB model}"))

    colgap!(fig.layout, 1, Relative(0.08))

    axs[1].title = L"B~\mathrm{[m]}"
    axs[2].title = L"V~\mathrm{[m/a]}"
    axs[3].title = L"H_{2016}~\mathrm{[m]}"
    axs[4].title = L"H_{2017} - H_{2016}~\mathrm{[m]}"
    axs[5].title = L"\text{mass balance mask}"
    
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
	
    fields.V[ice_mask_v]   .= NaN
    fields.H_old[ice_mask] .= NaN
    fields.H[ice_mask]     .= NaN

    hms = (heatmap!(axs[1], x_km, y_km, fields.B),
           heatmap!(axs[2], x_km, y_km, fields.V),
           heatmap!(axs[3], x_km, y_km, fields.H_old),
           heatmap!(axs[4], x_km, y_km, fields.H .- fields.H_old ),
           heatmap!(axs[5], x_km, y_km, fields.mb_mask))

    foreach(hms) do h
        h.interpolate = true
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
                     mb .* SECONDS_IN_YEAR; markersize=7,
                                            color=:red,
                                            label="data")

    # parametrised model
    lines!(axs[6], z ./ 1e3, b .* SECONDS_IN_YEAR; linewidth=2, label="model")
    
    scatter!(axs[6], scalars.ela / 1e3, 0; strokecolor=:black,
                                           strokewidth=2,
                                   		   color=:transparent,
                                   		   marker=:diamond,
                                   		   label="ELA")
    
    axislegend(axs[6]; position=:rb)

    cb = (Colorbar(fig[1, 1][1, 2], hms[1]),
          Colorbar(fig[1, 2][1, 2], hms[2]),
          Colorbar(fig[2, 1][1, 2], hms[3]),
          Colorbar(fig[2, 2][1, 2], hms[4]),
          Colorbar(fig[1, 3][1, 2], hms[5]))

    for (label, idx) in zip(("A", "B", "C", "D", "E", "F"),
                            ((1,1), (1,2), (1,3), (2,1), (2,2), (2,3)))
        Label(fig[idx..., TopLeft()], label; fontsize = 20,
											 font     = :bold,
											 padding  = (0, 10, 5, 0))
    end

    mkpath("../../figures")
    save("../../figures/aletsch_setup.pdf", fig)

    fig
end

# ╔═╡ Cell order:
# ╟─d80af3c8-36ec-4d09-9b83-46b45bb25374
# ╟─764a33ef-e288-45ec-960c-caf2eb56e3d3
# ╟─8bc9c1af-6f5e-40f7-9afb-a9134c56412a
# ╠═0f9575cb-b4b6-4d3a-97d4-dc5db7c23c12
# ╟─72db902c-10f3-4446-99d5-4eb0f810662b
# ╠═b6ccaa3e-d948-4485-9d4a-b18fce33cb1a
# ╟─a5871e66-3688-499d-b251-99b0f23e6726
# ╠═6f1088a6-93b3-44e2-9a50-611660edaa84
# ╟─d4ef10ab-fde9-43cc-96fc-6f31785ab85b
# ╟─ee1916bf-d3b4-41b8-b507-57b57a324bce
# ╠═055bd14d-a02f-4610-8212-3ea52ce4d7c0
# ╟─375feb17-9d84-42c8-b5b0-c7da73a17462
# ╠═8d1acf5a-4ada-43ed-a081-c7a4766c0219
# ╟─5b0af720-f416-438d-ab8d-42c2a4c4ca65
# ╠═37042783-bafe-444f-85ff-bc9f17d4ff53
# ╟─4af27194-4ebe-41a9-a716-142b3f4817e0
# ╠═24930db6-f893-4cc6-ad30-58e8bac0ee1c
# ╟─b0969b25-551f-4e72-b08c-63881e05f359
# ╠═77e4a6bc-8502-4328-a08f-c28796b3d55d
# ╟─20b667e7-2adf-428a-b7b2-fb684fb401b1
# ╟─f370b133-c07d-431b-a7ba-2252d16ded53
# ╠═9f5a6312-04c6-4881-81e0-4003f865828d
# ╟─0faf7da5-464b-4943-821b-5259714d8f87
# ╠═6c93121b-90e8-4a3b-adcd-d4ed118466f6
# ╟─b55c88fd-ef10-42a4-92ad-83ca671bd96a
# ╟─4b6fa9b4-4528-4e28-aaff-e13825a26037
# ╠═9c0af9c4-cb6a-46e8-b222-9e4a75eeb4c7
# ╟─0b99796a-6edb-4a53-a875-bd5d329643d7
# ╠═aca29b01-114a-492b-9598-84be41452183
# ╟─1262ffa4-1233-451a-82b0-f4c666e36bac
# ╠═578dbf49-b433-4cb3-8e94-6388beb3f68a
# ╟─137a9311-f5ed-4ce0-8493-c212ed161434
# ╠═926569d4-fdc7-491e-8759-437c0f6c588e
# ╟─6c73898b-b480-4efe-840b-c078120a5703
# ╠═be59f0e5-8a65-4e6e-87ac-b683670ee169
# ╟─442e8496-850f-45b9-80ab-addf317f9e27
# ╠═68a2f01e-88d1-4655-8125-baab7d7c025f
# ╟─ac8a9b9c-b0a5-42c3-96e0-de5b68abf66e
# ╠═b5624ea5-cd19-4b87-9634-8ab3682e2912
# ╟─fa29f186-68c4-41f1-9a4a-de0f895b41ca
# ╟─4c80c4b5-0a66-48c2-8617-20e33501f5ec
