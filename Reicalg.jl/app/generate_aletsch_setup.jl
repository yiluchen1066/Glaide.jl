### A Pluto.jl notebook ###
# v0.19.45

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
## Generating the input data for the Aletsch glacier

In this notebook, we will prepare the inputs for the Reicalg.jl model.

We will perform the following steps:

1. Download the datasets containing the elevation, velocity, and mass balance for the Aletsch glacier in years 2016--2017;
2. Crop and resample the raster data to match the specified extent and resolution
3. Create the best fit curve for the measured mass balance;
4. Create the mass balance mask to remove ice accumulation in the areas where there is no ice by the end of the modelled period.
"""

# ╔═╡ 8bc9c1af-6f5e-40f7-9afb-a9134c56412a
md"""
### Configuration

In this section we define all the parameters that can be tuned in the setup, such as filesystem paths and data extents.

First, define the path where the datasets will be stored. The path is relative to the location of the notebook:
"""

# ╔═╡ 0f9575cb-b4b6-4d3a-97d4-dc5db7c23c12
SOURCES_DIR = "../../datasets/_sources"; mkpath(SOURCES_DIR);

# ╔═╡ d5f674d4-5c84-495c-beda-07a955fe2508
md"""
Next, define the path where the resulting input file will be saved:
"""

# ╔═╡ 31a712cb-756c-4b03-b4ba-a081c3a7f762
ALETSCH_SETUP_PATH = "../../datasets/aletsch_setup.jld2";

# ╔═╡ a5871e66-3688-499d-b251-99b0f23e6726
md"""
Define the extents covering the Aletsch glacier in the Swiss coordinate system LV95:
"""

# ╔═╡ 6f1088a6-93b3-44e2-9a50-611660edaa84
ALETSCH_EXTENT = Extent(; X=(2.637e6, 2.651e6), Y=(1.1385e6, 1.158e6));

# ╔═╡ d4ef10ab-fde9-43cc-96fc-6f31785ab85b
md"""
> __Note__
> By changing the extents, you can generate input files for other Swiss glaciers than Aletsch. However, only the bedrock and velocity datasets covers entire Swiss Alps. You will need to provide elevation data and mass balance data separately.
"""

# ╔═╡ 5d84eab3-5eed-4a7a-b1b4-7d7708dedee1
md"""
Define the resolution in meters of the final input data:
"""

# ╔═╡ 7d181fae-f5c5-4b6d-93d3-9cdb5139c1d3
RESOLUTION_METERS = 50.0;

# ╔═╡ 4af27194-4ebe-41a9-a716-142b3f4817e0
 md"""
 ### Prerequisites

Unfortunately, it is impossible to automatically download the dataset for ice velocities from [Rabatel et al.](https://doi.org/10.3390/data8040066) so it needs to be downloaded manually using [this link](https://entrepot.recherche.data.gouv.fr/file.xhtml?persistentId=doi:10.57745/VJYARH&version=1.1) 
and saved to the `<sources_dir>/_sources` directory. The file path should be the same as this variable:
"""

# ╔═╡ 24930db6-f893-4cc6-ad30-58e8bac0ee1c
ALPS_VELOCITY_PATH = joinpath(SOURCES_DIR, "ALPES_wFLAG_wKT_ANNUALv2016-2021.nc");

# ╔═╡ b0969b25-551f-4e72-b08c-63881e05f359
md"""
The elevation data and mass balance data are distributed as the datasets accompanying the paper, they need to be downloaded manually and also stored at the sources directory:
"""

# ╔═╡ 77e4a6bc-8502-4328-a08f-c28796b3d55d
begin
	ALETSCH_SURFACE_2009_PATH = joinpath(SOURCES_DIR, "aletsch2009.asc")
    ALETSCH_SURFACE_2017_PATH = joinpath(SOURCES_DIR, "aletsch2017.asc")
    ALETSCH_MASS_BALANCE_PATH = joinpath(SOURCES_DIR, "aletsch_fix.dat")
end;

# ╔═╡ 094962a3-3f3c-442c-99fa-757be3ff8e3e
md"""
### Downloading the elevation data
"""

# ╔═╡ f370b133-c07d-431b-a7ba-2252d16ded53
md"""
Here, we define the URLs for downloading bed and surface elevation from [Grab et al. 2020](https://doi.org/10.1017/jog.2021.55):
"""

# ╔═╡ 9f5a6312-04c6-4881-81e0-4003f865828d
begin
	ALPS_BEDROCK_URL =  "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/434697/07_GlacierBed_SwissAlps.zip"
	ALPS_SURFACE_URL = "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/434697/08_SurfaceElevation_SwissAlps.zip"
end;

# ╔═╡ 0faf7da5-464b-4943-821b-5259714d8f87
md"""
Then we define which files we need from the datasets, and download these files using the `download_raster` function provided by Reicalg.jl. If the files already exist in the filesystem, they won't be downloaded.
"""

# ╔═╡ 6c93121b-90e8-4a3b-adcd-d4ed118466f6
begin
	# paths to the target GeoTIFF files in the ZIP archives:
	ALPS_BEDROCK_ZIP_PATH = "07_GlacierBed_SwissAlps/GlacierBed.tif"
	ALPS_SURFACE_ZIP_PATH = "08_SurfaceElevation_SwissAlps/SwissALTI3D_r2019.tif"

	# paths to the GeoTIFF files in the filesystem
    ALPS_BEDROCK_TIF_PATH = joinpath(SOURCES_DIR, ALPS_BEDROCK_ZIP_PATH)
    ALPS_SURFACE_TIF_PATH = joinpath(SOURCES_DIR, ALPS_SURFACE_ZIP_PATH)

	# download datasets from ETH research collection
    download_raster(ALPS_BEDROCK_URL, ALPS_BEDROCK_ZIP_PATH, ALPS_BEDROCK_TIF_PATH)
    download_raster(ALPS_SURFACE_URL, ALPS_SURFACE_ZIP_PATH, ALPS_SURFACE_TIF_PATH)
end;

# ╔═╡ b55c88fd-ef10-42a4-92ad-83ca671bd96a
md"""
### Processing the elevation data
"""

# ╔═╡ 4b6fa9b4-4528-4e28-aaff-e13825a26037
md"""
Bed elevation data from [Grab et al. 2020](https://doi.org/10.1017/jog.2021.55) is only defined at ice-covered regions. In our model, we need the bedrock to be defined everywhere in the computational domain. To achieve this, we combine the bedrock dataset with the surface elevation model at the ice-free regions.

We crop the bed and the surface data to match the provided extent. Then we replace the missing data points in the bedrock dataset with the corresponding values in the surface elevation dataset from ALTI3D. Finally, we resample the bedrock raster to target spatial resolution using cubic spline interpolation:
"""

# ╔═╡ 9c0af9c4-cb6a-46e8-b222-9e4a75eeb4c7
begin
	bedrock = crop(Raster(ALPS_BEDROCK_TIF_PATH); to=ALETSCH_EXTENT)
	surface = crop(Raster(ALPS_SURFACE_TIF_PATH); to=ALETSCH_EXTENT)
	
	# create mosaic to replace missing points with surface elevation from ALTI3D
    bedrock = mosaic(first, bedrock, surface)
    bedrock = resample(bedrock; to=ALETSCH_EXTENT, res=RESOLUTION_METERS, method=:cubicspline)
end;

# ╔═╡ 3e03b549-c455-4baa-a3d0-95399e6b95de
md"""
Next, we load the surface elevation data at known times. The data is georeferenced using the LV03 coordinate system:
"""

# ╔═╡ 0a614066-534b-462c-ae7c-7b69a4c05dc8
begin
	surface_2009 = Raster(ALETSCH_SURFACE_2009_PATH; crs=EPSG(21781))
	surface_2017 = Raster(ALETSCH_SURFACE_2017_PATH; crs=EPSG(21781))
end;

# ╔═╡ 0b99796a-6edb-4a53-a875-bd5d329643d7
md"""
We define a helper function to create ice thickness data raster from known surface and bed elevation. In this function, we perform three steps:

1. Resample the surface data to match the extent and resolution of the bed elevation model using cubic spline interpolation;
2. Clamp the values to prevent negative ice thickness;
3. Replace missing values with zeros, as the numerical model expects zero ice thickness in ice-free regions.
"""

# ╔═╡ aca29b01-114a-492b-9598-84be41452183
function create_ice_thickness_raster(surface, bedrock)
    # resample surface raster to match the bedrock extent and resolution
    surface = resample(surface; to=bedrock, method=:cubicspline)

    # clamp to prevent negative values
    thickness = max.(surface .- bedrock, 0.0)

    # where elevation data is missing, we assume that there is no ice: thickness is 0
    return replace_missing(thickness, 0.0)
end;

# ╔═╡ 7819d457-1d35-4d9d-befa-05226df059c0
md"""
Create ice thickness rasters:
"""

# ╔═╡ c9331fef-bf9c-48c9-8e49-027b432d0c7a
begin
    thickness_2009 = create_ice_thickness_raster(surface_2009, bedrock)
    thickness_2017 = create_ice_thickness_raster(surface_2017, bedrock)
end;

# ╔═╡ 37d51b73-7db4-45dd-a356-9004b695ef0c
md"""
Ice velocity data from [Rabatel et al. 2023](https://doi.org/10.3390/data8040066) is given on an annual basis. We create the intermediate ice thickness to match the velocity in years 2016--2017 by assuming linear variation between snapshots. Since the linear interpolation preserves monotonicity, no clamping is needed after this step:
"""

# ╔═╡ 9ce15475-54de-420f-98c4-1f9532774bf0
thickness_2016 = let t = (2016 - 2009) / (2017 - 2009)
     lerp.(thickness_2009, thickness_2017, t)
end;

# ╔═╡ 696633b5-280b-4eb9-b5a9-236c60b95aa4
md"""
The resulting dataset contains many small regions disconnected from the glacier, but having non-zero ice thickness. These areas might slow down the numerical convergence of both the forward model and inversion. We exclude the small patches of ice from the inversion by removing all connected regions of length 16 pixels or less:
"""

# ╔═╡ 1c0514db-4c3c-45b2-ac52-336c53694c4d
begin
	remove_components!(thickness_2016; min_length=16)
    remove_components!(thickness_2017; min_length=16)
end

# ╔═╡ 1262ffa4-1233-451a-82b0-f4c666e36bac
md"""
### Processing the velocity data

In this section, we load and process the velocity data from the dataset from [Rabatel et al. 2023](https://doi.org/10.3390/data8040066). The processing includes the following steps:

1. Load the data, georeferenced using the WGS 84 corrdinate system;
2. Flip the data array for consistency with the elevation data;
3. Replace missing values with zeros to run the numerical code;
4. Resample the data to match the bedrock extent and resolution using cubic spline interpolation;
5. The data is provided in m/a, convert to m/s;
6. Mask the velocity with the ice mask. This is the simplest way to ensure consistency between velocity and ice thickness datasets.
"""

# ╔═╡ 578dbf49-b433-4cb3-8e94-6388beb3f68a
begin
	# load velocity data from Rabatel et al.
    velocity = Raster(ALPS_VELOCITY_PATH; name=:v2016_2017, crs=EPSG(32632))

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
    velocity = mask(velocity; with=thickness_2017, missingval=0.0)
end;

# ╔═╡ 137a9311-f5ed-4ce0-8493-c212ed161434
md"""
### Smoothing the raster data

As a final pre-processing step, we apply several steps of laplacian smoothing to all the rasters that constitute the input data, i.e. the bed elevation, the ice thickness, and the velocity. This is needed to reduce the noise in the data. Note that the laplacian filter preserves monotonicity, no new local extrema can be created in the data.

Before we smooth the rasters, we save the ice mask, which we'll use for visualisation later:
"""

# ╔═╡ 016b8c37-2d20-45bb-b6e9-cc1fa8563b6a
begin
	ice_mask   = reverse(thickness_2017      .< eps(); dims=2)
	ice_mask_v = reverse(av4(thickness_2017) .< eps(); dims=2)
end;

# ╔═╡ bd08b785-ae66-4de8-8685-ce51361d9e75
md"""
Now we apply the function `smooth_lap!`, provided by Reicalg.jl:
"""

# ╔═╡ ac921c1d-1672-4b92-b44f-ec3ed740c3ae
for nsm in 1:2
	smooth_lap!(bedrock)
	smooth_lap!(thickness_2016)
	smooth_lap!(thickness_2017)
	smooth_lap!(velocity)
end

# ╔═╡ 6c73898b-b480-4efe-840b-c078120a5703
md"""
Processing the mass balance data

The surface mass balance (SMB) data is provided in the form of annual mass balance per elevation band. In Reicalg.jl, the SMB model is based on the simple altitude-dependent parametrisation:

```math
\dot{b}(z) = \min\left\{\beta (z - ELA), \dot{b}_\mathrm{max}\right\}~,
```

where $z$ is the altitude, $\beta$ is the rate of change of mass balance, $ELA$ is the equilibrium line altitude where $\dot{b}=0$, and $\dot{b}_\mathrm{max}$ is the maximum accumulation rate.


The data set covers time period from 1914 to 2022. In this section, we extract the SMB data for 2016-2017 hydrological year, and fit the parameters of the simple model:
"""

# ╔═╡ be59f0e5-8a65-4e6e-87ac-b683670ee169
begin
	local data = readdlm(ALETSCH_MASS_BALANCE_PATH; skipstart=4)
	
	# extract data for 2016-2017 hydrological year
	data = vec(data[102, :])
	
	elevation_bands = LinRange(data[11], data[12], 26)
	ela             = data[8]
	mass_balance    = data[(13+26):(13+2*26-1)] * 1e-3 * (1000 / 910) / SECONDS_IN_YEAR
	
	# remove NaNs from the mass balance data
	nan_mask = .!isnan.(mass_balance)
	
	mass_balance    = mass_balance[nan_mask]
	elevation_bands = elevation_bands[nan_mask]
	
	# find indices of minimal and maximal mass balance, we use the least squares fit for data between these bounds
	local imin = argmin(mass_balance)
	local imax = argmax(mass_balance)
	
	# skip 5 points to only take fit the linear part
	local nskip   = 5
	local fit_rng = imin:imax-nskip
	
	# fit the slope of the line crossing the 0 mass balance at ela
	β = lsq_fit(elevation_bands[fit_rng] .- ela, mass_balance[fit_rng])
	
	# fit the high altitude 
	local mass_balance_flat = mass_balance[(imax-nskip+1):end]
	b_max = sum(mass_balance_flat) / length(mass_balance_flat)
end;

# ╔═╡ 442e8496-850f-45b9-80ab-addf317f9e27
md"""
### Saving the input data

In this section, we convert the pre-processed data into a format recognisable by Reicalg.jl. We pack data arrays, scalar physics parameters, and numerical parameters into named tuples, then save these tuples as JLD2.

First, we convert all the elevation data to Float64:
"""

# ╔═╡ a4abc107-2b0f-4d1e-80e6-ba611090416f
begin
    H_old = Array{Float64}(thickness_2016)
    H     = Array{Float64}(thickness_2017)
    B     = Array{Float64}(bedrock)
end;

# ╔═╡ ba11815c-835d-4788-8226-6b3b41cf6e45
md"""
The velocity in Reicalg.jl is defined at the grid nodes, unlike ice thickness and bed elevation, which are defined at the cell centers. We thus interpolate the velocity data, but before than we convert it to Float64:
"""

# ╔═╡ 4386e6fe-b5c7-4c4d-8b0e-ecba41c7b80b
V = Array{Float64}(velocity) |> av4;

# ╔═╡ 0f3cb318-b17c-4921-8ab1-6fa39c2e1f06
md"""
The altitude-dependent SMB model that we use doesn't account for lateral variations of the mass balance. Therefore, in the numerical simulation the forward model might create non-zero ice thickness in the regions where the observed surface is ice-free. This is inconsistent with both the observed surface changes and velocity data that is used for the inversion. Here, we use the simple distributed correction for the mass balance: we introduce a mass balance mask, which remove ice accumulation in the regions where the observed ice thickness is zero:
"""

# ╔═╡ 97867263-f83d-462c-a195-e15b19e364f3
mb_mask = let S_old = B .+ H_old
	# the mask excludes areas where ice thickness is 0 and mass balance is positive
    bool_mask = @. !((H_old[2:end-1, 2:end-1] == 0) &&
				     (S_old[2:end-1, 2:end-1] - ela > 0))
	convert(Matrix{Float64}, bool_mask)
end;

# ╔═╡ 1040d082-30f3-40e5-bfd3-e9b6d512336a
md"""
Next, we pack all the relevant numerical parameters such as the grid resolution and coordinates of grid nodes and cell centers, into the named tuple:
"""

# ╔═╡ 923d432c-983e-4357-b6f3-b827c31d609d
begin
	# create numerics
    nx, ny = size(H)

    xd, yd = dims(bedrock)
    dx, dy = step(xd), -step(yd)

    lx, ly = nx * dx, ny * dy

    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)

	numerics = (; nx, ny, dx, dy, xc, yc)
end

# ╔═╡ ca332039-8edc-4dc1-a4df-337d9adc94b2
md"""
We pack all the scalar physics parameters into the named tuple:
"""

# ╔═╡ 280a0cf7-1fbd-457d-985f-52b0c4d62faf
scalars = (ρgn  = RHOG_N,
		   A    = GLEN_A,
		   npow = GLEN_N,
		   dt   = 1.0 * SECONDS_IN_YEAR,
		   lx,
		   ly,
		   β,
		   b_max,
		   ela)

# ╔═╡ fd109951-7c21-4352-9e01-c02e020b25f7
md"""
We pack the field arrays into the named tuple. The raster data is stored in the flipped order in the underlying array. We fix this by reversing all the arrays along the second dimension:
"""

# ╔═╡ 2c551d3d-490f-4fdc-a134-1954fc6cbf36
fields = map(x -> reverse(x, dims=2), (; H_old, H, B, V, mb_mask));

# ╔═╡ 61e382df-ab4b-4d1e-8592-c7470f198275
md"""
Finally, we save these named tuples into a JLD2 file:
"""

# ╔═╡ e46ba65d-1fe7-4db0-9412-8dc48eb9fbba
jldsave(ALETSCH_SETUP_PATH; fields, scalars, numerics);

# ╔═╡ fa29f186-68c4-41f1-9a4a-de0f895b41ca
md"""
### Visualising the data

Now we can visualise the input fields and save a nice figure for the publication:
"""

# ╔═╡ 4c80c4b5-0a66-48c2-8617-20e33501f5ec
with_theme(theme_latexfonts()) do
	fig = Figure(; size=(850, 500), fontsize=16)

	# convert to km
	x_km = xc ./ 1e3
	y_km = yc ./ 1e3
	
	ax = (Axis(fig[1, 1][1, 1]; aspect=DataAspect()),
		  Axis(fig[1, 2][1, 1]; aspect=DataAspect()),
		  Axis(fig[2, 1][1, 1]; aspect=DataAspect()),
		  Axis(fig[2, 2][1, 1]; aspect=DataAspect()),
		  Axis(fig[1, 3][1, 1]; aspect=DataAspect()),
		  Axis(fig[2, 3]      ; title=L"\text{SMB model}"))

	colgap!(fig.layout, 1, Relative(0.1))

	ax[1].title = L"B~\mathrm{[m]}"
	ax[2].title = L"V~\mathrm{[m/a]}"
	ax[3].title = L"H_{2016}~\mathrm{[m]}"
	ax[4].title = L"H_{2017} - H_{2016}~\mathrm{[m]}"
	ax[5].title = L"\text{mass balance mask}"
	
	ax[3].xlabel = L"x~\mathrm{[km]}"
	ax[4].xlabel = L"x~\mathrm{[km]}"

	ax[1].ylabel = L"y~\mathrm{[km]}"
	ax[3].ylabel = L"y~\mathrm{[km]}"

	ax[6].xlabel = L"z~\mathrm{[km]}"
	ax[6].ylabel = L"\dot{b}~\mathrm{[m/a]}"
	
	hidexdecorations!.((ax[1], ax[2], ax[5]))
	hideydecorations!.((ax[2], ax[4], ax[5]))

	ax[2].xgridvisible=true
	ax[2].ygridvisible=true
	ax[4].ygridvisible=true

	V = copy(fields.V).*SECONDS_IN_YEAR
	V[ice_mask_v] .= NaN

	H_old = copy(fields.H_old)
	H_old[ice_mask] .= NaN

	H = copy(fields.H)
	H[ice_mask] .= NaN

	hm = (heatmap!(ax[1], x_km, y_km, fields.B),
		  heatmap!(ax[2], x_km, y_km, V),
		  heatmap!(ax[3], x_km, y_km, H_old      ),
		  heatmap!(ax[4], x_km, y_km, H .- H_old ),
		  heatmap!(ax[5], x_km, y_km, fields.mb_mask))

	foreach(hm) do h
		h.interpolate = true
	end
	
	hm[1].colormap   = :terrain
	hm[1].colorrange = (1000, 4000)

	hm[2].colormap   = :turbo
	hm[2].colorrange = (0, 300)

	hm[3].colormap   = Reverse(:roma)
	hm[3].colorrange = (0, 900)

	hm[4].colormap   = Reverse(:roma)
	hm[4].colorrange = (-10, 0)

	hm[5].colormap = :grays
	
	z = LinRange(1900, 4150, 1000)
	b = @. min(β * (z - ela), b_max)

	# observational mass balance data
	scatter!(ax[6], elevation_bands ./ 1e3,
					mass_balance .* SECONDS_IN_YEAR; markersize=7,
												     color=:red,
               	 									 label="data")

	# parametrised model
	lines!(ax[6], z ./ 1e3, b .* SECONDS_IN_YEAR; linewidth=2, label="model")
	
	scatter!(ax[6], ela / 1e3, 0; strokecolor=:black,
			     				  strokewidth=2,
								  color=:transparent,
								  marker=:diamond,
								  label="ELA")
	
	axislegend(ax[6]; position=:rb)

	cb = (Colorbar(fig[1, 1][1, 2], hm[1]),
		  Colorbar(fig[1, 2][1, 2], hm[2]),
		  Colorbar(fig[2, 1][1, 2], hm[3]),
		  Colorbar(fig[2, 2][1, 2], hm[4]),
		  Colorbar(fig[1, 3][1, 2], hm[5]))

	for (label, idx) in zip(("A", "B", "C", "D", "E", "F"),
							   ((1,1), (1,2), (1,3), (2,1), (2,2), (2,3)))
		Label(fig[idx..., TopLeft()], label, fontsize = 20, font = :bold)
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
# ╟─d5f674d4-5c84-495c-beda-07a955fe2508
# ╠═31a712cb-756c-4b03-b4ba-a081c3a7f762
# ╟─a5871e66-3688-499d-b251-99b0f23e6726
# ╠═6f1088a6-93b3-44e2-9a50-611660edaa84
# ╟─d4ef10ab-fde9-43cc-96fc-6f31785ab85b
# ╟─5d84eab3-5eed-4a7a-b1b4-7d7708dedee1
# ╠═7d181fae-f5c5-4b6d-93d3-9cdb5139c1d3
# ╟─4af27194-4ebe-41a9-a716-142b3f4817e0
# ╠═24930db6-f893-4cc6-ad30-58e8bac0ee1c
# ╟─b0969b25-551f-4e72-b08c-63881e05f359
# ╠═77e4a6bc-8502-4328-a08f-c28796b3d55d
# ╟─094962a3-3f3c-442c-99fa-757be3ff8e3e
# ╟─f370b133-c07d-431b-a7ba-2252d16ded53
# ╠═9f5a6312-04c6-4881-81e0-4003f865828d
# ╟─0faf7da5-464b-4943-821b-5259714d8f87
# ╠═6c93121b-90e8-4a3b-adcd-d4ed118466f6
# ╟─b55c88fd-ef10-42a4-92ad-83ca671bd96a
# ╟─4b6fa9b4-4528-4e28-aaff-e13825a26037
# ╠═9c0af9c4-cb6a-46e8-b222-9e4a75eeb4c7
# ╟─3e03b549-c455-4baa-a3d0-95399e6b95de
# ╠═0a614066-534b-462c-ae7c-7b69a4c05dc8
# ╟─0b99796a-6edb-4a53-a875-bd5d329643d7
# ╠═aca29b01-114a-492b-9598-84be41452183
# ╟─7819d457-1d35-4d9d-befa-05226df059c0
# ╠═c9331fef-bf9c-48c9-8e49-027b432d0c7a
# ╟─37d51b73-7db4-45dd-a356-9004b695ef0c
# ╠═9ce15475-54de-420f-98c4-1f9532774bf0
# ╟─696633b5-280b-4eb9-b5a9-236c60b95aa4
# ╠═1c0514db-4c3c-45b2-ac52-336c53694c4d
# ╟─1262ffa4-1233-451a-82b0-f4c666e36bac
# ╠═578dbf49-b433-4cb3-8e94-6388beb3f68a
# ╟─137a9311-f5ed-4ce0-8493-c212ed161434
# ╠═016b8c37-2d20-45bb-b6e9-cc1fa8563b6a
# ╟─bd08b785-ae66-4de8-8685-ce51361d9e75
# ╠═ac921c1d-1672-4b92-b44f-ec3ed740c3ae
# ╟─6c73898b-b480-4efe-840b-c078120a5703
# ╠═be59f0e5-8a65-4e6e-87ac-b683670ee169
# ╟─442e8496-850f-45b9-80ab-addf317f9e27
# ╠═a4abc107-2b0f-4d1e-80e6-ba611090416f
# ╟─ba11815c-835d-4788-8226-6b3b41cf6e45
# ╠═4386e6fe-b5c7-4c4d-8b0e-ecba41c7b80b
# ╟─0f3cb318-b17c-4921-8ab1-6fa39c2e1f06
# ╠═97867263-f83d-462c-a195-e15b19e364f3
# ╟─1040d082-30f3-40e5-bfd3-e9b6d512336a
# ╠═923d432c-983e-4357-b6f3-b827c31d609d
# ╟─ca332039-8edc-4dc1-a4df-337d9adc94b2
# ╠═280a0cf7-1fbd-457d-985f-52b0c4d62faf
# ╟─fd109951-7c21-4352-9e01-c02e020b25f7
# ╠═2c551d3d-490f-4fdc-a134-1954fc6cbf36
# ╟─61e382df-ab4b-4d1e-8592-c7470f198275
# ╠═e46ba65d-1fe7-4db0-9412-8dc48eb9fbba
# ╟─fa29f186-68c4-41f1-9a4a-de0f895b41ca
# ╟─4c80c4b5-0a66-48c2-8617-20e33501f5ec
