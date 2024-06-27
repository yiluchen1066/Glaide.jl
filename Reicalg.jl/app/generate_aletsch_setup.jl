using Downloads
using ZipArchives
using Rasters, ArchGDAL, NCDatasets, Extents
using CairoMakie
using DelimitedFiles
using JLD2

# define consntants
const SECONDS_IN_YEAR = 3600 * 24 * 365

# source directory
const SOURCES_DIR = joinpath(pwd(), "datasets", "_sources")

# URLs for downloading bed and surface elevation from Grab et al. 2020
const ALPS_BEDROCK_URL = "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/434697/07_GlacierBed_SwissAlps.zip"
const ALPS_SURFACE_URL = "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/434697/08_SurfaceElevation_SwissAlps.zip"

# paths to the target GeoTIFF files in the ZIP archives
const ALPS_BEDROCK_ZIP_PATH = "07_GlacierBed_SwissAlps/GlacierBed.tif"
const ALPS_SURFACE_ZIP_PATH = "08_SurfaceElevation_SwissAlps/SwissALTI3D_r2019.tif"

# paths to the GeoTIFF files in the filesystem
const ALPS_BEDROCK_TIF_PATH = joinpath(SOURCES_DIR, "elevation", ALPS_BEDROCK_ZIP_PATH)
const ALPS_SURFACE_TIF_PATH = joinpath(SOURCES_DIR, "elevation", ALPS_SURFACE_ZIP_PATH)

# unfortunately, it is impossible to automatically download the dataset for ice velocities from Rabatel et al.
# so it needs to be downloaded manually using the following link:
# https://entrepot.recherche.data.gouv.fr/file.xhtml?persistentId=doi:10.57745/VJYARH&version=1.1
const ALPS_VELOCITY_NC_PATH = joinpath(SOURCES_DIR, "velocity", "ALPES_wFLAG_wKT_ANNUALv2016-2021.nc")

# this is internal data that we distribute along with the publication
const ALETSCH_SURFACE_2009_PATH = joinpath(SOURCES_DIR, "elevation", "aletsch2009.asc")
const ALETSCH_SURFACE_2017_PATH = joinpath(SOURCES_DIR, "elevation", "aletsch2017.asc")
const ALETSCH_MASS_BALANCE_PATH = joinpath(SOURCES_DIR, "mass_balance", "aletsch_fix.dat")

# the extents covering the Aletsch glacier in the Swiss coordinate system LV95
const ALETSCH_EXTENT = Extent(; X=(2.637e6, 2.651e6), Y=(1.1385e6, 1.158e6))

# helper functions

# linearly interpolate between values
lerp(a, b, t) = b * t + a * (oneunit(t) - t)

# download the dataset from Grab et al. 2020, and extract the GeoTIFF raster to specified location
function download_raster(url, zip_entry, path)
    if ispath(path)
        @info "path '$path' already exists, skipping download..."
        return
    end

    # download data into an in-memory buffer
    data = take!(Downloads.download(url, IOBuffer()))

    # interpret binary blob as ZIP archive (which it is)
    archive = ZipReader(data)

    dir = dirname(path)
    if !isdir(dir)
        mkpath(dir)
    end

    # save the file from the zip archive to disk
    open(path, "w") do io
        write(io, zip_readentry(archive, zip_entry))
    end

    return
end

# copied from GlacioTools.jl, workaround to avoid segfaults in Rasters.jl
function coords_as_ranges(raster_like; sigdigits=0)
    x, y = dims(raster_like)

    if sigdigits == 0
        x = X(LinRange(x[1], x[end], length(x)))
        y = Y(LinRange(y[1], y[end], length(y)))
    else
        x = X(LinRange(round(x[1]; sigdigits), round(x[end]; sigdigits), length(x)))
        y = Y(LinRange(round(y[1]; sigdigits), round(y[end]; sigdigits), length(y)))
    end

    dx = x[2] - x[1]
    dy = y[2] - y[1]

    @assert abs(dx) == abs(dy) "abs(dx) and abs(dy) not equal ($dx, $dy)"
    return x, y
end

# preprocess Alps velocity data from Rabatel et al. 2023, reverse axes and resample to bedrock
function preprocess_velocity_raster(velocity, bedrock)
    # the data in this dataset is stored in reversed order
    velocity = reverse(velocity; dims=2)

    # reproject the raster to the same crs (seems to be a bug in Rasters.jl, we get segfaults without this step)
    velocity = Raster(velocity; crs=crs(velocity), dims=coords_as_ranges(velocity))

    # need to replace missing values with something that is supported by GDAL
    velocity = replace_missing(velocity, 0.0)

    # resample using cubic spline interpolation
    velocity = resample(velocity; to=bedrock, method=:cubicspline)

    # convert units to m/s (original data is in m/a)
    @. velocity /= SECONDS_IN_YEAR

    return velocity
end

function create_ice_thickness_raster(surface, bedrock)
    # resample surface raster to match the bedrock extent and resolution
    surface = resample(surface; to=bedrock, method=:cubicspline)

    # clamp to prevent negative values
    thickness = max.(surface .- bedrock, 0.0)

    # where elevation data is missing, we assume that there is no ice: thickness is 0
    return replace_missing(thickness, 0.0)
end

# least squares fit of the data
lsq_fit(x, y) = (x' * x) \ (x' * y)

function create_mass_balance(data_path)
    data = readdlm(data_path; skipstart=4)

    # extract data for 2016-2017 hydrological year
    data = vec(data[102, :])

    elevation_bands = LinRange(data[11], data[12], 26)
    ela             = data[8]
    mass_balance    = data[(13+26):(13+2*26-1)] * 1e-3 * (1000 / 910)

    # remove NaNs from the mass balance data
    mask = .!isnan.(mass_balance)

    mass_balance    = mass_balance[mask]
    elevation_bands = elevation_bands[mask]

    # find indices of minimal and maximal mass balance, we use the least squares fit for data between these bounds
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

    return β, ela, b_max
end

function generate_aletsch_data(res_m)
    # download datasets from ETH research collection
    download_raster(ALPS_BEDROCK_URL, ALPS_BEDROCK_ZIP_PATH, ALPS_BEDROCK_TIF_PATH)
    download_raster(ALPS_SURFACE_URL, ALPS_SURFACE_ZIP_PATH, ALPS_SURFACE_TIF_PATH)

    bedrock = resample(Raster(ALPS_BEDROCK_TIF_PATH); to=ALETSCH_EXTENT, res=res_m, method=:cubicspline)
    surface = resample(Raster(ALPS_SURFACE_TIF_PATH); to=ALETSCH_EXTENT, res=res_m, method=:cubicspline)

    # create mosaic to replace missing points with surface elevation from ALTI3D
    bedrock = mosaic(first, bedrock, surface)

    # load raster for ice surface elevation model from 2009 and create ice thickness consistent with the bedrock topogaphy
    surface_2009   = Raster(ALETSCH_SURFACE_2009_PATH; crs=EPSG(21781))
    thickness_2009 = create_ice_thickness_raster(surface_2009, bedrock)

    # same for year 2017
    surface_2017   = Raster(ALETSCH_SURFACE_2009_PATH; crs=EPSG(21781))
    thickness_2017 = create_ice_thickness_raster(surface_2017, bedrock)

    # there is no data available for 2016, so we reconstruct it assuming linear variation in thickness
    thickness_2016 = lerp.(thickness_2009, thickness_2017, (2016 - 2009) / (2017 - 2009))

    # load velocity data from Rabatel et al.
    velocity_stack = RasterStack(ALPS_VELOCITY_NC_PATH; crs=EPSG(32632))

    # preprocess the velocity data by resampling it to the bedrock bounds and resolution
    velocity = preprocess_velocity_raster(velocity_stack.v2016_2017, bedrock)

    # mask velocity to exclude areas where ice thickness is 0
    velocity = mask(velocity; with=thickness_2017, missingval=0.0)

    # create mass balance model using the data from VAW
    β, ela, b_max = create_mass_balance(ALETSCH_MASS_BALANCE_PATH)

    # save the data as JLD2
    H_old = Array{Float64}(thickness_2016)
    H     = Array{Float64}(thickness_2017)
    B     = Array{Float64}(bedrock)

    # linearly interpolate velocity to the grid nodes
    #! format: off
    @views av4(A) = @. 0.25 * (A[1:end-1, 1:end-1] +
                               A[2:end  , 1:end-1] +
                               A[2:end  , 2:end  ] +
                               A[1:end-1, 2:end  ])
    #! format: on
    V = Array{Float64}(velocity) |> av4

    # ELA is assumed to be constant in space in this model
    ELA = fill(ela, size(H) .- 2)

    # create numerics
    nx, ny = size(H)

    xd, yd = dims(bedrock)
    dx, dy = step(xd), step(yd)

    lx, ly = nx * dx, ny * dy

    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)

    @views av1(a) = @. 0.5 * (a[1:end-1] + a[2:end])

    xv, yv = av1(xc), av1(yc)

    # pack parameters into named tuples and save as JLD2
    fields = (; H_old, H, B, V, ELA)
    scalars = (ρgn  = (910 * 9.81)^3,
               A    = 2.5e-24,
               npow = 3,
               dt   = 1 * SECONDS_IN_YEAR,
               β,
               b_max)
    numerics = (; nx, ny, dx, dy, xc, yc, xv, yv)

    ALETSCH_SETUP_DIR  = joinpath(pwd(), "datasets", "aletsch")
    ALETSCH_SETUP_PATH = joinpath(ALETSCH_SETUP_DIR, "aletsch_setup.jld2")

    mkpath(ALETSCH_SETUP_DIR)

    jldsave(ALETSCH_SETUP_PATH; fields, scalars, numerics)

    fig = Figure(; size=(1000, 500))

    ax = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="bedrock (Grab et al. 2020)"),
          Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="ALTI3D surface (SwissTopo, from Grab et al. 2020)"),
          Axis(fig[1, 3][1, 1]; aspect=DataAspect(), title="velocity 2016-2017 (Rabatel et al. 2023)"),
          Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="ice thickness 2009 (VAW)"),
          Axis(fig[2, 2][1, 1]; aspect=DataAspect(), title="ice thickness 2016 (interpolated)"),
          Axis(fig[2, 3][1, 1]; aspect=DataAspect(), title="ice thickness 2017 (VAW)"))

    hm = (heatmap!(ax[1], bedrock; colormap=:terrain, colorrange=(1000, 4000)),
          heatmap!(ax[2], surface; colormap=:terrain, colorrange=(1000, 4000)),
          heatmap!(ax[3], velocity; colormap=:turbo),
          heatmap!(ax[4], thickness_2009; colormap=:turbo, colorrange=(0, 800)),
          heatmap!(ax[5], thickness_2016; colormap=:turbo, colorrange=(0, 800)),
          heatmap!(ax[6], thickness_2017; colormap=:turbo, colorrange=(0, 800)))

    cb = (Colorbar(fig[1, 1][1, 2], hm[1]),
          Colorbar(fig[1, 2][1, 2], hm[2]),
          Colorbar(fig[1, 3][1, 2], hm[3]),
          Colorbar(fig[2, 1][1, 2], hm[4]),
          Colorbar(fig[2, 2][1, 2], hm[5]),
          Colorbar(fig[2, 3][1, 2], hm[6]))

    display(fig)

    return
end
