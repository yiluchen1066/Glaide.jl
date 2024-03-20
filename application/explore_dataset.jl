using Rasters
using GlacioTools
import NCDatasets
using CairoMakie
using ArchGDAL

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

function create_raster_stack()
    #load bedrock from GlacioTools    
    bedrock = Raster("Aletsch_bedrock.tif")
    bedrock = replace_missing(bedrock, NaN)

    #load surface from VAW data 
    surface_2017 = Raster("application/datasets/Aletsch/aletsch2017.asc"; crs = EPSG(21781))
    surface_2017 = resample(surface_2017; to=bedrock)
    surface_2017 = replace_missing(surface_2017, NaN)

    surface_2009 = Raster("application/datasets/Aletsch/aletsch2009.asc"; crs = EPSG(21781))
    surface_2009 = resample(surface_2009; to=bedrock)
    surface_2009 = replace_missing(surface_2009, NaN)

    y = 2016 
    t = (y - 2009) ./ (2017 - 2009)
    surface_2016 = surface_2017 * t + surface_2009 * (1 - t)

    mask_2016 = replace_missing(missingmask(surface_2016), false)
    #replace where we do not have surface with the value of bedrock
    surface_2016[.!mask_2016] .= bedrock[.!mask_2016]
    #replace where H is negative with the value of bedrock
    surface_2016[surface_2016 .< bedrock] .= bedrock[surface_2016 .< bedrock]

    mask_2017 = replace_missing(missingmask(surface_2017), false)
    surface_2017[.!mask_2017] .= bedrock[.!mask_2017]
    surface_2017[surface_2017 .< bedrock] .= bedrock[surface_2017 .< bedrock]

    #load velocity data from rabatel et al. 
    velocity_stack = RasterStack("application/datasets/Aletsch/velocity_data/ALPES_wFLAG_wKT_ANNUALv2016-2021.nc"; crs = EPSG(32632))
    velocity_2016_2017 = velocity_stack.v2016_2017
    velocity_2016_2017 = reverse(velocity_2016_2017; dims=2)
    velocity_2016_2017 = replace_missing(velocity_2016_2017, NaN)
    velocity_2016_2017 = Raster(velocity_2016_2017; dims = coords_as_ranges(velocity_2016_2017),
                                crs = crs(velocity_2016_2017))
    velocity_2016_2017 = resample(velocity_2016_2017; to=bedrock)

    stack = RasterStack((; bedrock, surface_2016, surface_2017, velocity_2016_2017)) 
    write("aletsch_data_2016_2017.nc", stack)
    return 
end 

create_raster_stack()

@views function main_data()
    stack = RasterStack("aletsch_data_2016_2017.nc")
    stack = replace_missing(stack, NaN)

    xr, yr = extrema.(dims(stack))
    lx, ly = xr[2] - xr[1], yr[2] - yr[1]

    aspect_ratio = ly/lx

    nx = 128 
    ny = ceil(Int, nx * aspect_ratio)

    stack = resample(stack; size=(nx, ny))
s
end 

main_data()
