using Rasters
using GlacioTools
import NCDatasets
using CairoMakie


# ice_thic, ice_surf, bed_surf = GlacioTools.geom_select("Rhone", "B43-03", "application/datasets/Rhone"; do_save=false)
ice_thic, ice_surf, bed_surf = GlacioTools.geom_select("Aletsch", "B36-26", "datasets/Aletsch"; do_save=false)
ice_thic = reverse(ice_thic[:,:,1], dims=2) # reads it into memory
ice_surf = reverse(ice_surf[:,:,1], dims=2) # reads it into memory
bed_surf = reverse(bed_surf[:,:,1], dims=2) # reads it into memory

dataset = RasterStack("datasets_vel/ALPES_wFLAG_wKT_ANNUALv2016-2021.nc",
    crs=EPSG(32632), mappedcrs=EPSG(32632)) # somehow it does not pic up the right coord-system,
                                            # so provide it manually
vmag_epsg = dataset.v2016_2017[:,:]
vmag_epsg = replace_missing(vmag_epsg, NaN)
# dims are with Missing which GDAL does not like
vmag_epsg = Raster(vmag_epsg, dims=GlacioTools.coords_as_ranges(vmag_epsg),
    crs=crs(vmag_epsg),
    metadata=vmag_epsg.metadata, missingval=vmag_epsg.missingval, name=vmag_epsg.name,
    refdims=vmag_epsg.refdims)

vmag = resample(vmag_epsg; to=ice_thic)
vmag = reverse(vmag, dims=2) # GDAL always reverses coordinates
# mask off ice (not sure you need this)
vmag[ice_thic.==0] .= NaN


fig = Figure()
ax  = Axis(fig[1,1][1,1]; aspect=DataAspect())
hm  = plot!(ax, vmag_epsg; colormap=:turbo)
Colorbar(fig[1,1][1,2], hm)

display(fig)

## My makie does not run.  But this shows all is good:
# using Plots
# plot(ice_thic)
# plot(vmag)
