using Rasters
using GlacioTools
import NCDatasets
using CairoMakie

# ice_thic, ice_surf, bed_surf = geom_select("Rhone", "B43-03", "application/datasets/Rhone"; do_save=false)
ice_thic, ice_surf, bed_surf = geom_select("Aletsch", "B36-26", "application/datasets/Aletsch"; do_save=false)

dataset = RasterStack("application/datasets_vel/ALPES_wFLAG_wKT_ANNUALv2016-2021.nc")
vmag_epsg = dataset.v2016_2017

# vmag = resample(vmag_epsg; to=ice_thic)
vmag = resample(vmag_epsg; crs=crs(ice_thic))

# bnd = X(4.15e5 .. 4.35e5), Y(5.138e6 .. 5.16e6)
# vmag_sub = vmag[bnd...]

# vmag_sub = crop(vmag; to=ice_th)

fig = Figure()
ax  = Axis(fig[1,1][1,1]; aspect=DataAspect())
hm  = plot!(ax, vmag; colormap=:turbo)
Colorbar(fig[1,1][1,2], hm)

display(fig)