using GlacioTools 
using Plots

name = "Rhone" 
SGI_ID = "B43-03"
datadir ="/scratch-1/yilchen/Msc-Inverse-SIA/Rhone_data/"

#data = fetch_glacier(name, SGI_ID; datadir)
#geom_select(name, SGI_ID, datadir; padding=400) 
#extract_geodata(Float64, name, datadir)
gl = GlacioTools.load_elevation(joinpath(datadir,"alps/data_Rhone.h5"))
@show(typeof(gl))
@show(gl.z_surf-gl.z_bed)

heatmap(gl.z_surf-gl.z_bed)

