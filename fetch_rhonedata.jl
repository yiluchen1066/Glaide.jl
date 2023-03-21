using GlacioTools 
using Plots

name = " Rhone" 
SGI_ID = "B43-03"

data = fetch_glacier(name, SGI_ID; datadir="/scratch-1/yilchen/Msc-Inverse-SIA/Rhone_data/")
geom_select(name, SGI_ID, "/scratch-1/yilchen/Msc-Inverse-SIA/Rhone_data/"; padding=400) 
extract_geodata(Float64, name, "/scratch-1/yilchen/Msc-Inverse-SIA/Rhone_data/")
gl = GlacioTools.load_elevation("/scratch-1/yilchen/Msc-Inverse-SIA/Rhone_data/alps/data_Rhone.h5")

heatmap(gl.z_surf-gl.z_bed)

