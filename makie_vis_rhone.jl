using CairoMakie
using DelimitedFiles
using LinearAlgebra
using Rasters

@views function load_data(bed_dat, surf_dat)
    z_bed  = reverse(dropdims(Raster(bed_dat); dims=3), dims=2)
    z_surf = reverse(dropdims(Raster(surf_dat); dims=3), dims=2)
    xy = DimPoints(dims(z_bed, (X, Y)))
    (x,y) = (first.(xy), last.(xy))
    return z_bed.data, z_surf.data, x.data[:,1], y.data[1,:]
end

B_rhone, S_rhone, xc, yc = load_data("Rhone_data_padding/Archive/Rhone_BedElev_cr.tif", "Rhone_data_padding/Archive/Rhone_SurfElev_cr.tif")
H_rhone = S_rhone .- B_rhone
nx      = size(B_rhone)[1]
ny      = size(B_rhone)[2]
B_off   = 0.1*ones(Float64, nx, ny)

S_rhone[H_rhone.==0.0] .= B_rhone[H_rhone.==0.0] .- 0.1

xc = xc .- xc[1]
yc = yc .- yc[1]

xc  = xc./1e3
yc = yc ./1e3

fig = Figure(resolution=(2800,1800), fontsize=42)
ax = Axis3(fig[1,1][1,1]; aspect=(1,1,0.5), title="Synthetic glacier")

surface!(ax,xc,yc,S_rhone;colormap=:ice         )
surface!(ax,xc,yc,B_rhone;colormap=:lightterrain)


save("rhone_glacier.png", fig)

# syntheic 
fig






