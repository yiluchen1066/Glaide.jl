using GLMakie
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

oz = minimum(B_rhone)

B_rhone .-= oz
S_rhone .-= oz

xc = xc .- xc[1]
yc = yc .- yc[1]

xc  = xc./1e3
yc = yc ./1e3

fig = Figure(resolution=(1300,900), fontsize=28)
ax = Axis3(fig[1,1][1,1]; aspect=(1,1,0.5), xlabel="X [km]", ylabel="Y [km]",zlabel="Z [m]")

ax.xlabeloffset[] = 60
ax.ylabeloffset[] = 60
ax.zlabeloffset[] = 100

plt = surface!(ax,xc,yc,S_rhone;colormap=:turbo,color=H_rhone)
surface!(ax,xc,yc,B_rhone;colormap=:lightterrain)

cb = Colorbar(fig[1,1][2,1],plt;vertical=false)
cb.tellwidth[] = false
cb.width[] = 500
cb.label[] = "H [m]"

save("rhone_glacier.png", fig)

# syntheic 
fig






