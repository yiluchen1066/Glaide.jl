using GLMakie
using DelimitedFiles
using LinearAlgebra

@views function load_data(bed_dat, surf_dat)
    z_bed  = reverse(dropdims(Raster(bed_dat); dims=3), dims=2)
    z_surf = reverse(dropdims(Raster(surf_dat); dims=3), dims=2)
    xy = DimPoints(dims(z_bed, (X, Y)))
    (x,y) = (first.(xy), last.(xy))
    return z_bed.data, z_surf.data, x.data[:,1], y.data[1,:]
end

B_rhone, S_rhone, xc, yc = load_data("Rhone_data_padding/Archive/Rhone_BedElev_cr.tif", "Rhone_data_padding/Archive/Rhone_SurfElev_cr.tif")
H_rhone = S_rhone .- B_rhone
S_rhone[H_rhone.==0.0] .= NaN

fig = Figure(resolution=(2500,25000), fontsize=42)
ax = Axis3(fig[1,1][1,1]; aspect=(1,1,0.5), title="Synthetic glacier")

GLMakie.surface!(ax,xc,yc,B_rhone;colormap=:lightterrain)
GLMakie.surface!(ax,xc,yc,S_rhone;colormap=:ice         )


save("rhone_glacier.png", fig)

# syntheic 
fig






