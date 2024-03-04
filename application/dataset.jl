using Rasters
using GlacioTools
import NCDatasets
using CairoMakie

@views function load_data(Glacier::AbstractString, SGI_ID::AbstractString, datadir::AbstractString, velocity_file; visu_data=false)
    #load ice surface data using geom_select
    ice_thic, ice_surf, bed_surf = GlacioTools.geom_select(Glacier, SGI_ID, joinpath(datadir, Glacier); do_save=false)
    ice_thic = reverse(ice_thic[:, :, 1]; dims=2) # reads it into memory
    ice_surf = reverse(ice_surf[:, :, 1]; dims=2) # reads it into memory
    bed_surf = reverse(bed_surf[:, :, 1]; dims=2) # reads it into memory
    #bed_offset = reverse(bed_surf[:, :, 1]; dims=2)

    velocity_path = joinpath(datadir, Glacier, velocity_file)

    #load vmag data 
    dataset = RasterStack(velocity_path; crs=EPSG(32632), mappedcrs=EPSG(32632))
    vmag_epsg = dataset.v2019_2020[:, :]
    vmag_epsg = replace_missing(vmag_epsg, NaN)
    vmag_epsg = Raster(vmag_epsg; dims=GlacioTools.coords_as_ranges(vmag_epsg),
                   crs=crs(vmag_epsg),
                   metadata=vmag_epsg.metadata, missingval=vmag_epsg.missingval, name=vmag_epsg.name,
                   refdims=vmag_epsg.refdims)
    vmag = resample(vmag_epsg; to=ice_thic)
    vmag = reverse(vmag; dims=2)

    #mask off

    vmag[ice_thic.==0] .= NaN
    ice_thic[ice_thic .== 0] .= NaN

    #maybe it would be a good check the difference in the size between vmag and ice_thic
    xy = DimPoints(dims(ice_thic, (X,Y)))
    (x, y) = (first.(xy), last.(xy))
    xc = x.data[:,1]
    yc = y.data[1,:]

    H_Alet = ice_thic.data
    S_Alet = ice_surf.data 
    B_Alet = bed_surf.data
    vmag_Alet = vmag.data 
    #Alet_offset = bed_offset.data

    oz = minimum(B_Alet)
    B_Alet .-= oz 
    S_Alet .-= oz 

    @show oz

    xc = xc .- xc[1]
    yc = yc .- yc[1]

    # visu
    # if visu_data
    #     fig = Figure(; size=(1000, 580), fontsize=22)
    #     axs = (H=Axis(fig[1, 1]; aspect=DataAspect(), xlabel=L"x\text{ [km]}", ylabel=L"y\text{ [km]}", title=L"H [m]"),
    #         vmag=Axis(fig[1, 2]; aspect=DataAspect(), xlabel=L"x\text{ [km]}", ylabel=L"y\text {[km]}", title=L"Vmag [m/s]"))
    #     plts = (H=plot!(axs.H, xc, yc, H; colormap=:turbo),
    #             vmag=plot!(axs.vmag, xc, yc, vmag; colormap=:turbo))
    #     axs.H.xticksize = 18
    #     axs.H.yticksize = 18
    #     axs.vmag.xticksize = 18
    #     axs.vmag.yticksize = 18
    #     Colorbar(fig[1, 1][1, 2], plts.H)
    #     Colorbar(fig[1, 2][1, 2], plts.vmag)
    #     colgap!(fig.layout, 7)
    #     display(fig)
    # end 
    return H_Alet, S_Alet, B_Alet, vmag_Alet, oz, xc, yc
end 



