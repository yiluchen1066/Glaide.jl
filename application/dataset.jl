using Rasters
using GlacioTools
import NCDatasets
using CairoMakie

@views function load_data(Glacier::AbstractString, SGI_ID::AbstractString, datadir::AbstractString; visu=nothing)
    #load ice surface data using geom_select
    ice_thic, ice_surf, bed_surf = GlacioTools.geom_select(Glacier, SGI_ID, datadir; do_save=false)
    ice_thic = reverse(ice_thic[:, :, 1]; dims=2) # reads it into memory
    ice_surf = reverse(ice_surf[:, :, 1]; dims=2) # reads it into memory
    bed_surf = reverse(bed_surf[:, :, 1]; dims=2) # reads it into memory

    #load vmag data 
    dataset = RasterStack("datasets_vel/ALPES_wFLAG_wKT_ANNUALv2016-2021.nc";
                      crs=EPSG(32632), mappedcrs=EPSG(32632))
    vmag_epsg = dataset.v2016_2017[:, :]
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

    # visualization 
    if !isnothing(visu)
        fig = Figure(; size=(1000, 580), fontsize=22)
        axs = (H=Axis(fig[1, 1]; aspect=DataAspect(), xlabel=L"x\text{ [m]}", ylabel=L"y\text{ [m]}", title=L"Ice thickness"),
            vmag=Axis(fig[1, 2]; aspect=DataAspect(), xlabel=L"x\text{ [m]}", ylabel=L"y\text {[m]}", title=L"Vmag"))
        plts = (H=plot!(axs.H, ice_thic; colormap=:turbo),
                vmag=plot!(axs.vmag, vmag; colormap=:turbo))
        axs.H.xticksize = 18
        axs.H.yticksize = 18
        axs.vmag.xticksize = 18
        axs.vmag.yticksize = 18
        Colorbar(fig[1, 1][1, 2], plts.H)
        Colorbar(fig[1, 2][1, 2], plts.vmag)
        colgap!(fig.layout, 7)
        display(fig)
    end 

    #maybe it would be a good check the difference in the size between vmag and ice_thic
    xy = DimPoints(dims(ice_thic, (X,Y)))
    (x, y) = (first.(xy), last.(xy))
    # xc, yc = x.data[:,1], y.data[1,:]

    # @show nx_1, ny_1
    # @show size(ice_thic.data)
    # @show size(vmag.data)

    # here I should return ice_thic, ice_surf, bed_surf, vmag, 
    return ice_thic.data, ice_surf.data, bed_surf.data, vmag.data, x.data[:,1], y.data[1,:]
end 



