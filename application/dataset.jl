using Rasters
using GlacioTools
import NCDatasets
using CairoMakie
using ArchGDAL

@views 

@views function load_data(Glacier::AbstractString, SGI_ID::AbstractString, datadir::AbstractString, velocity_file; visu_data=false)
    #load ice surface data using geom_select
    ice_thic, ice_surf, bed_surf = GlacioTools.geom_select(Glacier, SGI_ID, joinpath(datadir, Glacier); do_save=false)
    ice_thic = reverse(ice_thic[:, :, 1]; dims=2) # reads it into memory
    ice_surf = reverse(ice_surf[:, :, 1]; dims=2) # reads it into memory
    bed_surf = reverse(bed_surf[:, :, 1]; dims=2) # reads it into memory
    #bed_offset = reverse(bed_surf[:, :, 1]; dims=2)

    write("Aletsch_bedrock.tif", bed_surf)

    error("check")

    xy = DimPoints(dims(ice_thic, (X,Y)))
    (x, y) = (first.(xy), last.(xy))
    xc = x.data[:,1]
    yc = y.data[1,:]

    xc = xc .- xc[1]
    yc = yc .- yc[1]
    

    S_2009 = Raster("application/datasets/Aletsch/aletsch2009.asc"; crs=EPSG(2056), mappedcrs=EPSG(2056))
    S_2017 = Raster("application/datasets/Aletsch/aletsch2017.asc"; crs=EPSG(2056), mappedcrs=EPSG(2056))

    # create missing masks:
    mask_2009 = missingmask(S_2009)
    mask_2009 = replace_missing(mask_2009, 0.0)
    mask_2017 = missingmask(S_2017)
    mask_2017 = replace_missing(mask_2017, 0.0)

    replace_missing!(S_2009, NaN)
    replace_missing!(S_2017, NaN)

    S_2009 = Raster(S_2009; dims=GlacioTools.coords_as_ranges(S_2009), crs=crs(S_2009), metadata=S_2009.metadata, missingval=S_2009.missingval, name=S_2009.name,
    refdims=S_2009.refdims)
    S_2017 = Raster(S_2017; dims=GlacioTools.coords_as_ranges(S_2017), crs=crs(S_2017), metadata=S_2017.metadata, missingval=S_2017.missingval, name=S_2017.name,
    refdims=S_2017.refdims)


    @show size(S_2009)
    @show size(S_2017)

    S_2009_2 = resample(S_2009; to=ice_thic)
    S_2017_2 = resample(S_2017; to=ice_thic)

    @show size(S_2009_2)
    @show size(S_2017_2)
    @show size(bed_surf)

    fig = Figure(; size=(1500, 580), fontsize=22)
    axs = (S_2009=Axis(fig[1, 1]; aspect=DataAspect(), xlabel=L"x\text{ [km]}", ylabel=L"y\text{ [km]}", title=L"S(2009)"),
        S_2017=Axis(fig[1, 2]; aspect=DataAspect(), xlabel=L"x\text{ [km]}", ylabel=L"y\text {[km]}", title=L"S(2017)"), 
        B = Axis(fig[1, 3]; aspect=DataAspect(), xlabel=L"x\text{ [km]}", ylabel=L"y\text {[km]}", title=L"B"))
    plts = (S_2009=heatmap!(axs.S_2009, S_2009_2; colormap=:turbo),
            S_2017=heatmap!(axs.S_2017, S_2017_2; colormap=:turbo),
            B = heatmap!(axs.B, xc, yc, Array(bed_surf); colormap=:turbo))
    Colorbar(fig[1, 1][1, 2], plts.S_2009)
    Colorbar(fig[1, 2][1, 2], plts.S_2017)
    Colorbar(fig[1, 3][1, 2], plts.B)
    colgap!(fig.layout, 7)
    display(fig)

    error("check")


    S_2009 = bed_surf .* (1 .- mask_2009) .+ S_2009.* mask_2009
    S_2017 = bed_surf .* (1 .- mask_2017) .+ S_2017.* mask_2017

    S_2016 = S_2009 .+ (S_2017 .- S_2009)/(2017-2009)*(2016-2009)

    H_2016 = S_2016 .- bed_surf
    H_2017 = S_2017 .- bed_surf
 

    velocity_path = joinpath(datadir, Glacier, velocity_file)

    #load vmag data 
    dataset = RasterStack(velocity_path; crs=EPSG(32632), mappedcrs=EPSG(32632))
    vmag_epsg = dataset.v2017_2018[:, :]
    vmag_epsg = replace_missing(vmag_epsg, NaN)
    vmag_epsg = Raster(vmag_epsg; dims=GlacioTools.coords_as_ranges(vmag_epsg),
                   crs=crs(vmag_epsg),
                   metadata=vmag_epsg.metadata, missingval=vmag_epsg.missingval, name=vmag_epsg.name,
                   refdims=vmag_epsg.refdims)
    vmag = resample(vmag_epsg; to=ice_thic)
    vmag = reverse(vmag; dims=2)

    #mask off

    # is this neccesary?
    vmag[ice_thic.== 0] .= NaN
    ice_thic[ice_thic .== 0] .= NaN

    B_Alet = bed_surf.data
    S_Alet_2016 = S_2016.data
    S_Alet_2017 = S_2017.data
    H_Alet_2016 = H_2016.data
    H_Alet_2017 = H_2017.data
    vmag_Alet = vmag.data 
    #Alet_offset = bed_offset.data

    oz = minimum(B_Alet)
    B_Alet .-= oz 
    S_Alet .-= oz 

    @show typeof(H_Alet_2016)
    @show typeof(vmag_Alet)

    visu_data = true
    if visu_data
        fig = Figure(; size=(1500, 580), fontsize=22)
        axs = (H_2016=Axis(fig[1, 1]; aspect=DataAspect(), xlabel=L"x\text{ [km]}", ylabel=L"y\text{ [km]}", title=L"H(2016)"),
            S_2016=Axis(fig[1, 2]; aspect=DataAspect(), xlabel=L"x\text{ [km]}", ylabel=L"y\text {[km]}", title=L"S(2016)"),
            B=Axis(fig[1, 3]; aspect=DataAspect(), xlabel=L"x\text{ [km]}", ylabel=L"y\text{ [km]}", title=L"B"),
            H_2017=Axis(fig[2, 1]; aspect=DataAspect(), xlabel=L"x\text{ [km]}", ylabel=L"y\text {[km]}", title=L"H(2017)"),
            S_2017=Axis(fig[2, 2]; aspect=DataAspect(), xlabel=L"x\text{ [km]}", ylabel=L"y\text{ [km]}", title=L"S(2017)"),
            vmag=Axis(fig[2, 3]; aspect=DataAspect(), xlabel=L"x\text{ [km]}", ylabel=L"y\text {[km]}", title=L"Vmag [m/s]"))
        plts = (H_2016=heatmap!(axs.H_2016, xc, yc, Array(H_Alet_2016); colormap=:turbo),
                S_2016=heatmap!(axs.S_2016, xc, yc, Array(S_Alet_2016); colormap=:turbo),
                B=heatmap!(axs.B, xc, yc, Array(B_Alet); colormap=:turbo),
                H_2017=heatmap!(axs.H_2017, xc, yc, Array(H_Alet_2017); colormap=:turbo),
                S_2017=heatmap!(axs.S_2017, xc, yc, Array(S_Alet_2017); colormap=:turbo),
                vmag=heatmap!(axs.vmag, xc, yc, Array(vmag); colormap=:turbo))
        # axs.H.xticksize = 18
        # axs.H.yticksize = 18
        # axs.vmag.xticksize = 18
        # axs.vmag.yticksize = 18
        Colorbar(fig[1, 1][1, 2], plts.H_2016)
        Colorbar(fig[1, 2][1, 2], plts.S_2016)
        Colorbar(fig[1, 3][1, 2], plts.B)
        Colorbar(fig[2, 1][1, 2], plts.H_2017)
        Colorbar(fig[2, 2][1, 2], plts.S_2017)
        Colorbar(fig[2, 3][1, 2], plts.vmag)
        colgap!(fig.layout, 7)
        display(fig)
    end 
    return H_Alet_2016, H_Alet_2017, S_Alet_2016, S_Alet_2017, B_Alet, vmag_Alet, oz, xc, yc
end 



