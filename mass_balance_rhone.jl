using CUDA, BenchmarkTools 
using Plots, Plots.Measures, Printf
using DelimitedFiles, DataFrame 
using HDF5 

function interpolate_mb!(H, B, zs_sample, dz_sample, ms_sample, mb, nx, ny)
    @get_thread_idx(H)
    if ix <= nx-2 && iy <= ny-2 
        iz_f = clamp((H[ix+1, iy+1]+B[ix+1,iy+1]-zs_sample[1])/dz_sample, 0.0, Float64(length(ms_sample)-2))
        iz = floor(Int64, iz_f)+1
        f = iz_f - (iz-1) 
        mb[ix+1, iy+1] = ms_sample[iz]*(1.0-f) + ms_sample[iz+1]*f 
    end 
    return 
end 

function main()
    # load Rhone data: surface elevation and bedrock 

    rhone_data = h5open("Rhone_data_padding/alps/data_Rhone.h5", "r")
    xc = rhone_data["glacier/x"][:,1]
    yc = rhone_data["glacier/y"][1,:]
    B  = rhone_data["glacier/z_bed"][]
    S  = rhone_data["glacier/z_surf"][]

    close(rhone_data) 
    

    return 
end 

main()