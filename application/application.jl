using Statistics
using CUDA
using JLD2

CUDA.device!(1)

include("dataset.jl")
include("inversion_steadystate.jl")
include("inversion_snapshot.jl")
include("macros.jl")
include("load_massbalance.jl")

@views function smooth_2!(A, nsm)
    for _ in 1:nsm
        @inbounds A[2:end-1, 2:end-1] .= A[2:end-1, 2:end-1] .+
                                         1.0 / 4.1 .*
                                         (diff(diff(A[:, 2:end-1]; dims=1); dims=1) .+ diff(diff(A[2:end-1, :]; dims=2); dims=2))
        @inbounds A[[1, end], :] .= A[[2, end - 1], :]
        @inbounds A[:, [1, end]] .= A[:, [2, end - 1]]
    end
    return
end

@views function smooth_masked_2!(A, D, nsm)
    for _ in 1:nsm
        @inbounds A[2:end-1, 2:end-1] .= A[2:end-1, 2:end-1] .+
                                         1.0 / 4.1 .*
                                         D .* (diff(diff(A[:, 2:end-1]; dims=1); dims=1) .+ diff(diff(A[2:end-1, :]; dims=2); dims=2))
        @inbounds A[[1, end], :] .= A[[2, end - 1], :]
        @inbounds A[:, [1, end]] .= A[:, [2, end - 1]]
    end
    return
end

function preprocessing(Glacier::AbstractString, SGI_ID::AbstractString, datadir::AbstractString)
    ALETSCH_VELOCITY_FILE = "velocity_data/ALPES_wFLAG_wKT_ANNUALv2016-2021.nc"
    H_Alet, S_Alet, B_Alet, vmag_Alet, oz_Alet, xc_Alet, yc_Alet = load_data(Glacier, SGI_ID,  datadir, ALETSCH_VELOCITY_FILE)
    @show oz_Alet
    jldsave("Aletsch.jld2"; H_Alet, S_Alet, B_Alet, vmag_Alet, oz_Alet, xc_Alet, yc_Alet)
    return
end 

#preprocessing("Aletsch", "B36-26", "/scratch-1/yilchen/Msc-Inverse-SIA/application/datasets/")

function compute_∇S!(∇S, H, B, dx, dy)
    @get_indices
    @inbounds if ix <= size(∇S, 1) && iy <= size(∇S, 2)
        ∇Sx = 0.5 *
              ((B[ix + 1, iy + 1] - B[ix, iy + 1]) / dx +
               (H[ix + 1, iy + 1] - H[ix, iy + 1]) / dx +
               (B[ix + 1, iy] - B[ix, iy]) / dx +
               (H[ix + 1, iy] - H[ix, iy]) / dx)
        ∇Sy = 0.5 *
              ((B[ix + 1, iy + 1] - B[ix + 1, iy]) / dy +
               (H[ix + 1, iy + 1] - H[ix + 1, iy]) / dy +
               (B[ix, iy + 1] - B[ix, iy]) / dy +
               (H[ix, iy + 1] - H[ix, iy]) / dy)
        
        if H[ix, iy  ] == 0.0 && H[ix+1, iy  ] == 0.0 &&
           H[ix, iy+1] == 0.0 && H[ix+1, iy+1] == 0.0
            ∇S[ix, iy] = NaN
        else
            ∇S[ix, iy] = sqrt(∇Sx^2 + ∇Sy^2)
        end
    end
    return
end


function application()
    H_Alet, S_Alet, B_Alet, vmag_Alet, oz_Alet, xc_Alet, yc_Alet = load("Aletsch.jld2", "H_Alet", "S_Alet", "B_Alet", "vmag_Alet", "oz_Alet", "xc_Alet", "yc_Alet")
    # load the data 
    vmag_Alet ./= 365*24*3600 #m/s
    nx = size(H_Alet)[1]
    ny = size(H_Alet)[2]

    replace!(H_Alet, NaN => 0)
    nsm_topo = 100
    #bedrock smooth

    D_reg = H_Alet[2:end-1, 2:end-1] ./ maximum(H_Alet)

    smooth_masked_2!(H_Alet, D_reg, nsm_topo)
    @info "Smoothed the ice thickness"
    smooth_masked_2!(B_Alet, D_reg, nsm_topo)
    @info "Smoothed the bedrock"


    S_Alet .= B_Alet .+ H_Alet

    #load the mass balance data: m/s, m, 1/s
    b_max_Alet, ELA_Alet, β_Alet = load_massbalance()
    ELA_Alet -= oz_Alet

    M           = min.(β_Alet.*(H_Alet .+ B_Alet .- ELA_Alet), b_max_Alet)
    Mask        = ones(Float64, size(H_Alet))
    Mask[M.>0.0 .&& H_Alet.<=0.0] .= 0.0

    check_1_beforescaling = true

    if check_1_beforescaling == true
        fig = Figure(; size=(1000, 580), fontsize=22)
        axs = (H=Axis(fig[1, 1]; aspect=DataAspect(), xlabel=L"x\text{ [km]}", ylabel=L"y\text{ [km]}", title=L"H_{Alet} before scaling [m]"),
            vmag=Axis(fig[1, 2]; aspect=DataAspect(), xlabel=L"x\text{ [km]}", ylabel=L"y\text {[km]}", title=L"vmag_{Alet} before scaling [m/s]"))
        plts = (H=heatmap!(axs.H, xc_Alet, yc_Alet, Array(S_Alet .- B_Alet); colormap=:turbo),
                vmag=heatmap!(axs.vmag, xc_Alet, yc_Alet, vmag_Alet; colormap=:turbo))
        axs.H.xticksize = 18
        axs.H.yticksize = 18
        axs.vmag.xticksize = 18
        axs.vmag.yticksize = 18
        Colorbar(fig[1, 1][1, 2], plts.H)
        Colorbar(fig[1, 2][1, 2], plts.vmag)
        colgap!(fig.layout, 7)
        display(fig)
    end


    #real scale
    lsc_data = filter(!isnan, H_Alet) |> mean
    ρ = 910 #kg/m^3
    g = 9.81 #m/s^2
    A = 5e-26
    npow = 3
    #TODO 
    aρgn0_data = A * (ρ * g)^npow

    tsc_data = 1 / aρgn0_data / lsc_data^npow
    vsc_data = lsc_data / tsc_data

    #rescale
    lsc   = 1.0
    aρgn0 = 1.0

    tsc = 1 / aρgn0 / lsc^npow
    vsc = lsc / tsc

    #rescaling 
    # here we do not use H_Alet is because yeah H_Alet has many NaN
    H = max.(0.0, (S_Alet .- B_Alet)) ./ lsc_data .* lsc
    B = B_Alet ./ lsc_data .* lsc
    S = S_Alet ./ lsc_data .* lsc
    vmag = vmag_Alet ./ vsc_data .* vsc
    b_max = b_max_Alet ./ vsc_data .* vsc
    ELA   = ELA_Alet ./ lsc_data .* lsc
    β     = β_Alet .* tsc_data ./ tsc

    xc = xc_Alet ./ lsc_data .* lsc
    yc = yc_Alet ./ lsc_data .* lsc
    qmag = replace(vmag[2:end-1, 2:end-1], NaN => 0.0) .* H[2:end-1, 2:end-1]

    check_scaling = true
    if check_scaling
        fig = Figure(; size=(1000, 580), fontsize=22)
        axs = (H=Axis(fig[1, 1]; aspect=DataAspect(), xlabel=L"x", ylabel=L"y", title=L"H"),
            vmag=Axis(fig[1, 2]; aspect=DataAspect(), xlabel=L"x", ylabel=L"y", title=L"vmag"))
        plts = (H=heatmap!(axs.H, xc, yc, H; colormap=:turbo),
                vmag=heatmap!(axs.vmag, xc, yc, replace(vmag[2:end-1, 2:end-1], NaN => 0.0); colormap=:turbo))
        axs.H.xticksize = 18
        axs.H.yticksize = 18
        axs.vmag.xticksize = 18
        axs.vmag.yticksize = 18
        Colorbar(fig[1, 1][1, 2], plts.H)
        Colorbar(fig[1, 2][1, 2], plts.vmag)
        colgap!(fig.layout, 7)
        display(fig)
    end 

    #non-dimensional numbers
    s_f      = 1e3#1e-5 #as/a/lsc^2
    β_nd     = 0.12887130806152247 # β*tsc 
    b_max_nd = 0.4737783229663921 # b_max /lsc * tsc
    z_ela_l  = 50.244062744240225 # ELA / lsc 
    #non-dimensional nuemrics numbers 
    H_cut_l = 1.0e-2 #H_cut / lsc
    γ_nd = 1.0e-2 # γ0 / lsc^(2-2npow) * tsc^2

    asρgn0 = s_f * aρgn0 * lsc^2
    β = β_nd / tsc #0.12887130806152247
    ELA = z_ela_l * lsc #50.244062744240225
    b_max = b_max_nd * lsc / tsc #0.4737783229663921

    #numerics
    H_cut = H_cut_l * lsc #1.0e-2
    γ0 = γ_nd * lsc^(2 - 2npow) * tsc^(-2) # 1.0e-2

    ϵtol = (abs=1e-6, rel=1e-6)
    ϵtol_adj = 1e-8
    maxiter = 5 * nx^2
    Δγ = 0.25 #0.5 #0.25
    ngd = 500
    w_H_1, w_q_1 = 1.0, 0.0
    w_H_2, w_q_2 = 0.0, 1.0

    H_ini               = copy(H)
    As_ini              = asρgn0 * CUDA.ones(nx - 1, ny - 1)
    logAs_H_steadystate = log10.(As_ini)
    logAs_q_steadystate = log10.(As_ini)
    logAs_q_snapshot    = log10.(As_ini)

    H = CuArray(H)
    S = CuArray(S)
    B = CuArray(B)
    H_ini = CuArray(H_ini)
    H_obs = copy(H_ini)
    qmag = CuArray(qmag)
    qobs_mag = CuArray(qmag)
    β = CUDA.fill(β, nx, ny)
    ELA = CUDA.fill(ELA, nx, ny)
    mask = CuArray(Mask)

    check_qobs_mag = false

    if check_qobs_mag
        fig = Figure(; size=(1000, 580), fontsize=22)
        axs = (qobs_mag    = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel=L"x", ylabel=L"y", title=L"qobs_mag"),
               qmag  = Axis(fig[1, 2][1,1]; aspect=DataAspect(), xlabel=L"qmag"))
        plts = (qobs_mag=heatmap!(axs.qobs_mag, xc[2:end-1], yc[2:end-1], Array(qobs_mag); colormap=:turbo),
                qmag=heatmap!(axs.qmag, xc[2:end-1], yc[2:end-1], Array(qmag); colormap=:turbo))
        Colorbar(fig[1, 1][1, 2], plts.qobs_mag)
        Colorbar(fig[1, 2][1, 2], plts.qmag)
        colgap!(fig.layout, 7)
        display(fig)
    end 

    # pack 
    geometry = (; B, xc, yc)
    observed = (; H_obs, qobs_mag, mask)
    initial = (; H_ini, As_ini, qmag)
    physics = (; npow, aρgn0, β, ELA, b_max, H_cut, γ0)
    weights_H = (; w_H_1, w_q_1)
    weights_q = (; w_H_2, w_q_2)
    numerics = (; vsc, ϵtol, ϵtol_adj, maxiter)
    optim_params = (; Δγ, ngd)

    # run 3 inversions 
    
    #inversion_snapshot(logAs_q_snapshot, geometry, observed, initial, physics, numerics, optim_params)
    
    inversion_steadystate(logAs_H_steadystate, geometry, observed, initial, physics, weights_H, weights_q, numerics, optim_params; do_vis=true, do_thickness=true)

    # inversion_steadystate(logAs_q_steadystate, geometry, observed, initial, physics, weights_H, weights_q, numerics, optim_params; do_vis=false, do_thickness=false)

    # inversion_snapshot(logAs_q_snapshot, geometry, observed, initial, physics, numerics, optim_params)

    return
end

application()

