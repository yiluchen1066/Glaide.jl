using Statistics
using CUDA

include("dataset.jl")
include("inversion_steadystate.jl")
include("inversion_snapshot.jl")

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

function application()
    #init visu
    fig = Figure(; size=(1000, 580), fontsize=22)
    data_visu = (; fig)

    # load the data 
    H_Alet, S_Alet, B_Alet, vmag_Alet, xc_Alet, yc_Alet = load_data("Aletsch", "B36-26", "datasets/Aletsch"; visu=data_visu)
    nx = size(H_Alet)[1]
    ny = size(H_Alet)[2]

    #real scale
    lsc_data = mean(H_Alet)
    ρ = 910
    g = 9.81
    A = 1.9e-24
    npow = 3
    aρgn0_data = A * (ρ * g)^npow

    tsc_data = 1 / aρgn0_data / lsc_data^npow
    vsc_data = lsc_data / tsc_data

    #rescale
    lsc   = 1.0
    aρgn0 = 1.0

    tsc = 1 / aρgn0 / lsc^npow
    vsc = lsc / tsc

    #rescaling 
    H = H_Alet ./ lsc_data .* lsc
    B = B_Alet ./ lsc_data .* lsc
    S = S_Alet ./ lsc_data .* lsc
    vmag = vmag_Alet ./ vsc_data .* vsc

    xc = xc_Alet ./ lsc_data .* lsc
    yc = yc_Alet ./ lsc_data .* lsc
    # here vmag should still in the size of (nx, ny)
    qmag = vmag .* H

    #non-dimensional numbers
    s_f      = 0.01 #as/a/lsc^2
    β_nd     = 1.12e-7
    b_max_nd = 2.70e-8
    z_ela_l  = 2.93

    H_cut_l = 1.0e-6
    γ_nd = 1e2

    asρgn0 = s_f * aρgn0 * lsc^2
    β = β_nd / tsc
    ELA = z_ela_l * lsc
    b_max = b_max_nd * lsc / tsc

    #numerics
    H_cut = H_cut_l * lsc
    γ0 = γ_nd * lsc^(2 - 2npow) * tsc^(-2)
    ϵtol = (abs=1e-8, rel=1e-8)
    ϵtol_adj = 1e-8
    maxiter = 5 * nx^2
    Δγ = 0.2
    ngd = 100
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

    # pack 
    geometry = (; B, xc, yc)
    observed = (; H_obs, qobs_mag)
    initial = (; H_ini, As_ini)
    physics = (; npow, aρgn0, β, ELA, b_max, H_cut, γ0)
    weights_H = (; w_H_1, w_q_1)
    weights_q = (; w_H_2, w_q_2)
    numerics = (; ϵtol, ϵtol_adj, maxiter)
    optim_params = (; Δγ, ngd)

    # run 3 inversions 
    
    #inversion_snapshot(logAs_q_snapshot, geometry, observed, initial, physics, numerics, optim_params)
    
    inversion_steadystate(logAs_H_steadystate, geometry, observed, initial, physics, weights_H, weights_q, numerics, optim_params; do_vis=false, do_thickness=true)

    # error("check")
    # inversion_steadystate(logAs_q_steadystate, geometry, observed, initial, physics, weights_H, weights_q, numerics, optim_params; do_vis=false, do_thickness=false)

    # inversion_snapshot(logAs_q_snapshot, geometry, observed, initial, physics, numerics, optim_params)

    return
end

application()
