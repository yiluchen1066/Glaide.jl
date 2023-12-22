@views function load_data(bed_dat, surf_dat)
    z_bed  = reverse(dropdims(Raster(bed_dat); dims=3); dims=2)
    z_surf = reverse(dropdims(Raster(surf_dat); dims=3); dims=2)
    xy     = DimPoints(dims(z_bed, (X, Y)))
    (x, y) = (first.(xy), last.(xy))
    return z_bed.data, z_surf.data, x.data[:, 1], y.data[1, :]
end

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

function rhone_application()
    # load the data 
    B_rhone, S_rhone, xc_rhone, yc_rhone = load_data("datasets/Rhone/alps/Rhone_BedElev_cr.tif",
                                         "datasets/Rhone/alps/Rhone_SurfElev_cr.tif")
    # TODO load the vmag (nx-2,ny-2)

    




    H_rhone .= S_rhone .- B_rhone

    lsc_data = mean(H_rhone)
    ρ = 910
    g = 9.81
    A = 1.9e-24
    npow = 3
    aρgn0_data = A * (ρ*g)^npow

    tsc_data = 1 / aρgn0_data / lsc_data^npow
    vsc_data = lsc_data/tsc_data


    lsc   = 1.0
    aρgn0 = 1.0

    tsc = 1 / aρgn0 / lsc^npow
    vsc = lsc / tsc

    #rescaling 
    H = H_rhone / lsc_data * lsc
    B = B_rhone / lsc_data * lsc
    S = S_rhone / lsc_data * lsc
    vmag = vmag_rhone / vsc_data * vsc
    qmag = vmag .* H
    xc  = xc_rhone / lsc_data * lsc
    yc = yc_rhone / lsc_data * lsc

    #non-dimensional numbers
    s_f      = 0.01 #as/a/lsc^2
    β_nd     = 1.12e-7
    b_max_nd = 2.70e-8
    z_ela_l  = 2.93

    H_cut_l  = 1.0e-6

    asρgn0 = s_f * aρgn0 * lsc^2
    β = β_nd / tsc
    ELA = z_ela_l * lsc
    b_max = b_max_nd * l / tsc

    #numerics
    H_cut = H_cut_l * lsc
    ϵtol = (abs=1e-8, rel=1e-8)
    ϵtol_adj = 1e-8
    maxiter = 5 * nx^2
    Δγ = 0.2
    ngd = 100
    w_H_1, w_q_1 = 1.0, 0.0
    w_H_2, w_q_2 = 0.0, 1.0

    H_ini = copy(H)
    As_ini = asρgn0 * CUDA.ones(nx - 1, ny - 1)
    logAs_H_steadystate  = log10.(As_ini)
    logAs_q_steadystate  = log10.(As_ini)
    logAs_q_snapshot     = log10.(As_ini)

    # pack 
    geometry = (; B, xc, yc)
    observed = (; H, S, qmag)
    initial = (; H_ini, As_ini)
    physics = (; npow, aρgn0, β, ELA, b_max, H_cut)
    weights_H = (; w_H_1, w_q_1)
    weights_q = (; w_H_2, w_q_2)
    numerics = (; ϵtol, ϵtol_adj, maxiter)
    optim_params = (; Δγ, ngd)

    # run 3 inversions 
    inversion_steadystate(logAs_H_steadystate, geometry, observed, initial, physics, weights_H, numerics, optim_params; do_vis=false)

    inversion_steadystate(logAs_q_steadystate, geometry, observed, initial, physics, weights_q, numerics, optim_params; do_vis=false)

    inversion_snapshot(logAs_q_snapshot, geometry, observed, initial, physics, numerics, optim_params)

    #visualization for As_H_steady As_flux_steady As_snapshot

    return
end

