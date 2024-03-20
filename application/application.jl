using Statistics
using CUDA
using JLD2
using Rasters
using ArchGDAL
using NCDatasets

CUDA.device!(5)
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
    stack = RasterStack("aletsch_data_2016_2017.nc")
    stack = replace_missing(stack, NaN)

    xr, yr = extrema.(dims(stack))
    lx, ly = xr[2] - xr[1], yr[2] - yr[1]  

    aspect_ratio = ly/lx 

    nx = 128*2
    ny = ceil(Int, nx*aspect_ratio)
    stack = resample(stack; size=(nx, ny))

    xy = DimPoints(dims(stack.bedrock, (X, Y)))
    (x, y) = (first.(xy), last.(xy))
    xc = x.data[:, 1]
    yc = y.data[1, :]
    yc = reverse(yc; dims=1)

    vmag_obs   = stack.velocity_2016_2017.data
    B          = stack.bedrock.data
    S_old      = stack.surface_2016.data
    S_obs      = stack.surface_2017.data
    vmag_obs ./= 365*24*3600
    H_obs       = S_obs .- B
    H_old       = S_old .- B

    oz          = minimum(B)
    B         .-= oz

    vmag_obs    = reverse(vmag_obs; dims=2)
    B           = reverse(B; dims=2)
    S_old       = reverse(S_old; dims=2)
    S_obs       = reverse(S_obs; dims=2)
    H_obs       = reverse(H_obs; dims=2)
    H_old       = reverse(H_old; dims=2)

    #dt             # 1 year in seconds

    nsm_topo = 100
    #bedrock smooth
    D_reg = H_old[2:end-1, 2:end-1] ./ maximum(H_old)

    smooth_masked_2!(H_old, D_reg, nsm_topo)
    smooth_masked_2!(H_obs, D_reg, nsm_topo)
    @info "Smoothed the ice thickness"
    smooth_masked_2!(B, D_reg, nsm_topo)
    @info "Smoothed the bedrock"

    check_data = true 
    if check_data 
        fig = Figure(size=(800, 800))
        ax1 = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="Bed elevation [m.a.s.l.]")
        ax2 = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="Surface velocity (2016-2017) [m/y]")
        ax3 = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="Ice thickness (2016) [m.a.s.l.]")
        ax4 = Axis(fig[2, 2][1, 1]; aspect=DataAspect(), title="Ice thickness (2017) [m.a.s.l.]")

        hm1 = heatmap!(ax1, B; colormap=:terrain)
        hm2 = heatmap!(ax2, vmag_obs; colormap=:turbo)
        hm3 = heatmap!(ax3, H_old; colormap=:magma)
        hm4 = heatmap!(ax4, H_obs; colormap=:magma)

        Colorbar(fig[1, 1][1, 2], hm1)
        Colorbar(fig[1, 2][1, 2], hm2)
        Colorbar(fig[2, 1][1, 2], hm3)
        Colorbar(fig[2, 2][1, 2], hm4)

        display(fig)
    end

    #load the mass balance data: m/s, m, 1/s
    b_max_Alet, ELA_Alet, β_Alet = load_massbalance()
    ELA_Alet -= oz

    M           = min.(β_Alet.*(H_old .+ B .- ELA_Alet), b_max_Alet)
    Mask        = ones(Float64, size(H_old))
    Mask[M.>0.0 .&& H_old.<=0.0] .= 0.0

    #real scale
    lsc_data = mean(H_old)
    ρ = 910 #kg/m^3
    g = 9.81 #m/s^2
    A = 5e-26
    npow = 3
    dt   = 365*24*3600
    # TODO
    aρgn0_data = A * (ρ * g)^npow

    tsc_data = 1 / aρgn0_data / lsc_data^npow
    vsc_data = lsc_data / tsc_data

    #rescale
    lsc   = 1.0
    aρgn0 = 1.0

    tsc = 1 / aρgn0 / lsc^npow
    vsc = lsc / tsc

    #rescaling 
    H_old = H_old ./ lsc_data .* lsc
    H_obs = H_obs ./ lsc_data .* lsc
    B = B ./ lsc_data .* lsc
    S_old = S_old ./ lsc_data .* lsc
    S_obs = S_obs ./ lsc_data .* lsc
    vmag_obs = vmag_obs ./ vsc_data .* vsc
    b_max = b_max_Alet ./ vsc_data .* vsc
    ELA   = ELA_Alet ./ lsc_data .* lsc
    β     = β_Alet .* tsc_data ./ tsc
    xc       = xc ./ lsc_data .* lsc
    yc       = yc ./ lsc_data .* lsc
    dt       = dt  /tsc_data * tsc
    
    #dt       = dt /tsc_data*tsc
    qmag_obs = replace(vmag_obs[2:end-1, 2:end-1], NaN => 0.0) .* H_obs[2:end-1, 2:end-1]
    @show size(qmag_obs)
    check_data = true 
    if check_data 
        fig = Figure(size=(800, 800))
        ax1 = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="Bed elevation [m.a.s.l.] after rescaling")
        ax2 = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="Surface flux (2016-2017) after rescaling [m/y]")
        ax3 = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="Ice thickness (2016) after rescaling [m.a.s.l.]")
        ax4 = Axis(fig[2, 2][1, 1]; aspect=DataAspect(), title="Ice thickness (2017) after rescaling [m.a.s.l.]")

        hm1 = heatmap!(ax1, B; colormap=:terrain)
        hm2 = heatmap!(ax2, qmag_obs; colormap=:turbo)
        hm3 = heatmap!(ax3, H_old; colormap=:magma)
        hm4 = heatmap!(ax4, H_obs; colormap=:magma)

        Colorbar(fig[1, 1][1, 2], hm1)
        Colorbar(fig[1, 2][1, 2], hm2)
        Colorbar(fig[2, 1][1, 2], hm3)
        Colorbar(fig[2, 2][1, 2], hm4)

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
    Δγ = 0.1#0.25 #0.5 #0.25
    ngd = 500
    w_H_1, w_q_1 = 0.5, 0.5
    w_H_2, w_q_2 = 0.0, 1.0

    
    B           = CuArray(B)
    S_ini       = CuArray(S_old)
    S_obs       = CuArray(S_obs)
    H_obs       = CuArray(H_obs)
    H_ini       = CuArray(H_old)
    qmag_obs    = CuArray(qmag_obs)
    qmag        = CUDA.zeros(Float64,size(qmag_obs))
    vmag_obs    = CuArray(vmag_obs)
    vmag        = CUDA.zeros(Float64, size(vmag_obs))
    As_ini      = asρgn0 * CUDA.ones(nx-1, ny-1)
    logAs_steadystate = log10.(As_ini)
    logAs_snapshot   = log10.(As_ini)

    # okay now let's check where do I define qmag and the size of qmag

    # so here H and qmag are what are passed into model to solve, 
    # and for now I just want to have 0.5 0.5 weights
    #β = CUDA.fill(β, nx, ny)
    #ELA = CUDA.fill(ELA, nx, ny)
    mask = CuArray(Mask)

    # pack 
    geometry = (; B, xc, yc, nx, ny)
    observed = (; H_obs, qmag_obs, vmag_obs, mask)
    initial = (; H_ini, S_ini, As_ini, qmag, vmag)
    physics = (; npow, dt, aρgn0, β, ELA, b_max, H_cut, γ0)
    weights_H = (; w_H_1, w_q_1)
    weights_q = (; w_H_2, w_q_2)
    numerics = (; vsc, ϵtol, ϵtol_adj, maxiter)
    optim_params = (; Δγ, ngd)

    # run 3 inversions 
    
    inversion_snapshot(logAs_snapshot, geometry, observed, initial, physics, numerics, optim_params)
    
    #inversion_steadystate(logAs_steadystate, geometry, observed, initial, physics, weights_H, weights_q, numerics, optim_params; do_vis=true, do_thickness=true)

    # inversion_steadystate(logAs_q_steadystate, geometry, observed, initial, physics, weights_H, weights_q, numerics, optim_params; do_vis=false, do_thickness=false)

    # inversion_snapshot(logAs_q_snapshot, geometry, observed, initial, physics, numerics, optim_params)

    return
end

application()

