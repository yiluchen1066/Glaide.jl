using CairoMakie
using JLD2
using Rasters
using JLD2 
using ArchGDAL
using NCDatasets
using Statistics

include("macros.jl")
include("sia_forward_flux_implicit.jl")
include("sia_adjoint_flux_2D.jl")
include("sia_loss_flux_2D.jl")
include("load_massbalance.jl")

CUDA.allowscalar(false)

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

function adjoint_2D()
    # data loading and preprocessing 
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
    #bedrock, surface_2016, surface_2017, velocity_2016_2017, velocity_2017_2018
    vmag_old   = stack.velocity_2016_2017.data
    vmag_obs   = stack.velocity_2017_2018.data
    B          = stack.bedrock.data
    S_old      = stack.surface_2016.data
    S_obs      = stack.surface_2017.data
    S_2023     = stack.surface_2023.data
    vmag_old ./= 365*24*3600
    vmag_obs ./= 365*24*3600
    H_obs      = S_obs .- B
    H_old      = S_old .- B
    H_2023     = S_2023 .- B

    oz          = minimum(B)
    B         .-= oz

    vmag_old    = reverse(vmag_old; dims=2)
    vmag_obs    = reverse(vmag_obs; dims=2)
    B           = reverse(B; dims=2)
    S_old       = reverse(S_old; dims=2)
    S_obs       = reverse(S_obs; dims=2)
    S_2023      = reverse(S_2023; dims=2)
    H_obs       = reverse(H_obs; dims=2)
    H_old       = reverse(H_old; dims=2)
    H_2023      = reverse(H_2023; dims=2)

    nsm_topo = 100
    #bedrock smooth
    D_reg = H_old[2:end-1, 2:end-1] ./ maximum(H_old)

    smooth_masked_2!(H_old, D_reg, nsm_topo)
    smooth_masked_2!(H_obs, D_reg, nsm_topo)
    smooth_masked_2!(S_2023, D_reg, nsm_topo)
    smooth_masked_2!(H_2023, D_reg, nsm_topo)
    @info "Smoothed the ice thickness"
    smooth_masked_2!(B, D_reg, nsm_topo)
    @info "Smoothed the bedrock"

    crange = filter(!isnan, H_old) |> extrema

    check_data = true
    if check_data 
        fig = Figure(size=(800, 800))
        ax1 = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="Bed elevation [m.a.s.l.]")
        ax2 = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="Ice thickness (2023) [m.a.s.l.]")
        ax3 = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="Ice thickness (2016) [m.a.s.l.]")
        ax4 = Axis(fig[2, 2][1, 1]; aspect=DataAspect(), title="Ice thickness (2017) [m.a.s.l.]")

        hm1 = heatmap!(ax1, B; colormap=:terrain)
        hm2 = heatmap!(ax2, H_2023; colormap=:magma, colorrange=crange)
        hm3 = heatmap!(ax3, H_old; colormap=:magma, colorrange=crange)
        hm4 = heatmap!(ax4, H_obs; colormap=:magma, colorrange=crange)

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
    Mask[M .>0.0 .&& H_old.<= 0] .= 0.0

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

    nratio = (npow+1)/(npow+2)

    #rescale
    lsc   = 1.0
    aρgn0 = 1.0

    tsc = 1 / aρgn0 / lsc^npow
    vsc = lsc / tsc

    #rescaling 
    H_old = H_old ./ lsc_data .* lsc
    H_obs = H_obs ./ lsc_data .* lsc
    H_2023 = H_2023 ./ lsc_data .* lsc
    B = B ./ lsc_data .* lsc
    S_old = S_old ./ lsc_data .* lsc
    S_obs = S_obs ./ lsc_data .* lsc
    S_2023 = S_2023 ./ lsc_data .* lsc
    vmag_obs = vmag_obs ./ vsc_data .* vsc
    b_max = b_max_Alet ./ vsc_data .* vsc
    ELA   = ELA_Alet ./ lsc_data .* lsc
    β0    = β_Alet .* tsc_data ./ tsc
    xc       = xc ./ lsc_data .* lsc
    yc       = yc ./ lsc_data .* lsc
    dt       = dt  /tsc_data * tsc
    
    #dt       = dt /tsc_data*tsc
    qmag_obs = replace(vmag_obs[2:end-1, 2:end-1], NaN => 0.0) .* H_obs[2:end-1, 2:end-1] .* nratio

    # γ_nd     = 1e2
    s_f      = 1e3
    dt_nd    = 2.131382577069803e8   * 5e0
    t_total_nd = 2.131382577069803e8 * 5e0
    H_cut_l = 1.0e-6
    asρgn0           = s_f * aρgn0 * lsc^2 #5.0e-18*(ρg)^n = 3.54627498316e-6
    H_cut            = H_cut_l * lsc # 1.0e-2
    # γ0               = γ_nd * lsc^(2 - 2npow) * tsc^(-2) #1.0e-2
    dt               = dt_nd * tsc # 365*24*3600
    t_total          = t_total_nd * tsc

    ## numerics
    ϵtol       = (abs=1e-6, rel=1e-6)
    maxiter    = 5 * nx^2
    ncheck     = ceil(Int, 0.25 * nx^2)
    nthreads   = (16, 16)
    nblocks    = ceil.(Int, (nx, ny) ./ nthreads)
    ϵtol_adj   = 1e-8
    ncheck_adj = ceil(Int, 0.25 * nx^2)
    ngd        = 50
    bt_niter   = 5
    Δγ         = 1.0e-1

    w_H_nd = 0.5*sqrt(2)
    w_q_nd = 0.5*sqrt(2)

    ## pre-processing
    dx, dy = lx / nx, ly / ny
    xc_1 = xc[1:(end-1)]
    yc_1 = yc[1:(end-1)]

    w_H = w_H_nd/sum(H_obs .^ 2)
    w_q = w_q_nd/sum(qmag_obs .^ 2)

    D         = CUDA.zeros(Float64, nx - 1, ny - 1)
    qHx       = CUDA.zeros(Float64, nx - 1, ny - 2)
    qHy       = CUDA.zeros(Float64, nx - 2, ny - 1)
    qHx_obs   = CUDA.zeros(Float64, nx - 1, ny - 2)
    qHy_obs   = CUDA.zeros(Float64, nx - 2, ny - 1)

    As        = CUDA.fill(asρgn0, nx - 1, ny - 1)
    RH        = CUDA.zeros(Float64, nx, ny)
    Err_rel   = CUDA.zeros(Float64, nx, ny)
    Err_abs   = CUDA.zeros(Float64, nx, ny)
    As_ini    = copy(As)
    logAs     = copy(As)
    logAs_ini = copy(As)
    Lap_As    = copy(As)
    β         = CUDA.fill(β0, nx, ny)
    ELA       = CUDA.fill(ELA, nx, ny)
    mask      = CuArray(Mask)
    mb        = CuArray(M)
    #init adjoint storage
    q̄Hx = CUDA.zeros(Float64, nx - 1, ny - 2)
    q̄Hy = CUDA.zeros(Float64, nx - 2, ny - 1)
    D̄ = CUDA.zeros(Float64, nx - 1, ny - 1)
    H̄ = CUDA.zeros(Float64, nx, ny)
    R̄H = CUDA.zeros(Float64, nx, ny)
    Ās = CUDA.zeros(Float64, nx - 1, ny - 1)
    logĀs = CUDA.zeros(Float64, nx - 1, ny - 1)
    Tmp = CUDA.zeros(Float64, nx - 1, ny - 1)
    ψ_H = CUDA.zeros(Float64, nx, ny)
    ∂J_∂H = CUDA.zeros(Float64, nx, ny)
    ∂J_∂qx = CUDA.zeros(Float64, nx - 1, ny - 2)
    ∂J_∂qy = CUDA.zeros(Float64, nx - 2, ny - 1)
    qmag = CUDA.zeros(Float64, nx - 2, ny - 2)

    B    = CuArray(B)
    H_old = CuArray(H_old)
    H_obs    = CuArray(H_obs)
    qmag_obs = CuArray(qmag_obs)
    H        = copy(H_old)

    cost_evo = Float64[]
    iter_evo = Float64[]

    #pack parameters
    fwd_params = (fields           = (; H, H_old, B, β, ELA, D, qHx, qHy, As, RH, qmag, Err_rel, Err_abs),
                  scalars         = (; aρgn0, b_max, npow),
                  numerical_params = (; nx, ny, dx, dy, dt, t_total, maxiter, ncheck, ϵtol),
                  launch_config    = (; nthreads, nblocks))

    # fwd_visu = (; plts, fig)

    adj_params = (fields=(; q̄Hx, q̄Hy, D̄, R̄H, Ās, ψ_H, H̄),
                  numerical_params=(; ϵtol_adj, ncheck_adj, H_cut))

    loss_params = (fields=(; H_obs, qHx_obs, qHy_obs, qmag_obs, ∂J_∂H, ∂J_∂qx, ∂J_∂qy, Lap_As),
                   scalars=(; w_H, w_q))

    #this is to switch on/off regularization of the sensitivity 
    reg = (; nsm=5, α=5e-6, Tmp)
    
    logAs     = log10.(As)
    logAs_ini = log10.(As_ini)
    
    # setup visualisation
    begin
        #init visualization 
        fig = Figure(; size=(800, 400), fontsize=14)
        ax  = (
        q_s   = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="observed log |q|"),
        q_i   = Axis(fig[1, 2]; aspect=DataAspect(), xlabel="x", title="modeled log |q|"),
        As_i  = Axis(fig[2, 1]; aspect=DataAspect(), title="modeled log As"),
        conv  = Axis(fig[2, 2]; xlabel="#iter", ylabel="J/J₀", yscale=log10, title="convergence"))
        
        xlims!(ax.conv, 0, ngd+1)
        ylims!(ax.conv, 1e-4, 1e0)
        #xlims CairoMakie.xlims!()
        #ylims 

        nan_to_zero(x) = isnan(x) ? zero(x) : x


        idc_inn = findall(H[2:end-1,2:end-1] .≈ 0.0) |> Array

        H_vert = @. 0.25 * (H[1:end-1, 1:end-1] + H[2:end,1:end-1] + H[1:end-1,2:end] + H[2:end,2:end])
        idv = findall(H_vert .≈ 0.0) |> Array

        As_v = Array(logAs)
        As_v[idv] .= NaN

        qmag_obs_v = Array(qmag_obs)
        qmag_obs_v[idc_inn] .= NaN

        qmag_v = Array(qmag)
        qmag_v[idc_inn] .= NaN

        As_crange = filter(!isnan, As_v) |> extrema
        q_crange = filter(!isnan, qmag_obs_v) |> extrema

        plts = (
                q_s   = heatmap!(ax.q_s, xc[2:end-1], yc[2:end-1], qmag_obs_v; colormap=:turbo, colorrange=q_crange),
                q_i   = heatmap!(ax.q_i, xc[2:end-1], yc[2:end-1], qmag_v; colormap=:turbo, colorrange=q_crange),
                As_i  = (heatmap!(ax.As_i, xc_1, yc_1, As_v; colormap=:turbo, colorrange=As_crange),
                contour!(ax.As_i, xc, yc, Array(H); levels=0.001:0.001, color=:white, linestyle=:dash),
                contour!(ax.As_i, xc, yc, Array(H_obs); levels=0.001:0.001, color=:red)),
                conv  = scatterlines!(ax.conv, Point2.(iter_evo, cost_evo); linewidth=2))

        #lg = axislegend(ax.slice; labelsize=10, rowgap=-5, height=40)
        
        # Colorbar(fig[1, 1][1, 2], plts.As_s[1])
        # Colorbar(fig[1, 2][1, 2], plts.As_i[1])
        # Colorbar(fig[2,1][1,2], plts.q_s)
        # Colorbar(fig[2,2][1,2], plts.q_i)
        # display(fig)
    end


    As    .= As_ini
    logAs .= log10.(As)
    fwd_visu =(; plts, fig)
    #Define loss functions 
    J(_logAs) = loss(logAs, fwd_params, loss_params;  visu=fwd_visu)
    ∇J!(_logĀs, _logAs) = ∇loss!(logĀs, logAs, fwd_params, adj_params, loss_params;reg)
    @info "inversion for As"

    
    # γ = γ0
    J_old = 0.0
    J_new = 0.0
    #solve for H with As and compute J_old
    H .= 0.0

    J_old = J(logAs)
    J_ini = J_old

    #ispath("output_steadystate") && rm("output_steadystate", recursive=true)
    #mkdir("output_steadystate")
    #jldsave("output_steadystate/static.jld2"; qmag_obs, H_obs, As_ini, As_syn, xc, yc, xv, yv)

    iframe = 1
    @info "Gradient descent - inversion for As"
    for igd in 1:ngd
        #As_ini .= As
        println("GD iteration $igd \n")
        ∇J!(logĀs, logAs)
        γ = Δγ / maximum(abs.(logĀs))
        # γ = min(γ, 1 / reg.α)
        @. logAs -= γ * logĀs

        push!(iter_evo, igd)
        push!(cost_evo, J(logAs)/J_ini)
        #visualization 

        @printf " min(As) = %1.2e \n" minimum(exp10.(logAs))
        @printf " --> Loss J = %1.2e (γ = %1.2e) \n" last(cost_evo) / first(cost_evo) γ

        As_v = Array(logAs)

        As_rng = filter(!isnan, logAs) |> extrema
        @show As_rng

        H_vert = @. 0.25 * (H[1:end-1, 1:end-1] + H[2:end,1:end-1] + H[1:end-1,2:end] + H[2:end,2:end])
        idv = findall(H_vert .≈ 0.0) |> Array
        idc_inn = findall(H[2:end-1,2:end-1] .≈ 0.0) |> Array

        As_v[idv] .= NaN

        qmag_v = Array(qmag)
        qmag_v[idc_inn] .= NaN
        plts.q_i[3] = qmag_v
        plts.As_i[1][3] = As_v
        plts.As_i[2][3] = Array(H)
        plts.conv[1] = Point2.(iter_evo, cost_evo)

        #jldsave("output_steadystate/step_$iframe.jld2"; As_v, H, qmag_v, iter_evo, cost_evo)
        iframe += 1
        display(fig)

        # if igd == ngd 
        #     display(fig)
        #     jldsave("synthetic_timedepedent.jld2"; logAs_timedepedent = As_v, qmag_timedepedent = qmag_v, H_timedepedent = Array(H), xc=xc, yc= yc, xv=xv, yv=yv)
    
        # end
    end
    return
end

adjoint_2D()