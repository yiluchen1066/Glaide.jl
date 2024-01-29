using CairoMakie
using JLD2

include("sia_forward_flux_2D.jl")
include("sia_adjoint_flux_2D.jl")
include("sia_loss_flux_2D.jl")

CUDA.allowscalar(false)

function adjoint_2D()
    ## physics
    # power law exponent
    npow = 3

    # dimensionally independent physics 
    lsc   = 1.0 #1e4 # length scale  
    aρgn0 = 1.0 #1.3517139631340709e-12 # A*(ρg)^n = 1.9*10^(-24)*(910*9.81)^3

    # time scale
    tsc = 1 / aρgn0 / lsc^npow
    # non-dimensional numbers 
    s_f_syn  = 1e-4                 # sliding to ice flow ratio: s_f   = asρgn0/aρgn0/lx^2
    s_f      = 0.08#0.026315789473684213 # sliding to ice flow ratio: s_f   = asρgn0/aρgn0/lx^2
    b_max_nd = 4.706167536706325e-12 # m_max/lx*tsc m_max = 2.0 m/a lx = 1e3 tsc = 
    β1tsc    = 2.353083768353162e-10 # ratio between characteristic time scales of ice flow and accumulation/ablation βtsc = β0*tsc 
    β2tsc    = 3.5296256525297436e-10
    γ_nd     = 1e2

    # geometry 
    lx_l, ly_l           = 25.0, 20.0  # horizontal length to characteristic length ratio
    w1_l, w2_l           = 100.0, 10.0 # width to charactertistic length ratio 
    B0_l                 = 0.35  # maximum bed rock elevation to characteristic length ratio
    z_ela_l_1, z_ela_l_2 = 0.215, 0.09 # ela to domain length ratio z_ela_l = 
    #numerics 
    H_cut_l = 1.0e-6

    # dimensionally dependent parameters 
    lx, ly           = lx_l * lsc, ly_l * lsc  # 250000, 200000
    w1, w2           = w1_l * lsc^2, w2_l * lsc^2 # 1e10, 1e9
    z_ELA_0, z_ELA_1 = z_ela_l_1 * lsc, z_ela_l_2 * lsc # 2150, 900
    B0               = B0_l * lsc # 3500
    asρgn0_syn       = s_f_syn * aρgn0 * lsc^2 #5.7e-20*(ρg)^n = 5.7e-20*(910*9.81)^3 = 4.055141889402214e-8
    asρgn0           = s_f * aρgn0 * lsc^2 #5.0e-18*(ρg)^n = 3.54627498316e-6
    b_max            = b_max_nd * lsc / tsc  #2.0 m/a = 6.341958396752917e-8 m/s
    β0, β1           = β1tsc / tsc, β2tsc / tsc  # 3.1709791983764586e-10, 4.756468797564688e-10
    H_cut            = H_cut_l * lsc # 1.0e-2
    γ0               = γ_nd * lsc^(2 - 2npow) * tsc^(-2) #1.0e-2

    ## numerics
    nx, ny     = 128, 128
    ϵtol       = (abs=1e-6, rel=1e-6)
    maxiter    = 5 * nx^2
    ncheck     = ceil(Int, 0.25 * nx^2)
    nthreads   = (16, 16)
    nblocks    = ceil.(Int, (nx, ny) ./ nthreads)
    ϵtol_adj   = 1e-8
    ncheck_adj = ceil(Int, 0.25 * nx^2)
    ngd        = 50
    bt_niter   = 5
    Δγ         = 10.0

    w_H = 0.5
    w_q = 0.5

    ## pre-processing
    dx, dy = lx / nx, ly / ny
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)
    xc_1 = xc[1:(end-1)]
    yc_1 = yc[1:(end-1)]

    xv = LinRange(-lx/2 + dx, lx/2 - dx, nx-1)
    yv = LinRange(-ly/2 + dy, ly/2 - dy, ny-1)

    ## init arrays
    # ice thickness
    H     = CUDA.zeros(Float64, nx, ny)
    H_obs = CUDA.zeros(Float64, nx, ny)

    # bedrock elevation
    ω = 8 # TODO: check!
    B = (@. B0 * (exp(-xc^2 / w1 - yc'^2 / w2) +
                  exp(-xc^2 / w2 - (yc' - ly / ω)^2 / w1))) |> CuArray

    # other fields
    β   = CUDA.fill(β0, nx, ny) .+ β1 .* atan.(xc ./ lx)
    ELA = fill(z_ELA_0, nx, ny) .+ z_ELA_1 .* atan.(yc' ./ ly .+ 0 .* xc) |> CuArray

    D         = CUDA.zeros(Float64, nx - 1, ny - 1)
    qHx       = CUDA.zeros(Float64, nx - 1, ny - 2)
    qHy       = CUDA.zeros(Float64, nx - 2, ny - 1)
    qHx_obs   = CUDA.zeros(Float64, nx - 1, ny - 2)
    qHy_obs   = CUDA.zeros(Float64, nx - 2, ny - 1)
    # As_syn    = CUDA.fill(asρgn0_syn, nx - 1, ny - 1)

    As_syn    = asρgn0_syn .* (0.5 .* cos.(5π .* xv ./ lx) .* sin.(5π .* yv' ./ ly) .+ 1.0) |> CuArray

    As        = CUDA.fill(asρgn0, nx - 1, ny - 1)
    RH        = CUDA.zeros(Float64, nx, ny)
    Err_rel   = CUDA.zeros(Float64, nx, ny)
    Err_abs   = CUDA.zeros(Float64, nx, ny)
    As_ini    = copy(As)
    logAs     = copy(As)
    logAs_syn = copy(As)
    logAs_ini = copy(As)
    Lap_As    = copy(As)
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
    qobs_mag = CUDA.zeros(Float64, nx - 2, ny - 2)

    cost_evo = Float64[]
    iter_evo = Float64[]

    #pack parameters
    fwd_params = (fields           = (; H, B, β, ELA, D, qHx, qHy, As, RH, qmag, Err_rel, Err_abs),
                  scalars          = (; aρgn0, b_max, npow),
                  numerical_params = (; nx, ny, dx, dy, maxiter, ncheck, ϵtol),
                  launch_config    = (; nthreads, nblocks))

    # fwd_visu = (; plts, fig)

    adj_params = (fields=(; q̄Hx, q̄Hy, D̄, R̄H, Ās, ψ_H, H̄),
                  numerical_params=(; ϵtol_adj, ncheck_adj, H_cut))

    loss_params = (fields=(; H_obs, qHx_obs, qHy_obs, qobs_mag, ∂J_∂H, ∂J_∂qx, ∂J_∂qy, Lap_As),
                   scalars=(; w_H, w_q))

    #this is to switch on/off the smooth of the sensitivity 
    reg = (; nsm=10, α=1e-4, Tmp)

    @info "synthetic solve"
    logAs     = log10.(As)
    logAs_syn = log10.(As_syn)
    logAs_ini = log10.(As_ini)
    #initial guess
    #solve for H_obs
    @show maximum(As_syn)
    #solve_sia!(As_syn, fwd_params...; visu=fwd_visu)
    H .= 0.0
    solve_sia!(logAs_syn, fwd_params...)
    H_obs .= H
    qHx_obs .= qHx
    qHy_obs .= qHy
    qobs_mag .= qmag
    # plts.q_s[3] = Array(qobs_mag)
    # plts.As_i[3][3] = Array(H_obs)
    @info "synthetic solve done"

    As    .= As_ini
    logAs .= log10.(As)

    #Define loss functions 
    J(_logAs) = loss(logAs, fwd_params, loss_params)
    ∇J!(_logĀs, _logAs) = ∇loss!(logĀs, logAs, fwd_params, adj_params, loss_params; reg)
    @info "inversion for As"

    
    γ = γ0
    J_old = 0.0
    J_new = 0.0
    #solve for H with As and compute J_old
    H .= 0.0

    @show maximum(logAs)
    J_old = J(logAs)
    J_ini = J_old

    # setup visualisation
    begin
        #init visualization 
        fig = Figure(; size=(800, 400), fontsize=14)
        ax  = (As_s  = Axis(fig[1, 1]; aspect=DataAspect(), ylabel="y", title="observed log As"),
        As_i  = Axis(fig[1, 2]; aspect=DataAspect(), title="modeled log As"),
        q_s   = Axis(fig[2, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="observed log |q|"),
        q_i   = Axis(fig[2, 2]; aspect=DataAspect(), xlabel="x", title="modeled log |q|"),
        slice = Axis(fig[1, 3]; xlabel="Aₛ"),
        conv  = Axis(fig[2, 3]; xlabel="#iter", ylabel="J/J₀", yscale=log10, title="convergence"))
        
        xlims!(ax.conv, 0, ngd+1)
        ylims!(ax.conv, 1e-4, 1e0)
        #xlims CairoMakie.xlims!()
        #ylims 

        nan_to_zero(x) = isnan(x) ? zero(x) : x

        xc_1 = xc[1:(end-1)]
        yc_1 = yc[1:(end-1)]

        linkyaxes!(ax.slice, ax.As_i, ax.As_s, ax.q_s, ax.q_i)

        idc_inn = findall(H[2:end-1,2:end-1] .≈ 0.0) |> Array

        H_vert = @. 0.25 * (H[1:end-1, 1:end-1] + H[2:end,1:end-1] + H[1:end-1,2:end] + H[2:end,2:end])
        idv = findall(H_vert .≈ 0.0) |> Array

        As_syn_v = Array(logAs_syn)
        As_syn_v[idv] .= NaN

        As_v = Array(logAs)
        As_v[idv] .= NaN

        qobs_mag_v = Array(log10.(qobs_mag))
        qobs_mag_v[idc_inn] .= NaN

        qmag_v = Array(log10.(qmag))
        qmag_v[idc_inn] .= NaN

        As_crange = filter(!isnan, As_syn_v) |> extrema
        q_crange = filter(!isnan, qobs_mag_v) |> extrema

        plts = (As_s  = (heatmap!(ax.As_s, xv, yv, As_syn_v; colormap=:turbo, colorrange=As_crange),
                vlines!(ax.As_s, xc[nx÷2]; linestyle=:dash, color=:magenta)),
                As_i  = (heatmap!(ax.As_i, xv, yv, As_v; colormap=:turbo, colorrange=As_crange),
                contour!(ax.As_i, xc, yc, Array(H); levels=0.001:0.001, color=:white, linestyle=:dash),
                contour!(ax.As_i, xc, yc, Array(H_obs); levels=0.001:0.001, color=:red)),
                q_s   = heatmap!(ax.q_s, xc[2:end-1], yc[2:end-1], qobs_mag_v; colormap=:turbo, colorrange=q_crange),
                q_i   = heatmap!(ax.q_i, xc[2:end-1], yc[2:end-1], qmag_v; colormap=:turbo, colorrange=q_crange),
                slice = (lines!(ax.slice, As_syn_v[nx÷2, :], yv; label="observed"),
                lines!(ax.slice, As_v[nx÷2, :], yv; label="modeled")),
                conv  = scatterlines!(ax.conv, Point2.(iter_evo, cost_evo); linewidth=2))

        lg = axislegend(ax.slice; labelsize=10, rowgap=-5, height=40)

        Colorbar(fig[1, 1][1, 2], plts.As_s[1])
        Colorbar(fig[1, 2][1, 2], plts.As_i[1])
        Colorbar(fig[2,1][1,2], plts.q_s)
        Colorbar(fig[2,2][1,2], plts.q_i)


        display(fig)
    end

    ispath("output_steadystate") && rm("output_steadystate", recursive=true)
    mkdir("output_steadystate")
    jldsave("output_steadystate/static.jld2"; qobs_mag, H_obs, As_ini, As_syn, xc, yc, xv, yv)

    iframe = 1
    @info "Gradient descent - inversion for As"
    for igd in 1:ngd
        #As_ini .= As
        println("GD iteration $igd \n")
        ∇J!(logĀs, logAs)
        γ = Δγ / maximum(abs.(logĀs))
        γ = min(γ, 1 / reg.α)
        @. logAs -= γ * logĀs

        push!(iter_evo, igd)
        push!(cost_evo, J(logAs))
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

        qmag_v = Array(log10.(qmag))
        qmag_v[idc_inn] .= NaN

        plts.As_i[1][3] = As_v
        plts.As_i[2][3] = Array(H)
        plts.q_i[3] = qmag_v
        plts.slice[2][1][] = Point2.(As_v[nx÷2, :],yv)
        plts.conv[1] = Point2.(iter_evo, cost_evo)



        jldsave("output_steadystate/step_$iframe.jld2"; As_v, H, qmag_v, iter_evo, cost_evo)
        iframe += 1
        display(fig)
    end
    return
end

adjoint_2D()