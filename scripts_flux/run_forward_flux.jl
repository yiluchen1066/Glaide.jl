include("sia_forward_flux_2D.jl")

function main()
    ## physics
    # power law exponent
    npow = 3

    # dimensionally independent physics 
    lsc   = 1.0 #1e4 # length scale  
    aρgn0 = 1.0 #1.3517139631340709e-12 # A*(ρg)^n = 1.9*10^(-24)*(910*9.81)^3

    # time scale
    tsc = 1 / aρgn0 / lsc^npow

    # non-dimensional numbers 
    s_f_syn  = 0.01
    s_f      = 0.026315789473684213 # sliding to ice flow ratio: s_f   = asρgn0/aρgn0/lx^2
    b_max_nd = 4.706167536706325e-12 # m_max/lx*tsc m_max = 2.0 m/a lx = 1e3 tsc = 
    β1tsc    = 2.353083768353162e-10 # ratio between characteristic time scales of ice flow and accumulation/ablation βtsc = β0*tsc 
    β2tsc    = 3.5296256525297436e-10

    # geometry 
    lx_l, ly_l           = 25.0, 20.0  # horizontal length to characteristic length ratio
    w1_l, w2_l           = 100.0, 10.0 # width to charactertistic length ratio 
    B0_l                 = 0.35  # maximum bed rock elevation to characteristic length ratio
    z_ela_l_1, z_ela_l_2 = 0.215, 0.09 # ela to domain length ratio z_ela_l = 

    # dimensionally dependent parameters 
    lx, ly           = lx_l * lsc, ly_l * lsc  # 250000, 200000
    w1, w2           = w1_l * lsc^2, w2_l * lsc^2 # 1e10, 1e9
    z_ELA_0, z_ELA_1 = z_ela_l_1 * lsc, z_ela_l_2 * lsc # 2150, 900
    B0               = B0_l * lsc # 3500
    asρgn0           = s_f * aρgn0 * lsc^2 #5.7e-20*(ρg)^n = 5.7e-20*(910*9.81)^3 = 4.055141889402214e-8
    asρgn0_syn       = s_f_syn * aρgn0 * lsc^2 #5.0e-18*(ρg)^n = 3.54627498316e-6
    b_max            = b_max_nd * lsc / tsc  #2.0 m/a = 6.341958396752917e-8 m/s
    β0, β1           = β1tsc / tsc, β2tsc / tsc  # 3.1709791983764586e-10, 4.756468797564688e-10

    ## numerics
    nx, ny   = 128, 128
    ϵtol     = (abs=1e-8, rel=1e-8)
    maxiter  = 5 * nx^2
    ncheck   = ceil(Int, 0.25 * nx^2)
    nthreads = (16, 16)
    nblocks  = ceil.(Int, (nx, ny) ./ nthreads)

    #check 

    #check 
    @show(asρgn0_syn)
    @show(asρgn0)
    @show(lx)
    @show(ly)
    @show(w1)
    @show(w2)
    @show(B0)
    @show(z_ELA_0)
    @show(z_ELA_1)
    @show(b_max)
    @show(β0)
    @show(β1)
    #@show(H_cut)
    @show(nx)
    @show(ny)
    @show(ϵtol)
    @show(maxiter)
    @show(ncheck)
    @show(nthreads)
    @show(nblocks)
    #@show(ϵtol_adj)
    #@show(ncheck_adj)

    ## pre-processing
    dx, dy = lx / nx, ly / ny
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)

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

    D       = CUDA.zeros(Float64, nx - 1, ny - 1)
    qHx     = CUDA.zeros(Float64, nx - 1, ny - 2)
    qHy     = CUDA.zeros(Float64, nx - 2, ny - 1)
    As      = CUDA.fill(asρgn0, nx - 1, ny - 1)
    As_syn  = CUDA.fill(asρgn0_syn, nx - 1, ny - 1)
    RH      = CUDA.zeros(Float64, nx, ny)
    Err_rel = CUDA.zeros(Float64, nx, ny)
    Err_abs = CUDA.zeros(Float64, nx, ny)

    ## init visualization 
    fig = Figure(; resolution=(1200, 800), fontsize=32)

    axs = (H   = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="Ice thickness"),
           As  = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="log10(Sliding coefficient)"),
           Err = Axis(fig[2, 1]; aspect=1, yscale=log10, title="Convergence history"))

    ylims!(axs.Err, 0.1ϵtol.rel, 2)

    plt = (H=heatmap!(axs.H, xc, yc, Array(H); colormap=:turbo),
           As=heatmap!(axs.As, xc, yc, Array(log10.(As)); colormap=:viridis),
           Err=(scatterlines!(axs.Err, Point2{Float64}[]),
                scatterlines!(axs.Err, Point2{Float64}[])))

    Colorbar(fig[1, 1][1, 2], plt.H)
    Colorbar(fig[1, 2][1, 2], plt.As)

    ## pack parameters
    fwd_params = (fields           = (; H, B, β, ELA, D, qHx, qHy, RH, Err_rel, Err_abs),
                  scalars          = (; aρgn0, b_max, npow),
                  numerical_params = (; nx, ny, dx, dy, maxiter, ncheck, ϵtol),
                  launch_config    = (; nthreads, nblocks))
    fwd_visu = (; plt, fig)

    @info "Solve synthetic SIA"
    @show maximum(As_syn)
    solve_sia!(As_syn, fwd_params...; visu=fwd_visu)
    H_obs .= H

    @info "solve SIA"
    @show maximum(As)
    solve_sia!(As, fwd_params...; visu=fwd_visu)

    @show(maximum(H .- H_obs))

    return
end

main()
