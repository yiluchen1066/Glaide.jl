include("sia_forward_2D.jl")

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
    s_f      = 0.01  # sliding to ice flow ratio: s_f   = asρgn0/aρgn0/lx^2
    b_max_nd = 4.706167536706325e-12 # m_max/lx*tsc m_max = 2.0 m/a lx = 1e3 tsc = 
    βtsc     = 2.353083768353162e-10 # ratio between characteristic time scales of ice flow and accumulation/ablation βtsc = β0*tsc 
    β1tsc    = 3.5296256525297436e-10

    # geometry 
    lx_l      = 25.0  # horizontal length to characteristic length ratio
    ly_l      = 20.0  # horizontal length to characteristic length ratio 
    w1_l      = 100.0 #width to charactertistic length ratio 
    w2_l      = 10.0  # width to characteristic length ratio
    B0_l      = 0.35  # maximum bed rock elevation to characteristic length ratio
    z_ela_l   = 0.215 # ela to domain length ratio z_ela_l = 
    z_ela_1_l = 0.09

    # dimensionally dependent parameters 
    lx      = lx_l * lsc #250000
    ly      = ly_l * lsc #200000
    w1      = w1_l * lsc^2 #1e10
    w2      = w2_l * lsc^2 #1e9
    z_ELA_0 = z_ela_l * lsc # 2150
    z_ELA_1 = z_ela_1_l * lsc #900
    B0      = B0_l * lsc # 3500
    asρgn0  = s_f * aρgn0 * lsc^2 #5.7e-20*(ρg)^n = 5.7e-20*(910*9.81)^3 = 4.055141889402214e-8
    b_max   = b_max_nd * lsc / tsc  #2.0 m/a = 6.341958396752917e-8 m/s
    β0      = βtsc / tsc  #0.01 /a = 3.1709791983764586e-10
    β1      = β1tsc / tsc #0.015/3600/24/365 = 4.756468797564688e-10

    ## numerics
    nx       = 128
    ny       = 128
    ϵtol     = (abs=1e-6, rel=1e-6)
    maxiter  = 5 * nx^2
    ncheck   = ceil(Int, 0.1 * nx^2)
    nthreads = (16, 16)
    nblocks  = ceil.(Int, (nx, ny) ./ nthreads)

    ## pre-processing
    dx = lx / nx
    dy = lx / ny
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)

    ## init arrays
    # ice thickness
    H = CUDA.zeros(Float64, nx, ny)

    # bedrock elevation
    ω = 8 # TODO: check!
    B = (@. B0 * (exp(-xc^2 / w1 - yc'^2 / w2) +
                  exp(-xc^2 / w2 - (yc' - ly / ω)^2 / w1))) |> CuArray

    # other fields
    β       = CUDA.fill(β0, nx, ny) .+ β1 .* atan.(xc ./ lx)
    ELA     = CUDA.fill(z_ELA_0, nx, ny) .+ z_ELA_1 .* atan.(yc' ./ ly .+ 0 .* xc)
    D       = CUDA.zeros(Float64, nx - 1, ny - 1)
    qHx     = CUDA.zeros(Float64, nx - 1, ny - 2)
    qHy     = CUDA.zeros(Float64, nx - 2, ny - 1)
    As      = CUDA.fill(asρgn0, nx - 1, ny - 1)
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
    fwd_params = (fields           = (; H, B, β, ELA, D, qHx, qHy, As, RH, Err_rel, Err_abs),
                  scalars          = (; aρgn0, b_max, npow),
                  numerical_params = (; nx, ny, dx, dy, maxiter, ncheck, ϵtol),
                  launch_config    = (; nthreads, nblocks))
    fwd_visu = (; plt, fig)

    @info "Solve SIA"
    solve_sia!(fwd_params...; visu=fwd_visu)

    write("output/synthetic_new.dat", Array(H))

    return
end

main()