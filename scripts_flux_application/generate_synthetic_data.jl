using CairoMakie
using JLD2

include("sia_forward_flux_implicit.jl")
include("sia_forward_flux_steadystate.jl")
include("sia_forward_flux_explicit.jl")
include("sia_adjoint_flux_2D.jl")
include("sia_loss_flux_2D.jl")

CUDA.allowscalar(false)

function adjoint_2D()
    ## physics
    # power law exponent
    npow = 3

    # dimensionally independent physics 
    lsc   = 1.0#1e4 # length scale  
    aρgn0 = 1.0#1.3517139631340709e-12 # A*(ρg)^n = 1.9*10^(-24)*(910*9.81)^3

    # time scale
    tsc = 1 / aρgn0 / lsc^npow
    # non-dimensional numbers 
    s_f_syn  = 1e-4                 # sliding to ice flow ratio: s_f   = asρgn0/aρgn0/lx^2
    s_f      = 0.08#0.026315789473684213 # sliding to ice flow ratio: s_f   = asρgn0/aρgn0/lx^2
    b_max_nd = 4.706167536706325e-12 # m_max/lx*tsc m_max = 2.0 m/a lx = 1e3 tsc = 
    β1tsc    = 2.353083768353162e-10 # ratio between characteristic time scales of ice flow and accumulation/ablation βtsc = β0*tsc 
    β2tsc    = 3.5296256525297436e-10
    γ_nd     = 1e2
    dt_nd    = 2.131382577069803e8   * 5e0
    t_total_nd = 2.131382577069803e8 * 5e0

    # geometry 
    lx_l, ly_l           = 25.0, 20.0  # horizontal length to characteristic length ratio
    w1_l, w2_l           = 100.0, 10.0 # width to charactertistic length ratio 
    B0_l                 = 0.35  # maximum bed rock elevation to characteristic length ratio
    z_ela_l_1_1, z_ela_l_2_1 = 0.215, 0.09 # ela to domain length ratio z_ela_l
    z_ela_l_1_2, z_ela_l_2_2 = 1.2 * z_ela_l_1_1, 1.2 * 0.09
    #numerics 
    H_cut_l = 1.0e-6

    # dimensionally dependent parameters 
    lx, ly           = lx_l * lsc, ly_l * lsc  # 250000, 200000
    w1, w2           = w1_l * lsc^2, w2_l * lsc^2 # 1e10, 1e9
    z_ELA_0_1, z_ELA_1_1 = z_ela_l_1_1 * lsc, z_ela_l_2_1 * lsc # 2150, 900
    z_ELA_0_2, z_ELA_1_2 = z_ela_l_1_2 * lsc, z_ela_l_2_2 * lsc # 2000, 800
    B0               = B0_l * lsc # 3500
    asρgn0_syn       = s_f_syn * aρgn0 * lsc^2 #5.7e-20*(ρg)^n = 5.7e-20*(910*9.81)^3 = 4.055141889402214e-8
    asρgn0           = s_f * aρgn0 * lsc^2 #5.0e-18*(ρg)^n = 3.54627498316e-6
    b_max            = b_max_nd * lsc / tsc  #2.0 m/a = 6.341958396752917e-8 m/s
    β0, β1           = β1tsc / tsc, β2tsc / tsc  # 3.1709791983764586e-10, 4.756468797564688e-10
    H_cut            = H_cut_l * lsc # 1.0e-2
    γ0               = γ_nd * lsc^(2 - 2npow) * tsc^(-2) #1.0e-2
    dt               = dt_nd * tsc # 365*24*3600
    t_total          = t_total_nd * tsc # 365*24*3600

    # error("check")

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
    H_old = CUDA.zeros(Float64, nx, ny)
    H_obs_explicit = CUDA.zeros(Float64, nx, ny)
    qmag_old = CUDA.zeros(nx-2, ny-2)
    qmag_obs = CUDA.zeros(nx-2, ny-2)
    qmag_obs_explicit = CUDA.zeros(Float64, nx-2, ny-2)

    # bedrock elevation
    ω = 8 # TODO: check!
    B = (@. B0 * (exp(-xc^2 / w1 - yc'^2 / w2) +
                  exp(-xc^2 / w2 - (yc' - ly / ω)^2 / w1))) |> CuArray

    # other fields
    β   = CUDA.fill(β0, nx, ny) .+ β1 .* atan.(xc ./ lx)
    ELA_1 = fill(z_ELA_0_1, nx, ny) .+ z_ELA_1_1 .* atan.(yc' ./ ly .+ 0 .* xc) |> CuArray
    ELA_2 = fill(z_ELA_0_2, nx, ny) .+ z_ELA_1_2 .* atan.(yc' ./ ly .+ 0 .* xc) |> CuArray
    ELA   = CUDA.zeros(Float64, nx, ny)

    D         = CUDA.zeros(Float64, nx - 1, ny - 1)
    qHx       = CUDA.zeros(Float64, nx - 1, ny - 2)
    qHy       = CUDA.zeros(Float64, nx - 2, ny - 1)
    qmag = CUDA.zeros(Float64, nx - 2, ny - 2)

    As_syn    = asρgn0_syn .* (0.5 .* cos.(5π .* xv ./ lx) .* sin.(5π .* yv' ./ ly) .+ 1.0) |> CuArray
    As        = CUDA.fill(asρgn0, nx - 1, ny - 1)
    RH        = CUDA.zeros(Float64, nx, ny)
    Err_rel   = CUDA.zeros(Float64, nx, ny)
    Err_abs   = CUDA.zeros(Float64, nx, ny)
    logAs     = copy(As)
    logAs_syn = copy(As)

    cost_evo = Float64[]
    iter_evo = Float64[]

    #pack parameters
    fwd_params = (fields            = (; H, H_old, B, β, ELA, ELA_1, ELA_2, D, qHx, qHy, As, RH, qmag, Err_rel, Err_abs),
                  scalars          = (; aρgn0, b_max, npow),
                  numerical_params = (; nx, ny, dx, dy, dt, t_total, maxiter, ncheck, ϵtol),
                  launch_config     = (; nthreads, nblocks))

    # fwd_visu = (; plts, fig)
    
    logAs     = log10.(As)
    logAs_syn = log10.(As_syn)
    #TODO generate the H_old by the steady-state forward solver
    @info "generate a steady-state of H_old"
    H .= 0.0
    solve_sia_steadystate!(logAs_syn, fwd_params...)
    H_old .= H
    qmag_old .= qmag
    @info "steady state solve for H_old done"

    @info "generate H_obs by explicit transient model"
    H .= H_old
    solve_sia_explicit!(logAs_syn, fwd_params...)
    H_obs_explicit .= H
    qmag_obs_explicit .= qmag
    @info "steady state solve for H_old done"

    #TODO generate the H_obs by the transient forward solver
    #TODO what is the initial state of H in the transient forward solver?
    @info "generate H_obs"
    H .= H_old
    solve_sia_implicit!(logAs_syn, fwd_params...)
    H_obs .= H
    qmag_obs .= qmag
    @info "transient solve for H_obs done"

    v_obs = copy(qmag_obs)
    v_old = copy(qmag_old)
    n_ratio = (n+2)/(n+1)
    v_obs .= qmag_obs ./ H_obs[2:end-1, 2:end-1]  .* n_ratio
    v_old .= qmag_old ./ H_old[2:end-1, 2:end-1]  .* n_ratio

    #rescaling 
    lsc_data      = 1e4
    aρgn0_data    = 1.3517139631340709e-12
    tsc_data      = 1 / aρgn0_data / lsc_data^n
    s_f_syn       = 1e-4
    lx_l          = 25.0
    ly_l          = 20.0

    H_obs          = H_obs * lsc_data
    H_old          = H_old * lsc_data
    v_obs          = v_obs * lsc_data / tsc_data
    v_old          = v_old * lsc_data / tsc_data

    lx             = lx_l * lsc_data
    ly             = ly_l * lsc_data 
    nx             = 128
    ny             = 128 
    dx, dy = lx / nx, ly / ny
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)
    xv = LinRange(-lx/2 + dx, lx/2 - dx, nx-1)
    yv = LinRange(-ly/2 + dy, ly/2 - dy, ny-1)

    v_obs .*= 365*24*3600
    v_old .*= 365*24*3600

    H_vert = @. 0.25 * (H_obs[1:end-1, 1:end-1] + H_obs[2:end,1:end-1] + H_obs[1:end-1,2:end] + H_obs[2:end,2:end])
    idv = findall(H_vert .≈ 0.0) |> Array
    H_obs[idv] .= NaN
    H_old[idv] .= NaN


    #TODO visulize H_old H_obs and H_old .- H_obs
    fig = Figure(; size=(800, 500), fontsize=14)
    ax  = (H_old  = Axis(fig[1, 1]; aspect=DataAspect(), ylabel="Y [km]",title="a"),
        H_obs  = Axis(fig[1, 2]; aspect=DataAspect(), title="b"),
        diff_H   = Axis(fig[1, 3][1, 1]; aspect=DataAspect(), title="c"),
        v_old  = Axis(fig[2, 1]; aspect=DataAspect(), xlabel="X [km]", ylabel="Y [km]",title="d"),
        v_obs  = Axis(fig[2, 2]; aspect=DataAspect(), xlabel="X [km]",title="e"),
        diff_v   = Axis(fig[2, 3][1, 1]; aspect=DataAspect(), xlabel="X [km]",title="f"))

    xlims!(ax.H_old, -100, 100)
    xlims!(ax.H_obs, -100, 100)
    xlims!(ax.diff_H, -100, 100)
    xlims!(ax.v_old, -100, 100)
    xlims!(ax.v_obs, -100, 100)
    xlims!(ax.diff_v, -100, 100)

    hidexdecorations!(ax.H_old; grid=false)
    hidexdecorations!(ax.H_obs; grid=false) 
    hidexdecorations!(ax.diff_H; grid=false)

    hideydecorations!(ax.H_obs; grid=false)
    hideydecorations!(ax.diff_H; grid=false)
    hideydecorations!(ax.v_obs; grid=false)
    hideydecorations!(ax.diff_v; grid=false)

    colgap!(fig.layout, Relative(1/128))

    plts = (H_old  = heatmap!(ax.H_old, xc./1000, yc./1000, Array(H_old); colormap=:turbo),
            H_obs  = heatmap!(ax.H_obs, xc./1000, yc./1000, Array(H_obs); colormap=:turbo),
            diff_H   = heatmap!(ax.diff_H, xc./1000, yc./1000, Array(abs.(H_obs .- H_old)); colormap=:turbo, colorrange=(1.0, 350.0)),
            v_old  = heatmap!(ax.v_old, xc[1:end-1]./1000, yc[1:end-1]./1000, Array(v_old); colormap=:turbo, colorrange=(1.0, 350.0)),
            v_obs  = heatmap!(ax.v_obs, xc[1:end-1]./1000, yc[1:end-1]./1000, Array(v_obs); colormap=:turbo, colorrange=(1.0, 350.0)),
            diff_v   = heatmap!(ax.diff_v, xc[1:end-1]./1000, yc[1:end-1]./1000, Array(abs.(v_obs .- v_old)); colormap=:turbo, colorrange=(1.0, 350.0)))

    #lg = axislegend(ax.slice; labelsize=10, rowgap=-5, height=40)

    # Colorbar(fig[1, 1][1, 2], plts.H_old)
    # Colorbar(fig[1, 2][1, 2], plts.H_obs)
    Colorbar(fig[1, 3][1, 2], plts.diff_H)
    # Colorbar(fig[2, 1][1, 2], plts.v_old)
    # Colorbar(fig[2, 2][1, 2], plts.v_obs)
    Colorbar(fig[2, 3][1, 2], plts.diff_v)
    display(fig)
    save("generated_synthetic.png", fig)

    #TODO save to disk
    #jldsave("synthetic_data_generated.jld2"; B=Array(B), H_old =Array(H_old), qmag_old = Array(qmag_old), H_obs= Array(H_obs), qmag_obs = Array(qmag_obs))

    
    return
end

adjoint_2D()