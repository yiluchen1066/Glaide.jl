using CairoMakie
using Enzyme

include("macros.jl")
include("sia_forward_flux_2D.jl")

@inline ∇(fun, args...) = (Enzyme.autodiff_deferred(Enzyme.Reverse, fun, Const, args...); return)
const DupNN = DuplicatedNoNeed

function laplacian!(As, As2)
    @get_indices
    if ix >= 2 && ix <= size(As, 1) - 1 && iy >= 2 && iy <= size(As, 2) - 1
        ΔAs = As[ix-1, iy] + As[ix+1, iy] + As[ix, iy-1] + As[ix, iy+1] - 4.0 * As[ix, iy]
        As2[ix, iy] = As[ix, iy] + 1 / 8 * ΔAs
    end
    return
end

function smooth!(As, As2, nsm, nthreads, nblocks)
    for _ in 1:nsm
        @cuda threads = nthreads blocks = nblocks laplacian!(As, As2)
        As2[[1, end], :] .= As2[[2, end - 1], :]
        As2[:, [1, end]] .= As2[:, [2, end - 1]]
        As, As2 = As2, As
    end
    return
end 

function forward_model!(logAs, fwd_params)
    (; D, H, B, As, qHx, qHy, qmag) = fwd_params.fields
    (; aρgn0, npow) = fwd_params.scalars
    (; dx, dy) = fwd_params.numerical_params
    (; nthreads, nblocks) = fwd_params.launch_config

    As .= exp10.(logAs)
    @cuda threads = nthreads blocks = nblocks compute_D!(D, H, B, As, aρgn0, npow, dx, dy)
    @cuda threads = nthreads blocks = nblocks compute_q!(qHx, qHy, D, H, B, dx, dy)
    @. qmag = sqrt($avx(qHx)^2 + $avy(qHy)^2)
    return
end

function ∂J_∂qx_vec!(q̄Hx, qmag, qobs_mag, qHx)
    q̄Hx                .= 0
    @. q̄Hx[1:end-1, :] += (qmag - qobs_mag) * $avx(qHx) / (2 * qmag + (qmag==0))
    @. q̄Hx[2:end, :]   += (qmag - qobs_mag) * $avx(qHx) / (2 * qmag + (qmag==0))
    return
end

function ∂J_∂qy_vec!(q̄Hy, qmag, qobs_mag, qHy)
    q̄Hy                .= 0
    @. q̄Hy[:, 1:end-1] += (qmag - qobs_mag) * $avy(qHy) / (2 * qmag + (qmag==0))
    @. q̄Hy[:, 2:end]   += (qmag - qobs_mag) * $avy(qHy) / (2 * qmag + (qmag==0))
    return
end

function loss(logAs, fwd_params, loss_params)
    (; qobs_mag) = loss_params.fields
    (; qmag)     = fwd_params.fields
    forward_model!(logAs, fwd_params)
    return 0.5 * sum((qmag .- qobs_mag).^2)
end

function ∇loss!(logĀs, logAs, fwd_params, adj_params, loss_params; reg=nothing)
    #unpack forward parameters 
    (; H, B, β, ELA, D, qHx, qHy, As, qmag) = fwd_params.fields
    (; aρgn0, npow) = fwd_params.scalars
    (; dx, dy) = fwd_params.numerical_params
    (; nthreads, nblocks) = fwd_params.launch_config
    #unpack adjoint parameters
    (; q̄Hx, q̄Hy, D̄, H̄) = adj_params.fields
    #unpack loss parameters
    (; qobs_mag) = loss_params.fields

    forward_model!(logAs, fwd_params)

    ∂J_∂qx_vec!(q̄Hx, qmag, qobs_mag, qHx)
    ∂J_∂qy_vec!(q̄Hy, qmag, qobs_mag, qHy)

    logĀs .= 0.0
    D̄ .= 0.0
    H̄ .= 0.0

    @cuda threads = nthreads blocks = nblocks ∇(compute_q!,
                                                DupNN(qHx, q̄Hx),
                                                DupNN(qHy, q̄Hy),
                                                DupNN(D, D̄),
                                                Const(H),
                                                Const(B), Const(dx), Const(dy))
    @cuda threads = nthreads blocks = nblocks ∇(compute_D!,
                                                DupNN(D, D̄),
                                                Const(H), Const(B),
                                                DupNN(As, logĀs), Const(aρgn0), Const(npow), Const(dx), Const(dy))

    logĀs[[1, end], :] = logĀs[[2, end - 1], :]
    logĀs[:, [1, end]] = logĀs[:, [2, end - 1]]

    #smoothing 
    
    # convert to dJ/dlogAs
    logĀs .*= As

    return
end

function adjoint_2D()
    ## physics
    # power law exponent
    npow = 3

    # dimensionally independent physics 
    lsc   = 80.0 #1e4 # length scale  
    aρgn0 = 1.0 #1.3517139631340709e-12 # A*(ρg)^n = 1.9*10^(-24)*(910*9.81)^3

    # time scale
    tsc = 1 / aρgn0 / lsc^npow
    # non-dimensional numbers 
    s_f_syn  = 1e-4                  # sliding to ice flow ratio: s_f   = asρgn0/aρgn0/lx^2
    s_f      =  0.01 # sliding to ice flow ratio: s_f   = asρgn0/aρgn0/lx^2
    b_max_nd = 4.706167536706325e-12 # m_max/lx*tsc m_max = 2.0 m/a lx = 1e3 tsc = 
    β1tsc    = 2.353083768353162e-10 # ratio between characteristic time scales of ice flow and accumulation/ablation βtsc = β0*tsc 
    β2tsc    = 3.5296256525297436e-10

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
    
    ## numerics
    nx, ny     = 128, 128
    ϵtol       = (abs=1e-8, rel=1e-8)
    maxiter    = 5 * nx^2
    ncheck     = ceil(Int, 0.25 * nx^2)
    nthreads   = (16, 16)
    nblocks    = ceil.(Int, (nx, ny) ./ nthreads)
    ngd        = 500
    bt_niter   = 5
    Δγ         = 1e-1

    ## pre-processing
    dx, dy = lx / nx, ly / ny
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)
    xc_1 = xc[1:(end-1)]
    yc_1 = yc[1:(end-1)]

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
    As_syn    = CUDA.fill(asρgn0_syn, nx - 1, ny - 1)
    As        = CUDA.fill(asρgn0, nx - 1, ny - 1)
    RH        = CUDA.zeros(Float64, nx, ny)
    Err_rel   = CUDA.zeros(Float64, nx, ny)
    Err_abs   = CUDA.zeros(Float64, nx, ny)
    As_ini    = copy(As)
    logAs     = copy(As)
    logAs_syn = copy(As)
    logAs_ini = copy(As)
    #init adjoint storage
    q̄Hx      = CUDA.zeros(Float64, nx - 1, ny - 2)
    q̄Hy      = CUDA.zeros(Float64, nx - 2, ny - 1)
    D̄        = CUDA.zeros(Float64, nx - 1, ny - 1)
    H̄        = CUDA.zeros(Float64, nx, ny)
    logĀs    = CUDA.zeros(Float64, nx - 1, ny - 1)
    Tmp      = CUDA.zeros(Float64, nx - 1, ny - 1)
    qmag     = CUDA.zeros(Float64, nx - 2, ny - 2)
    qobs_mag = CUDA.zeros(Float64, nx - 2, ny - 2)

    cost_evo = Float64[]
    iter_evo = Float64[]

    # setup visualisation
    begin
        #init visualization 
        fig = Figure(size=(1000, 800), fontsize=32)
        #opts    = (xaxisposition=:top,) # save for later 

        axs = (H    = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel=L"x", ylabel=L"y", title=L"H"),
               H_s  = Axis(fig[1, 2]; aspect=1, xlabel=L"H"),
               As   = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel=L"x", title=L"\log_{10}(A_s)"),
               As_s = Axis(fig[2, 2]; aspect=1, xlabel=L"\log_{10}(A_s)"),
               err  = Axis(fig[2, 3]; yscale=log10, title=L"convergence", xlabel="iter", ylabel=L"error"))

        #xlims CairoMakie.xlims!()
        #ylims 

        nan_to_zero(x) = isnan(x) ? zero(x) : x

        # As_ini .= CuArray(asρgn0.*(1.0 .+ 1e1.*CUDA.rand(Float64, size(As_ini))))

        logAs     = log10.(As)
        logAs_syn = log10.(As_syn)
        logAs_ini = log10.(As_ini)

        xc_1 = xc[1:(end-1)]
        yc_1 = yc[1:(end-1)]

        plts = (H    = heatmap!(axs.H, xc, yc, Array(H); colormap=:turbo),
                H_v  = vlines!(axs.H, xc[nx÷2]; color=:magenta, linewidth=4, linestyle=:dash),
                H_s  = (lines!(axs.H_s, Point2.(Array(H_obs[nx÷2, :]), yc); linewidth=4, color=:red, label="synthetic"),
                lines!(axs.H_s, Point2.(Array(H[nx÷2, :]), yc); linewidth=4, color=:blue, label="current")),
                As   = heatmap!(axs.As, xc_1, yc_1, Array(logAs); colormap=:viridis),
                As_v = vlines!(axs.As, xc_1[nx÷2]; linewidth=4, color=:magenta, linewtyle=:dash),
                As_s = (lines!(axs.As_s, Point2.(Array(logAs[nx÷2, :]), yc_1); linewith=4, color=:blue, label="current"),
                lines!(axs.As_s, Point2.(Array(logAs_ini[nx÷2, :]), yc_1); linewidth=4, color=:green, label="initial"),
                lines!(axs.As_s, Point2.(Array(logAs_syn[nx÷2, :]), yc_1); linewidth=4, color=:red, label="synthetic")),
                err  = scatterlines!(axs.err, Point2.(iter_evo, cost_evo); linewidth=4))

        Colorbar(fig[1, 1][1, 2], plts.H)
        Colorbar(fig[2, 1][1, 2], plts.As)
    end

    #pack parameters
    fwd_params = (fields           = (; H, B, β, ELA, D, qHx, qHy, As, RH, Err_rel, Err_abs, qmag),
                  scalars          = (; aρgn0, b_max, npow),
                  numerical_params = (; nx, ny, dx, dy, maxiter, ncheck, ϵtol),
                  launch_config    = (; nthreads, nblocks))
    adj_params = (fields=(; q̄Hx, q̄Hy, D̄, H̄),)
    loss_params = (fields=(; qHx_obs, qHy_obs, qobs_mag),)

    fwd_visu = (; plts, fig)

    @info "synthetic solve"
    @show maximum(As_syn)
    H .= 0.0
    solve_sia!(logAs_syn, fwd_params...)
    H_obs .= H
    qHx_obs .= qHx
    qHy_obs .= qHy
    @. qobs_mag = sqrt($avx(qHx_obs)^2 + $avy(qHy_obs)^2)
    @show size(qobs_mag)
    @info "synthetic solve done"

    # define reg
    reg = (; nsm=100, Tmp)

    J(_logAs) = loss(logAs, fwd_params, loss_params)
    ∇J!(_logĀs, _logAs) = ∇loss!(logĀs, logAs, fwd_params, adj_params, loss_params; reg)

    #initial guess 
    As .= As_ini
    @show maximum(As)
    logAs .= log10.(As)
    J0 = J(logAs)

    push!(cost_evo, 1.0)
    push!(iter_evo, 0)

    @info "Gradient descent - inversion for As"
    for igd in 1:ngd
        println("GD iteration $igd \n")
        ∇J!(logĀs, logAs)
        @show maximum(exp10.(logAs))
        @show maximum(abs.(logĀs))
        γ = Δγ / maximum(abs.(logĀs))
        @. logAs -= γ * logĀs
        As .= exp10.(logAs)

        if !isnothing(reg)
            (; nsm, Tmp) = reg
            Tmp .= As
            smooth!(As, Tmp, nsm, nthreads, nblocks)
        end

        push!(iter_evo, igd)
        push!(cost_evo, J(logAs)/J0)

        @printf " min(As) = %1.2e \n" minimum(exp10.(logAs))
        @printf " --> Loss J = %1.2e (γ = %1.2e) \n" last(cost_evo) γ

        if igd % 10 == 0
            plts.H[3]       = Array(H)
            plts.H_s[2][1]  = Point2.(Array(H[nx÷2, :]), yc)
            plts.As[3]      = Array(log10.(As))
            plts.As_s[1][1] = Point2.(Array(log10.(As[nx÷2, :])), yc_1)
            plts.err[1]     = Point2.(iter_evo, cost_evo)
            display(fig)
        end
    end
    return
end

adjoint_2D()