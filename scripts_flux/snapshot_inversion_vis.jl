using CairoMakie
using Enzyme
using JLD2

include("macros.jl")
include("sia_forward_flux_steadystate.jl")

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

function ∂J_∂qx_vec!(q̄Hx, qmag, qmag_obs, qHx)
    q̄Hx                .= 0
    @. q̄Hx[1:end-1, :] += (qmag - qmag_obs) * $avx(qHx) / (2 * qmag + (qmag==0))
    @. q̄Hx[2:end, :]   += (qmag - qmag_obs) * $avx(qHx) / (2 * qmag + (qmag==0))
    return
end

function ∂J_∂qy_vec!(q̄Hy, qmag, qmag_obs, qHy)
    q̄Hy                .= 0
    @. q̄Hy[:, 1:end-1] += (qmag - qmag_obs) * $avy(qHy) / (2 * qmag + (qmag==0))
    @. q̄Hy[:, 2:end]   += (qmag - qmag_obs) * $avy(qHy) / (2 * qmag + (qmag==0))
    return
end

function loss(logAs, fwd_params, loss_params)
    (; qmag_obs) = loss_params.fields
    (; qmag)     = fwd_params.fields
    forward_model!(logAs, fwd_params)
    return 0.5 * sum((qmag .- qmag_obs).^2)
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
    (; qmag_obs) = loss_params.fields

    forward_model!(logAs, fwd_params)

    ∂J_∂qx_vec!(q̄Hx, qmag, qmag_obs, qHx)
    ∂J_∂qy_vec!(q̄Hy, qmag, qmag_obs, qHy)

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
    lsc   = 1.0#80.0 #1e4 # length scale  
    aρgn0 = 1.0#1.9*10^(-24)*(910*9.81)^3#1.0 #1.3517139631340709e-12 # A*(ρg)^n = 1.9*10^(-24)*(910*9.81)^3

    # time scale
    tsc = 1 / aρgn0 / lsc^npow
    # non-dimensional numbers 
    s_f_syn  = 1e-4                  # sliding to ice flow ratio: s_f   = asρgn0/aρgn0/lx^2
    s_f      =  1e-4#0.08 # sliding to ice flow ratio: s_f   = asρgn0/aρgn0/lx^2
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

    # dimensionally dependent parameters 
    lx, ly           = lx_l * lsc, ly_l * lsc  # 250000, 200000
    w1, w2           = w1_l * lsc^2, w2_l * lsc^2 # 1e10, 1e9
    z_ELA_0, z_ELA_1 = z_ela_l_1 * lsc, z_ela_l_2 * lsc # 2150, 900
    B0               = B0_l * lsc # 3500
    asρgn0_syn       = s_f_syn * aρgn0 * lsc^2 #5.7e-20*(ρg)^n = 5.7e-20*(910*9.81)^3 = 4.055141889402214e-8
    asρgn0           = s_f * aρgn0 * lsc^2 #5.0e-18*(ρg)^n = 3.54627498316e-6
    b_max            = b_max_nd * lsc / tsc  #2.0 m/a = 6.341958396752917e-8 m/s
    β0, β1           = β1tsc / tsc, β2tsc / tsc  # 3.1709791983764586e-10, 4.756468797564688e-10
    γ0               = γ_nd * lsc^(2 - 2npow) * tsc^(-2) #1.0e-2
    
    ## numerics
    nx, ny     = 128, 128
    ϵtol       = (abs=1e-8, rel=1e-8)
    maxiter    = 5 * nx^2
    ncheck     = ceil(Int, 0.25 * nx^2)
    nthreads   = (16, 16)
    nblocks    = ceil.(Int, (nx, ny) ./ nthreads)
    ngd        = 50
    bt_niter   = 5
    Δγ         = 1e-1#2e-1

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
    #As_syn    = CUDA.fill(asρgn0_syn, nx - 1, ny - 1)

    As_syn    = asρgn0_syn .* (0.5 .* cos.(5π .* xv ./ lx) .* sin.(5π .* yv' ./ ly) .+ 1.0) |> CuArray

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
    qmag_obs = CUDA.zeros(Float64, nx - 2, ny - 2)

    cost_evo = Float64[]
    iter_evo = Float64[]

    logAs     = log10.(As)
    logAs_syn = log10.(As_syn)
    logAs_ini = log10.(As_ini)

    (B, H_old, qmag_old, H_obs, qmag_obs) = load("synthetic_data_generated.jld2", "B", "H_old", "qmag_old", "H_obs", "qmag_obs")

    B = CuArray(B)
    H_obs = CuArray(H_obs)
    qmag_obs = CuArray(qmag_obs)
    H        = copy(H_obs)
    qmag     = copy(qmag_obs)
        #pack parameters
    fwd_params = (fields           = (; H, B, β, ELA, D, qHx, qHy, As, RH, Err_rel, Err_abs, qmag),
                  scalars          = (; aρgn0, b_max, npow),
                  numerical_params = (; nx, ny, dx, dy, maxiter, ncheck, ϵtol),
                  launch_config    = (; nthreads, nblocks))
    adj_params = (fields=(; q̄Hx, q̄Hy, D̄, H̄),)
    loss_params = (fields=(; qHx_obs, qHy_obs, qmag_obs),)
     # define reg
    reg = (; nsm=500, Tmp)

    J(_logAs) = loss(logAs, fwd_params, loss_params)
    ∇J!(_logĀs, _logAs) = ∇loss!(logĀs, logAs, fwd_params, adj_params, loss_params; reg)

     #initial guess 
    As .= As_ini
    @show maximum(As)
    logAs .= log10.(As)
    γ = γ0
    J0 = J(logAs)

    # setup visualisation
    begin
        #init visualization 
        fig = Figure(; size=(800, 400), fontsize=18)
        ax  = (As_s  = Axis(fig[1, 1]; aspect=DataAspect(), ylabel="y", title="observed log As"),
        As_i  = Axis(fig[1, 2]; aspect=DataAspect(), title="modeled log As"),
        q_s   = Axis(fig[2, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="observed log |q|"),
        q_i   = Axis(fig[2, 2]; aspect=DataAspect(), xlabel="x", title="modeled log |q|"),
        slice = Axis(fig[1, 3]; xlabel="Aₛ"),
        conv  = Axis(fig[2, 3]; xlabel="#iter", ylabel="J/J₀", yscale=log10, title="convergence"))
        
        # xlims!(ax.slice, -20, 80)
        xlims!(ax.conv, 0, ngd + 1)
        ylims!(ax.conv, 1e-8, 1e0)
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

        qmag_obs_v = Array(qmag_obs)
        qmag_obs_v[idc_inn] .= NaN

        qmag_v = Array(qmag)
        qmag_v[idc_inn] .= NaN

        As_crange = filter(!isnan, As_syn_v) |> extrema
        q_crange = filter(!isnan, qmag_obs_v) |> extrema

        plts = (As_s  = (heatmap!(ax.As_s, xv, yv, As_syn_v; colormap=:turbo, colorrange=As_crange),
                vlines!(ax.As_s, xc[nx÷2]; linestyle=:dash, color=:magenta)
                ),
                As_i  = heatmap!(ax.As_i, xv, yv, As_v; colormap=:turbo, colorrange=As_crange),
                q_s   = heatmap!(ax.q_s, xc[2:end-1], yc[2:end-1], qmag_obs_v; colormap=:turbo, colorrange=q_crange),
                q_i   = heatmap!(ax.q_i, xc[2:end-1], yc[2:end-1], qmag_v; colormap=:turbo, colorrange=q_crange),
                slice = (lines!(ax.slice, As_syn_v[nx÷2, :], yv; label="observed"),
                lines!(ax.slice, As_v[nx÷2, :], yv; label="modeled")),
                conv  = scatterlines!(ax.conv, Point2.(iter_evo, cost_evo); linewidth=2))

        lg = axislegend(ax.slice; labelsize=10, rowgap=-5, height=40)

        Colorbar(fig[1, 1][1, 2], plts.As_s[1])
        Colorbar(fig[1, 2][1, 2], plts.As_i)
        Colorbar(fig[2,1][1,2], plts.q_s)
        Colorbar(fig[2,2][1,2], plts.q_i)
    end

    # push!(cost_evo, 1.0)
    # push!(iter_evo, 0)

    @info "Gradient descent - inversion for As"
    #CairoMakie.record(fig, "snapshot.mp4", 1:ngd; framerate=1) do igd
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

        As_v = Array(logAs)
        As_v[idv] .= NaN

        qmag_v = Array(qmag)
        qmag_v[idc_inn] .= NaN

        plts.As_i[3] = As_v
        plts.q_i[3] = qmag_v
        plts.slice[2][1][] = Point2.(As_v[nx÷2, :],yv)
        plts.conv[1] = Point2.(iter_evo, cost_evo./first(cost_evo))

        @show first(cost_evo)
        @show last(cost_evo)

        if igd == ngd 
            display(fig)
            jldsave("synthetic_snapshot.jld2"; logAs_syn = As_syn_v, logAs_snapshot = As_v, qmag_obs=qmag_obs_v, qmag_snapshot = qmag_v, H_obs=Array(H_obs), H_snapshot=Array(H), iter_evo, cost_evo)
        end
    end
    return
end

adjoint_2D()