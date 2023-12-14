using CairoMakie

include("scripts_flux/sia_forward_flux_2D.jl")
include("scripts_flux/sia_adjoint_flux_2D.jl")
include("scripts_flux/sia_loss_flux_2D.jl")


function inversion_steadystate(geometry, observed, initial, physics, numerics, optim_params; do_vis=false)
    (; B, xc, yc) = geometry
    (; H_obs, qobs_mag) = observed
    (; H_ini, As_ini) = initial
    (; npow, aρgn0, β, ELA, b_max, H_cut, w_H, w_q) = physics
    (; ϵtol, ϵtol_adj, maxiter) = numerics
    (; Δγ, ngd) = optim_params

    # pre-processing
    nx = length(xc)
    ny = length(yc)
    dx = xc[2] - xc[1]
    dy = yc[2] - yc[1]

    ncheck     = ceil(Int, 0.25 * nx^2)
    ncheck_adj = ceil(Int, 0.25 * nx^2)
    nthreads   = (16, 16)
    nblocks    = ceil.(Int, (nx, ny) ./ nthreads)

    # init forward
    H         = copy(H_ini)
    D         = CUDA.zeros(Float64, nx - 1, ny - 1)
    As        = CUDA.zeros(Float64, nx - 1, ny - 1)
    qx        = CUDA.zeros(Float64, nx - 1, ny - 2)
    qy        = CUDA.zeros(Float64, nx - 2, ny - 1)
    qmag      = CUDA.zeros(Float64, nx - 2, ny - 2)
    RH        = CUDA.zeros(Float64, nx, ny)
    Err_rel   = CUDA.zeros(Float64, nx, ny)
    Err_abs   = CUDA.zeros(Float64, nx, ny)
    logAs     = copy(As)
    logAs_ini = copy(As)
    # init adjoint
    q̄x    = CUDA.zeros(Float64, nx - 1, ny - 2)
    q̄y    = CUDA.zeros(Float64, nx - 2, ny - 1)
    D̄     = CUDA.zeros(Float64, nx - 1, ny - 1)
    H̄     = CUDA.zeros(Float64, nx, ny)
    R̄H    = CUDA.zeros(Float64, nx, ny)
    logĀs = CUDA.zeros(Float64, nx - 1, ny - 1)
    Tmp    = CUDA.zeros(Float64, nx - 1, ny - 1)
    ψ_H    = CUDA.zeros(Float64, nx, ny)
    ∂J_∂H  = CUDA.zeros(Float64, nx, ny)


    # init convergence history
    cost_evo = Float64[]
    iter_evo = Float64[]

    # setup visualisation
    if do_vis
        #init visualization 
        fig = Figure(; resolution=(1600, 1200), fontsize=32)

        axs = (H    = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel=L"x", ylabel=L"y", title=L"H"),
               H_s  = Axis(fig[1, 2]; aspect=1, xlabel=L"H"),
               As   = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel=L"x", title=L"\log_{10}(A_s)"),
               As_s = Axis(fig[2, 2]; aspect=1, xlabel=L"\log_{10}(A_s)"),
               err  = Axis(fig[2, 3]; yscale=log10, title=L"convergence", xlabel="iter", ylabel=L"error"))

        nan_to_zero(x) = isnan(x) ? zero(x) : x

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
    fwd_params = (fields           = (; H, B, β, ELA, D, qx, qy, As, RH, qmag, Err_rel, Err_abs),
                  scalars          = (; aρgn0, b_max, npow),
                  numerical_params = (; nx, ny, dx, dy, maxiter, ncheck, ϵtol),
                  launch_config    = (; nthreads, nblocks))

    fwd_visu = (; plts, fig)

    adj_params = (fields=(; q̄x, q̄y, D̄, R̄H, ψ_H, H̄),
                  numerical_params=(; ϵtol_adj, ncheck_adj, H_cut))

    loss_params = (fields=(; H_obs, qobs_mag, ∂J_∂H),
                   scalars=(; w_H, w_q))

    #this is to switch on/off the smooth of the sensitivity 
    reg = (; nsm=20, Tmp)

    #Define loss functions 
    J(_logAs) = loss(logAs, fwd_params, loss_params)
    ∇J!(_logĀs, _logAs) = ∇loss!(logĀs, logAs, fwd_params, adj_params, loss_params; reg)
    @info "inversion for As"

    #initial guess
    As .= As_ini
    logAs .= log10.(As)
    γ = γ0
    J_old = 0.0
    J_new = 0.0
    #solve for H with As and compute J_old
    H .= 0.0
    @show maximum(logAs)
    J_old = J(logAs)
    J_ini = J_old

    @info "Gradient descent - inversion for As"
    for igd in 1:ngd
        #As_ini .= As
        println("GD iteration $igd \n")
        ∇J!(logĀs, logAs)
        γ = Δγ / maximum(abs.(logĀs))
        @. logAs -= γ * logĀs

        push!(iter_evo, igd)
        push!(cost_evo, J(logAs))
        #visualization 

        @printf " min(As) = %1.2e \n" minimum(exp10.(logAs))
        @printf " --> Loss J = %1.2e (γ = %1.2e) \n" last(cost_evo) / first(cost_evo) γ

        plts.H[3]       = Array(H)
        plts.H_s[2][1]  = Point2.(Array(H[nx÷2, :]), yc)
        plts.As[3]      = Array(log10.(As))
        plts.As_s[1][1] = Point2.(Array(log10.(As[nx÷2, :])), yc_1)
        plts.err[1]     = Point2.(iter_evo, cost_evo ./ 0.99cost_evo[1])
        display(fig)
    end
    return
end
