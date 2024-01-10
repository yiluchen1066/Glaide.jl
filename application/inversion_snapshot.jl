using CairoMakie
using Enzyme

include("macros.jl")
include("snapshot_loss.jl")

function inversion_snapshot(logAs, geometry, observed, initial, physics, numerics, optim_params)
    (; B, xc, yc) = geometry
    (; H_obs, qobs_mag) = observed
    (; H_ini, As_ini) = initial
    (; npow, aρgn0, β, ELA, γ0) = physics
    (; ϵtol, maxiter) = numerics
    (; Δγ, ngd) = optim_params

    #pre-processing 
    nx       = length(xc)
    ny       = length(yc)
    dx       = xc[2] - xc[1]
    dy       = yc[2] - yc[1]
    nthreads = (16, 16)
    nblocks  = ceil.(Int, (nx, ny) ./ nthreads)

    ## init arrays
    # ice thickness
    H         = copy(H_ini)
    D         = CUDA.zeros(Float64, nx - 1, ny - 1)
    As        = CUDA.zeros(Float64, nx-1, ny - 1)
    qx        = CUDA.zeros(Float64, nx - 1, ny - 2)
    qy        = CUDA.zeros(Float64, nx - 2, ny - 1)
    logAs_ini = copy(logAs)
    # init adjoint storage
    q̄x      = CUDA.zeros(Float64, nx - 1, ny - 2)
    q̄y      = CUDA.zeros(Float64, nx - 2, ny - 1)
    D̄       = CUDA.zeros(Float64, nx - 1, ny - 1)
    H̄       = CUDA.zeros(Float64, nx, ny)
    logĀs   = CUDA.zeros(Float64, nx - 1, ny - 1)
    Tmp      = CUDA.zeros(Float64, nx - 1, ny - 1)
    qmag     = CUDA.zeros(Float64, nx - 2, ny - 2)
    qobs_mag = CUDA.zeros(Float64, nx - 2, ny - 2)

    cost_evo = Float64[]
    iter_evo = Float64[]

    # setup visualisation
    begin
        #init visualization 
        fig = Figure(; size=(1000, 800), fontsize=32)
        #opts    = (xaxisposition=:top,) # save for later 

        axs = (H    = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel=L"x", ylabel=L"y", title=L"H"),
               H_s  = Axis(fig[1, 2]; aspect=1, xlabel=L"H"),
               As   = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel=L"x", title=L"\log_{10}(A_s)"),
               As_s = Axis(fig[2, 2]; aspect=1, xlabel=L"\log_{10}(A_s)"),
               err  = Axis(fig[2, 3]; yscale=log10, title=L"convergence", xlabel="iter", ylabel=L"error"))

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

    #here I wrapped As into fwd_params, with 
    #pack parameters
    fwd_params = (fields           = (; H, B, β, ELA, D, qx, qy, As, qmag),
                  scalars          = (; aρgn0, npow),
                  numerical_params = (; nx, ny, dx, dy, maxiter, ϵtol),
                  launch_config    = (; nthreads, nblocks))
    adj_params = (fields=(; q̄x, q̄y, D̄, H̄),)
    loss_params = (fields=(; qx_obs, qy_obs, qobs_mag),)

    # define reg
    reg = (; nsm=100, Tmp)

    J(_logAs) = loss_snapshot(logAs, fwd_params, loss_params)
    ∇J!(_logĀs, _logAs) = ∇loss_snapshot!(logĀs, logAs, fwd_params, adj_params, loss_params; reg)

    #initial guess 
    As .= As_ini
    logAs .= log10.(As)
    γ = γ0
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
        push!(cost_evo, J(logAs) / J0)

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
