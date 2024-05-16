using CairoMakie
using Enzyme


include("macros.jl")
include("snapshot_loss.jl")

# TODO then pass nx and ny instead of xc yc

function inversion_snapshot(logAs, geometry, observed, initial, physics, numerics, optim_params)
    (; B, xc, yc, nx, ny) = geometry
    (; qmag_obs, vmag_obs) = observed
    (; H_ini, As_ini, qmag, vmag) = initial
    (; npow, aρgn0) = physics
    (; ϵtol, maxiter) = numerics
    (; Δγ, ngd) = optim_params


    #pre-processing 
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
    surf_grad = CUDA.zeros(Float64, nx - 2, ny - 2)

    cost_evo = Float64[]
    iter_evo = Float64[]

    logAs     = log10.(As)
    logAs_ini = log10.(As_ini)

    # setup visualisation
    begin
        #init visualization 
        fig = Figure(; size=(800, 525))
        #opts    = (xaxisposition=:top,) # save for later 

        axs = (qmag_obs    = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x [km]", ylabel="y [km]", title=L"|q^\mathrm{obs}|"),
               qmag  = Axis(fig[1, 2][1,1]; aspect=DataAspect(), xlabel="x [km]", ylabel="y [km]",title="|q|"),
               As   = Axis(fig[1, 3][1, 1]; aspect=DataAspect(), xlabel="x [km]", ylabel="y [km]", title=L"\log_{10}(A_s)"),
               err  = Axis(fig[2, 3]; yscale=log10, title="convergence", xlabel="iter", ylabel=L"J/J_0"))

        nan_to_zero(x) = isnan(x) ? zero(x) : x

        # As_ini .= CuArray(asρgn0.*(1.0 .+ 1e1.*CUDA.rand(Float64, size(As_ini))))

        xc_1 = xc[1:(end-1)]    
        yc_1 = yc[1:(end-1)]

        logAs     = log10.(As)
        logAs_ini = log10.(As_ini)

        #qrng = extrema(replace(qmag_obs, NaN => 0.0))

        plts = (qmag_obs    = heatmap!(axs.qmag_obs, xc[2:end-1]./1000, yc[2:end-1]./1000, Array(qmag_obs); colormap=:turbo),
                qmag    = heatmap!(axs.qmag, xc[2:end-1]./1000, yc[2:end-1]./1000, Array(qmag); colormap=:turbo),
                # D    = heatmap!(axs.D, xc[1:end-1], yc[1:end-1], Array(D); colormap=:turbo),
                As   = heatmap!(axs.As, xc_1./1000, yc_1./1000, Array(logAs); colormap=:turbo),
                # As_v = vlines!(axs.As, xc_1[nx÷2]; linewidth=4, color=:magenta, linewtyle=:dash),
                #As_s = (lines!(axs.As_s, Point2.(Array(logAs[nx÷2, :]), yc_1); linewith=4, color=:blue, label="current"),
                #        lines!(axs.As_s, Point2.(Array(logAs_ini[nx÷2, :]), yc_1); linewidth=4, color=:green, label="initial")),
                err  = scatterlines!(axs.err, Point2.(iter_evo, cost_evo); linewidth=4))
    end


    fwd_params = (fields           = (; H, B, D, qx, qy, As, qmag),
                  scalars          = (; aρgn0, npow),
                  numerical_params = (; nx, ny, dx, dy, maxiter, ϵtol),
                  launch_config    = (; nthreads, nblocks))
    adj_params = (fields=(; q̄x, q̄y, D̄, H̄),)
    loss_params = (fields=(; qmag_obs),)

    # define reg
    reg = (; nsm=1, Tmp)

    J(_logAs) = loss_snapshot(logAs, fwd_params, loss_params)
    ∇J!(_logĀs, _logAs) = ∇loss_snapshot!(logĀs, logAs, fwd_params, adj_params, loss_params; reg)

    #initial guess 
    As .= As_ini
    logAs .= log10.(As)
    J0 = J(logAs)

    push!(cost_evo, 1.0)
    push!(iter_evo, 0)

    Mask = CUDA.ones(Float64, size(B))
    Mask[B.<= 0.0] .= 0.0

    @show typeof(Mask)
    @show typeof(logAs)

    #jldsave("output_snapshot/static.jld2"; qmag_obs, As_ini)

    @info "Gradient descent - inversion for As"
    for igd in 1:ngd
        println("GD iteration $igd \n")
        ∇J!(logĀs, logAs)
        # zero out where H is zero 
        γ = Δγ / maximum(abs.(logĀs))
        @. logAs -= γ * logĀs

        logAs .*= Mask[1:end-1, 1:end-1]

        if !isnothing(reg)
            (; nsm, Tmp) = reg
            Tmp .= logAs
            smooth!(logAs, Tmp, nsm, nthreads, nblocks)
        end
        As .= exp10.(logAs)


        push!(iter_evo, igd)
        J_new = J(logAs)
        push!(cost_evo, J_new / J0)

        @printf " min(As) = %1.2e \n" minimum(exp10.(logAs))
        @printf " --> Abs loss J = %1.2e Rel loss J = %1.2e (γ = %1.2e) \n" J_new last(cost_evo) γ

        if igd % 25 == 0
            #jldsave("/output_snapshot/step_$iframe.jld2"; As, qmag)
            # qmag_obs = replace(vmag_obs[2:end-1, 2:end-1], NaN => 0.0) .* H_obs[2:end-1, 2:end-1]
            # then vmag is also not defined
            vmag  = qmag ./ H[2:end-1, 2:end-1]
            q_vis = Array(qmag)
            v_vis = Array(vmag)
            H_vis = Array(H[2:end-1,2:end-1])
            H_vis_1 = Array(H[1:end-1,1:end-1])


            qobs_vis = Array(qmag_obs)
            vobs_vis = Array(vmag_obs[2:end-1, 2:end-1])

            q_vis[H_vis .< 0.1] .= NaN
            qobs_vis[H_vis .< 0.1] .= NaN
            v_vis[H_vis .< 0.1] .= NaN
            vobs_vis[H_vis .< 0.1] .= NaN

            logAs_vis = Array(logAs)
            logAs_vis[H_vis_1 .< 0.1] .= NaN
            
            plts.qmag[3]    = q_vis
            plts.qmag_obs[3]    = qobs_vis
            # plts.D[3]       = Array(D)
            plts.As[3]      = Array(logAs_vis)
            #plts.As_s[1][1] = Point2.(Array(logAs[nx÷2, :]), yc_1)
            plts.err[1]     = Point2.(iter_evo, cost_evo)

            if igd == 100
                Colorbar(fig[1, 1][1, 2], plts.qmag_obs)
                Colorbar(fig[1, 2][1, 2], plts.qmag)
                Colorbar(fig[1, 3][1, 2], plts.As)
            end

            if igd == 500
                jldsave("snapshot_Aletsch.jld2"; logAs = logAs_vis, q_obs = qobs_vis, q = q_vis, H = H_vis, xc = xc, yc = yc)
                save("snapshot_Aletsch.png", fig)
                display(fig)
            end
        end
    end
    return
end
