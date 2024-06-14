function create_adjoint_debug_visualisation(params)
    # unpack
    (; nx, ny, dx, dy) = params.numerics
    (; ψ, H̄) = params.fields

    # preprocessing
    lx, ly = dx * nx, dy * ny
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)

    vis_fields = (ψ=Array(ψ),
                  H̄=Array(H̄))

    conv_hist = Point2{Float64}[]

    # make figure and title
    fig = Figure(; size=(700, 500), fontsize=12)
    Label(fig[0, 1:2], "DEBUG VISUALISATION (ADJOINT SOLVE)"; color=:red, font=:bold)

    # make axes
    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="adjoint state"),
           Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="adjoint residual"),
           Axis(fig[2, 2]; yscale=log10, title="convergence"))

    # make heatmaps
    hms = (heatmap!(axs[1], xc, yc, vis_fields.ψ; colormap=:turbo),
           heatmap!(axs[2], xc, yc, vis_fields.H̄; colormap=:turbo))

    # make line plots
    plt = scatterlines!(axs[3], conv_hist; label="relative")

    # make colorbars
    cbr = (Colorbar(fig[1, 1][1, 2], hms[1]),
           Colorbar(fig[1, 2][1, 2], hms[2]))

    display(fig)

    return (; fig, axs, hms, plt, cbr, vis_fields, conv_hist)
end

function update_adjoint_debug_visualisation!(vis, params, iter, errs)
    (; fig, axs, hms, plt, vis_fields, conv_hist) = vis

    # update convergence history
    push!(conv_hist, Point2(iter, errs.err_rel))

    # copy data from GPU to CPU for visualisation
    copy!(vis_fields.ψ, params.fields.ψ)
    copy!(vis_fields.H̄, params.fields.H̄)

    # update heatmaps
    hms[1][3] = vis_fields.ψ
    hms[2][3] = vis_fields.H̄

    # update plots
    plt[1] = conv_hist

    # update axis limits for plots
    # autolimits!(axs[3])

    display(fig)
    return
end
