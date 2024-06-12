function create_debug_visualisation(fields, numerics)
    # unpack
    (; nx, ny, dx, dy) = numerics
    (; H, qx, qy, As)  = fields

    # preprocessing
    lx, ly = dx * nx, dy * ny
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)

    vis_fields = (q  = Array(vmag(qx, qy)),
                  H  = Array(H),
                  As = Array(log10.(As)))

    conv_hist = (err_abs = Point2{Float64}[],
                 err_rel = Point2{Float64}[])

    # make figure and title
    fig = Figure(; size=(800, 500), fontsize=12)
    Label(fig[0, 1:2], "DEBUG VISUALISATION"; color=:red, font=:bold)

    # make axes
    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="ice thickness"),
           Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="flux magnitude"),
           Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="sliding coefficient (log10)"),
           Axis(fig[2, 2]; yscale=log10, title="convergence"))

    # make heatmaps
    hms = (heatmap!(axs[1], xc, yc, vis_fields.H; colormap=:turbo),
           heatmap!(axs[2], xc, yc, vis_fields.q; colormap=:turbo),
           heatmap!(axs[3], xc, yc, vis_fields.As; colormap=:turbo))

    # make line plots
    plt = (scatterlines!(axs[4], conv_hist.err_abs; label="absolute"),
           scatterlines!(axs[4], conv_hist.err_rel; label="relative"))

    # make colorbars
    cbr = (Colorbar(fig[1, 1][1, 2], hms[1]),
           Colorbar(fig[1, 2][1, 2], hms[2]),
           Colorbar(fig[2, 1][1, 2], hms[3]))

    display(fig)

    return (; fig, axs, hms, plt, cbr, vis_fields, conv_hist)
end

function update_debug_visualisation!(vis, fields, iter, errs)
    (; fig, axs, hms, plt, cbr, vis_fields, conv_hist) = vis

    # update convergence history
    push!(conv_hist.err_abs, Point2(iter, errs.err_abs))
    push!(conv_hist.err_rel, Point2(iter, errs.err_rel))

    # copy data from GPU to CPU for visualisation
    copy!(vis_fields.H, fields.H)
    copy!(vis_fields.q, vmag(fields.qx, fields.qy))
    copy!(vis_fields.As, log10.(fields.As))

    # update heatmaps
    hms[1][3] = vis_fields.H
    hms[2][3] = vis_fields.q
    hms[3][3] = vis_fields.As

    # update plots
    plt[1][1] = conv_hist.err_abs
    plt[2][1] = conv_hist.err_rel

    # update axis limits for plots
    autolimits!(axs[4])

    display(fig)
    return
end