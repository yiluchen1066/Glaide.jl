# forward solver visualisation

function create_debug_visualisation(model)
    # unpack
    (; H, B, As, V)    = model.fields
    (; A, ρgn, npow)   = model.scalars
    (; dx, dy, xc, yc) = model.numerics

    surface_velocity!(V, H, B, As, A, ρgn, npow, dx, dy)

    vis_fields = (V  = Array(V),
                  H  = Array(H),
                  As = Array(log10.(As)))

    conv_hist = (err_abs = Point2{Float64}[],
                 err_rel = Point2{Float64}[])

    # make figure and title
    fig = Figure(; size=(800, 500), fontsize=12)
    Label(fig[0, 1:2], "DEBUG VISUALISATION"; color=:red, font=:bold)

    # make axes
    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="ice thickness"),
           Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="surface velocity"),
           Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="sliding coefficient (log10)"),
           Axis(fig[2, 2]; yscale=log10, title="convergence"))

    # make heatmaps
    hms = (heatmap!(axs[1], xc, yc, vis_fields.H; colormap=:turbo),
           heatmap!(axs[2], xc, yc, vis_fields.V; colormap=:turbo),
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

function update_debug_visualisation!(vis, model, iter, errs)
    (; fig, axs, hms, plt, vis_fields, conv_hist) = vis

    # unpack
    (; H, B, As, V)  = model.fields
    (; A, ρgn, npow) = model.scalars
    (; dx, dy)       = model.numerics

    surface_velocity!(V, H, B, As, A, ρgn, npow, dx, dy)

    # update convergence history
    push!(conv_hist.err_abs, Point2(iter, errs.err_abs))
    push!(conv_hist.err_rel, Point2(iter, errs.err_rel))

    # copy data from GPU to CPU for visualisation
    copy!(vis_fields.H, H)
    copy!(vis_fields.V, V)
    copy!(vis_fields.As, log10.(As))

    # update heatmaps
    hms[1][3] = vis_fields.H
    hms[2][3] = vis_fields.V
    hms[3][3] = vis_fields.As

    # update plots
    plt[1][1] = conv_hist.err_abs
    plt[2][1] = conv_hist.err_rel

    # update axis limits for plots
    autolimits!(axs[4])

    display(fig)
    return
end

# adjoint solver visualisation

function create_adjoint_debug_visualisation(model)
    # unpack
    (; ψ, H̄)  = model.adjoint_fields
    (; xc, yc) = model.numerics

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

function update_adjoint_debug_visualisation!(vis, model, iter, errs)
    (; fig, hms, plt, vis_fields, conv_hist) = vis

    # update convergence history
    push!(conv_hist, Point2(iter, errs.err_rel))

    # copy data from GPU to CPU for visualisation
    copy!(vis_fields.ψ, model.adjoint_fields.ψ)
    copy!(vis_fields.H̄, model.adjoint_fields.H̄)

    # update heatmaps
    hms[1][3] = vis_fields.ψ
    hms[2][3] = vis_fields.H̄

    # update plots
    plt[1] = conv_hist

    display(fig)
    return
end
