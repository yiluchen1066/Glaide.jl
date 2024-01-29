using CairoMakie

# load the data
#jldsave("output_steadystate/static.jld2"; qobs_mag, H_obs, As_ini, As_syn, xc, yc, xv, yv)
(qobs_mag, H_obs, As_ini, As_syn, xc, yc, xv, yv) = load("output_steadystate/static.jld2", "qobs_mag", "H_obs", "As_ini", "As_syn","xc", "yc", "xv", "yv")


gd_niter = 25

fig = Figure(; size=(800, 400), fontsize=18)
ax  = (As_s  = Axis(fig[1, 1]; aspect=DataAspect(), ylabel="y", title="observed"),
As_i  = Axis(fig[1, 2]; aspect=DataAspect(), title="modeled"),
q_s   = Axis(fig[2, 1]; aspect=DataAspect(), xlabel="x", ylabel="y"),
q_i   = Axis(fig[2, 2]; aspect=DataAspect(), xlabel="x"),
slice = Axis(fig[1, 3]; xlabel="x", ylabel="Aₛ"),
conv  = Axis(fig[2, 3]; xlabel="#iter", ylabel="J/J₀", yscale=log10))

#ylims!(ax.slice, -ly / 2, ly / 2)

linkyaxes!(ax.slice, ax.As_i, ax.As_s, ax.q_s, ax.q_i)

plts = (As_s  = (heatmap!(ax.As_s, xc, yc, Array(As_syn); colormap=:turbo),
        vlines!(ax.As_s, xc[nx÷2]; linestyle=:dash, color=:magenta)
        ),
        As_i  = (heatmap!(ax.As_i, xc, yc, Array(As); colormap=:turbo),
        contour!(ax.As_i, xc, yc, Array(H); levels=0.01:0.01, color=:white, linestyle=:dash),
        contour!(ax.As_i, xc, yc, Array(H_obs); levels=0.01:0.01, color=:red)),
        q_i   = heatmap!(ax.q_i, xc, yc, Array(qmag); colormap=:turbo),
        q_s   = heatmap!(ax.q_s, xc, yc, Array(qobs_mag); colormap=:turbo),
        slice = (lines!(ax.slice, Array(As_syn[nx÷2, :]), yc; label="observed"),
        lines!(ax.slice, Array(As[nx÷2, :]), yc; label="modeled")),
        conv  = scatterlines!(ax.conv, Point2.(iter_evo, cost_evo); linewidth=4))

lg = axislegend(ax.slice; labelsize=10, rowgap=-5, height=40)

display(fig)

record(fig, "steady_state.mp4", 1:gd_niter, framerate=1) do iframe
    file = "output_steadystate/step_$iframe.jld2"
    As, H, qmag, iter_evo, cost_evo = load(file, "As", "H", "qmag", "iter_evo", "cost_evo")
    plts.As_i[1][3] = Array(As)
    plts.As_i[3][3] = Array(H)
    plts.q_i[3] = Array(qmag)
    plts.conv[1] = Point2.(iter_evo, cost_evo)
end