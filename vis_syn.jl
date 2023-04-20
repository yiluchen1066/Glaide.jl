using JLD2
using CairoMakie


(B, H_obs, as_syn, as_ini_vis, xc, yc, nx, ny,gd_niter) = load("synthetic_data_output/synthetic_static.jld2", "B", "H_obs", "as_syn", "as_ini_vis", "xc", "yc","nx", "ny", "gd_niter")

#xc = xc .- xc[1]
#yc = yc .- yc[1]
xc_1 = xc[1:end-1]
yc_1 = yc[1:end-1]

fig   = Figure(resolution=(1600, 580), fontsize =32)
opts = (xaxisposition=:top,)

axs  = (
    # xticks
    H    = Axis(fig[1,1]; xticks=-10:5:10, aspect=DataAspect(), xlabel=L"x",ylabel=L"y",opts...),
    H_s  = Axis(fig[1,2]; xticks=0.005:0.01:0.02,aspect=1, xlabel=L"H", opts...),
    As   = Axis(fig[1,3]; xticks=-10:5:10, aspect=DataAspect(), xlabel=L"x",opts...),
    As_s = Axis(fig[1,4]; aspect=1, xlabel=L"\log_{10}(A_s)",opts...),
)

# xlims
CairoMakie.xlims!(axs.H, -10, 10)
CairoMakie.xlims!(axs.H_s, 0.003, 0.02)
CairoMakie.xlims!(axs.As, -10,10)
CairoMakie.xlims!(axs.As_s, -2.2, -1.4)

#ylims 
for axname in eachindex(axs)
    CairoMakie.ylims!(axs[axname], -10, 10)
end 

#hide decorations
#hidedecorations!(axs.H_s, ticks=false)
#hidedecorations!(axs.As, ticks=(-10, 5, 10))
#hidedecorations!(axs.As_s, ticks=false)

nan_to_zero(x) = isnan(x) ? zero(x) : x

(H, as) = load("synthetic_data_output/synthetic_1.jld2", "H", "as")
as          = log10.(as)

H_s         = H[nx÷2,:]
H_obs_s     = H_obs[nx÷2,:]
as_ini_vis  = log10.(as_ini_vis)
as_syn      = log10.(as_syn)
as_s        = as[nx÷2, :]
as_ini_vis_s = as_ini_vis[nx÷2,:]
as_syn_s    = as_syn[nx÷2,:]


plts = (
    #colorrange=(0, 400)
    H = CairoMakie.heatmap!(axs.H, xc, yc, H; colormap=:turbo,colorrange=(0.002, 0.02)),
    H_v = CairoMakie.vlines!(axs.H, xc[nx÷2]; color=:magenta, linewidth=4, linestyle=:dash),
    H_s = (
        CairoMakie.lines!(axs.H_s, H_obs_s, yc; linewidth=4, color=:red, label="initial"), 
        CairoMakie.lines!(axs.H_s, H_s, yc; linewidth=4, color=:blue, label="current"),
        
    ), 
    #colorrange= (-30, -10)
    As  = CairoMakie.heatmap!(axs.As, xc_1, yc_1, as; colormap=:viridis,colorrange=(-2.0, -1.55)), 
    As_v = CairoMakie.vlines!(axs.As, xc_1[nx÷2]; linewidth=4, color=:magenta, linestyle=:dash), 
    As_s = (
        CairoMakie.lines!(axs.As_s, as_s, yc_1; linewidth =4, color=:blue, label="current"), 
        CairoMakie.lines!(axs.As_s, as_ini_vis_s, yc_1; linewidth=4, color=:green, label="initial"), 
        CairoMakie.lines!(axs.As_s, as_syn_s, yc_1; linewidth=4, color=:red, label="synthetic"), 
    ),
)

axislegend(axs.H_s; position=:lb, labelsize=20)
axislegend(axs.As_s; position=:lb,labelsize=20)

cb = Colorbar(fig[2,1], plts.H; vertical=false, label=L"H \text{ [m]}", ticksize=4.0)
Colorbar(fig[2,3], plts.As; vertical=false, label=L"\log_{10}(A_s)", ticksize=4.0)

colgap!(fig.layout, 50)

colsize!(fig.layout, 1, axs.H.scene.px_area[].widths[1])
colsize!(fig.layout, 2, axs.H.scene.px_area[].widths[1])
colsize!(fig.layout, 3, axs.As.scene.px_area[].widths[1])
colsize!(fig.layout, 4, axs.As.scene.px_area[].widths[1])

CairoMakie.record(fig, "adjoint_synthetic_2D.mp4", 1:gd_niter; framerate=3) do gd_iter
    (H, as) = load("synthetic_data_output/synthetic_$gd_iter.jld2", "H", "as")
    as   = log10.(as)
    as_s = as[nx÷2,:]
    H_s  = H[nx÷2,:]
    as_s = as[nx÷2,:] 


    plts.H[3][] = H
    plts.As[3][] = as

    plts.H_s[2][1][]  = Point2.(H_s,yc)
    plts.As_s[1][1][] = Point2.(as_s,yc_1)
end

#gap between colorbar 
#

#= anim = @animate for gd_iter = 1:gd_niter 
    (H, as) = load("synthetic_data/synthetic_$gd_iter.jld2", "H", "as")
    p1=heatmap(xc, yc, Array(H'); xlabel ="X", ylabel="Y", title ="Ice thickness", xlims=extrema(xc), ylims=extrema(yc),levels=20, color =:turbo, aspect_ratio = 1,cbar=true)
    vline!(xc[nx ÷2])
    p2=plot(Array(H[nx÷2,:]),yc;xlabel="H",ylabel="Y", label="Current H (cross section)", legend=:bottom)
    plot!(Array(H_obs[nx÷2,:]),yc; xlabel="H", ylabel="Y", title="Ice thickness", label="Synthetic H (cross section)",  legend=:bottom)
             
    p3=heatmap(xc[1:end-1], yc[1:end-1], Array(log10.(as)'); xlabel="X", ylabel="Y", xlims=extrema(xc), ylims=extrema(yc),label="as", title="Sliding coefficient as", aspect_ratio=1)
    p4=plot(Array(log10.(as[nx÷2,:])),yc[1:end-1]; xlabel="as", ylabel="Y", title="Sliding coefficient as",color=:blue, lw = 3, label="Current as (cross section)", legend=true)
    plot!(Array(log10.(as_ini_vis[nx÷2,:])),yc[1:end-1]; xlabel="as", ylabel="Y", color=:green, lw=3, label="Initial as for inversion", legend=true)
    plot!(Array(log10.(as_syn[nx÷2,:])),yc[1:end-1];xlabel="as", ylabel="Y", color=:red, lw= 3, label="Synthetic as", legend=true)
    #p5 = plot(iter_evo, J_evo; shape = :circle, xlabel="Iterations", ylabel="J_old/J_ini", title="misfit", yaxis=:log10)
    display(plot(p1,p2,p3,p4; layout=(2,2)))
    #display(plot(p5;  size=(490,490)))
             
end 

gif(anim, "adjoint_bench_2D.gif"; fps=5)
 =#