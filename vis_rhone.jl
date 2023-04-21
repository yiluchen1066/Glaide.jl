using JLD2
using CairoMakie
# default(size=(1200,300),framestyle=:box,label=true,grid=true,margin=3mm,lw=3.5, xrotation = 45, labelfontsize=11,tickfontsize=11,titlefontsize=14)


(B, H_obs, H_ini_vis, xc, yc, nx, ny,gd_niter) = load("rhone_data_output/rhone_static.jld2", "B", "H_obs", "H_ini_vis",  "xc", "yc","nx", "ny", "gd_niter")

xc = xc .- xc[1]
yc = yc .- yc[1]

xc_1 = xc.*1e-3
yc_1 = yc.*1e-3
xc_2 = xc[1:end-1].*1e-3
yc_2 = yc[1:end-1].*1e-3

fig  = Figure(resolution=(1600,800),fontsize=36)
opts = (xaxisposition=:top,)

axs = (
    H    = Axis(fig[1,1];xticks = 0:3:6,aspect=DataAspect(),xlabel=L"x\text{ [km]}",ylabel=L"y\text{ [km]}",opts...),
    H_s  = Axis(fig[1,2];xticks = -20:20:20,xlabel=L"H-H_\mathrm{obs}\text{ [m]}",opts...),
    As   = Axis(fig[1,3];xticks = 0:3:6,aspect=DataAspect(),xlabel=L"x\text{ [km]}",opts...),
    As_s = Axis(fig[1,4];xlabel=L"\log_{10}(A_s)\text{ [Pa}^{-3}\text{m}^{2}\text{s}^{-1}\text{]}",opts...),
)

CairoMakie.xlims!(axs.H,0,6)
CairoMakie.xlims!(axs.H_s,-30,30)
CairoMakie.xlims!(axs.As,0,6)
CairoMakie.xlims!(axs.As_s, -30,-15)

for axname in eachindex(axs)
    CairoMakie.ylims!(axs[axname],0,10)
end

hideydecorations!(axs.H_s, ticks = false)
hideydecorations!(axs.As, ticks = false)
hideydecorations!(axs.As_s, ticks = false)

nan_to_zero(x) = isnan(x) ? zero(x) : x

(H, as) = load("rhone_data_output/rhone_1.jld2", "H", "as")
as      = log10.(as./(910*9.81)^3)

ΔH_ini  = replace!(nan_to_zero,H_obs[nx÷2,:].-H_ini_vis)
ΔH      = replace!(nan_to_zero,H_obs[nx÷2,:].-H[nx÷2,:])
as_s    = as[nx÷2,:]

plts = (
    H   = CairoMakie.heatmap!(axs.H ,xc_1, yc_1, H;colormap=:turbo,colorrange=(0,400)),
    H_v = CairoMakie.vlines!(axs.H,xc_1[nx ÷ 2];color=:magenta,linewidth=4,linestyle=:dash),
    H_s = (
        CairoMakie.lines!(axs.H_s,ΔH_ini, yc_1; linewidth=4,color=:blue,label="initial"),
        CairoMakie.lines!(axs.H_s,ΔH, yc_1; linewidth=4,color=:red,label="current"),
    ),
    As   = CairoMakie.heatmap!(axs.As,xc_2, yc_2, as;colormap=:viridis,colorrange=(-30,-15)),
    As_v = CairoMakie.vlines!(axs.As,xc_2[nx ÷ 2];color=:magenta,linewidth=4,linestyle=:dash),
    As_s = (
        CairoMakie.lines!(axs.As_s,as_s, yc_2; linewidth=4, color=:green),
    ),
)

axislegend(axs.H_s;position=:rb,labelsize=30)

cb = Colorbar(fig[2,1],plts.H;vertical=false,label=L"H \text{ [m]}")
Colorbar(fig[2,3],plts.As;vertical=false,label=L"\log_{10}(A_s)\text{ [Pa}^{-3}\text{m}^{2}\text{s}^{-1}\text{]}")

colgap!(fig.layout, 50)

colsize!(fig.layout, 1, axs.H.scene.px_area[].widths[1])
colsize!(fig.layout, 2, axs.H.scene.px_area[].widths[1])
colsize!(fig.layout, 3, axs.As.scene.px_area[].widths[1])
colsize!(fig.layout, 4, axs.As.scene.px_area[].widths[1])

# resize_to_layout!(fig)

CairoMakie.record(fig, "adjoint_rhone_2D.mp4", 1:gd_niter; framerate=3) do gd_iter
    (H, as) = load("rhone_data_output/rhone_$gd_iter.jld2", "H", "as")
    as   = log10.(as./(910*9.81)^3)
    ΔH      = replace!(nan_to_zero,H_obs[nx÷2,:].-H[nx÷2,:])
    as_s    = as[nx÷2,:]

    plts.H[3][] = H
    plts.As[3][] = as

    plts.H_s[2][1][]  = Point2.(ΔH,yc_1)
    plts.As_s[1][1][] = Point2.(as_s,yc_2)
end

# anim = @animate for gd_iter = 1:1
#     (H, as) = load("rhone_data_output/rhone_$gd_iter.jld2", "H", "as")
#     as   = log10.(as./(910*9.81)^3)
#     p1=heatmap(xc_1, yc_1, Array(H'); xlabel ="X (km)", ylabel="Y (km)", xlims=extrema(xc_1), ylims=extrema(yc_1), title ="Ice thickness", levels=20, color =:turbo, aspect_ratio = 1,cbar=true)
#     vline!(p1, [xc_1[nx ÷ 2]], label=false)
#     p2=plot(Array((H_obs[nx÷2,:].-H[nx÷2,:])),yc_1; xlabel="ΔH", ylabel="Y (km)", xlims=(-30, 30),ylims=extrema(yc_1), title="H_H_obs", label="Current ΔH",  legend=true)
#     plot!(Array(H_obs[nx÷2,:]).-H_ini_vis,yc_1;xlabel="ΔH (m)",ylabel="Y (km)", ylims=extrema(yc_1), label="Initial ΔH", legend=true)
#     p3=heatmap(xc_2, yc_2, Array(as'); xlabel="X (km)", ylabel="Y (km)", xlims=extrema(xc_2), ylims=extrema(yc_2),label="as", title="Sliding coefficient as", aspect_ratio=1)
#     p4=plot(Array(as[nx÷2,:]),yc_2; xlabel="log10(as) (Pa^(−3) m^2 s^(−1))", ylabel="Y (km)", title="Sliding coefficient as",color=:blue, label=false)
#     #p5 = plot(iter_evo, J_evo; shape = :circle, xlabel="Iterations", ylabel="J_old/J_ini", title="misfit", yaxis=:log10)
#     display(plot(p1,p2,p3,p4; layout=grid(1,4,widths=[0.35,0.15,0.35,0.15])))
             
# end 

# gif(anim, "adjoint_rhone_2D.gif"; fps=1)
