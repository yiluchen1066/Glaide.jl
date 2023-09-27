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

fig  = Figure(resolution=(1000,580),fontsize=22)
opts = (xaxisposition=:top,)

@show(gd_niter)

axs = (
    # H_1    = Axis(fig[1,1];xticks = 0:3:6,aspect=DataAspect(),xlabel=L"x\text{ [km]},\text{iter=1}",ylabel=L"y\text{ [km]}",opts...),
    # H_5    = Axis(fig[1,2];xticks = 0:3:6,aspect=DataAspect(),xlabel=L"x\text{ [km]},\text{iter=7}",ylabel=L"y\text{ [km]}",opts...),
    # H_15    = Axis(fig[1,3];xticks = 0:3:6,aspect=DataAspect(),xlabel=L"x\text{ [km]},\text{iter=15}",ylabel=L"y\text{ [km]}",opts...),
    # H_s_1  = Axis(fig[1,1];xticks = -20:20:20,xlabel=L"H-H_\mathrm{obs}\text{ [m]}, \text{iter=1}",ylabel=L"y\text{ [km]}",opts...),
    # H_s_5  = Axis(fig[1,2];xticks = -20:20:20,xlabel=L"H-H_\mathrm{obs}\text{ [m]}, \text{iter=7}",ylabel=L"y\text{ [km]}",opts...),
    # H_s_15  = Axis(fig[1,3];xticks = -20:20:20,xlabel=L"H-H_\mathrm{obs}\text{ [m]}, \text{iter=15}",ylabel=L"y\text{ [km]}",opts...),
    # As_1   = Axis(fig[1,1];xticks = 0:3:6,aspect=DataAspect(),xlabel=L"x\text{ [km]}, \text{iter=1}",ylabel=L"y\text{ [km]}",opts...),
    # As_5   = Axis(fig[1,2];xticks = 0:3:6,aspect=DataAspect(),xlabel=L"x\text{ [km]}, \text{iter=7}",ylabel=L"y\text{ [km]}",opts...),
    # As_15   = Axis(fig[1,3];xticks = 0:3:6,aspect=DataAspect(),xlabel=L"x\text{ [km]}, \text{iter=15}",ylabel=L"y\text{ [km]}",opts...),
    As_s_1 = Axis(fig[1,1];xlabel=L"\text{iter=1}",ylabel=L"y\text{ [km]}",opts...),
    As_s_5 = Axis(fig[1,2];xlabel=L"\text{iter=7}",ylabel=L"y\text{ [km]}",opts...),
    As_s_15 = Axis(fig[1,3];xlabel=L"\text{iter=15}",ylabel=L"y\text{ [km]}",opts...),
)

# CairoMakie.xlims!(axs.H_1,0,6)
# CairoMakie.xlims!(axs.H_5,0,6)
# CairoMakie.xlims!(axs.H_15,0,6)
# CairoMakie.xlims!(axs.H_s_1,-30,30)
# CairoMakie.xlims!(axs.H_s_5,-30,30)
# CairoMakie.xlims!(axs.H_s_15,-30,30)
# CairoMakie.xlims!(axs.As_1,0,6)
# CairoMakie.xlims!(axs.As_5,0,6)
# CairoMakie.xlims!(axs.As_15,0,6)
CairoMakie.xlims!(axs.As_s_1, -30,-15)
CairoMakie.xlims!(axs.As_s_5, -30,-15)
CairoMakie.xlims!(axs.As_s_15, -30,-15)

for axname in eachindex(axs)
    CairoMakie.ylims!(axs[axname],0,10)
end

#hideydecorations!(axs.H_s, ticks = false)
#hideydecorations!(axs.As, ticks = false)
#hideydecorations!(axs.As_s, ticks = false)

nan_to_zero(x) = isnan(x) ? zero(x) : x

(H_1, as_1) = load("rhone_data_output/rhone_1.jld2", "H", "as")
(H_5, as_5) = load("rhone_data_output/rhone_7.jld2", "H", "as")
(H_15, as_15) = load("rhone_data_output/rhone_15.jld2", "H", "as")
as_1      = log10.(as_1./(910*9.81)^3)
as_5      = log10.(as_5./(910*9.81)^3)
as_15      = log10.(as_15./(910*9.81)^3)

ΔH_ini  = replace!(nan_to_zero,H_obs[nx÷2,:].-H_ini_vis)
ΔH_1    = replace!(nan_to_zero,H_obs[nx÷2,:].-H_1[nx÷2,:])
ΔH_5    = replace!(nan_to_zero,H_obs[nx÷2,:].-H_5[nx÷2,:])
ΔH_15   = replace!(nan_to_zero,H_obs[nx÷2,:].-H_15[nx÷2,:])
as_s_1  = as_1[nx÷2,:]
as_s_5  = as_5[nx÷2,:]
as_s_15 = as_15[nx÷2,:]

plts = (
    # H_1   = CairoMakie.heatmap!(axs.H_1 ,xc_1, yc_1, H_1;colormap=:turbo,colorrange=(0,400)),
    # H_5   = CairoMakie.heatmap!(axs.H_5 ,xc_1, yc_1, H_5;colormap=:turbo,colorrange=(0,400)),
    # H_15  = CairoMakie.heatmap!(axs.H_15 ,xc_1, yc_1, H_15;colormap=:turbo,colorrange=(0,400)),
    #H_v = CairoMakie.vlines!(axs.H,xc_1[nx ÷ 2];color=:magenta,linewidth=4,linestyle=:dash),
    # H_s_1 = (
    #    CairoMakie.lines!(axs.H_s_1,ΔH_ini, yc_1; linewidth=4,color=:blue,label=L"\text{initial}"),
    #    CairoMakie.lines!(axs.H_s_1,ΔH_1, yc_1; linewidth=4,color=:red,label=L"\text{current}"),
    # ),
    # H_s_5 = (
    #    CairoMakie.lines!(axs.H_s_5,ΔH_ini, yc_1; linewidth=4,color=:blue,label=L"\text{initial}"),
    #    CairoMakie.lines!(axs.H_s_5,ΔH_5, yc_1; linewidth=4,color=:red,label=L"\text{current}"),
    # ),
    # H_s_15 = (
    #    CairoMakie.lines!(axs.H_s_15,ΔH_ini, yc_1; linewidth=4,color=:blue,label=L"\text{initial}"),
    #   CairoMakie.lines!(axs.H_s_15,ΔH_15, yc_1; linewidth=4,color=:red,label=L"\text{current}"),
    # ),
    # As_1   = CairoMakie.heatmap!(axs.As_1,xc_2, yc_2, as_1;colormap=:viridis,colorrange=(-30,-15)),
    # As_5   = CairoMakie.heatmap!(axs.As_5,xc_2, yc_2, as_5;colormap=:viridis,colorrange=(-30,-15)),
    # As_15  = CairoMakie.heatmap!(axs.As_15,xc_2, yc_2, as_15;colormap=:viridis,colorrange=(-30,-15)),
    #As_v = CairoMakie.vlines!(axs.As,xc_2[nx ÷ 2];color=:magenta,linewidth=4,linestyle=:dash),
    As_s_1 = (
       CairoMakie.lines!(axs.As_s_1,as_s_1, yc_2; linewidth=4, color=:green),
    ),
    As_s_5 = (
       CairoMakie.lines!(axs.As_s_5,as_s_5, yc_2; linewidth=4, color=:green),
    ),
    As_s_15 = (
       CairoMakie.lines!(axs.As_s_15,as_s_15, yc_2; linewidth=4, color=:green),
    ),
)

# axislegend(axs.H_s_1;position=:rb,labelsize=22)
# axislegend(axs.H_s_5;position=:rb,labelsize=22)
# axislegend(axs.H_s_15;position=:rb,labelsize=22)

# Colorbar(fig[2,:];limits=(0,400), vertical=false,label=L"H \text{ [m]}", labelsize)
# Colorbar(fig[2,:];limits=(-30,-15),vertical=false,label=L"\log_{10}(A_s)\text{ [Pa}^{-3}\text{m}^{2}\text{s}^{-1}\text{]}")

colgap!(fig.layout, 5)

#fig[2, :] = Label(fig, L"\log_{10}(A_s)\text{ [Pa}^{-3}\text{m}^{2}\text{s}^{-1}\text{]}")

#colsize!(fig.layout, 1, axs.H.scene.px_area[].widths[1])
#colsize!(fig.layout, 2, axs.H.scene.px_area[].widths[1])
#colsize!(fig.layout, 3, axs.As.scene.px_area[].widths[1])
#colsize!(fig.layout, 4, axs.As.scene.px_area[].widths[1])

#plts.H[3][] = H
#plts.As[3][] = as

#plts.H_s[2][1][]  = Point2.(ΔH,yc_1)
#plts.As_s[1][1][] = Point2.(as_s,yc_2)

save("adjoint_rhone_2D_As_s.png", fig)

# resize_to_layout!(fig)

#=
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
=#
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
