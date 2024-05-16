#init visualization 
fig     = Figure(resolution=(1600, 500), fontsize=32)
#opts    = (xaxisposition=:top,) # save for later 

axs     = (
    H   = Axis(fig[1,1]; aspect=DataAspect(), xlabel=L"x", ylabel=L"y"), 
    H_s = Axis(fig[1,2]; aspect=1, xlabel=L"H"),
    As  = Axis(fig[1,3]; aspect=DataAspect(), xlabel=L"x"),
    As_s= Axis(fig[1,4]; aspect=1, xlabel=L"\log_{10}(A_s)"),
)

#xlims CairoMakie.xlims!()
#ylims 

nan_to_zero(x) = isnan(x) ? zero(x) : x

logAs          = log10.(As)
logAs_syn      = log10.(As_syn)

H_s         = H[nx÷2,:]
H_obs_s     = H_obs[nx÷2,:]
As_ini_vis  = log10.(As_ini_vis)
As_s        = As[nx÷2,:]
As_ini_vis_s= As_ini_vis[nx÷2,:]
As_syn_s    = As_syn[nx÷2,:]

xc_1        = xc[1:end-1]
yc_1        = yc[1:end-1]


plts        =(
    H       = heatmap!(axs.H, xc, yc, Array(H); colormap=:turbo), 
    H_v     = vlines!(axs.H, xc[nx÷2]; color=:magenta, linewidth=4, linestyle=:dash),
    H_s     = (
        lines!(axs.H_s, Array(H_obs_s), yc; linewidth=4, color=:red, label="synthetic"),
        lines!(axs.H_s, Array(H_s), yc; linewidth=4, color=:blue, label="current"),
    ),
    As      = heatmap!(axs.As, xc_1, yc_1, Array(logAs); colormap=:viridis),
    As_v    = vlines!(axs.As, xc_1[nx÷2]; linewidth=4, color=:magenta, linewtyle=:dash),
    As_s    = (
        lines!(axs.As_s, Array(As_s), yc_1; linewith=4, color=:blue, label="current"),
        lines!(axs.As_s, Array(As_ini_vis_s), yc_1; linewidth=4, color=:green, label="initial"),
        lines!(axs.As_s, Array(As_syn_s), yc_1; linewidth=4, color=:red, label="synthetic"),
    ),
)

Colorbar(fig[1, 1][1, 2], plts.H)
Colorbar(fig[1, 3][1, 2], plts.As)