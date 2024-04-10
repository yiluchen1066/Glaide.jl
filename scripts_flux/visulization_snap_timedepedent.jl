using CairoMakie
using JLD2

n     = 3
lsc   = 1.0
lx_l, ly_l           = 25.0, 20.0

lx, ly     = lx_l * lsc, ly_l * lsc
nx, ny     = 128, 128
dx, dy     = lx / nx, ly / ny
xc         = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
yc         = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)
xc_1       = xc[1:(end-1)]
yc_1       = yc[1:(end-1)]
xv = LinRange(-lx/2 + dx, lx/2 - dx, nx-1)
yv = LinRange(-ly/2 + dy, ly/2 - dy, ny-1)

#jldsave("synthetic_snapshot.jld2"; logAs_syn = As_syn_v, logAs_snapshot = As_v, qmag_obs=qmag_obs_v, qmag_snapshot = qmag_v, H_obs=Array(H_obs), H=Array(H))
(logAs_syn, logAs_snapshot, qmag_obs, qmag_snapshot, H_obs, H_snapshot) = 
    load("synthetic_snapshot.jld2", "logAs_syn", "logAs_snapshot", "qmag_obs", "qmag_snapshot", "H_obs", "H_snapshot")
#jldsave("synthetic_timedepedent.jld2"; logAs_timedepedent = As_v, qmag_timedepedent = qmag_v, H = Array(H))
(logAs_timedepedent, qmag_timedepedent, H_timedepedent) = 
    load("synthetic_timedepedent.jld2", "logAs_timedepedent", "qmag_timedepedent", "H_timedepedent")

# set H_obs, H_snapshot and H_timedepedent to be NaN where H is zeros
H_vert = @. 0.25 * (H_obs[1:end-1, 1:end-1] + H_obs[2:end,1:end-1] + H_obs[1:end-1,2:end] + H_obs[2:end,2:end])
idv = findall(H_vert .≈ 0.0) |> Array
H_obs[idv] .= NaN
H_snapshot[idv] .= NaN
H_timedepedent[idv] .= NaN

#convert from qmag = v 
v_obs = copy(qmag_obs)
v_snapshot = copy(qmag_snapshot)
v_timedepedent = copy(qmag_timedepedent)
n_ratio = (n+2)/(n+1)
v_obs .= qmag_obs ./ H_obs[2:end-1, 2:end-1]  .* n_ratio
v_snapshot .= qmag_snapshot ./ H_snapshot[2:end-1, 2:end-1] .* n_ratio
v_timedepedent .= qmag_timedepedent ./ H_timedepedent[2:end-1, 2:end-1] .* n_ratio
v_obs .*= 365*24*3600
v_snapshot .*= 365*24*3600
v_timedepedent .*= 365*24*3600

fig = Figure(; size=(1200, 900), fontsize=14)
ax  = (
    As_syn  = Axis(fig[1, 1][1,1]; aspect=DataAspect(), ylabel = "Y", title="Synthetic As"),
    qmag_obs  = Axis(fig[1, 2][1,1]; aspect=DataAspect(), title="Observed v"),
    H_obs   = Axis(fig[1, 3][1,1]; aspect=DataAspect(), title="Observed H"),

    As_snapshot  = Axis(fig[2, 1][1,1]; aspect=DataAspect(), ylabel = "Y",title="Inverted As"),
    qmag_snapshot  = Axis(fig[2, 2][1,1]; aspect=DataAspect(), title="Modeled v - observed v"),
    H_snapshot   = Axis(fig[2, 3][1,1]; aspect=DataAspect(), title="Modeled H"),

    As_timedepedent  = Axis(fig[3, 1][1,1]; aspect=DataAspect(), ylabel = "Y", xlabel="X", title="Inverted As"),
    qmag_timedepedent  = Axis(fig[3, 2][1,1]; aspect=DataAspect(), xlabel="X", title="Modeled v  - observed v"),
    H_timedepedent   = Axis(fig[3, 3][1,1]; aspect=DataAspect(), xlabel="X", title="Modeled H"))

As_crange = filter(!isnan, logAs_syn) |> extrema
v_crange = filter(!isnan, v_obs) |> extrema
v_obs_max = filter(!isnan, v_obs) |> maximum
H_obs_max = filter(!isnan, H_obs) |> maximum


plts = (
    As_syn    = heatmap!(ax.As_syn, xv, yv, logAs_syn; colormap=:turbo, colorrange=As_crange),
    qmag_obs  = heatmap!(ax.qmag_obs, xc_1, yc_1, v_obs; colormap=:turbo, colorrange=(0.0, 3e-2)),
    H_obs     = heatmap!(ax.H_obs, xc, yc, H_obs; colormap=:turbo),
    
    As_snapshot    = heatmap!(ax.As_snapshot, xv, yv, logAs_snapshot; colormap=:turbo, colorrange=As_crange),
    qmag_snapshot  = heatmap!(ax.qmag_snapshot, xc_1, yc_1, abs.(v_snapshot .- v_obs)./v_obs_max; colormap=:turbo, colorrange=(0.0, 0.001)),
    H_snapshot     = heatmap!(ax.H_snapshot, xc, yc, abs.(H_snapshot .- H_obs) ./ H_obs_max; colormap=:turbo),

    As_timedepedent    = heatmap!(ax.As_timedepedent, xv, yv, logAs_timedepedent; colormap=:turbo, colorrange=As_crange),
    qmag_timedepedent  = heatmap!(ax.qmag_timedepedent, xc_1, yc_1, abs.(v_timedepedent .- v_obs)./v_obs_max; colormap=:turbo, colorrange=(0.0, 0.001)),
    H_timedepedent     = heatmap!(ax.H_timedepedent, xc, yc, abs.(H_timedepedent .- H_obs) ./ H_obs; colormap=:turbo, colorrange=(0.0, 0.001)))

Colorbar(fig[1,1][1,2], plts.As_syn)
Colorbar(fig[1,2][1,2], plts.qmag_obs)
Colorbar(fig[1,3][1,2], plts.H_obs)
Colorbar(fig[2,1][1,2], plts.As_snapshot)
Colorbar(fig[2,2][1,2], plts.qmag_snapshot)
Colorbar(fig[2,3][1,2], plts.H_snapshot)
Colorbar(fig[3,1][1,2], plts.As_timedepedent)
Colorbar(fig[3,2][1,2], plts.qmag_timedepedent)
Colorbar(fig[3,3][1,2], plts.H_timedepedent)


display(fig)
save("synthetic.png", fig)