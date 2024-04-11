using CairoMakie
using JLD2

n          = 3
ρ          = 910 
g          = 9.81

(logAs, q_obs, q, H) = load("snapshot_Aletsch.jld2", "logAs", "q_obs", "q", "H")

#from qmag convert to vmag 
v_obs = copy(q_obs)
q = copy(q)
n_ratio = (n+2)/(n+1)
v_obs .= q_obs ./ H  .* n_ratio
q     .= q ./ H .* n_ratio

#from logAs convert to As
As          = exp10.(logAs)

#rescaling 
lsc_data      = 1e4
aρgn0_data    = 1.3517139631340709e-12
tsc_data      = 1 / aρgn0_data / lsc_data^n
s_f_syn       = 1e-4
lx_l          = 25.0
ly_l          = 20.0

As_syn        = As_syn * s_f_syn * aρgn0_data * lsc_data^n
As_snapshot   = As_snapshot * s_f_syn *aρgn0_data * lsc_data^n
As_timedepedent= As_timedepedent * s_f_syn *aρgn0_data * lsc_data^n

As_syn        = As_syn / (ρ*g)^n
As_snapshot   = As_snapshot / (ρ*g)^n
As_timedepedent= As_timedepedent / (ρ*g)^n


H_obs          = H_obs * lsc_data
H_snapshot     = H_snapshot * lsc_data
H_timedepedent = H_timedepedent * lsc_data 

v_obs          = v_obs * lsc_data / tsc_data
v_snapshot     = v_snapshot * lsc_data / tsc_data
v_timedepedent = v_timedepedent * lsc_data / tsc_data

lx             = lx_l * lsc_data
ly             = ly_l * lsc_data 
nx             = 128
ny             = 128 
dx, dy = lx / nx, ly / ny
xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)
xv = LinRange(-lx/2 + dx, lx/2 - dx, nx-1)
yv = LinRange(-ly/2 + dy, ly/2 - dy, ny-1)


# set H_obs, H_snapshot and H_timedepedent to be NaN where H is zeros
H_vert = @. 0.25 * (H_obs[1:end-1, 1:end-1] + H_obs[2:end,1:end-1] + H_obs[1:end-1,2:end] + H_obs[2:end,2:end])
idv = findall(H_vert .≈ 0.0) |> Array
H_obs[idv] .= NaN
H_snapshot[idv] .= NaN
H_timedepedent[idv] .= NaN

#convert the unit of v from m/s to m/a 
#just before visualization
v_obs .*= 365*24*3600
v_snapshot .*= 365*24*3600
v_timedepedent .*= 365*24*3600

fig = Figure(; size=(1200, 900), fontsize=14)
ax  = (
    As_syn  = Axis(fig[1, 1][1,1]; aspect=DataAspect(), subtitle="(a)", ylabel = "Y [km]", title=L"Synthetic As $[pa^{-3}m^2s^{-1}]$"),
    qmag_obs  = Axis(fig[1, 2][1,1]; aspect=DataAspect(), subtitle="(b)", title=L"Observed v $[m/a]$"),
    H_obs   = Axis(fig[1, 3][1,1]; aspect=DataAspect(), subtitle="(c)",title=L"Observed H $[m]$"),

    As_snapshot  = Axis(fig[2, 1][1,1]; aspect=DataAspect(), ylabel = "Y [km]",subtitle="(d)",title=L"Snapshot inverted As $[pa^{-3}m^2s^{-1}]$"),
    qmag_snapshot  = Axis(fig[2, 2][1,1]; aspect=DataAspect(), subtitle="(e)",title=L"Snapshot $\Delta v$ $[m/a]$"),
    H_snapshot   = Axis(fig[2, 3][1,1]; aspect=DataAspect(), subtitle="(f)",title=L"Snapshot $\Delta H$ $[m]$"),

    As_timedepedent  = Axis(fig[3, 1][1,1]; aspect=DataAspect(), ylabel = "Y [km]", xlabel="X [km]", subtitle="(g)",title=L"Time-dependent inverted As $[pa^{-3}m^2s^{-1}]$"),
    qmag_timedepedent  = Axis(fig[3, 2][1,1]; aspect=DataAspect(), xlabel="X [km]", subtitle="(h)",title=L"Time-dependent $\Delta v$ $[m/a]$"),
    H_timedepedent   = Axis(fig[3, 3][1,1]; aspect=DataAspect(), xlabel="X [km]", subtitle="(i)",title=L"Time-dependet $\Delta H$ $[m]$"))

As_crange = filter(!isnan, As_syn) |> extrema
v_crange = filter(!isnan, v_obs) |> extrema
v_obs_max = filter(!isnan, v_obs) |> maximum
H_obs_max = filter(!isnan, H_obs) |> maximum

plts = (
    As_syn    = heatmap!(ax.As_syn, xv./1000, yv./1000, As_syn; colormap=:turbo, colorrange=As_crange),
    qmag_obs  = heatmap!(ax.qmag_obs, xc[1:end-1]./1000, yc[1:end-1]./1000, v_obs; colormap=:turbo, colorrange=(1.0, 350.0)),
    H_obs     = heatmap!(ax.H_obs, xc./1000, yc./1000, H_obs; colormap=:turbo),
    
    As_snapshot    = heatmap!(ax.As_snapshot, xv./1000, yv./1000, As_snapshot; colormap=:turbo, colorrange=As_crange),
    qmag_snapshot  = heatmap!(ax.qmag_snapshot, xc[1:end-1]./1000, yc[1:end-1]./1000, abs.(v_snapshot .- v_obs)./v_obs_max; colormap=:turbo, colorrange=(0.0, 0.001)),
    H_snapshot     = heatmap!(ax.H_snapshot, xc./1000, yc./1000, abs.(H_snapshot .- H_obs) ./ H_obs_max; colormap=:turbo),

    As_timedepedent    = heatmap!(ax.As_timedepedent, xv./1000, yv./1000, As_timedepedent; colormap=:turbo, colorrange=As_crange),
    qmag_timedepedent  = heatmap!(ax.qmag_timedepedent, xc[1:end-1]./1000, yc[1:end-1]./1000, abs.(v_timedepedent .- v_obs)./v_obs_max; colormap=:turbo, colorrange=(0.0, 0.001)),
    H_timedepedent     = heatmap!(ax.H_timedepedent, xc./1000, yc./1000, abs.(H_timedepedent .- H_obs) ./ H_obs; colormap=:turbo, colorrange=(0.0, 0.001)))

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