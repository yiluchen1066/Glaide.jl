using CairoMakie
using JLD2

n          = 3
ρ          = 910 
g          = 9.81

#jldsave("synthetic_snapshot.jld2"; logAs_syn = As_syn_v, logAs_snapshot = As_v, qmag_obs=qmag_obs_v, qmag_snapshot = qmag_v, H_obs=Array(H_obs), H=Array(H))
(logAs_syn, logAs_snapshot, qmag_obs, qmag_snapshot, H_obs, H_snapshot) = 
    load("synthetic_snapshot.jld2", "logAs_syn", "logAs_snapshot", "qmag_obs", "qmag_snapshot", "H_obs", "H_snapshot")
#jldsave("synthetic_timedepedent.jld2"; logAs_timedepedent = As_v, qmag_timedepedent = qmag_v, H = Array(H))
(logAs_timedepedent, qmag_timedepedent, H_timedepedent) = 
    load("synthetic_timedepedent.jld2", "logAs_timedepedent", "qmag_timedepedent", "H_timedepedent")

#from qmag convert to vmag 
v_obs = copy(qmag_obs)
v_snapshot = copy(qmag_snapshot)
v_timedepedent = copy(qmag_timedepedent)
n_ratio = (n+2)/(n+1)
v_obs .= qmag_obs ./ H_obs[2:end-1, 2:end-1]  .* n_ratio
v_snapshot .= qmag_snapshot ./ H_snapshot[2:end-1, 2:end-1] .* n_ratio
v_timedepedent .= qmag_timedepedent ./ H_timedepedent[2:end-1, 2:end-1] .* n_ratio

#from logAs convert to As
As_syn          = exp10.(logAs_syn)
As_snapshot     = exp10.(logAs_snapshot)
As_timedepedent = exp10.(logAs_timedepedent)

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
    As_syn  = Axis(fig[1, 1][1,1]; aspect=DataAspect(), ylabel = "Y [km]"),
    qmag_obs  = Axis(fig[1, 2][1,1]; aspect=DataAspect()),
    H_obs   = Axis(fig[1, 3][1,1]; aspect=DataAspect()),

    As_timedepedent  = Axis(fig[2, 1][1,1]; aspect=DataAspect(), ylabel = "Y [km]", xlabel="X [km]"),
    qmag_timedepedent  = Axis(fig[2, 2][1,1]; aspect=DataAspect(), xlabel="X [km]"),
    H_timedepedent   = Axis(fig[2, 3][1,1]; aspect=DataAspect(), xlabel="X [km]"),

    As_snapshot  = Axis(fig[3, 1][1,1]; aspect=DataAspect(), ylabel = "Y [km]"),
    qmag_snapshot  = Axis(fig[3, 2][1,1]; aspect=DataAspect()))

As_crange = filter(!isnan, As_syn) |> extrema
v_crange = filter(!isnan, v_obs) |> extrema
v_obs_max = filter(!isnan, v_obs) |> maximum
H_obs_max = filter(!isnan, H_obs) |> maximum

xlims!(ax.As_syn, -100, 100)
xlims!(ax.qmag_obs, -100, 100)
xlims!(ax.H_obs, -100, 100)
xlims!(ax.As_snapshot, -100, 100)
xlims!(ax.qmag_snapshot, -100, 100)
xlims!(ax.As_timedepedent, -100, 100)
xlims!(ax.qmag_timedepedent, -100, 100)
xlims!(ax.H_timedepedent, -100, 100)

hidexdecorations!(ax.As_syn; grid=false)
hidexdecorations!(ax.qmag_obs; grid=false)
hidexdecorations!(ax.H_obs; grid=false)
hidexdecorations!(ax.As_snapshot; grid=false)
hidexdecorations!(ax.qmag_snapshot; grid=false)


hideydecorations!(ax.qmag_obs; grid=false)
hideydecorations!(ax.H_obs; grid=false)
hideydecorations!(ax.qmag_snapshot; grid=false)
hideydecorations!(ax.qmag_timedepedent; grid=false)
hideydecorations!(ax.H_timedepedent; grid=false)

rowgap!(fig.layout, Relative(1/16))

plts = (
    As_syn    = heatmap!(ax.As_syn, xv./1000, yv./1000, log10.(As_syn); colormap=:turbo),
    qmag_obs  = heatmap!(ax.qmag_obs, xc[1:end-1]./1000, yc[1:end-1]./1000, v_obs; colormap=:turbo, colorrange=(1.0, 350.0)),
    H_obs     = heatmap!(ax.H_obs, xc./1000, yc./1000, H_obs; colormap=:turbo),

    As_timedepedent    = heatmap!(ax.As_timedepedent, xv./1000, yv./1000, log10.(As_timedepedent); colormap=:turbo),
    qmag_timedepedent  = heatmap!(ax.qmag_timedepedent, xc[1:end-1]./1000, yc[1:end-1]./1000, (abs.(v_timedepedent .- v_obs)./v_obs_max).*100; colormap=:turbo, colorrange=(0, 0.2)),
    H_timedepedent     = heatmap!(ax.H_timedepedent, xc./1000, yc./1000, (abs.(H_timedepedent .- H_obs) ./ H_obs).*100; colormap=:turbo, colorrange=(0, 0.2)),
    
    As_snapshot    = heatmap!(ax.As_snapshot, xv./1000, yv./1000, log10.(As_snapshot); colormap=:turbo),
    qmag_snapshot  = heatmap!(ax.qmag_snapshot, xc[1:end-1]./1000, yc[1:end-1]./1000, (abs.(v_snapshot .- v_obs)./v_obs_max).*100; colormap=:turbo, colorrange=(0, 0.2)))

Colorbar(fig[1,1][1,2], plts.As_syn)
Colorbar(fig[1,2][1,2], plts.qmag_obs)
Colorbar(fig[1,3][1,2], plts.H_obs)
Colorbar(fig[2,1][1,2], plts.As_timedepedent)
Colorbar(fig[2,2][1,2], plts.qmag_timedepedent)
Colorbar(fig[2,3][1,2], plts.H_timedepedent)
Colorbar(fig[3,1][1,2], plts.As_snapshot)
Colorbar(fig[3,2][1,2], plts.qmag_snapshot)



display(fig)
save("synthetic.png", fig)