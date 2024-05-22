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

fig = Figure(; size=(1300, 550), fontsize=14)
ax  = (
    As_snapshot     = Axis(fig[1, 1][1,1]; aspect=DataAspect(), ylabel = "Y [km]", title="a"),
    ΔAs_snapshot    = Axis(fig[1, 2][1,1]; aspect=DataAspect(), title="b"),
    qmag_snapshot   = Axis(fig[1, 3][1,1]; aspect=DataAspect(),  title="c"),
    As_timedepedent   = Axis(fig[2, 1][1,1]; aspect=DataAspect(), ylabel = "Y [km]", xlabel="X [km]", title="d"),
    ΔAs_timedepedent  = Axis(fig[2, 2][1,1]; aspect=DataAspect(), xlabel="X [km]", title="e"),
    qmag_timedepedent  = Axis(fig[2, 3][1,1]; aspect=DataAspect(), xlabel="X [km]", title="f"),
    H_timedepedent   = Axis(fig[2, 4][1,1]; aspect=DataAspect(), xlabel="X [km]", title="g"))

As_crange = filter(!isnan, As_syn) |> extrema
As_syn_max = filter(!isnan, As_syn) |> maximum
v_crange = filter(!isnan, v_obs) |> extrema
v_obs_max = filter(!isnan, v_obs) |> maximum
H_obs_max = filter(!isnan, H_obs) |> maximum


xlims!(ax.As_snapshot, -100, 100)
xlims!(ax.ΔAs_snapshot, -100, 100)
xlims!(ax.As_timedepedent, -100, 100)
xlims!(ax.ΔAs_timedepedent, -100, 100)
xlims!(ax.qmag_snapshot, -100, 100)
xlims!(ax.qmag_timedepedent, -100, 100)
xlims!(ax.H_timedepedent, -100, 100)

rowgap!(fig.layout, Relative(1/16))

hidexdecorations!(ax.As_snapshot, grid=false)
hidexdecorations!(ax.ΔAs_snapshot, grid=false)
hidexdecorations!(ax.qmag_snapshot, grid=false)
hideydecorations!(ax.qmag_snapshot, grid=false)
hideydecorations!(ax.ΔAs_snapshot, grid=false)
hideydecorations!(ax.ΔAs_timedepedent, grid=false)
hideydecorations!(ax.qmag_timedepedent, grid=false)
hideydecorations!(ax.H_timedepedent, grid=false)



plts = (
    As_snapshot     = heatmap!(ax.As_snapshot, xv./1000, yv./1000, log10.(As_snapshot); colormap=:GnBu_9),
    ΔAs_snapshot    = heatmap!(ax.ΔAs_snapshot, xv./1000, yv./1000, (abs.(As_snapshot-As_syn)./As_syn_max).*100; colormap=:GnBu_9, colorrange=(0, 10)),
    qmag_snapshot  = heatmap!(ax.qmag_snapshot, xc[1:end-1]./1000, yc[1:end-1]./1000, (abs.(v_snapshot .- v_obs)./v_obs_max).*100; colormap=:GnBu_9, colorrange=(0, 0.2)),
    As_timedepedent = heatmap!(ax.As_timedepedent, xv./1000, yv./1000, log10.(As_timedepedent); colormap=:GnBu_9),
    ΔAs_timedepedent  = heatmap!(ax.ΔAs_timedepedent, xv./1000, yv./1000, (abs.(As_timedepedent-As_syn)./As_syn_max).*100; colormap=:GnBu_9, colorrange=(0, 10)),
    qmag_timedepedent  = heatmap!(ax.qmag_timedepedent, xc[1:end-1]./1000, yc[1:end-1]./1000, (abs.(v_timedepedent .- v_obs)./v_obs_max).*100; colormap=:GnBu_9, colorrange=(0, 0.2)),
    H_timedepedent     = heatmap!(ax.H_timedepedent, xc./1000, yc./1000, (abs.(H_timedepedent .- H_obs) ./ H_obs_max).*100; colormap=:GnBu_9, colorrange=(0, 0.2)))

Colorbar(fig[1,1][1,2], plts.As_snapshot, label=L"(Pa^{-3}m^2s^{-1})")
Colorbar(fig[1,3][1,2], plts.qmag_snapshot, label=L"%")
Colorbar(fig[2,1][1,2], plts.As_timedepedent, label=L"(Pa^{-3}m^2s^{-1})")
Colorbar(fig[2,4][1,2], plts.H_timedepedent, label=L"%")



@show filter(!isnan, (abs.(v_snapshot .- v_obs)./v_obs_max).*100) |> extrema
@show filter(!isnan, (abs.(v_timedepedent .- v_obs)./v_obs_max).*100) |> extrema 
@show filter(!isnan, (abs.(H_timedepedent .- H_obs) ./ H_obs).*100) |> extrema 

display(fig)
save("synthetic_misfit.png", fig)