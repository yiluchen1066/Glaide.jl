using CairoMakie
using JLD2

n          = 3
ρ          = 910 
g          = 9.81
(logAs_sp_Aletsch, qmag_obs, qmag_snapshot, H_snapshot, xc, yc, iter_sp, cost_sp) = 
    load("snapshot_Aletsch.jld2", "logAs", "q_obs", "q", "H", "xc", "yc","iter_evo", "cost_evo")

(logAs_td_Aletsch, qmag_td_Aletsch, H_td_Aletsch, qmag_Aletsch_obs, H_Aletsch_obs, iter_evo_td, cost_evo_td, xc, yc) = 
    load("output_TD_Aletsch/step_50.jld2", "logAs_td_Aletsch", "qmag_td_Aletsch", "H_td_Aletsch", "qmag_Aletsch_obs", "H_Aletsch_obs", "iter_evo", "cost_evo", "xc", "yc")

(aρgn0_Aletsch, tsc_Aletsch, lsc_Aletsch, s_f) = 
    load("Altesch_scaling.jld2", "aρgn0_data", "tsc_data", "lsc_data", "s_f")

#from logAs convert to As
As_snapshot     = exp10.(logAs_sp_Aletsch)
As_timedepedent = exp10.(logAs_td_Aletsch)

As_snapshot   = As_snapshot * s_f *aρgn0_Aletsch * lsc_Aletsch^n
As_timedepedent= As_timedepedent * s_f *aρgn0_Aletsch * lsc_Aletsch^n

As_snapshot   = As_snapshot / (ρ*g)^n
As_timedepedent= As_timedepedent / (ρ*g)^n

#from qmag convert to vmag: qmag_obs, qmag_snapshot, qmag_td_Aletsch, qmag_Aletsch_obs
v_sp_obs = copy(qmag_obs)
v_sp = copy(qmag_snapshot)
v_td_obs = copy(qmag_Aletsch_obs)
v_td = copy(qmag_td_Aletsch)
n_ratio = (n+2)/(n+1)

v_sp_obs .= qmag_obs ./ H_snapshot[2:end-1, 2:end-1]  .* n_ratio
v_sp     .= qmag_snapshot ./ H_snapshot[2:end-1, 2:end-1] .* n_ratio
v_td_obs .= qmag_Aletsch_obs ./ H_Aletsch_obs[2:end-1, 2:end-1] .* n_ratio
v_td     .= qmag_td_Aletsch ./ H_td_Aletsch[2:end-1, 2:end-1] .* n_ratio


H_snapshot     = H_snapshot * lsc_Aletsch
H_Aletsch_obs  = H_Aletsch_obs * lsc_Aletsch
H_td_Aletsch   = H_td_Aletsch * lsc_Aletsch 

v_sp_obs       = v_sp_obs * lsc_Aletsch / tsc_Aletsch
v_sp           = v_sp * lsc_Aletsch / tsc_Aletsch
v_td_obs       = v_td_obs * lsc_Aletsch / tsc_Aletsch
v_td           = v_td * lsc_Aletsch / tsc_Aletsch

xc = xc .* lsc_Aletsch
yc = yc .* lsc_Aletsch

# set H_obs, H_snapshot and H_timedepedent to be NaN where H is zeros
# H_vert = @. 0.25 * (H_obs[1:end-1, 1:end-1] + H_obs[2:end,1:end-1] + H_obs[1:end-1,2:end] + H_obs[2:end,2:end])
# idv = findall(H_vert .≈ 0.0) |> Array
# H_obs[idv] .= NaN
# H_snapshot[idv] .= NaN
# H_timedepedent[idv] .= NaN

#convert the unit of v from m/s to m/a 
v_sp_obs .*= 365*24*3600
v_sp     .*= 365*24*3600
v_td_obs .*= 365*24*3600
v_td     .*= 365*24*3600

# subplots of logAs_snapshot_Aletsch, logAs_td_Aletsch
fig = Figure(; size=(750, 550), fontsize=14)
ax  = (
    As_snapshot       = Axis(fig[1, 1][1,1]; aspect=DataAspect(),  ylabel="Y (km)", title="a"),
    As_timedepedent   = Axis(fig[1, 2][1,1]; aspect=DataAspect(),  title="b"),
    #ΔAs               = Axis(fig[1, 3][1,1]; aspect=DataAspect(),  title="c"),
    qmag_snapshot     = Axis(fig[2, 1][1,1]; aspect=DataAspect(), xlabel="X (km)", ylabel="Y (km)", title="c"),
    qmag_timedepedent  = Axis(fig[2, 2][1,1]; aspect=DataAspect(), xlabel="X (km)", title="d"),
    H_timedepedent   = Axis(fig[2, 3][1,1]; aspect=DataAspect(), xlabel="X (km)", title="e"))

# As_crange = filter(!isnan, As_syn) |> extrema
# As_syn_max = filter(!isnan, As_syn) |> maximum

v_sp_max = filter(!isnan, v_sp_obs) |> maximum
v_td_max = filter(!isnan, v_td_obs) |> maximum
H_td_max = filter(!isnan, H_Aletsch_obs) |> maximum

hidexdecorations!(ax.As_snapshot; grid=false)
hidexdecorations!(ax.As_timedepedent; grid=false)
#hidexdecorations!(ax.ΔAs; grid=false)
hideydecorations!(ax.As_timedepedent; grid=false)
#hideydecorations!(ax.ΔAs; grid=false)
hideydecorations!(ax.qmag_timedepedent; grid=false)
hideydecorations!(ax.H_timedepedent; grid=false)

#rowgap!(fig.layout, Relative(1/16))
plts = (
    As_snapshot     = heatmap!(ax.As_snapshot, xc[1:end-1]./1000, yc[1:end-1]./1000, log10.(As_snapshot); colormap=:GnBu_9),
    As_timedepedent = heatmap!(ax.As_timedepedent, xc[1:end-1]./1000, yc[1:end-1]./1000, log10.(As_timedepedent); colormap=:GnBu_9),
    #ΔAs             = heatmap!(ax.ΔAs, xc[1:end-1]./1000, yc[1:end-1]./1000, (abs.(log10.(As_snapshot)-log10.(As_timedepedent))./As_sp_max).*100; colormap=:GnBu_9),
    qmag_snapshot  = heatmap!(ax.qmag_snapshot, xc[1:end-1]./1000, yc[1:end-1]./1000, (abs.(v_sp.- v_sp_obs)./v_sp_max).*100; colormap=:GnBu_9, colorrange=(0.0,10.0)),
    qmag_timedepedent  = heatmap!(ax.qmag_timedepedent, xc[1:end-1]./1000, yc[1:end-1]./1000, (abs.(v_td.- v_td_obs)./v_td_max).*100; colormap=:GnBu_9, colorrange=(0.0,10.0)),
    H_timedepedent     = heatmap!(ax.H_timedepedent, xc./1000, yc./1000, (abs.(H_td_Aletsch.- H_Aletsch_obs)./H_td_max).*100 ; colormap=:GnBu_9, colorrange=(0.0, 10.0)))

Colorbar(fig[1,2][1,2], plts.As_timedepedent, label=L"(Pa^{-3}m^2s^{-1})")
#Colorbar(fig[1,3][1,2], plts.ΔAs, label=L"(Pa^{-3}m^2s^{-1})")
# # :GnBu_9
# Colorbar(fig[1,1][1,2], plts.qmag_snapshot)
# Colorbar(fig[1,2][1,2], plts.qmag_timedepedent)
Colorbar(fig[2,3][1,2], plts.H_timedepedent, label=L"%")

display(fig)
save("Aletsch_sp_td.png", fig)

