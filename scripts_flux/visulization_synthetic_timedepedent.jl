using CairoMakie
using JLD2

n          = 3
ρ          = 910 
g          = 9.81

#jldsave("synthetic_timedepedent.jld2"; logAs_timedepedent = As_v, qmag_timedepedent = qmag_v, H = Array(H))
(logAs_syn, logAs_snapshot, qmag_obs, qmag_snapshot, H_obs, H_snapshot) = 
    load("synthetic_snapshot.jld2", "logAs_syn", "logAs_snapshot", "qmag_obs", "qmag_snapshot", "H_obs", "H_snapshot")
(logAs_timedepedent_01, qmag_timedepedent_01, H_timedepedent_01) = 
    load("synthetic_timedepedent_01.jld2", "logAs_timedepedent", "qmag_timedepedent", "H_timedepedent")
(logAs_timedepedent_10, qmag_timedepedent_10, H_timedepedent_10) = 
    load("synthetic_timedepedent_10.jld2", "logAs_timedepedent", "qmag_timedepedent", "H_timedepedent")
(logAs_timedepedent_23, qmag_timedepedent_23, H_timedepedent_23) = 
    load("synthetic_timedepedent_23.jld2", "logAs_timedepedent", "qmag_timedepedent", "H_timedepedent")


# #from qmag convert to vmag 
v_obs = copy(qmag_obs)
v_timedepedent_01 = copy(qmag_timedepedent_01)
v_timedepedent_10 = copy(qmag_timedepedent_10)
v_timedepedent_23 = copy(qmag_timedepedent_23)
n_ratio = (n+2)/(n+1)
v_obs .= qmag_obs ./ H_obs[2:end-1, 2:end-1]  .* n_ratio
v_timedepedent_01 .= qmag_timedepedent_01 ./ H_timedepedent_01[2:end-1, 2:end-1] .* n_ratio
v_timedepedent_10 .= qmag_timedepedent_10 ./ H_timedepedent_10[2:end-1, 2:end-1] .* n_ratio
v_timedepedent_23 .= qmag_timedepedent_23 ./ H_timedepedent_23[2:end-1, 2:end-1] .* n_ratio

#from logAs convert to As
As_syn                      = exp10.(logAs_syn)
As_timedepedent_01          = exp10.(logAs_timedepedent_01)
As_timedepedent_10          = exp10.(logAs_timedepedent_10)
As_timedepedent_23          = exp10.(logAs_timedepedent_23)

#rescaling 
lsc_data      = 1e4
aρgn0_data    = 1.3517139631340709e-12
tsc_data      = 1 / aρgn0_data / lsc_data^n
s_f_syn       = 1e-4
lx_l          = 25.0
ly_l          = 20.0

As_syn                   = As_syn * s_f_syn * aρgn0_data * lsc_data^n
As_timedepedent_01       = As_timedepedent_01 * s_f_syn * aρgn0_data * lsc_data^n
As_timedepedent_10       = As_timedepedent_10 * s_f_syn *aρgn0_data * lsc_data^n
As_timedepedent_23       = As_timedepedent_23 * s_f_syn *aρgn0_data * lsc_data^n

As_syn                    = As_syn / (ρ*g)^n
As_timedepedent_01        = As_timedepedent_01 / (ρ*g)^n
As_timedepedent_10        = As_timedepedent_10 / (ρ*g)^n
As_timedepedent_23        = As_timedepedent_23 / (ρ*g)^n


H_obs                     = H_obs * lsc_data
H_timedepedent_01         = H_timedepedent_01 * lsc_data
H_timedepedent_10         = H_timedepedent_10 * lsc_data 
H_timedepedent_23         = H_timedepedent_23 * lsc_data

v_obs                     = v_obs * lsc_data / tsc_data
v_timedepedent_01         = v_timedepedent_01 * lsc_data / tsc_data
v_timedepedent_10         = v_timedepedent_10 * lsc_data / tsc_data
v_timedepedent_23         = v_timedepedent_23 * lsc_data / tsc_data

lx             = lx_l * lsc_data
ly             = ly_l * lsc_data 
nx             = 128
ny             = 128 
dx, dy = lx / nx, ly / ny
xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)
xv = LinRange(-lx/2 + dx, lx/2 - dx, nx-1)
yv = LinRange(-ly/2 + dy, ly/2 - dy, ny-1)


# # set H_obs, H_snapshot and H_timedepedent to be NaN where H is zeros
# H_vert = @. 0.25 * (H_obs[1:end-1, 1:end-1] + H_obs[2:end,1:end-1] + H_obs[1:end-1,2:end] + H_obs[2:end,2:end])
# idv = findall(H_vert .≈ 0.0) |> Array
# H_obs[idv] .= NaN
# H_snapshot[idv] .= NaN
# H_timedepedent[idv] .= NaN

# #convert the unit of v from m/s to m/a 
# #just before visualization
v_obs .*= 365*24*3600
v_timedepedent_01 .*= 365*24*3600
v_timedepedent_10 .*= 365*24*3600
v_timedepedent_23 .*= 365*24*3600

fig = Figure(; size=(990,600), fontsize=14)
ax  = (
    As_timedepedent_01  = Axis(fig[1, 1][1,1]; aspect=DataAspect(), ylabel = "Y [km]", xlabel="X [km]", title="a"),
    ΔAs_timedepedent_01 = Axis(fig[1, 2][1,1]; aspect=DataAspect(), ylabel = "Y [km]", xlabel="X [km]", title="d"),
    qmag_timedepedent_01  = Axis(fig[1, 3][1,1]; aspect=DataAspect(),  title="c"),
    H_timedepedent_01   = Axis(fig[1, 4][1,1]; aspect=DataAspect(), xlabel="X [km]", title="g"),
    As_timedepedent_10  = Axis(fig[2, 1][1,1]; aspect=DataAspect(), ylabel = "Y [km]", xlabel="X [km]", title="a"),
    ΔAs_timedepedent_10 = Axis(fig[2, 2][1,1]; aspect=DataAspect(), ylabel = "Y [km]", xlabel="X [km]", title="d"),
    qmag_timedepedent_10  = Axis(fig[2, 3][1,1]; aspect=DataAspect(),  title="c"),
    H_timedepedent_10   = Axis(fig[2, 4][1,1]; aspect=DataAspect(), xlabel="X [km]", title="g"),
    As_timedepedent_23  = Axis(fig[3, 1][1,1]; aspect=DataAspect(), ylabel = "Y [km]", xlabel="X [km]", title="a"),
    ΔAs_timedepedent_23 = Axis(fig[3, 2][1,1]; aspect=DataAspect(), ylabel = "Y [km]", xlabel="X [km]", title="d"),
    qmag_timedepedent_23  = Axis(fig[3, 3][1,1]; aspect=DataAspect(),  title="c"),
    H_timedepedent_23   = Axis(fig[3, 4][1,1]; aspect=DataAspect(), xlabel="X [km]", title="g"))


# As_crange = filter(!isnan, As_syn) |> extrema
# v_crange = filter(!isnan, v_obs) |> extrema



xlims!(ax.As_timedepedent_01, -100, 100)
xlims!(ax.ΔAs_timedepedent_01, -100, 100)
xlims!(ax.qmag_timedepedent_01, -100, 100)
xlims!(ax.H_timedepedent_01, -100, 100)
xlims!(ax.As_timedepedent_10, -100, 100)
xlims!(ax.ΔAs_timedepedent_10, -100, 100)
xlims!(ax.qmag_timedepedent_10, -100, 100)
xlims!(ax.H_timedepedent_10, -100, 100)
xlims!(ax.As_timedepedent_23, -100, 100)
xlims!(ax.ΔAs_timedepedent_23, -100, 100)
xlims!(ax.qmag_timedepedent_23, -100, 100)
xlims!(ax.H_timedepedent_23, -100, 100)


hidexdecorations!(ax.As_timedepedent_01; grid=false)
hidexdecorations!(ax.ΔAs_timedepedent_01; grid=false)
hidexdecorations!(ax.qmag_timedepedent_01; grid=false)
hidexdecorations!(ax.H_timedepedent_01; grid=false)
hidexdecorations!(ax.As_timedepedent_10; grid=false)
hidexdecorations!(ax.ΔAs_timedepedent_10; grid=false)
hidexdecorations!(ax.qmag_timedepedent_10; grid=false)
hidexdecorations!(ax.H_timedepedent_10; grid=false)
hideydecorations!(ax.ΔAs_timedepedent_01; grid=false)
hideydecorations!(ax.qmag_timedepedent_01; grid=false)
hideydecorations!(ax.H_timedepedent_01; grid=false)
hideydecorations!(ax.ΔAs_timedepedent_10; grid=false)
hideydecorations!(ax.qmag_timedepedent_10; grid=false)
hideydecorations!(ax.H_timedepedent_10; grid=false)
hideydecorations!(ax.ΔAs_timedepedent_23; grid=false)
hideydecorations!(ax.qmag_timedepedent_23; grid=false)
hideydecorations!(ax.H_timedepedent_23; grid=false)

As_syn_max = filter(!isnan, As_syn) |> maximum
v_obs_max = filter(!isnan, v_obs) |> maximum
H_obs_max = filter(!isnan, H_obs) |> maximum
colgap!(fig.layout, Relative(1/256))

plts = (
    As_timedepedent_01     = heatmap!(ax.As_timedepedent_01, xv./1000, yv./1000, log10.(As_timedepedent_01); colormap=:GnBu_9),
    ΔAs_timedepedent_01    = heatmap!(ax.ΔAs_timedepedent_01, xv./1000, yv./1000, (abs.(As_timedepedent_01-As_syn)./As_syn_max).*100; colormap=:GnBu_9, colorrange=(0, 10)),
    qmag_timedepedent_01   = heatmap!(ax.qmag_timedepedent_01, xc[1:end-1]./1000, yc[1:end-1]./1000, (abs.(v_timedepedent_01 .- v_obs)./v_obs_max).*100; colormap=:GnBu_9, colorrange=(0, 0.2)),
    H_timedepedent_01      = heatmap!(ax.H_timedepedent_01, xc./1000, yc./1000, replace!((abs.(H_timedepedent_01 .- H_obs) ./ H_obs_max).*100, 0.0=>NaN); colormap=:GnBu_9, colorrange=(0, 0.2)),
    As_timedepedent_10 = heatmap!(ax.As_timedepedent_10, xv./1000, yv./1000, log10.(As_timedepedent_10); colormap=:GnBu_9),
    ΔAs_timedepedent_10  = heatmap!(ax.ΔAs_timedepedent_10, xv./1000, yv./1000, (abs.(As_timedepedent_10-As_syn)./As_syn_max).*100; colormap=:GnBu_9, colorrange=(0, 10)),
    qmag_timedepedent_10  = heatmap!(ax.qmag_timedepedent_10, xc[1:end-1]./1000, yc[1:end-1]./1000, (abs.(v_timedepedent_10 .- v_obs)./v_obs_max).*100; colormap=:GnBu_9, colorrange=(0, 0.2)),
    H_timedepedent_10     = heatmap!(ax.H_timedepedent_10, xc./1000, yc./1000, replace!((abs.(H_timedepedent_10 .- H_obs) ./ H_obs_max), 0.0=>NaN).*100; colormap=:GnBu_9, colorrange=(0, 0.2)),
    As_timedepedent_23 = heatmap!(ax.As_timedepedent_23, xv./1000, yv./1000, log10.(As_timedepedent_23); colormap=:GnBu_9),
    ΔAs_timedepedent_23  = heatmap!(ax.ΔAs_timedepedent_23, xv./1000, yv./1000, (abs.(As_timedepedent_23-As_syn)./As_syn_max).*100; colormap=:GnBu_9, colorrange=(0, 10)),
    qmag_timedepedent_23  = heatmap!(ax.qmag_timedepedent_23, xc[1:end-1]./1000, yc[1:end-1]./1000, (abs.(v_timedepedent_23 .- v_obs)./v_obs_max).*100; colormap=:GnBu_9, colorrange=(0, 0.2)),
    H_timedepedent_23     = heatmap!(ax.H_timedepedent_23, xc./1000, yc./1000, replace!((abs.(H_timedepedent_23 .- H_obs) ./ H_obs_max).*100, 0.0=>NaN); colormap=:GnBu_9, colorrange=(0, 0.2)))


Colorbar(fig[1,1][1,2], plts.As_timedepedent_01, label=L"(Pa^{-3}m^2s^{-1})")
Colorbar(fig[2,1][1,2], plts.As_timedepedent_10, label=L"(Pa^{-3}m^2s^{-1})")
Colorbar(fig[3,1][1,2], plts.As_timedepedent_23, label=L"(Pa^{-3}m^2s^{-1})")
Colorbar(fig[1,4][1,2], plts.H_timedepedent_01, label=L"%")
Colorbar(fig[2,4][1,2], plts.H_timedepedent_10, label=L"%")
Colorbar(fig[3,4][1,2], plts.H_timedepedent_23, label=L"%")



display(fig)
save("timedepedent_compare.png", fig)
