using CairoMakie
using JLD2

n          = 3
ρ          = 910 
g          = 9.81

#jldsave("synthetic_timedepedent.jld2"; logAs_timedepedent = As_v, qmag_timedepedent = qmag_v, H = Array(H))
(logAs_timedepedent_01, qmag_timedepedent_01, H_timedepedent_01) = 
    load("synthetic_timedepedent_01.jld2", "logAs_timedepedent", "qmag_timedepedent", "H_timedepedent")
(logAs_timedepedent_10, qmag_timedepedent_10, H_timedepedent_10) = 
    load("synthetic_timedepedent_10.jld2", "logAs_timedepedent", "qmag_timedepedent", "H_timedepedent")
(logAs_timedepedent_23, qmag_timedepedent_23, H_timedepedent_23) = 
    load("synthetic_timedepedent_23.jld2", "logAs_timedepedent", "qmag_timedepedent", "H_timedepedent")

# #from qmag convert to vmag 
# v_obs = copy(qmag_obs)
# v_snapshot = copy(qmag_snapshot)
# v_timedepedent = copy(qmag_timedepedent)
# n_ratio = (n+2)/(n+1)
# v_obs .= qmag_obs ./ H_obs[2:end-1, 2:end-1]  .* n_ratio
# v_snapshot .= qmag_snapshot ./ H_snapshot[2:end-1, 2:end-1] .* n_ratio
# v_timedepedent .= qmag_timedepedent ./ H_timedepedent[2:end-1, 2:end-1] .* n_ratio

#from logAs convert to As
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

As_timedepedent_01       = As_timedepedent_01 * s_f_syn * aρgn0_data * lsc_data^n
As_timedepedent_10       = As_timedepedent_10 * s_f_syn *aρgn0_data * lsc_data^n
As_timedepedent_23       = As_timedepedent_23 * s_f_syn *aρgn0_data * lsc_data^n

As_timedepedent_01        = As_timedepedent_01 / (ρ*g)^n
As_timedepedent_10        = As_timedepedent_10 / (ρ*g)^n
As_timedepedent_23        = As_timedepedent_23 / (ρ*g)^n


# H_obs          = H_obs * lsc_data
# H_snapshot     = H_snapshot * lsc_data
# H_timedepedent = H_timedepedent * lsc_data 

# v_obs          = v_obs * lsc_data / tsc_data
# v_snapshot     = v_snapshot * lsc_data / tsc_data
# v_timedepedent = v_timedepedent * lsc_data / tsc_data

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
# v_obs .*= 365*24*3600
# v_snapshot .*= 365*24*3600
# v_timedepedent .*= 365*24*3600


fig = Figure(; size=(850,300), fontsize=14)
ax  = (
    As_timedepedent_10  = Axis(fig[1, 1][1,1]; aspect=DataAspect(), ylabel = "Y [km]", xlabel="X [km]", title="a"),
    As_timedepedent_23  = Axis(fig[1, 2][1,1]; aspect=DataAspect(), ylabel = "Y [km]", xlabel="X [km]", title="b"),
    As_timedepedent_01  = Axis(fig[1, 3][1,1]; aspect=DataAspect(), ylabel = "Y [km]", xlabel="X [km]", title="c"))


# As_crange = filter(!isnan, As_syn) |> extrema
# v_crange = filter(!isnan, v_obs) |> extrema
# v_obs_max = filter(!isnan, v_obs) |> maximum
# H_obs_max = filter(!isnan, H_obs) |> maximum


xlims!(ax.As_timedepedent_01, -100, 100)
xlims!(ax.As_timedepedent_10, -100, 100)
xlims!(ax.As_timedepedent_23, -100, 100)

# hidexdecorations!(ax.As_syn; grid=false)
# hidexdecorations!(ax.qmag_obs; grid=false)
# hidexdecorations!(ax.H_obs; grid=false)
# hidexdecorations!(ax.As_snapshot; grid=false)
# hidexdecorations!(ax.qmag_snapshot; grid=false)

hideydecorations!(ax.As_timedepedent_23; grid=false)
hideydecorations!(ax.As_timedepedent_01; grid=false)


colgap!(fig.layout, Relative(1/256))

plts = (
    As_timedepedent_10    = heatmap!(ax.As_timedepedent_10, xv./1000, yv./1000, log10.(As_timedepedent_10); colormap=:turbo),
    As_timedepedent_23    = heatmap!(ax.As_timedepedent_23, xv./1000, yv./1000, log10.(As_timedepedent_23); colormap=:turbo),
    As_timedepedent_01    = heatmap!(ax.As_timedepedent_01, xv./1000, yv./1000, log10.(As_timedepedent_01); colormap=:turbo))


Colorbar(fig[1,3][1,2], plts.As_timedepedent_23, label=L"(Pa^{-3}m^2s^{-1})")


display(fig)
save("timedepedent_compare.png", fig)
