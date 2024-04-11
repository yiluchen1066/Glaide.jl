using CairoMakie
using JLD2

n          = 3
ρ          = 910 
g          = 9.81

(aρgn0_data, tsc_data, lsc_data, s_f) = load("Altesch_scaling.jld2", "aρgn0_data", "tsc_data", "lsc_data","s_f")
(logAs, q_obs, q, H, xc, yc) = load("snapshot_Aletsch.jld2", "logAs", "q_obs", "q", "H", "xc", "yc")

#from qmag convert to vmag 
v_obs = copy(q_obs)
v = copy(q)
n_ratio = (n+2)/(n+1)
v_obs .= q_obs ./ H  .* n_ratio
v     .= q ./ H .* n_ratio

#from logAs convert to As
As          = exp10.(logAs)

# #rescaling 
As        = As * s_f * aρgn0_data * lsc_data^n
As        = As / (ρ*g)^n

H         = H * lsc_data
v_obs     = v_obs * lsc_data / tsc_data
v         = v * lsc_data / tsc_data
xc        = xc .* lsc_data
yc        = yc .* lsc_data


#convert the unit of v from m/s to m/a 
#just before visualization
v_obs .*= 365*24*3600
v .*= 365*24*3600

fig = Figure(; size=(600, 600), fontsize=14)
ax  = (
    v_obs  = Axis(fig[1, 1][1,1]; aspect=DataAspect(), ylabel = "Y [km]"),
    v      = Axis(fig[1, 2][1,1]; aspect=DataAspect()),
    As  = Axis(fig[2, 1][1,1]; aspect=DataAspect(), xlabel = "X [km]",ylabel = "Y [km]"),
    Δv  = Axis(fig[2, 2][1,1]; aspect=DataAspect(), xlabel = "X [km]",ylabel = "X [km]"))

As_crange = filter(!isnan, As) |> extrema
q_crange  = filter(!isnan, q_obs) |> extrema
v_crange = filter(!isnan, v_obs) |> extrema
# v_obs_max = filter(!isnan, v_obs) |> maximum
# H_obs_max = filter(!isnan, H_obs) |> maximum

hidexdecorations!(ax.v_obs; grid=false)
hidexdecorations!(ax.v; grid=false)
hideydecorations!(ax.v; grid=false)
hideydecorations!(ax.Δv; grid=false)

rowgap!(fig.layout, Relative(1/8))

plts = (
    v_obs    = heatmap!(ax.v_obs, xc[2:end-1]./1000, yc[2:end-1]./1000, v_obs; colormap=:turbo, colorrange=v_crange),
    v        = heatmap!(ax.v, xc[2:end-1]./1000, yc[2:end-1]./1000, v; colormap=:turbo, colorrange=v_crange),
    As     = heatmap!(ax.As, xc[1:end-1]./1000, yc[1:end-1]./1000, log10.(As); colormap=:turbo),
    Δv    = heatmap!(ax.Δv, xc[1:end-1]./1000, yc[1:end-1]./1000, abs.(q .- q_obs)./q_obs; colormap=:turbo,colorrange=(0.0, 0.10)))

Colorbar(fig[1,1][1,2], plts.v_obs)
Colorbar(fig[1,2][1,2], plts.v)
Colorbar(fig[2,1][1,2], plts.As)
Colorbar(fig[2,2][1,2], plts.Δv)


display(fig)
save("Altesch_snapshot.png", fig)

#rowgap!()
#hidexdecorations!()
