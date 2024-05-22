using CairoMakie
using JLD2

n          = 3
ρ          = 910 
g          = 9.81

(iter_evo_sp, cost_evo_sp) = 
    load("synthetic_snapshot.jld2", "iter_evo", "cost_evo")
(iter_evo_td, cost_evo_td) = 
    load("synthetic_timedepedent.jld2", "iter_evo", "cost_evo")
(iter_evo_td_01, cost_evo_td_01) = 
    load("synthetic_timedepedent_01.jld2", "iter_evo", "cost_evo")
(iter_evo_td_10, cost_evo_td_10) = 
    load("synthetic_timedepedent_10.jld2", "iter_evo", "cost_evo")
(iter_evo_td_23, cost_evo_td_23) = 
    load("synthetic_timedepedent_23.jld2", "iter_evo", "cost_evo")

fig = Figure(; size=(520,480), fontsize=16)
# how do I text the title
ax = Axis(fig[1,1]; xlabel="#iter", ylabel=L"J/J_0",yscale=log10)
ylabel=L"$Annual$ $mass$ $balance$ $[\text{m}\cdot \text{a}^{-1}]$"
convergence_plots = scatterlines!(ax, Point2.(iter_evo_sp, cost_evo_sp./first(cost_evo_sp)); label=L"snapshot", linewidth=2)
                    scatterlines!(ax, Point2.(iter_evo_td, cost_evo_td./first(cost_evo_td)); label=L"$time-dependent$ $(\text{\omega_H=\omega_q})$", linewidth=2)
                    scatterlines!(ax, Point2.(iter_evo_td_01, cost_evo_td_01./first(cost_evo_td_01)); label=L"$time-dependent$ $(\text{\omega_q=1, \omega_H=0})$",linewidth=2)
                    scatterlines!(ax, Point2.(iter_evo_td_10, cost_evo_td_10./first(cost_evo_td_10)); label=L"$time-dependent$ $(\text{\omega_H=1, \omega_q=0})$",linewidth=2)
                    scatterlines!(ax, Point2.(iter_evo_td_23, cost_evo_td_23./first(cost_evo_td_23)); label=L"$time-dependent$ $(\text{\omega_q=2 \cdot \omega_q})$",linewidth=2)

axislegend(ax; position=:rt, labelsize=14)
display(fig)
save("synthetic_convergence.png", fig)

