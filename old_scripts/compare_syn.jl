using CairoMakie

nx, ny = 128, 128

fld_old_syn = (
    H_obs   = zeros(nx, ny),
    D       = zeros(nx - 1, ny - 1),
    As      = zeros(nx - 1, ny - 1),
    ELA     = zeros(nx, ny),
    β       = zeros(nx, ny),
)

fld_new_syn = (
    H_obs   = zeros(nx, ny),
    D       = zeros(nx - 1, ny - 1),
    As      = zeros(nx - 1, ny - 1),
    ELA     = zeros(nx, ny),
    β       = zeros(nx, ny),
)

fld_old_fwd = (
    H   = zeros(nx, ny), 
    D   = zeros(nx-1, ny-1), 
    As  = zeros(nx-1,ny-1), 
    ELA = zeros(nx, ny), 
    β   = zeros(nx, ny), 
)

fld_new_fwd = (
    H   = zeros(nx, ny), 
    D   = zeros(nx-1, ny-1), 
    As  = zeros(nx-1, ny-1),
    ELA = zeros(nx, ny), 
    β   = zeros(nx, ny), 
)

#r, R, dR_qHx, dR_qHy,dq_D
fld_old_adj = (
    r       = zeros(nx, ny), 
    dR      = zeros(nx, ny), 
    dR_qHx  = zeros(nx-1, ny-2), 
    dR_qHy  = zeros(nx-2, ny-1), 
    dq_D    = zeros(nx-1, ny-1),
)

fld_new_adj = (
    ψ_H     = zeros(nx, ny), 
    H̄       = zeros(nx, ny), 
    q̄Hx     = zeros(nx-1, ny-2), 
    q̄Hy     = zeros(nx-2, ny-1), 
    D̄       = zeros(nx-1, ny-1),
)

open("output/synthetic_old.dat", "r") do io
    read!(io, fld_old_syn.H_obs)
    read!(io, fld_old_syn.D)
    read!(io, fld_old_syn.As)
    read!(io, fld_old_syn.ELA)
    read!(io, fld_old_syn.β)
end

open("output/synthetic_new.dat", "r") do io
    read!(io, fld_new_syn.H_obs)
    read!(io, fld_new_syn.D)
    read!(io, fld_new_syn.As)
    read!(io, fld_new_syn.ELA)
    read!(io, fld_new_syn.β)
end

open("output/forward_old.dat", "r") do io
    read!(io, fld_old_fwd.H)
    read!(io, fld_old_fwd.D)
    read!(io, fld_old_fwd.As)
    read!(io, fld_old_fwd.ELA)
    read!(io, fld_old_fwd.β)
end

open("output/forward_new.dat", "r") do io
    read!(io, fld_new_fwd.H)
    read!(io, fld_new_fwd.D)
    read!(io, fld_new_fwd.As)
    read!(io, fld_new_fwd.ELA)
    read!(io, fld_new_fwd.β)
end

open("output/adjoint_old.dat", "r") do io 
    read!(io, fld_old_adj.r)
    read!(io, fld_old_adj.dR)
    read!(io, fld_old_adj.dR_qHx)
    read!(io, fld_old_adj.dR_qHy)
    read!(io, fld_old_adj.dq_D)
end 

open("output/adjoint_new.dat", "r") do io 
    read!(io, fld_new_adj.ψ_H)
    read!(io, fld_new_adj.H̄)
    read!(io, fld_new_adj.q̄Hx)
    read!(io, fld_new_adj.q̄Hy)
    read!(io, fld_new_adj.D̄)
end

ΔH_old = fld_old_syn.H_obs - fld_old_fwd.H
ΔH_new = fld_new_syn.H_obs - fld_new_fwd.H


@show maximum(abs.(fld_old_fwd.H       .- fld_new_fwd.H))
@show maximum(abs.(fld_old_fwd.D       .- fld_new_fwd.D))
@show maximum(abs.(fld_old_fwd.As      .- fld_new_fwd.As))
@show maximum(abs.(fld_old_fwd.ELA     .- fld_new_fwd.ELA))
@show maximum(abs.(fld_old_fwd.β       .- fld_new_fwd.β))
@show maximum(abs.(fld_old_adj.r       .- fld_new_adj.ψ_H))
@show maximum(abs.(fld_old_adj.dR      .- fld_new_adj.H̄))
@show maximum(abs.(fld_old_adj.dR_qHx  .- fld_new_adj.q̄Hx))
@show maximum(abs.(fld_old_adj.dR_qHy  .- fld_new_adj.q̄Hy))
@show maximum(abs.(fld_old_adj.dq_D    .- fld_new_adj.D̄))

fig = Figure(; resolution=(1200, 800), fontsize=32)

#ax  = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="ΔH")
#plt = heatmap!(ax, fld_old.H .- fld_new.H; colormap=:turbo)

axs = (
    H_syn   = Axis(fig[1,1]; aspect=DataAspect(), title="ΔH_syn"), 
    H_fwd   = Axis(fig[1,2]; aspect=DataAspect(), title="ΔH_fwd"),
    ΔH      = Axis(fig[2,1]; aspect=DataAspect(), title="ΔH"),
    Δψ_H    = Axis(fig[2,2]; aspect=DataAspect(), title="Δψ_H")
)

plts = (
    H_syn = heatmap!(axs.H_syn, fld_old_syn.H_obs-fld_new_syn.H_obs; colormap =:turbo), 
    H_fwd = heatmap!(axs.H_fwd, fld_old_fwd.H-fld_new_fwd.H; colormap =:turbo),
    ΔH    = heatmap!(axs.ΔH, ΔH_old-ΔH_new; colormap =:turbo), 
    Δψ_H  = heatmap!(axs.Δψ_H, fld_old_adj.r- fld_new_adj.ψ_H; colormap=:turbo)
    
)

#Colorbar(fig[1, 1][1, 2][1,3], plts)

display(fig)
