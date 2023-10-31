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
    dR_H    = zeros(nx, ny),
    dq_H    = zeros(nx, ny), 
    dD_H    = zeros(nx, ny),
)

fld_new_adj = (
    ψ_H     = zeros(nx, ny), 
    H̄       = zeros(nx, ny), 
    q̄Hx     = zeros(nx-1, ny-2), 
    q̄Hy     = zeros(nx-2, ny-1), 
    D̄       = zeros(nx-1, ny-1),
    H̄_1     = zeros(nx, ny),
    H̄_2     = zeros(nx, ny),
    H̄_3     = zeros(nx, ny),
)

fld_old_adj_start = (
    tmp2    = zeros(nx, ny), 
    dR_qHx  = zeros(nx-1, ny-2), 
    dR_qHy  = zeros(nx-2, ny-1), 
    dR      = zeros(nx, ny), 
    dq_D    = zeros(nx-1, ny-1), 
    dR_H    = zeros(nx, ny), 
    dq_H    = zeros(nx, ny), 
    dD_H    = zeros(nx, ny),
)

fld_new_adj_start = (
    R̄H      = zeros(nx, ny),
    q̄Hx     = zeros(nx-1, ny-2), 
    q̄Hy     = zeros(nx-2, ny-1), 
    H̄       = zeros(nx, ny), 
    D̄       = zeros(nx-1, ny-1),
    H̄_1     = zeros(nx, ny), 
    H̄_2     = zeros(nx, ny), 
    H̄_3     = zeros(nx, ny),
)
 
fld_adjoint_old_start_1 = (
    tmp2    = zeros(nx, ny),
    dR_qHx  = zeros(nx-1, ny-2), 
    dR_qHy  = zeros(nx-2, ny-1),
    dR_H    = zeros(nx, ny)
)

fld_adjoint_new_start_1 = (
    R̄H      = zeros(nx, ny), 
    q̄Hx     = zeros(nx-1, ny-2), 
    q̄Hy     = zeros(nx-2, ny-1), 
    H̄_1     = zeros(nx, ny), 
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

open("output/adjoint_old_5.dat", "r") do io 
    read!(io, fld_old_adj.r)
    read!(io, fld_old_adj.dR)
    read!(io, fld_old_adj.dR_qHx)
    read!(io, fld_old_adj.dR_qHy)
    read!(io, fld_old_adj.dq_D)
    read!(io, fld_old_adj.dR_H)
    read!(io, fld_old_adj.dq_H)
    read!(io, fld_old_adj.dD_H)
end 

open("output/adjoint_new_5.dat", "r") do io 
    read!(io, fld_new_adj.ψ_H)
    read!(io, fld_new_adj.H̄)
    read!(io, fld_new_adj.q̄Hx)
    read!(io, fld_new_adj.q̄Hy)
    read!(io, fld_new_adj.D̄)
    read!(io, fld_new_adj.H̄_1)
    read!(io, fld_new_adj.H̄_2)
    read!(io, fld_new_adj.H̄_3)
end




# open("output/adjoint_old_start.dat","r") do io     
#     read!(io,fld_new_adj_start.tmp2)
#     read!(io,fld_new_adj_start.dR_qHx)
#     read!(io,fld_new_adj_start.dR_qHy)
#     read!(io,fld_new_adj_start.dR)
#     read!(io,fld_new_adj_start.dq_D)
#     read!(io,fld_new_adj_start.dR_H)
#     read!(io,fld_new_adj_start.dq_H)
#     read!(io,fld_new_adj_start.dq_D)
# end 


# #write("output/adjoint_new_start.dat", Array(R̄H), Array(q̄Hx), Array(q̄Hy), Array(H̄), Array(D̄), Array(H̄_1), Array(H̄_2), Array(H̄_3))
# #write("output/adjoint_old_start.dat", Array(tmp2), Array(dR_qHx), Array(dR_qHy),Array(dR), Array(dq_D), Array(dR_H), Array(dq_H), Array(dD_H))
# open("output/adjoint_new_start.dat","r") do io     
#     read!(io,fld_new_adj_start.R̄H)
#     read!(io,fld_new_adj_start.q̄Hx)
#     read!(io,fld_new_adj_start.q̄Hy)
#     read!(io,fld_new_adj_start.H̄)
#     read!(io,fld_new_adj_start.D̄)
#     read!(io,fld_new_adj_start.H̄_1)
#     read!(io,fld_new_adj_start.H̄_2)
#     read!(io,fld_new_adj_start.H̄_3)
# end 

#write("output/adjoint_old_start_1_$(iter).dat", Array(tmp2), Array(dR_qHx), Array(dR_qHy), Array(dR_H))
#write("output/adjoint_new_start_1_$(iter).dat", Array(R̄H), Array(q̄Hx), Array(q̄Hy), Array(H̄_1))
# so here the routine is you first define a field (name tuple of fields) and then you read the file and filling in the

# open("output/adjoint_old_start_1_1.dat", "r") do io
#     read!(io,fld_old_adj_start.tmp2)
#     read!(io,fld_old_adj_start.dR_qHx)
#     read!(io,fld_old_adj_start.dR_qHy)
#     read!(io,fld_old_adj_start.dR_H)
# end 

# open("output/adjoint_new_start_1_1.tdat", "r") do io
#     read!(io,fld_old_adj_start.R̄H)
#     read!(io,fld_old_adj_start.q̄Hx)
#     read!(io,fld_old_adj_start.q̄Hy)
#     read!(io,fld_old_adj_start.H̄_1)
# end 


@show maximum(abs.(fld_old_adj_start.tmp2 - fld_new_adj_start.R̄H))
@show maximum(abs.(fld_old_adj_start.dR_qHx - fld_new_adj_start.q̄Hx))
@show maximum(abs.(fld_old_adj_start.dR_qHy - fld_new_adj_start.q̄Hy))
@show maximum(abs.(fld_old_adj_start.dR - fld_new_adj_start.H̄))
@show maximum(abs.(fld_old_adj_start.dq_D - fld_new_adj_start.D̄))
@show maximum(abs.(fld_old_adj_start.dR_H - fld_new_adj_start.H̄_1))
@show maximum(abs.(fld_old_adj_start.dq_H - fld_new_adj_start.H̄_2))
@show maximum(abs.(fld_old_adj_start.dD_H - fld_new_adj_start.H̄_3))



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

@show maximum(abs.(fld_old_adj.dR))
@show maximum(abs.(fld_new_adj.H̄))

@show maximum(abs.(fld_old_adj.r))
@show maximum(abs.(fld_new_adj.ψ_H))

@show maximum(abs.(fld_old_adj.dR_H     .- fld_new_adj.H̄_1))
@show maximum(abs.(fld_old_adj.dq_H     .- fld_new_adj.H̄_2))
@show maximum(abs.(fld_old_adj.dD_H     .- fld_new_adj.H̄_3))
@show maximum(abs.(fld_old_adj.dq_D    .- fld_new_adj.D̄))



fig = Figure(; resolution=(1200, 800), fontsize=32)

#ax  = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="ΔH")
#plt = heatmap!(ax, fld_old.H .- fld_new.H; colormap=:turbo)

axs = (
    H_syn   = Axis(fig[1,1][1,1]; aspect=DataAspect(), title="ΔH_syn"), 
    H_fwd   = Axis(fig[1,2][1,1]; aspect=DataAspect(), title="ΔH_fwd"),
    ΔH      = Axis(fig[2,1][1,1]; aspect=DataAspect(), title="ΔH"),
    Δψ_H    = Axis(fig[2,2][1,1]; aspect=DataAspect(), title="Δψ_H")
)

plts = (
    H_syn = heatmap!(axs.H_syn, fld_old_syn.H_obs-fld_new_syn.H_obs; colormap =:turbo), 
    H_fwd = heatmap!(axs.H_fwd, fld_old_fwd.H-fld_new_fwd.H; colormap =:turbo),
    ΔH    = heatmap!(axs.ΔH, fld_old_adj.dR-fld_new_adj.H̄; colormap =:turbo), 
    Δψ_H  = heatmap!(axs.Δψ_H, fld_old_adj.r- fld_new_adj.ψ_H; colormap=:turbo)
    
)

Colorbar(fig[1, 1][1, 2], plts.H_syn)
Colorbar(fig[1, 2][1, 2], plts.H_fwd)
Colorbar(fig[2, 1][1, 2], plts.ΔH)
Colorbar(fig[2, 2][1, 2], plts.Δψ_H)

display(fig)
