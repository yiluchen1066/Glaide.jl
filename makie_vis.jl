using GLMakie
using DelimitedFiles
using LinearAlgebra

#synthetic glacier 
nx = 128 
ny = 128 

lx = 250000
ly = 250000
lz = 1e3


dx = lx/nx
dy = ly/ny


xc = LinRange(-lx/2+dx/2, lx/2-dx/2, nx)
yc = LinRange(-ly/2+dy/2, ly/2-dy/2, ny)


w1 = 1e10
w2 = 1e9 
B0 = 3500 
ω  = 8 

B = zeros(Float64, nx, ny)

B = @. B0*(exp(-xc^2/w1 - yc'^2/w2) + exp(-xc^2/w2-(yc'-ly/ω)^2/w1))

xc = xc .- xc[1]
yc = yc .- yc[1]

xc = xc ./1e3
yc = yc ./1e3


fig = Figure(resolution=(3000,2500), fontsize=42)
ax = Axis3(fig[1,1][1,1]; aspect=(1,1,0.5), xlabel="X [km]", ylabel="Y [km]")

ax.xlabeloffset[] = 60
ax.ylabeloffset[] = 60
ax.zlabeloffset[] = 100

surface!(ax, xc, yc, B; colormap=:lightterrain)
#xlabel!(ax,"X in m")
#GLMakie.ylabel!(ax,"Y in m")
#GLMakie.zlabel!(ax,"Height in m")
display(fig)
save("synthetic_glacier.png", fig)

# syntheic 
fig






