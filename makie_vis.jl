using GLMakie
using DelimitedFiles
using LinearAlgebra

nx = 128 
ny = 128 

lx = 250000
ly = 250000
lz = 1e3

dx = xc[2] - xc[1]
dy = yc[2] - yc[1]

xc = LinRange(-lx/2+dx/2, lx/2-dx/2, nx)
yc = LinRange(-ly/2+dy/2, ly/2-dy/2, ny)


w1 = 1e10
w2 = 1e9 
B0 = 3500 
ω  = 8 

B = zeros(Float64, nx, ny)

B = @. B0*(exp(-xc^2/w1 - yc'^2/w2) + exp(-xc^2/w2-(yc'-ly/ω)^2/w1))

fig = Figure(resolution=(2500,1200), fontsize=42)
ax = Axis3(fig[1,1][1,1]; aspect_ratio=:data, title="Synthetic glacier")

surface!(ax, xc, yc, B; colormap=:lightterrain)
savefig("synthetic_glacier.png", fig)

fig

