using Plots, Plots.Measures, Printf
using PyPlot

function sin_wave()
    ω = 0.1
    lx,ly = 30e3, 30e3 
    nx = 128 
    ny = 128 
    dx = lx/nx
    dy = ly/ny
    xc = LinRange(dx/2, lx-dx/2, nx)
    yc = LinRange(dy/2, ly-dy/2, ny) 
    x0 = xc[round(Int,nx/2)]
    y0 = yc[round(Int,ny/2)]

    H = zeros(Float64, nx,ny)
    B = zeros(Float64, nx,ny)
    S = zeros(Float64, nx,ny)
    @show(size(H))

    B = @. sin(ω*pi*(xc+yc'))
    
    @show(size(B))
    p1 = plot(xc, yc, B',st=:surface,camera=(-30,30))
    display(plot(p1, size=(980,980)))
    return 
end 

sin_wave()