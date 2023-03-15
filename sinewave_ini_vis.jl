using Plots, Plots.Measures, Printf
using PyPlot

function sin_wave()
    B0 = 500
    ω = 0.0001
    lx,ly = 30e3, 30e3 
    nx = 128 
    ny = 128 
    dx = lx/nx
    dy = ly/ny
    xc = LinRange(dx/2, lx-dx/2, nx)
    yc = LinRange(dy/2, ly-dy/2, ny) 
    x0 = xc[round(Int,nx/2)]
    y0 = yc[round(Int,ny/2)]
    w = 0.4*lx
    w1= 0.45*lx

    H = zeros(Float64, nx,ny)
    B = zeros(Float64, nx,ny)
    S = zeros(Float64, nx,ny)
    @show(size(H))

    #B = @. B0*sin(ω*pi*(xc+yc'))+B0
    B = @. B0*(exp(-((xc-x0)/w)^2-((yc'-y0)/w)^2))*sin(ω*pi*(xc+yc'))+B0
    H = @. (B0+100)*(exp(-((xc-x0)/w1)^2-((yc'-y0)/w1)^2))*sin(ω*pi*(xc+yc'))+(B0+100)
    #H = @. 200*exp(-((xc-x0)/5000)^2-((yc'-y0)/5000)^2)

    @show(size(B))
    p1 = Plots.plot(xc, yc, B',st=:surface,camera=(20,25), aspect_ratio =1)
    Plots.plot!(xc,yc, H', st=:surface, camera=(20,25),aspect_ratio=1)
    p2 = Plots.contour(xc, yc, B'; levels =20, aspect_ratio=1)
    p3 = Plots.contourf(xc, yc,H'; levels=20, aspect_ratio=1)
    p4 = Plots.contourf(xc,yc, (B.+H)'; levels=20, aspect_ratio=1)
    display(plot(p1,p2,p3,p4; layout=(2,2), size=(980,980)))
    return 
end 

sin_wave()