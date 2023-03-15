using Plots, Plots.Measures, Printf

function sin_wave()
    B0 = 500
    ω = 0.0001
    lx,ly,lz = 30e3, 30e3, 1e3 
    ox, oy, oz = -lx/2, -ly/2, 0.0 
    nx = 128 
    ny = 128 
    dx = lx/nx
    dy = ly/ny

    xv = LinRange(ox, ox+lx, nx+1)
    yv = LinRange(oy, oy+ly, ny+1)
    xc = 0.5*(xv[1:end-1]+xv[2:end])
    yc = 0.5*(yv[1:end-1]+yv[2:end])

    @show(size(xc))
    @show(size(yc))

    x0 = xc[round(Int,nx/2)]
    y0 = yc[round(Int,ny/2)]

    # bedrock properties
    wx1, wy1 = 0.4lx, 0.2ly 
    wx2, wy2 = 0.15lx, 0.15ly 
    ω1 = 4 
    ω2 = 6


    fun = @. 0.4exp(-(xc/wx1)^2-((yc-0.2ly)'/wy1)^2) + 0.2*exp(-(xc/wx2)^2-((yc-0.1ly)'/wy2)^2)*(cos(ω1*π*(xc/lx + 0*yc'/ly-0.25)) + 1)
    @. fun += 0.025exp(-(xc/wx2)^2-(yc'/wy2)^2)*cos(ω2*π*(0*xc/lx + yc'/ly)) 
    @. fun += 0xc + 0.15*(yc'/ly)
    zmin,zmax = extrema(fun)
    @. fun = (fun - zmin)/(zmax-zmin)

    @show(size(fun))

    B = zeros(nx,ny)
    @. B += oz + (lz-oz)*fun

    H = zeros(Float64, nx,ny)
    S = zeros(Float64, nx,ny)

    @show(size(B))
    p1 = Plots.plot(xc, yc, B',st=:surface,camera=(-30,25), aspect_ratio =1)
    Plots.plot!(xc,yc, H', st=:surface, camera=(20,25),aspect_ratio=1)
    p2 = Plots.contour(xc, yc, B'; levels =20, aspect_ratio=1)
    p3 = Plots.contourf(xc, yc,H'; levels=20, aspect_ratio=1)
    p4 = Plots.contourf(xc,yc, (B.+H)'; levels=20, aspect_ratio=1)
    display(plot(p1,p2,p3,p4; layout=(2,2), size=(980,980)))
    return 
end 


# using GLMakie

# lx,ly,lz = 30000.0,30000.0,1000.0
# ox,oy,oz = -lx/2,-ly/2,0.0

# nx,ny = 128,128

# xv,yv = LinRange(ox,ox+lx,nx+1),LinRange(oy,oy+ly,ny+1)
# xc,yc = 0.5*(xv[1:end-1]+xv[2:end]),0.5*(yv[1:end-1]+yv[2:end])

# # bedrock properties
# wx1,wy1 = 0.4lx,0.2ly
# wx2,wy2 = 0.15lx,0.15ly
# ω1 = 4
# ω2 = 6

# fun = @. 0.4exp(-(xc/wx1)^2-((yc-0.2ly)'/wy1)^2) + 0.2*exp(-(xc/wx2)^2-((yc-0.1ly)'/wy2)^2)*(cos(ω1*π*(xc/lx + 0*yc'/ly-0.25)) + 1)
# @. fun += 0.025exp(-(xc/wx2)^2-(yc'/wy2)^2)*cos(ω2*π*(0*xc/lx + yc'/ly)) 
# @. fun += 0xc + 0.15*(yc'/ly)
# zmin,zmax = extrema(fun)
# @. fun = (fun - zmin)/(zmax-zmin)

# B = zeros(nx,ny)
# @. B += oz + (lz-oz)*fun

# # make figure
# fig = Figure(resolution=(2000,1000),fontsize=32)
# ax = Axis3(fig[1,1];aspect=(5,5,1))
# surface!(ax,xc,yc,B)

# display(fig)


sin_wave()