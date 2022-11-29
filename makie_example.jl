using GLMakie
using DelimitedFiles
using LinearAlgebra

xc = vec(readdlm("xc_rhone.txt"))#LinRange(-1,1,201)
yc = vec(readdlm("yc_rhone.txt"))#LinRange(-1,1,201)

dx,dy = xc[2]-xc[1], yc[2]-yc[1]

B = readdlm("B_rhone.txt")  # @. 0.1*sin(2π*xc)*cos(2π*yc')
H = readdlm("H_rhone.txt")  # @. 0.5*sqrt(max(0.6 - xc^2 - yc'^2,0.0))
S = readdlm("S_rhone.txt"); @. S[H == 0.0] = NaN  # @. B + Hs

Vx = readdlm("vx.txt")
Vy = readdlm("vy.txt")
Vz = 0.0.*Vx

for iy in axes(B,2)[2:end-1]
    for ix in axes(B,1)[2:end-1]
        ∇Sx = (S[ix+1,iy] - S[ix-1,iy])/(2dx)
        ∇Sy = (S[ix,iy+1] - S[ix,iy-1])/(2dy)
        N         = normalize(Vec3(∇Sx,∇Sy,-1.0))
        V         = Vec3(Vx[ix,iy],Vy[ix,iy],0.0)
        Vz[ix,iy] = dot(N,V)
    end
end


fig = Figure(resolution=(2500,1200),fontsize=42)
ax  = Axis3(fig[1,1][1,1];aspect=:data,title="Rhone glacier")

xc2 = [x for x in xc, _ in yc]
yc2 = [y for _ in xc, y in yc]

st = 30
ps = [Point3f(xc2[ix,iy],yc2[ix,iy],S[ix,iy] + 10.0) for ix in 1:st:size(B,1) for iy in 1:st:size(B,2)]
ns = [150*normalize(Vec3f(Vx[ix,iy], Vy[ix,iy], Vz[ix,iy])) for ix in 1:st:size(B,1) for iy in 1:st:size(B,2)]

idx = findall(v -> v[2] > 200.0, ps)
ps = ps[idx]
ns = ns[idx]

surface!(ax,xc,yc,B;colormap=:lightterrain)
surface!(ax,xc,yc,S;colormap=:ice         )
arrows!(ax,ps,ns;fxaa=true, # turn on anti-aliasing
           linecolor = :white, arrowcolor = :white,
           linewidth = 20.0,arrowsize = Vec3f(70, 70, 100))

save("rhone.png",fig)

fig