using GLMakie

xc = LinRange(-1,1,101)
yc = LinRange(-1,1,101)

B = @. sin(2π*xc)*cos(2π*yc')
H = @. sqrt(max(0.6 - xc^2 - yc'^2,0.0))
S = @. B + H; @. S[H == 0.0] = NaN

fig = Figure(resolution=(2000,1000),fontsize=32)
ax  = Axis3(fig[1,1][1,1];aspect=:data,title="Rhone glacier")

surface!(ax,B;colormap=:terrain)
surface!(ax,S;colormap=:ice    )

save(fig,"rhone.png")