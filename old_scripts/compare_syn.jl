using CairoMakie

H_old = zeros(128,128)
H_new = zeros(128,128)

read!("synthetic_old.dat", H_old)
read!("synthetic_new.dat", H_new)

fig  = Figure(resolution=(2000, 1500), fontsize=32)

ax  = Axis(fig[1,1][1,1]; aspect=DataAspect(), title="ΔH")
plt = heatmap!(ax, H_old.-H_new; colormap=:turbo)

Colorbar(fig[1,1][1,2], plt)

display(fig)
