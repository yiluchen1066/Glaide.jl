using CairoMakie

nx, ny = 128, 128

fld_old = (
    H   = zeros(nx, ny),
    D   = zeros(nx - 1, ny - 1),
    As  = zeros(nx - 1, ny - 1),
    ELA = zeros(nx, ny),
    β   = zeros(nx, ny),
)

fld_new = (
    H   = zeros(nx, ny),
    D   = zeros(nx - 1, ny - 1),
    As  = zeros(nx - 1, ny - 1),
    ELA = zeros(nx, ny),
    β   = zeros(nx, ny),
)

open("output/synthetic_old.dat", "r") do io
    read!(io, fld_old.H)
    read!(io, fld_old.D)
    read!(io, fld_old.As)
    read!(io, fld_old.ELA)
    read!(io, fld_old.β)
end

open("output/synthetic_new.dat", "r") do io
    read!(io, fld_new.H)
    read!(io, fld_new.D)
    read!(io, fld_new.As)
    read!(io, fld_new.ELA)
    read!(io, fld_new.β)
end

@show maximum(abs.(fld_old.H   .- fld_new.H))
@show maximum(abs.(fld_old.D   .- fld_new.D))
@show maximum(abs.(fld_old.As  .- fld_new.As))
@show maximum(abs.(fld_old.ELA .- fld_new.ELA))
@show maximum(abs.(fld_old.β   .- fld_new.β))

fig = Figure(; resolution=(1000, 800), fontsize=32)

ax  = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="ΔH")
plt = heatmap!(ax, fld_old.H .- fld_new.H; colormap=:turbo)

Colorbar(fig[1, 1][1, 2], plt)

display(fig)
