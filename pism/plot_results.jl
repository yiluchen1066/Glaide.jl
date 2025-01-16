using CairoMakie
using NCDatasets

include("bedrock_glaide.jl")

function visualise(input, filename::String, output::String)

    Lx ,Ly ,res ,B_0 ,B_a ,W_1 ,W_2 = input

    # Read the NetCDF file
    ds = NCDataset("input_glaide.nc", "r")
    B  = ds["bedrock_elevation"][:, :] # dimensions: x, y
    xc = ds["x"][:] # dimensions: x
    yc = ds["y"][:] # dimensions: y
    close(ds)

    ds = NCDataset(filename, "r")
    S = ds["usurf"][:, :, 1] # dimensions: x, y, time
    close(ds)

    # thickness
    H = S .- B

    @info "maximum ice thickness (m): $(maximum(H))"

    # Plotting
    fig = Figure(size = (500, 800))
    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y"),
           Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y"),
           Axis(fig[3, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y"))
    hms = (heatmap!(axs[1], xc, yc, B; colormap=:roma),
           heatmap!(axs[2], xc, yc, S; colormap=:davos),
           heatmap!(axs[3], xc, yc, H; colormap=:turbo))
    cbs = (Colorbar(fig[1, 1][1, 2], hms[1]),
           Colorbar(fig[2, 1][1, 2], hms[2]),
           Colorbar(fig[3, 1][1, 2], hms[3]))

    save(output, fig)
end

function (@main)(ARGS)

    filename, output = ARGS[1:2]

    visualise(define_input(), filename, output)
    return
end
