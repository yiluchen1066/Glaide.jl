using CairoMakie
using NCDatasets

function visualise(filename::String, output::String)
    # Read the NetCDF file
    ds = NCDataset("pism_input.nc", "r")
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
    fig = Figure(size = (1100, 300))
    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="Bedrock elevation", xlabel="x", ylabel="y"),
           Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="Surface elevation", xlabel="x"),
           Axis(fig[1, 3][1, 1]; aspect=DataAspect(), title="Ice thickness", xlabel="x"))
    hms = (heatmap!(axs[1], xc, yc, B; colormap=:roma),
           heatmap!(axs[2], xc, yc, S; colormap=:davos),
           heatmap!(axs[3], xc, yc, H; colormap=:turbo))
    cbs = (Colorbar(fig[1, 1][1, 2], hms[1]),
           Colorbar(fig[1, 2][1, 2], hms[2]),
           Colorbar(fig[1, 3][1, 2], hms[3]))

    save(output, fig)
end

function (@main)(ARGS)

    filename, output = ARGS[1:2]

    visualise(filename, output)
    return
end
