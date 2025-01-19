using NCDatasets
using CairoMakie

const SECONDS_IN_YEAR = 60 * 60 * 24 * 365

define_input() = (Lx  = 20e3,    # meters
                  Ly  = 20e3,    # meters
                  res = 100.0,    # meters
                  B_0 = 1000.0,  # meters
                  B_a = 3000.0,  # meters
                  W_1 = 1e4,
                  W_2 = 3e3)

# SMB - directly passed as input to PISM exe
# β     = 0.01 / SECONDS_IN_YEAR
# b_max = 2.5  / SECONDS_IN_YEAR
# ela   = 1800.0

"Create a NetCDF file that can be used with PISM"
function create_pism_input(input, filename::String)

    Lx, Ly, resolution, B_0, B_a, W_1, W_2 = input

    dx, dy = resolution, resolution
    nx, ny = ceil(Int, Lx / dx), ceil(Int, Ly / dy)

    @info "Resolution nx, ny = $((nx, ny))"

    # if the resolution is fixed, domain extents need to be corrected
    lx, ly = nx * dx, ny * dy

    # grid cell center coordinates
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)

    "Bed elevation in meters"
    B = @. B_0 + 0.5 * B_a * (exp(-(xc / W_1)^2 - (yc' / W_2)^2) +
                              exp(-(xc / W_2)^2 - (yc' / W_1)^2))

    NCDataset(filename, "c") do ds
        defDim(ds, "x", length(xc))
        defDim(ds, "y", length(yc))

        var_x = defVar(ds, "x", Float64, ["x"])
        var_x.attrib["units"] = "meter"
        var_x[:] = xc

        var_y = defVar(ds, "y", Float64, ["y"])
        var_y.attrib["units"] = "meter"
        var_y[:] = yc

        bed_var = defVar(ds, "bedrock_elevation", Float64, ["y", "x"])
        bed_var.attrib["units"] = "meter"
        bed_var.attrib["standard_name"] = "bedrock_altitude"
        bed_var[:, :] .= B

        temp_var = defVar(ds, "ice_surface_temp", Float64, ["y", "x"])
        temp_var.attrib["units"] = "Celsius"
        temp_var.attrib["long_name"] = "ice surface temperature"
        temp_var[:, :] .= 0.0
    end
    return
end

function (@main)(ARGS)

    create_pism_input(define_input(), ARGS[1])

    return
end
