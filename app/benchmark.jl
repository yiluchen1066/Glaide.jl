using Glaide, CairoMakie, Unitful, CUDA, Logging, Printf, JLD2, DelimitedFiles

function benchmark(resolution; save_results=false)
    # reduction factor
    E = 1.0

    # rescaled units
    ρgnA = (RHOG^GLEN_N * E * GLEN_A) * (L_REF^GLEN_N * T_REF) |> NoUnits

    # geometry
    Lx = 20u"km" / L_REF |> NoUnits
    Ly = 20u"km" / L_REF |> NoUnits

    # bed elevation parameters
    B_0 = 1u"km" / L_REF |> NoUnits
    B_a = 3u"km" / L_REF |> NoUnits
    W_1 = 10u"km" / L_REF |> NoUnits
    W_2 = 3u"km" / L_REF |> NoUnits

    # mass balance parameters
    b      = 0.01u"yr^-1" * T_REF |> NoUnits
    mb_max = 2.5u"m*yr^-1" / L_REF * T_REF |> NoUnits
    ela    = 1.8u"km" / L_REF |> NoUnits

    h = resolution * 1u"m" / L_REF |> NoUnits

    # numerical parameters
    dx, dy = h, h
    nx, ny = ceil(Int, Lx / dx), ceil(Int, Ly / dy)

    reg = 5e-8u"s^-1" * T_REF |> NoUnits

    # if the resolution is fixed, domain extents need to be corrected
    lx, ly = nx * dx, ny * dy

    # grid cell center coordinates
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)

    # default scalar parameters for a steady state (dt = ∞)
    scalars = TimeDependentScalars(; n=GLEN_N, ρgnA, lx, ly, dt=2u"yr" / T_REF |> NoUnits, b, mb_max, ela)

    # default solver parameters
    numerics = TimeDependentNumerics(nx, ny; reg)

    model = TimeDependentSIA(scalars, numerics)

    # set the bed elevation
    copy!(model.fields.B, @. B_0 + 0.5 * B_a * (exp(-(xc / W_1)^2 - (yc' / W_2)^2) +
                                                exp(-(xc / W_2)^2 - (yc' / W_1)^2)))

    # set As to the zero value for compatibility with PISM
    fill!(model.fields.ρgnAs, 0.0)

    # accumulation allowed everywhere
    fill!(model.fields.mb_mask, 1.0)

    for it in 1:50
        # copy the current ice thickness to the old ice thickness
        copyto!(model.fields.H_old, model.fields.H)

        # solve the model
        solve!(model)
    end

    if save_results
        jldsave("../output/pism_benchmark_$(resolution)m.jld2"; H=Array(model.fields.H))
    end

    return
end

function run_bench()
    exec_times  = Float64[]
    resolutions = [200, 100, 50, 25, 10]

    for iwarmup in 1:3
        benchmark(resolutions[1])
    end

    for resolution in resolutions
        ttot = @elapsed benchmark(resolution)
        nx   = ceil(Int, 20e3 / resolution)

        @info @sprintf("Resolution: %d m; nx, ny = (%d, %d), total time: %.3f s", resolution, nx, nx, ttot)

        push!(exec_times, ttot)
    end

    println("save data")
    benchmark(50; save_results=true)
    benchmark(25; save_results=true)

    # save the execution times
    writedlm("../output/pism_benchmark_times2.txt", hcat(resolutions, exec_times))

    return
end

run_bench()
