using Glaide, CairoMakie, Unitful, CUDA, Logging, Printf, JLD2
CUDA.device!(3)

function benchmark(resolution; save_results=false)
    # physics
    n  = 3
    ρ  = 910.0u"kg/m^3"
    g  = 9.81u"m/s^2"
    A₀ = 2.5e-24u"Pa^-3*s^-1"

    # reduction factor
    E = 1.0

    # rescaled units
    ρgnA = ((ρ * g)^n * E * A₀) * (L_REF^n * T_REF) |> NoUnits

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
    scalars = TimeDependentScalars(; n, ρgnA, lx, ly, dt=2u"yr" / T_REF |> NoUnits, b, mb_max, ela)

    # default solver parameters
    numerics = TimeDependentNumerics(xc, yc; dmpswitch=ceil(Int, 0.5nx), reg)

    model = TimeDependentSIA(scalars, numerics; debug_vis=false)

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

    # with_theme(theme_latexfonts()) do
    #     B = Array(model.fields.B) .* ustrip(u"m", L_REF)
    #     H = Array(model.fields.H) .* ustrip(u"m", L_REF)
    #     V = Array(model.fields.V) .* ustrip(u"m/yr", L_REF / T_REF)

    #     ice_mask = H .== 0

    #     H_v = copy(H)
    #     V_v = copy(V)

    #     # mask out ice-free pixels
    #     H_v[ice_mask] .= NaN
    #     V_v[ice_mask] .= NaN

    #     fig = Figure(; size=(320, 600), fontsize=16)

    #     axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect()),
    #            Axis(fig[2, 1][1, 1]; aspect=DataAspect()),
    #            Axis(fig[3, 1][1, 1]; aspect=DataAspect()))

    #     hidexdecorations!.((axs[1], axs[2]))

    #     for ax in axs
    #         ax.xgridvisible = true
    #         ax.ygridvisible = true
    #         limits!(ax, -10, 10, -10, 10)
    #     end

    #     axs[1].ylabel = L"y~\mathrm{[km]}"
    #     axs[2].ylabel = L"y~\mathrm{[km]}"
    #     axs[3].ylabel = L"y~\mathrm{[km]}"

    #     axs[3].xlabel = L"x~\mathrm{[km]}"

    #     axs[1].title = L"B~\mathrm{[m]}"
    #     axs[2].title = L"H~\mathrm{[m]}"
    #     axs[3].title = L"V~\mathrm{[m/a]}"

    #     # convert to km for plotting
    #     xc_km = xc .* ustrip(u"km", L_REF)
    #     yc_km = yc .* ustrip(u"km", L_REF)

    #     hms = (heatmap!(axs[1], xc_km, yc_km, B),
    #            heatmap!(axs[2], xc_km, yc_km, H_v),
    #            heatmap!(axs[3], xc_km, yc_km, V_v))

    #     # enable interpolation for smoother picture
    #     # foreach(hms) do h
    #     #     h.interpolate = true
    #     # end

    #     hms[1].colormap = :terrain
    #     hms[2].colormap = Reverse(:ice)
    #     hms[3].colormap = :matter

    #     hms[1].colorrange = (1000, 4000)
    #     hms[2].colorrange = (0, 150)
    #     hms[3].colorrange = (0, 300)

    #     cbs = (Colorbar(fig[1, 1][1, 2], hms[1]),
    #            Colorbar(fig[2, 1][1, 2], hms[2]),
    #            Colorbar(fig[3, 1][1, 2], hms[3]))

    #     fig
    # end |> display

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
    writedlm("../output/pism_benchmark_times.txt", hcat(resolutions, exec_times))

    return
end

run_bench()
