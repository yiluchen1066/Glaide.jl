using Glaide, CairoMakie

function main()
    # reference scales
    Tref = SECONDS_IN_HOUR

    # our scales
    Tsc = 1.0
    Lsc = 1.0

    # geometry
    Lx, Ly     = 20e3, 20e3
    resolution = 50.0

    # ice flow parameters
    As_0 = 1e-22
    As_a = 2.0
    ω    = 3π

    # bed elevation parameters
    B_0 = 1000.0
    B_a = 3000.0
    W_1 = 1e4
    W_2 = 3e3

    # mass balance parameters
    b      = 0.01 / SECONDS_IN_YEAR
    mb_max = 2.5 / SECONDS_IN_YEAR
    ela    = 1800.0

    # numerical parameters
    # dx, dy = resolution, resolution
    nx, ny = ceil(Int, Lx / resolution), ceil(Int, Ly / resolution)
    dx, dy = Lx / nx, Ly / ny

    # if the resolution is fixed, domain extents need to be corrected
    lx, ly = nx * dx, ny * dy

    # grid cell center coordinates
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)

    # default scalar parameters for a steady state (dt = ∞)
    scalars = TimeDependentScalars(; lx, ly, dt=Inf, b, mb_max, ela)

    # default solver parameters
    numerics = TimeDependentNumerics(xc, yc; dmpswitch=5nx)

    model = TimeDependentSIA(scalars, numerics; report=true, debug_vis=false)

    # set the bed elevation
    copy!(model.fields.B, @. B_0 + 0.5 * B_a * (exp(-(xc / W_1)^2 - (yc' / W_2)^2) +
                                                exp(-(xc / W_2)^2 - (yc' / W_1)^2)))

    # set As to the background value
    fill!(model.fields.As, As_0)

    # accumulation allowed everywhere
    fill!(model.fields.mb_mask, 1.0)

    # solve the model to steady state
    @time solve!(model)

    # save geometry and surface velocity
    model.fields.H_old .= model.fields.H
    V_old = Array(model.fields.V)

    # sliding parameter perturbation
    As_synthetic = @. 10^(log10(As_0) + As_a * cos(ω * xc / lx) * sin(ω * yc' / ly))

    copyto!(model.fields.As, As_synthetic)

    # step change in mass balance (ELA +20%)
    model.scalars.ela = ela * 1.2

    # finite time step (15y)
    model.scalars.dt = 15 * SECONDS_IN_YEAR

    # solve again
    @time solve!(model)

    # Ās = zero(model.fields.As)

    # @. model.adjoint_fields.H̄ = 1.0e-2 * model.fields.H / $maximum(abs, model.fields.H)
    # @. model.adjoint_fields.V̄ = 1.0e-2 * model.fields.V / $maximum(abs, model.fields.V)

    # @time Glaide.solve_adjoint!(Ās, model)

    with_theme(theme_latexfonts()) do
        B       = Array(model.fields.B)
        H       = Array(model.fields.H)
        H_old   = Array(model.fields.H_old)
        V       = Array(model.fields.V)
        As      = Array(model.fields.As)
        mb_mask = Array(model.fields.mb_mask)

        ice_mask_old = H_old .== 0
        ice_mask     = H .== 0

        H_old_v = copy(H_old)
        H_v     = copy(H)
        As_v    = copy(As)
        # convert to m/a
        V_old_v = copy(V_old) .* SECONDS_IN_YEAR
        V_v     = copy(V) .* SECONDS_IN_YEAR

        # mask out ice-free pixels
        H_old_v[ice_mask_old] .= NaN
        V_old_v[ice_mask_old] .= NaN

        H_v[ice_mask]  .= NaN
        As_v[ice_mask] .= NaN
        V_v[ice_mask]  .= NaN

        fig = Figure(; size=(800, 450), fontsize=16)

        axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect()),
               Axis(fig[2, 1][1, 1]; aspect=DataAspect()),
               Axis(fig[1, 2][1, 1]; aspect=DataAspect()),
               Axis(fig[2, 2][1, 1]; aspect=DataAspect()),
               Axis(fig[1, 3][1, 1]; aspect=DataAspect()),
               Axis(fig[2, 3][1, 1]; aspect=DataAspect()))

        hidexdecorations!.((axs[1], axs[3], axs[5]))
        hideydecorations!.((axs[3], axs[4], axs[5], axs[6]))

        for ax in axs
            ax.xgridvisible = true
            ax.ygridvisible = true
            limits!(ax, -10, 10, -10, 10)
        end

        axs[1].ylabel = L"y~\mathrm{[km]}"
        axs[2].ylabel = L"y~\mathrm{[km]}"

        axs[2].xlabel = L"x~\mathrm{[km]}"
        axs[4].xlabel = L"x~\mathrm{[km]}"
        axs[6].xlabel = L"x~\mathrm{[km]}"

        axs[1].title = L"B~\mathrm{[m]}"
        axs[2].title = L"\log_{10}(As)"
        axs[3].title = L"H_\mathrm{old}~\mathrm{[m]}"
        axs[4].title = L"H~\mathrm{[m]}"
        axs[5].title = L"V_\mathrm{old}~\mathrm{[m/a]}"
        axs[6].title = L"V~\mathrm{[m/a]}"

        # convert to km for plotting
        xc_km, yc_km = xc / 1e3, yc / 1e3

        hms = (heatmap!(axs[1], xc_km, yc_km, B),
               heatmap!(axs[2], xc_km, yc_km, log10.(As_v)),
               heatmap!(axs[3], xc_km, yc_km, H_old_v),
               heatmap!(axs[4], xc_km, yc_km, H_v),
               heatmap!(axs[5], xc_km, yc_km, V_old_v),
               heatmap!(axs[6], xc_km, yc_km, V_v))

        # enable interpolation for smoother picture
        foreach(hms) do h
            h.interpolate = true
        end

        hms[1].colormap = :terrain
        hms[2].colormap = Makie.Reverse(:roma)
        hms[3].colormap = :vik
        hms[4].colormap = :vik
        hms[5].colormap = :turbo
        hms[6].colormap = :turbo

        hms[1].colorrange = (1000, 4000)
        hms[2].colorrange = (-24, -20)
        hms[3].colorrange = (0, 150)
        hms[4].colorrange = (0, 150)
        hms[5].colorrange = (0, 300)
        hms[6].colorrange = (0, 300)

        cbs = (Colorbar(fig[1, 1][1, 2], hms[1]),
               Colorbar(fig[2, 1][1, 2], hms[2]),
               Colorbar(fig[1, 2][1, 2], hms[3]),
               Colorbar(fig[2, 2][1, 2], hms[4]),
               Colorbar(fig[1, 3][1, 2], hms[5]),
               Colorbar(fig[2, 3][1, 2], hms[6]))

        fig
    end |> display

    return
end

main()
