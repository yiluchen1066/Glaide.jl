using Glaide, CairoMakie, Unitful

function main()
    # physics
    n   = 3
    ρ   = 910.0u"kg/m^3"
    g   = 9.81u"m/s^2"
    A₀  = 2.5e-24u"Pa^-3*s^-1"
    Aₛ₀ = 1e-22u"Pa^-3*s^-1*m"

    # reduction factor
    E = 1.0

    # dimensionless amplitude and frequency of the perturbation
    ρgnAs_a = 2.0
    ω       = 3π

    # reference scales
    Lref = 1u"hm"
    Tref = 1u"yr"

    # rescaled units
    @show ρgnA    = ((ρ * g)^n * E * A₀) * (Lref^n * Tref) |> NoUnits
    @show ρgnAs_0 = ((ρ * g)^n * Aₛ₀) * (Lref^(n - 1) * Tref) |> NoUnits

    # geometry
    @show Lx = 20u"km" / Lref |> NoUnits
    @show Ly = 20u"km" / Lref |> NoUnits

    @show resolution = 50u"m" / Lref |> NoUnits

    # bed elevation parameters
    @show B_0 = 1u"km" / Lref |> NoUnits
    @show B_a = 3u"km" / Lref |> NoUnits
    @show W_1 = 10u"km" / Lref |> NoUnits
    @show W_2 = 3u"km" / Lref |> NoUnits

    # mass balance parameters
    @show b      = 0.01u"yr^-1" * Tref |> NoUnits
    @show mb_max = 2.5u"m*yr^-1" / Lref * Tref |> NoUnits
    @show ela    = 1.8u"km" / Lref |> NoUnits

    # numerical parameters
    dx, dy = resolution, resolution
    nx, ny = ceil(Int, Lx / dx), ceil(Int, Ly / dy)
    # dx, dy = Lx / nx, Ly / ny

    @show reg = 5e-8u"s^-1" * Tref |> NoUnits

    # if the resolution is fixed, domain extents need to be corrected
    lx, ly = nx * dx, ny * dy

    # grid cell center coordinates
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)

    # default scalar parameters for a steady state (dt = ∞)
    scalars = TimeDependentScalars(; n, ρgnA, lx, ly, dt=Inf, b, mb_max, ela)

    # default solver parameters
    numerics = TimeDependentNumerics(xc, yc; dmpswitch=2nx, reg)

    model = TimeDependentSIA(scalars, numerics; report=true, debug_vis=false)

    # set the bed elevation
    copy!(model.fields.B, @. B_0 + 0.5 * B_a * (exp(-(xc / W_1)^2 - (yc' / W_2)^2) +
                                                exp(-(xc / W_2)^2 - (yc' / W_1)^2)))

    # set As to the background value
    fill!(model.fields.ρgnAs, ρgnAs_0)

    # accumulation allowed everywhere
    fill!(model.fields.mb_mask, 1.0)

    # solve the model to steady state
    @time solve!(model)

    # save geometry and surface velocity
    model.fields.H_old .= model.fields.H
    V_old = Array(model.fields.V)

    # sliding parameter perturbation
    ρgnAs_synthetic = @. 10^(log10(ρgnAs_0) + ρgnAs_a * cos(ω * xc / lx) * sin(ω * yc' / ly))

    copyto!(model.fields.ρgnAs, ρgnAs_synthetic)

    # step change in mass balance (ELA +20%)
    model.scalars.ela = ela * 1.2

    # finite time step (15y)
    model.scalars.dt = 15u"yr" / Tref |> NoUnits

    # solve again
    @time solve!(model)

    ρgnĀs = zero(model.fields.ρgnAs)

    @. model.adjoint_fields.H̄ = 1.0e-1 * model.fields.H / $maximum(abs, model.fields.H)
    @. model.adjoint_fields.V̄ = 1.0e-1 * model.fields.V / $maximum(abs, model.fields.V)

    @time Glaide.solve_adjoint!(ρgnĀs, model)

    with_theme(theme_latexfonts()) do
        ofunit(ut, x, ref) = x * ustrip(ut, ref)

        B     = ofunit.(u"m", Array(model.fields.B), Lref)
        H     = ofunit.(u"m", Array(model.fields.H), Lref)
        H_old = ofunit.(u"m", Array(model.fields.H_old), Lref)
        V     = ofunit.(u"m/yr", Array(model.fields.V), Lref / Tref)
        V_old = ofunit.(u"m/yr", Array(V_old), Lref / Tref)

        ρgnAs   = ofunit.(u"m^-2*s^-1", Array(model.fields.ρgnAs), Lref^(-2) * Tref^(-1))
        As      = ρgnAs ./ ustrip((ρ * g)^n)
        mb_mask = model.fields.mb_mask

        ice_mask_old = H_old .== 0
        ice_mask     = H .== 0

        H_old_v = copy(H_old)
        H_v     = copy(H)
        As_v    = copy(As)
        V_old_v = copy(V_old)
        V_v     = copy(V)

        # mask out ice-free pixels
        H_old_v[ice_mask_old] .= NaN
        V_old_v[ice_mask_old] .= NaN

        H_v[ice_mask] .= NaN
        As_v[ice_mask] .= NaN
        V_v[ice_mask] .= NaN

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
        axs[2].title = L"\log_{10}(A_\mathrm{s})"
        axs[3].title = L"H_\mathrm{old}~\mathrm{[m]}"
        axs[4].title = L"H~\mathrm{[m]}"
        axs[5].title = L"V_\mathrm{old}~\mathrm{[m/a]}"
        axs[6].title = L"V~\mathrm{[m/a]}"

        # convert to km for plotting
        xc_km = ofunit.(u"km", xc, Lref)
        yc_km = ofunit.(u"km", yc, Lref)

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
        hms[3].colormap = Reverse(:ice)
        hms[4].colormap = Reverse(:ice)
        hms[5].colormap = :matter
        hms[6].colormap = :matter

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
