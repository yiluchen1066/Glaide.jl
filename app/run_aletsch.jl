using Glaide, CairoMakie, Unitful, JLD2

function main(input_file)
    reg   = 5e-6u"s^-1" * T_REF |> NoUnits
    model = TimeDependentSIA(input_file; reg, report=true, debug_vis=false, dmpswitch=2000)

    model.scalars.ρgnA *= 0.25
    # model.scalars.dt = 1.0

    (; xc, yc) = model.numerics

    V_old = copy(model.fields.V)
    H_old = copy(model.fields.H_old)

    # set As to the background value
    Aₛ₀     = 1e-22u"Pa^-3*s^-1*m"
    ρgnAs_0 = ((RHOG)^GLEN_N * Aₛ₀) * (L_REF^(GLEN_N - 1) * T_REF) |> NoUnits

    X = load("../output/snapshot_aletsch_25m/step_0100.jld2", "X")

    # fill!(model.fields.ρgnAs, ρgnAs_0)
    copy!(model.fields.ρgnAs, exp.(X))

    # solve the model to steady state
    @time solve!(model)

    with_theme(theme_latexfonts()) do
        ofunit(ut, x, ref) = x * ustrip(ut, ref)

        B     = ofunit.(u"m", Array(model.fields.B), L_REF)
        H     = ofunit.(u"m", Array(model.fields.H), L_REF)
        H_old = ofunit.(u"m", Array(model.fields.H_old), L_REF)
        V     = ofunit.(u"m/yr", Array(model.fields.V), L_REF / T_REF)
        V_old = ofunit.(u"m/yr", Array(V_old), L_REF / T_REF)

        ρgnAs   = ofunit.(u"m^-2*s^-1", Array(model.fields.ρgnAs), L_REF^(-2) * T_REF^(-1))
        As      = ρgnAs ./ ustrip(RHOG^GLEN_N)
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
        xc_km = ofunit.(u"km", xc, L_REF)
        yc_km = ofunit.(u"km", yc, L_REF)

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
        hms[3].colorrange = (0, 900)
        hms[4].colorrange = (0, 900)
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

main("../datasets/aletsch_25m.jld2")
