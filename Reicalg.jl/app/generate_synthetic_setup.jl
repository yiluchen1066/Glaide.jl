using Reicalg
using CUDA
using CairoMakie
using JLD2

@views av1(a) = @. 0.5 * (a[1:end-1] + a[2:end])

function generate_synthetic_data(dst_dir, res_m; vis=true)
    # sliding parameter background value
    As0 = 1e-22

    # sliding parameter variation amplitude
    As_amp = 4.0

    # synthetic geometry parameters
    lx, ly = 20e3, 20e3
    B0     = 1000.0
    B_amp  = 3000.0

    # equilibrium line altitude
    β     = 0.01 / SECONDS_IN_YEAR
    b_max = 2.0 / SECONDS_IN_YEAR
    ela   = 1800.0

    # preprocessing
    dx, dy = res_m, res_m
    nx, ny = ceil(Int, lx / dx), ceil(Int, ly / dy)
    lx, ly = nx * dx, ny * dy

    # default scalar parameters for a steady state (dt = ∞)
    scalars = TimeDependentScalars(; lx, ly, dt=Inf, β, b_max, ela)

    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)

    # default solver parameters
    numerics = TimeDependentNumerics(xc, yc; dmp1=0.9, dmp2=0.95)

    xv, yv = av1(xc), av1(yc)

    # arrays
    model = TimeDependentSIA(scalars, numerics)

    (; B, H, H_old, V, As, mb_mask) = model.fields

    # initialise

    # two bumps similar to (Visnjevic et al., 2018)
    copy!(B, @. B0 + B_amp * (0.5 * exp(-xc^2 / 1e8 - yc'^2 / 1e7) +
                              0.5 * exp(-xc^2 / 1e7 - yc'^2 / 1e8)))

    #  initialise As with a background value
    fill!(As, As0)

    # accumulation allowed everywhere
    fill!(mb_mask, 1.0)

    # solve for a steady state to get initial synthetic geometry
    solve!(model)

    # save geometry and surface velocity
    H_old .= H
    V_old = copy(V)

    # step change in mass balance (ELA +20%)
    scalars.ela *= 1.2

    # sliding parameter variation
    copy!(As, @. exp(log(As0) + As_amp * cos(3π * xv / lx) * sin(3π * yv' / ly)))

    # finite time step (10y)
    scalars.dt = 10 * SECONDS_IN_YEAR

    # update scalar parameters
    (; npow, A, ρgn, ela, dt) = scalars

    # solve again
    solve!(model)

    # transfer arrays to CPU
    H       = Array(H)
    H_old   = Array(H_old)
    B       = Array(B)
    As      = Array(As)
    V       = Array(V)
    V_old   = Array(V_old)
    mb_mask = Array(mb_mask)

    # pack everything into named tuples for saving
    fields   = (; B, H, H_old, V, V_old, As, mb_mask)
    numerics = (; nx, ny, dx, dy, xc, yc)
    scalars  = (; lx, ly, β, b_max, ela, dt, npow, A, ρgn)

    mkpath(dst_dir)
    jldsave(joinpath(dst_dir, "synthetic_setup.jld2"); fields, scalars, numerics)

    if vis
        # generate ice mask and mask all data
        #! format: off
        ice_mask = Array(H[1:end-1, 1:end-1] .== 0 .||
                         H[2:end  , 1:end-1] .== 0 .||
                         H[1:end-1, 2:end  ] .== 0 .||
                         H[2:end  , 2:end  ] .== 0)
        #! format: on

        As[ice_mask] .= NaN

        fig = Figure(; size=(800, 400), fontsize=12)

        axs = (B     = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="B"),
               As    = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="log10(As)"),
               H_old = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="H_old"),
               H     = Axis(fig[2, 2][1, 1]; aspect=DataAspect(), title="H"),
               V_old = Axis(fig[1, 3][1, 1]; aspect=DataAspect(), title="V_old"),
               V     = Axis(fig[2, 3][1, 1]; aspect=DataAspect(), title="V"))

        # convert to km for plotting
        xc_km, yc_km = xc / 1e3, yc / 1e3

        #! format: off
        hms = (B     = heatmap!(axs.B    , xc_km, yc_km, B                       ; colormap=:terrain),
               As    = heatmap!(axs.As   , xc_km, yc_km, log10.(As)              ; colormap=:roma),
               H_old = heatmap!(axs.H_old, xc_km, yc_km, H_old                   ; colormap=:turbo, colorrange=(0, 250)),
               H     = heatmap!(axs.H    , xc_km, yc_km, H                       ; colormap=:turbo, colorrange=(0, 250)),
               V_old = heatmap!(axs.V_old, xc_km, yc_km, V_old .* SECONDS_IN_YEAR; colormap=:vik  , colorrange=(0, 300)),
               V     = heatmap!(axs.V    , xc_km, yc_km, V     .* SECONDS_IN_YEAR; colormap=:vik  , colorrange=(0, 300)))
        #! format: on

        cbs = (B     = Colorbar(fig[1, 1][1, 2], hms.B),
               As    = Colorbar(fig[2, 1][1, 2], hms.As),
               H_old = Colorbar(fig[1, 2][1, 2], hms.H_old),
               H     = Colorbar(fig[2, 2][1, 2], hms.H),
               V_old = Colorbar(fig[1, 3][1, 2], hms.V_old),
               V     = Colorbar(fig[2, 3][1, 2], hms.V))

        display(fig)
    end

    return
end
