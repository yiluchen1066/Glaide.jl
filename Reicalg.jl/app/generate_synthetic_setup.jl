using Reicalg
using CUDA
using CairoMakie
using JLD2

const SECONDS_IN_YEAR = 3600 * 24 * 365

@views av1(a) = @. 0.5 * (a[1:end-1] + a[2:end])

function generate_synthetic_data(nx, ny; vis=true)
    # set CUDA device
    CUDA.device!(1)

    # ice flow parameters
    npow = 3                 # glen's power law exponent
    A    = 2.5e-24           # ice flow parameter
    As0  = 1e-21             # sliding flow parameter
    ρgn  = (910 * 9.81)^npow # gravity (pre-exponentiated)

    # synthetic geometry parameters
    lx, ly = 200e3, 200e3
    B0     = 3500.0

    # synthetic mass balance parameters
    β     = 0.01 / SECONDS_IN_YEAR
    b_max = 2.0 / SECONDS_IN_YEAR
    ELA0  = 2150.0

    # time step (infinte since we want steady state)
    dt = Inf

    # numerics
    cfl     = 1 / 6.1
    maxiter = 100 * max(nx, ny)
    ncheck  = 5nx
    ϵtol    = 1e-6

    # preprocessing
    dx, dy = lx / nx, ly / ny
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dx / 2, ly / 2 - dx / 2, ny)

    xv, yv = av1(xc), av1(yc)

    # arrays
    fields = SIA_fields(nx, ny)

    (; B, H, H_old, V, As, ELA, mb_mask) = fields

    # initialise

    # two bumps as in (Visnjevic et al., 2018)
    copy!(B, @. B0 * (exp(-xc^2 / 1e10 - yc'^2 / 1e9) +
                      exp(-xc^2 / 1e9 - (yc' - ly / 8)^2 / 1e10)))

    copy!(As, @. exp(log(As0) * (0.15 * cos(5π * xv / lx) * sin(5π * yv' / ly) + 1.0)))
    copy!(ELA, @. ELA0 + 900 * atan(0 * xc[2:end-1] + yc[2:end-1]' / ly))

    fill!(mb_mask, 1.0)

    # pack solver parameters into named tuples
    scalars  = (; ρgn, A, npow, β, b_max, dt)
    numerics = (; nx, ny, dx, dy, cfl, maxiter, ncheck, ϵtol)

    # solve for a steady state to get initial synthetic geometry
    solve_sia!((; fields, scalars, numerics); debug_vis=true)

    # save geometry and surface velocity
    H_old .= H
    V_old = copy(V)

    # step change in mass balance (ELA and β +20%)
    ELA .*= 1.2
    β *= 1.2

    # finite time step (50y)
    dt = 50 * SECONDS_IN_YEAR

    # scalars are copied by value, need to override
    scalars = merge(scalars, (; β, dt))

    # solve again
    solve_sia!((; fields, scalars, numerics); debug_vis=true)

    # transfer arrays to CPU
    H       = Array(H)
    H_old   = Array(H_old)
    B       = Array(B)
    As      = Array(As)
    V       = Array(V)
    V_old   = Array(V_old)
    ELA     = Array(ELA)
    mb_mask = Array(mb_mask)

    fields   = (; B, H, H_old, V, V_old, As, ELA, mb_mask)
    numerics = (; nx, ny, dx, dy, xc, yc, xv, yv)

    # remove existing data
    outdir = joinpath(pwd(), "datasets", "synthetic")
    ispath(outdir) && rm(outdir; recursive=true)
    mkpath(outdir)

    jldsave(joinpath(outdir, "synthetic_setup.jld2"); fields, scalars, numerics)

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

        xc_km, yc_km = xc / 1e3, yc / 1e3

        #! format: off
        hms = (B     = heatmap!(axs.B    , xc_km, yc_km, B                       ; colormap=:terrain),
               As    = heatmap!(axs.As   , xc_km, yc_km, log10.(As)              ; colormap=:roma),
               H_old = heatmap!(axs.H_old, xc_km, yc_km, H_old                   ; colormap=:turbo, colorrange=(0, 450)),
               H     = heatmap!(axs.H    , xc_km, yc_km, H                       ; colormap=:turbo, colorrange=(0, 450)),
               V_old = heatmap!(axs.V_old, xc_km, yc_km, V_old .* SECONDS_IN_YEAR; colormap=:vik  , colorrange=(0, 600)),
               V     = heatmap!(axs.V    , xc_km, yc_km, V     .* SECONDS_IN_YEAR; colormap=:vik  , colorrange=(0, 600)))
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

generate_synthetic_data(256, 256)
