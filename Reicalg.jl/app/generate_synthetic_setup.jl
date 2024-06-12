using Reicalg
using CUDA
using CairoMakie

const SECONDS_IN_YEAR = 3600 * 24 * 365

@views av1(a) = @. 0.5 * (a[1:end-1] + a[2:end])

function generate_synthetic_data(nx, ny)
    # set CUDA device
    CUDA.device!(1)

    # ice flow parameters
    npow = 3                 # glen's power law exponent
    A    = 1.9e-24           # ice flow parameter
    As0  = 5.7e-20           # sliding flow parameter
    ρgn  = (910 * 9.81)^npow # gravity (pre-exponentiated)

    # synthetic geometry parameters
    lx, ly = 220e3, 200e3
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
    B       = CUDA.zeros(Float64, nx, ny)
    H       = CUDA.zeros(Float64, nx, ny)
    H_old   = CUDA.zeros(Float64, nx, ny)
    qx      = CUDA.zeros(Float64, nx - 1, ny - 2)
    qy      = CUDA.zeros(Float64, nx - 2, ny - 1)
    D       = CUDA.zeros(Float64, nx - 1, ny - 1)
    As      = CUDA.zeros(Float64, nx - 1, ny - 1)
    mb_mask = CUDA.zeros(Float64, nx - 2, ny - 2)
    r_H     = CUDA.zeros(Float64, nx - 2, ny - 2)
    d_H     = CUDA.zeros(Float64, nx - 2, ny - 2)
    dH_dτ   = CUDA.zeros(Float64, nx - 2, ny - 2)
    ELA     = CUDA.zeros(Float64, nx - 2, ny - 2)

    # initialise

    # two bumps as per (Visnjevic et al., 2018)
    copy!(B, @. B0 * (exp(-xc^2 / 1e10 - yc'^2 / 1e9) +
                      exp(-xc^2 / 1e9 - (yc' - ly / 8)^2 / 1e10)))

    copy!(As, @. exp(log(As0) * (0.1 * cos(5π * xv / lx) * sin(5π * yv' / ly) + 1.0)))
    copy!(ELA, @. ELA0 + 900 * atan(0 * xc[2:end-1] + yc[2:end-1]' / ly))

    fill!(mb_mask, 1.0)

    # figures

    fields       = (; B, H, H_old, qx, qy, D, As, r_H, d_H, dH_dτ)
    scalars      = (; ρgn, A, npow, dt)
    mass_balance = (; β, ELA, b_max, mb_mask)
    numerics     = (; nx, ny, dx, dy, cfl, maxiter, ncheck, ϵtol)

    # solve for a steady state to get initial synthetic geometry
    solve_sia!(fields, scalars, mass_balance, numerics; debug_vis=false)

    # save geometry
    H_old = copy(H)

    # step change in mass balance (ELA and β +20%)
    mass_balance = merge(mass_balance, (; ELA=1.2 .* ELA, β=1.2 * β))

    # start from current geometry
    fields = merge(fields, (; H_old))

    # finite time step (50y)
    scalars = merge(scalars, (; dt=50 * SECONDS_IN_YEAR))

    # solve again
    solve_sia!(fields, scalars, mass_balance, numerics; debug_vis=false)

    # transfer arrays to CPU
    H     = Array(H)
    H_old = Array(H_old)
    B     = Array(B)
    As    = Array(As)

    # generate ice mask and mask all data
    #! format: off
    ice_mask = Array(H[1:end-1, 1:end-1] .== 0 .||
                     H[2:end  , 1:end-1] .== 0 .||
                     H[1:end-1, 2:end  ] .== 0 .||
                     H[2:end  , 2:end  ] .== 0)
    #! format: on

    As[ice_mask] .= NaN

    fig = Figure(; size=(600, 450), fontsize=12)

    axs = (B     = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="bedrock"),
           H_old = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="H_old"),
           H     = Axis(fig[2, 2][1, 1]; aspect=DataAspect(), title="H"),
           As    = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="sliding parameter (log10)"))

    xc_km, yc_km = xc / 1e3, yc / 1e3

    #! format: off
    hms = (B     = heatmap!(axs.B    , xc_km, yc_km, B         ; colormap=:terrain),
           H_old = heatmap!(axs.H_old, xc_km, yc_km, H_old     ; colormap=:turbo, colorrange=(0, 450)),
           H     = heatmap!(axs.H    , xc_km, yc_km, H         ; colormap=:turbo, colorrange=(0, 450)),
           As    = heatmap!(axs.As   , xc_km, yc_km, log10.(As); colormap=:turbo))
    #! format: on

    cbs = (B     = Colorbar(fig[1, 1][1, 2], hms.B),
           H_old = Colorbar(fig[1, 2][1, 2], hms.H_old),
           H     = Colorbar(fig[2, 2][1, 2], hms.H),
           As    = Colorbar(fig[2, 1][1, 2], hms.As))

    display(fig)

    return
end

generate_synthetic_data(256, 256)
