using Glaide, CairoMakie, BenchmarkTools, CUDA, Printf

function run(resolution)
    Lx, Ly = 20e3, 20e3

    # ice flow parameters
    As_0 = 1e-22

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
    nx = cld(nx, 32) * 32
    ny = cld(ny, 32) * 32

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

    model = TimeDependentSIA(scalars, numerics; report=true, debug_vis=true)

    B_h = @. B_0 + 0.5 * B_a * (exp(-(xc / W_1)^2 - (yc' / W_2)^2) +
                                exp(-(xc / W_2)^2 - (yc' / W_1)^2))

    # set the bed elevation
    copy!(model.fields.B, B_h)

    # set As to the background value
    fill!(model.fields.As, As_0)
    
    # accumulation allowed everywhere
    fill!(model.fields.mb_mask, 1.0)

    (; r, z, B, H, H_old, As, mb_mask) = model.fields
    (; ρgn, A, n, b, mb_max, ela, dt) = model.scalars
    
    # @time Glaide.residual!(r, z, B, H, H_old, A, As, ρgn, n, b, ela, mb_max, mb_mask, dt, dx, dy, Glaide.ComputeResidual())
    # @time Glaide.residual2!(r, z, B, H, H_old, A, As, ρgn, n, b, ela, mb_max, mb_mask, dt, dx, dy, Glaide.ComputeResidual())

    tt = @belapsed begin
        Glaide.residual!($r,
                         $z,
                         $B,
                         $H,
                         $H_old,
                         $A,
                         $As,
                         $ρgn,
                         $n,
                         $b,
                         $ela,
                         $mb_max,
                         $mb_mask,
                         $dt,
                         $dx,
                         $dy,
                         Glaide.ComputeResidual())
        CUDA.synchronize()
    end

    Aeff = nx * ny * 6 * sizeof(Float64) / 1e9
    Teff = Aeff / tt

    @printf("   resolution: %.0f m, time: %g s, effective memory bandwidth: %.3f GB/s\n", resolution, tt, Teff)

    return
end

run(10.0)
