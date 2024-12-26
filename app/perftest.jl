using Glaide, BenchmarkTools, CUDA, Printf

function runme(resolution)
    Lx, Ly = 20e3, 20e3

    # ice flow parameters
    ρgnAs_0 = RHOGN * 1e-22

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
    numerics = TimeDependentNumerics(xc, yc; dmpswitch=1nx)

    model = TimeDependentSIA(scalars, numerics; report=true, debug_vis=true)

    B_h = @. B_0 + 0.5 * B_a * (exp(-(xc / W_1)^2 - (yc' / W_2)^2) +
                                exp(-(xc / W_2)^2 - (yc' / W_1)^2))

    # set the bed elevation
    copy!(model.fields.B, B_h)

    # set As to the background value
    fill!(model.fields.ρgnAs, ρgnAs_0)

    # accumulation allowed everywhere
    fill!(model.fields.mb_mask, 1.0)

    (; r, z, B, H, H_old, ρgnAs, mb_mask) = model.fields
    (; ρgnA, n, b, mb_max, ela, dt)       = model.scalars

    ρgnA2_n2 = 2 / (n + 2) * ρgnA
    _n3      = inv(n + 3)
    _n2      = inv(n + 2)
    _dt      = inv(dt)
    _dx      = inv(dx)
    _dy      = inv(dy)
    nm1      = n - oneunit(n)

    mode = Glaide.ComputeResidual()
    # mode = Glaide.ComputePreconditionedResidual()
    # mode = Glaide.ComputePreconditioner()

    fun = @cuda launch = false Glaide._residual!(r, z, B, H, H_old, ρgnAs, mb_mask, ρgnA2_n2, b, mb_max, ela, _dt, _n3, _n2, _dx, _dy, n, nm1, mode)

    @show CUDA.registers(fun)

    tt = @belapsed begin
        Glaide.residual!($r, $z, $B, $H, $H_old, $ρgnAs, $mb_mask, $ρgnA, $n, $b, $mb_max, $ela, $dt, $dx, $dy, $mode)
        CUDA.synchronize()
    end

    Aeff = nx * ny * (6 * sizeof(Float64)) / 1e9
    Teff = Aeff / tt

    @printf("   resolution: %.0f m, time: %g s, effective memory bandwidth: %.3f GB/s\n", resolution, tt, Teff)

    return
end

runme(5.0)
