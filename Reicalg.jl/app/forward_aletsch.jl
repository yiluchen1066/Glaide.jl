using Reicalg

const SECONDS_IN_YEAR = 3600 * 24 * 365

function forward_aletsch(filepath::AbstractString, As_init, E, dt)
    model = TimeDependentSIA(filepath)

    model.scalars.A *= E
    model.scalars.dt = dt

    @. model.fields.As = As_init

    solve!(model; debug_vis=true, report=true)

    return
end

forward_aletsch("datasets/aletsch_setup.jld2", 1e-22, 0.25, 1.0 * SECONDS_IN_YEAR)
