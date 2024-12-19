struct SnapshotSIA{FF<:SnapshotFields,
                   SS<:SnapshotScalars,
                   SN<:SnapshotNumerics,
                   AF<:SnapshotAdjointFields}
    fields::FF
    scalars::SS
    numerics::SN
    adjoint_fields::AF
end

function SnapshotSIA(scalars, numerics)
    fields         = SnapshotFields(numerics.nx, numerics.ny)
    adjoint_fields = SnapshotAdjointFields(numerics.nx, numerics.ny)

    return SnapshotSIA(fields, scalars, numerics, adjoint_fields)
end

function SnapshotSIA(path::AbstractString)
    data = load(path)

    dfields = data["fields"]

    (; nx, ny, xc, yc)    = data["numerics"]
    (; lx, ly, n, A, ρgn) = data["scalars"]

    fields   = SnapshotFields(nx, ny)
    scalars  = SnapshotScalars(lx, ly, n, A, ρgn)
    numerics = SnapshotNumerics(xc, yc)

    adjoint_fields = SnapshotAdjointFields(nx, ny)

    copy!(fields.H, dfields.H)
    copy!(fields.B, dfields.B)
    copy!(fields.V, dfields.V)

    return SnapshotSIA(fields, scalars, numerics, adjoint_fields)
end

function solve!(model::SnapshotSIA; kwargs...)
    (; H, B, As, V) = model.fields
    (; A, ρgn, n)   = model.scalars
    (; dx, dy)      = model.numerics

    surface_velocity!(V, H, B, As, A, ρgn, n, dx, dy)

    return
end

function solve_adjoint!(Ās, model::SnapshotSIA; kwargs...)
    (; H, B, As, V) = model.fields
    (; A, ρgn, n)   = model.scalars
    (; dx, dy)      = model.numerics
    (; V̄)          = model.adjoint_fields

    ∇surface_velocity!(DupNN(V, V̄), Const(H),
                       Const(B), DupNN(As, Ās),
                       A, ρgn, n, dx, dy)

    return
end
