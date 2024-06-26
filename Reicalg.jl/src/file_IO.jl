using JLD2

function load_from_file(path)
    scalars, numerics = load(path, "scalars", "numerics")
    fields = SIA_fields(numerics.nx, numerics.ny)

    host_fields = load(path, "fields")

    for name in intersect(keys(fields), keys(host_fields))
        copy!(fields[name], host_fields[name])
    end

    return fields, scalars, numerics
end
