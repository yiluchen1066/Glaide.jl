module Reicalg

export ela_mass_balance
export diffusivity!, residual!, update_ice_thickness!, surface_velocity!

export solve_sia!

export avx, avy, vmag

using Printf
using CairoMakie
using CUDA

# surface mass balance model
ela_mass_balance(z, β, ela, b_max) = min(β * (z - ela), b_max)

# CUDA launch configuration heuristics
function launch_config(sz::Integer)
    nthreads = 256
    nblocks  = cld(sz, nthreads)
    return nthreads, nblocks
end

function launch_config(sz::NTuple{2,Integer})
    nthreads = 32, 8
    nblocks  = cld.(sz, nthreads)
    return nthreads, nblocks
end

# array utils
@views avx(q) = @. 0.5 * (q[1:end-1, :] + q[2:end, :])
@views avy(q) = @. 0.5 * (q[:, 1:end-1] + q[:, 2:end])

vmag(qx, qy) = sqrt.(avx(qx) .^ 2 .+ avy(qy) .^ 2)

include("macros.jl")
include("sia.jl")
include("debug_visualisation.jl")
include("forward_solver.jl")

end # module Reicalg
