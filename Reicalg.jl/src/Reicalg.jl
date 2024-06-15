module Reicalg

export ela_mass_balance
export diffusivity!, residual!, update_ice_thickness!, surface_velocity!

export SIA_fields, SIA_adjoint_fields
export solve_sia!, adjoint_sia!

export objective_time_dependent!, grad_objective_time_dependent!

using Printf
using CairoMakie
using CUDA
using Enzyme

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

# Enzyme utils
@inline ∇(fun, args...) = (Enzyme.autodiff_deferred(Enzyme.Reverse, fun, Const, args...); return)
const DupNN = DuplicatedNoNeed

include("macros.jl")
include("fields.jl")
include("sia.jl")
include("debug_visualisation.jl")
include("forward_solver.jl")

include("sia_adjoint.jl")
include("adjoint_debug_visualisation.jl")
include("adjoint_solver.jl")

include("objective.jl")
include("regularisation.jl")

end # module Reicalg
