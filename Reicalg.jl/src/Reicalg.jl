module Reicalg

export SnapshotSIA, SnapshotObjective
export SnapshotScalars, SnapshotNumerics

export TimeDependentSIA, TimeDependentObjective
export TimeDependentScalars, TimeDependentNumerics, TimeDependentAdjointNumerics

export Regularisation

export solve!, gradient_descent

using Printf
using CairoMakie
using CUDA
using Enzyme
using JLD2

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

# helper macros for calculating finite differences
include("macros.jl")

include("boundary_conditions.jl")

include("sia.jl")
include("sia_adjoint.jl")

include("regularisation.jl")

include("gradient_descent.jl")

# snapshot inverison
include("snapshot/parameters.jl")
include("snapshot/model.jl")
include("snapshot/objective.jl")

# time-dependent inversion
include("time_dependent/debug_visualisation.jl")
include("time_dependent/parameters.jl")
include("time_dependent/model.jl")
include("time_dependent/objective.jl")

end # module Reicalg
