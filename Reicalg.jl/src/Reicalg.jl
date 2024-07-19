module Reicalg

export SnapshotSIA, SnapshotObjective
export SnapshotScalars, SnapshotNumerics

export TimeDependentSIA, TimeDependentObjective
export TimeDependentScalars, TimeDependentNumerics, TimeDependentAdjointNumerics

export Regularisation

export BacktrackingLineSearch, OptimisationOptions, OptmisationState
export solve!, optimise

export SECONDS_IN_YEAR

using LinearAlgebra
using Printf
using CairoMakie
using CUDA
using Enzyme
using JLD2

# define consntants
const SECONDS_IN_YEAR = 3600 * 24 * 365

# surface mass balance model
ela_mass_balance(z, β, ela, b_max) = min(β * (z - ela), b_max)

# CUDA launch configuration heuristics in 1D
function launch_config(sz::Integer)
    nthreads = 256
    nblocks  = cld(sz, nthreads)
    return nthreads, nblocks
end

# CUDA launch configuration heuristics in 2D
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

# homogeneous and non-homogeneous Neumann boundary conditions
include("boundary_conditions.jl")

# SIA model and adjoint calls
include("sia.jl")
include("sia_adjoint.jl")

# gradient-based optimisation
include("optimise.jl")

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
