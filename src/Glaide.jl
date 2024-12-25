module Glaide

# main solver function
export solve!

# snapshot model
export SnapshotSIA, SnapshotObjective
export SnapshotScalars, SnapshotNumerics

# time-dependent model
export TimeDependentSIA, TimeDependentObjective
export TimeDependentScalars, TimeDependentNumerics, TimeDependentAdjointNumerics

# optimisation
export BacktrackingLineSearch, OptimisationOptions, OptmisationState, OptimisationResult
export optimise

# utils
export lerp
export download_raster
export coords_as_ranges
export av1, av4
export lsq_fit
export laplacian_smoothing, laplacian_smoothing!
export remove_components, remove_components!

# constants
export SECONDS_IN_YEAR, GLEN_A, GLEN_N, RHOGN

using LinearAlgebra
using Printf
using CairoMakie
using CUDA
using Enzyme
using StaticArrays
using JLD2
using ZipArchives
using Downloads
using Rasters
using ImageMorphology
using LLVM

# define consntants
const SECONDS_IN_YEAR = 3600 * 24 * 365
const GLEN_N          = 3
const RHOGN           = (910 * 9.81)^GLEN_N
const RHOGNA          = RHOGN * 2.5e-24

# surface mass balance model
@inline ela_mass_balance(z, b, ela, mb_max) = min(b * (z - ela), mb_max)

# CUDA launch configuration heuristics
function launch_config(sz::NTuple{2,Integer}, ty=4)
    nthreads = 32, ty
    nblocks  = cld.(sz, nthreads)
    return nthreads, nblocks
end

# Enzyme utils
∇(fun, args...) = (Enzyme.autodiff_deferred(Enzyme.Reverse, Const(fun), Const, args...); return)
const DupNN = DuplicatedNoNeed

# helper functions to simplify the data processing
include("utils.jl")

# helper macros for calculating finite differences
include("macros.jl")

# finite difference operators
include("finite_difference.jl")

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

end # module Glaide
