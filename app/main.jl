# Main script to run all the simulations and produce the figures
include("generate_synthetic_setup.jl")
include("generate_aletsch_setup.jl")
include("snapshot_inversion.jl")
include("time_dependent_inversion.jl")
include("forward_aletsch.jl")
include("make_figures.jl")
