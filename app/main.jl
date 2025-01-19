# Main script to run all the simulations and produce the figures
@info "generating synthetic setup"
include("generate_synthetic_setup.jl")

@info "generating Aletsch setup"
include("generate_aletsch_setup.jl")

@info "running snapshot inversion"
include("snapshot_inversion.jl")

@info "running time-dependent inversion"
include("time_dependent_inversion.jl")

@info "running forward Aletsch model"
include("forward_aletsch.jl")

@info "running benchmark"
include("benchmark.jl")

@info "making figures"
include("make_figures.jl")
