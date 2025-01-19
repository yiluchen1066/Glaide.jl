#!/bin/bash

# spack
. /data/spack/share/spack/setup-env.sh # change this to your local spack install activation path
spack load pism@2.1.1

# cleanup
rm -f *.png *.nc *.log *~

# create input
julia --project -e "using Pkg; Pkg.instantiate()"
julia --project generate_synthetic_setup.jl pism_input.nc

# run PISM
time bash -x run_pism.sh

echo "PISM run done"

# visualise
julia --project visualise_results.jl pism_output.nc pism_out.png
