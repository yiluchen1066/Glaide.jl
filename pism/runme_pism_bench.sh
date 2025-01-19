#!/bin/bash

# spack
. /data/spack/share/spack/setup-env.sh # change this to your local spack install activation path
spack load pism@2.1.1

# cleanup
rm -f *.nc *.log *~

# create input
julia --project generate_synthetic_setup.jl input_pism.nc

# run PISM
time bash -x run_pism.sh

echo "PISM run done"

# visualise
julia --project plot_results.jl pism_output.nc figure_jl.png
