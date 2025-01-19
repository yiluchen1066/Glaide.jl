#!/bin/bash

# spack
. /data/spack/share/spack/setup-env.sh
spack load pism@2.1.1

# cleanup
rm -f *.nc *.log *~

# create input
julia --project bedrock_glaide.jl input_glaide.nc

# run PISM
time bash -x run_glaide.sh

echo "PISM run done"

# visualise
julia --project plot_results.jl bedrock_glaide.nc figure_jl.png
