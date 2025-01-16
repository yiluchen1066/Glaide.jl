#!/bin/bash

# Run PISM for 50000 years to approximate the steady state.

variables=thk,usurf

mpiexec -n 4 \
pismr \
  -bootstrap \
  -constants.ice.density 910 \
  -constants.standard_gravity 9.81 \
  -energy none \
  -flow_law.isothermal_Glen.ice_softness 7.884e-17.Pa-3.year-1 \
  -surface elevation -climatic_mass_balance -13.0,2.5,500,1800,2050 \
  -grid.Lz 200 \
  -grid.Mz 41 \
  -i input_glaide.nc \
  -output.extra.file ex.nc \
  -output.extra.times 1 \
  -output.extra.vars ${variables} \
  -output.file bedrock_glaide.nc \
  -output.timeseries.filename ts.nc \
  -output.timeseries.times 100 \
  -stress_balance.ice_free_thickness_standard 0.01 \
  -stress_balance.model sia \
  -stress_balance.sia.Glen_exponent 3 \
  -stress_balance.sia.bed_smoother.range 0 \
  -stress_balance.sia.flow_law isothermal_glen \
  -y 100 \
  > bedrock_glaide.log \
  ;
