# Performance benchmarking against PISM
This directory contains scripts to benchmark performance of the SIA solver against [PISM](http://www.pism.io/) using the synthetic topography configuration.

## PISM installation and run
Install PISM using [Spack](https://spack.readthedocs.io/en/latest/getting_started.html#installation). Upon following the [Spack install instruction](https://www.pism.io/docs/installation/spack.html), install PISM by
```
spack install pism@2.1.1 ^petsc~metis~hdf5~hypre~superlu-dist
```

To run PISM benchmarks, Julia is needed to produce the NETCDF input file and visualise the output.

## Model configuration and output
Running PISM's SIA solver without sliding for 100 years using a bilinear surface mass balance over a synthetic topography



## Benchmark results
Running PISM on an AMD EPYC 7282 16-Core Processor using 16 MPI ranks on 16 cores. Timing is performed using bash's built-in `time` functionality.

| Resolution \[m\] | wall-time \[m-s\] |
| ---------------- | ----------------: |
| 125              |         0m26.80s  |
| 100              |         0m59.62s  |
| 50               |        14m53.89s  |
| 25               |       237m54.07s  |
