# Performance Benchmarking Against PISM
This directory contains scripts to benchmark the performance of the SIA solver against [PISM](http://www.pism.io/) using a synthetic topography configuration.

## PISM Installation and Execution

### Installing PISM
PISM can be installed using [Spack](https://spack.readthedocs.io/en/latest/getting_started.html#installation). Follow the [PISM installation guide](https://www.pism.io/docs/installation/spack.html) and install PISM with the following command:

```bash
spack install pism@2.1.1 ^petsc~metis~hdf5~hypre~superlu-dist
```

### Running PISM Benchmarks
To run the benchmarks, you will need Julia to generate the NetCDF input file and to visualize the output.

Run the following command to benchmark PISM for 100 years at a 100 m grid cell resolution using 4 MPI processes:

```bash
bash runme_pism_bench.sh
```

This script will execute the benchmark, which typically takes a few minutes, and will produce a PNG visualization named `pism_out.png`.

## Benchmark Results
The benchmarks were conducted on an AMD EPYC 7282 16-Core Processor using 16 MPI ranks on 16 cores. Wall-time measurements were obtained using the built-in `time` functionality in bash.

| Resolution (m) | Wall-Time (m:s)  |
|----------------|-----------------:|
| 125            |        0:26.80   |
| 100            |        0:59.62   |
| 50             |       14:53.89   |
| 25             |      237:54.07   |

---

For further details or issues, please refer to the [PISM documentation](http://www.pism.io/docs.html).
