# Glaide.jl
| code | data |
| :--- | :--- |
| [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13793630.svg)](https://doi.org/10.5281/zenodo.13793630) | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13133070.svg)](https://doi.org/10.5281/zenodo.13133070) |

**Gla**cier sl**ide** &ndash; Snapshot and time-dependent inversions of glacier basal sliding using automatic generation of adjoint code on graphics processing units.

Glaide.jl provides a collection of inversion tools to reconstruct spatially distributed glacier sliding coefficient at high-resolution on GPUs. The approach combines the adjoint state method and the powerful automatic differentiation capabilities unique to the Julia language using [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) and [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) to target Nvidia GPUs.

This Julia package is companion to the paper titled _"Snapshot and time-dependent inversions of basal sliding using automatic generation of adjoint code on graphics processing units"_ submitted for publication to the Journal of Glaciology and allows to reproduce all the figures from the paper upon running a set of interactive [Pluto.jl](https://plutojl.org) notebooks.

The Aletsch glacier data used in the scripts is available for download from the [ETH research collection](https://www.research-collection.ethz.ch) and from [Zenodo](https://doi.org/10.5281/zenodo.13133070).

## Getting started
We provide a collection of [Pluto.jl](https://plutojl.org) notebooks located in the [**app/**](./app/) directory to reproduce all the figures from the paper and run snapshot and time-dependent inversions on synthetic and real ([Aletsch glacier in the Swiss Alps](https://s.geo.admin.ch/may6x854g9bn)) geometries.

> [!WARNING]
> You need a server with a **CUDA capable GPU** in order to run Glaide.jl and perform any computational steps beyond setup generation.

1. To get started, download or clone the current repository on a GPU-hosting machine or server
```
git clone https://github.com/yiluchen1066/Glaide.jl.git
```

2. Change directory and start Julia activating the current project:
```
cd Glaide.jl

julia --project=.
```

3. From Julia, instantiate the project
```julia-repl
julia> ]

(Glaide.jl) pkg> instantiate
```

4. Activate Pluto and launch the Pluto server:
```julia-repl
julia> using Pluto

julia> Pluto.run()
```

5. After executing these commands, a Pluto web application should open in your default web-browser. From there, select the Pluto notebook to run, e.g., `app/generate_synthetic_setup.jl` and hit "open". This should launch the selected notebook.

    <img src="assets/pluto_ui.png" width=50%/>

## Reproducing the study

To run all simulations and reproduce all figures from the study, you need to run the notebooks / scripts in `app/`. This can be done by running the script `app/main.jl` **from within** the `app/` folder (`cd app` first).

Or, run the Pluto notebooks located in the [**app/**](./app/) folder in the following sequence:
1. `generate_synthetic_setup.jl`
2. `generate_aletsch_setup.jl`
3. `snapshot_inversion.jl`
4. `time_dependent_inversion.jl`
5. `forward_aletsch.jl`
6. `make_figures.jl`

> [!NOTE]
> - You will first need to execute the `generate_synthetic_setup.jl` and/or `generate_aletsch_setup.jl` notebooks in order to create the data needed to further run the inversion workflows.
> - Fetching the input files will download ~4 GB of data. Make sure to have a sufficiently good internet connection and some time ahead and grab a drink while it's processing.
> - Generating the Aletsch dataset for various spatial resolutions may consume up to 10 GB of system RAM. Make sure to have sufficient free memory.
> - The time-dependent inversions of high-resolution Aletsch may consume a significant amount of compute resources and take some time, reason why we provide a "plain" Julia script `time_dependent_inversion.jl` instead of a Pluto notebook. The time-dependent synthetic experiments can nevertheless be launched using the `time_dependent_inversion.jl` Pluto notebook if desired.
