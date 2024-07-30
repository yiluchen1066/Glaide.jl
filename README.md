# Glaide.jl
**Gla**cier sl**ide** - Snapshot and time-dependent inversions of glacier basal sliding using automatic generation of adjoint code on graphics processing units

Glaide.jl provides a collection of inversion tools to reconstruct spatially distributed glacier sliding coefficient at high-resolution on GPUs. The approach combines the adjoint state method, the pseudo-transient method and leverages the powerful automatic differentiation capabilities unique to the Julia language using [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) and [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) to target Nvidia GPUs.

This Julia package is companion to the paper titled _"Snapshot and time-dependent inversions of basal sliding using automatic generation of adjoint code on graphics processing units"_ submitted for publication to the Journal of Glaciology and allows to reproduce all the figures from the paper upon running a set of interactive [Pluto.jl](https://plutojl.org) notebooks.

The Aletsch glacier data used in the scripts is available for download from the [ETH research collection](https://www.research-collection.ethz.ch) and from Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13133070.svg)](https://doi.org/10.5281/zenodo.13133070).

## Getting started
We provide a collection of [Pluto.jl](https://plutojl.org) notebooks located in the [*app/*](./app/) directory to reproduce most of the figure from the paper and run snapshot and time-dependent inversions on synthetic and Aletsch glacier (Swiss Alps) geometries.

> [!WARNING]
> You need a server with a CUDA capable GPU in order to run Glaide.jl and perform the following any computational steps beyond setup generation.

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

5. ideally, Pluto fires-up a server on `localhost` that should open on your default web-browser. From there, select the Pluto notebook to run, e.g., `app/generate_synthetic_setup.jl` and hit "run". This should launch the selected notebook.

> [!NOTE]
> You will first need to execute the `generate_synthetic_setup.jl` and/or `generate_aletsch_setup.jl` notebooks in order to create the data needed to further run the inversion workflows.
