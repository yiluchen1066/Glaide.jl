# Msc-Inverse-SIA

## Synthetic Case

To manage tasks and files, perform the following runs:

- **Snapshot Inversion:**
  - Run `include("scripts_flux/snapshot_inversion_vis.jl")` to perform snapshot inversion on a synthetic glacier. This will save the results to `"synthetic_snapshot.jld2"`.

- **Time-Dependent Inversion:**
  - Run `include("scripts_flux/main_flux.jl")` for time-dependent inversion on a synthetic glacier. Results are saved to `"synthetic_timedependent.jld2"`.

- **Visualization:**
  - Run `include("scripts_flux/visualization_snap_timedependent.jl")` for visualization of snapshot and time-dependent results.
  - Run `include("scripts_flux/visualization_synthetic_timedependent.jl")` to visualize the time-dependent approach with different weights.

## Aletsch Case

For operations related to the Aletsch glacier:

- **Snapshot Inversion:**
  - Run `include("application/application.jl")` for snapshot inversion on the Aletsch glacier. This saves the results to `"snapshot_Aletsch.jld2"`.

- **Time-Dependent Inversion:**
  - Run `include("scripts_flux_application/main_flux_application.jl")` for time-dependent inversion on the Aletsch glacier. The results are saved incrementally to `"output_TD_Aletsch/step_$iframe.jld2"`.

- **Visualization:**
  - Run `include("scripts_flux_application/visu_Aletsch_snap_td.jl")` for visualization of both snapshot and time-dependent results for Aletsch glacier.

- **Data Requirement:**
  - The file `"aletsch_data_2016_2017.nc"` is required for running `"application/application.jl"` and `"scripts_flux_application/main_flux_application.jl"`, available on my suerzack.

## Visualization on MS

For generating figures, execute the following scripts:

- **Figure 1:**
  - Run `include("scripts_flux_application/generated_synthetic_data.jl")`.

- **Figure 4:**
  - Run `include("scripts_flux/visualization_snap_timedependent.jl")`.

- **Figure 5:**
  - Run `include("scripts_flux/visualization_on_synthetic_timedependent.jl")`.

- **Figure 6:**
  - Run `include("scripts_flux/synthetic_convergence.jl")`.

- **Figure 7:**
  - Run `include("scripts_flux_application/visu_Aletsch_snap_td.jl")`.
