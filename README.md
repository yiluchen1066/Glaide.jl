# Msc-Inverse-SIA

## synthetic case

run include("scripts_flux/snapshot_inversion_vis.jl") for snapshot inversion on synthetic glacier, which save the results to "synthetic_snapshot.jld2" 

run include("scripts_flux/main_flux.jl") for time-dependent inversion on synthetic glacier, which save the results to "synthetic_timedepedent.jld2"

run include("scripts_flux/visulization_snap_timedepedent.jl") for visualization 

run include("scripts_flux/visulization_synthetic_timedepedent.jl") for visulization of time depedent approach with different weights 

## Aletsch case 

run include("application/application.jl") for snapshot inversion on Aletsch glacier, which save the results to "snapshot_Aletsch.jld2"

run include("scripts_flux_application/main_flux_application.jl") for time dependent inversion on Aletsch glacier, which save the results to "output_TD_Aletsch/step_$iframe.jld2"

run include("scirpts_flux_application/visu_Aletsch_snap_td.jl") for visulization for Aletsch glacier.

"aletsch_data_2016_2017.nc" is needed to run "application/application.jl" and "scripts_flux_application/main_flux_application.jl", which is on my suerzack. 
