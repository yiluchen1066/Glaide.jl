using GlacioTools

const SGI_IDS = Dict("Rhone" => "B43-03",
                     "Aletsch" => "B36-26",
                     "PlaineMorte" => "A55f-03",
                     "Morteratsch" => "E22-03",
                     "Arolla" => "B73-14",
                     "ArollaHaut" => "B73-12")

const DATASET_DIR = "datasets"

function save_glacier_data(glacier_name)
    datadir = joinpath(@__DIR__, DATASET_DIR, glacier_name)
    data    = fetch_glacier(glacier_name, SGI_IDS[glacier_name]; datadir)
    return data
end

save_glacier_data("Aletsch")


# Download velocity  Rabatel et al, https://doi.org/10.3390/data8040066
# at https://doi.org/10.57745/XHQ7TL
