
function downloads_data(SGI_ID)
    datadir = joinpath(@__DIR__,"Rhone_data/")
    data = fetch_glacier("Rhone", SGI_ID; datadir)
    return 
end 

downloads_data("B43-03")

# SGI_ID = "B43-03"
# datadir = joinpath(@__DIR__,"mydata/")
# data = fetch_glacier("Rhone", SGI_ID; datadir)
