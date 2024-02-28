using CSV 
using DataFrames
using CairoMakie
using GLM

#this function could make the plots, and also return b_max_Alet (m/a), ELA_Alet (m), β_Alet (m/a)
@views function load_massbalance()

    df = open("/scratch-1/yilchen/Msc-Inverse-SIA/application/datasets/Aletsch/aletsch_fix.dat") do io 
        h1 = readline(io)
        h2 = readline(io)
        h3 = readline(io)
        h4 = readline(io)
    
        CSV.File(io; delim=" ", ignorerepeated=true, header=false) |> DataFrame
    end 
    
    df2019 = df[106,:] |> Vector
    z = LinRange(df2019[11], df2019[12], 26)
    ELA = df2019[8]
    mb2019 = df2019[13+26:13+26*2-1]* 1e-3 * (1000/910)
    b_max  = maximum(filter(!isnan, mb2019))
    mb_min = minimum(filter(!isnan, mb2019))
    
    index_min = findfirst(isequal(mb_min), mb2019)
    index_max = argmin(abs.(filter(!isnan, mb2019)))
    
    z_lg = z[index_min:index_max]
    mb2019_lg = mb2019[index_min:index_max]
    #fit a linear regression model
    model = lm(@formula(y ~ x), DataFrame(x=z_lg, y=mb2019_lg))
    coefficients = coef(model)
    intercept = coefficients[1]
    slope = coefficients[2]
    
    mb_predict = zeros(length(z_lg)) |> Vector
    mb_predict .= slope.* z_lg .+ intercept
    L"x\text{ [km]}"
    fig = Figure()
    # how do I text the title
    ax = Axis(fig[1,1]; xlabel=L"Elevation \text{ [m]}", ylabel=L"$Annual$ $mass$ $balance$ $[\text{m}\cdot \text{a}^{-1}]$")
    
    mb_plots = scatterlines!(ax, z, mb2019; label=L"$real$ $mass$ $balance$", linewidth=3)
    scatterlines!(ax, z_lg, mb_predict; color=:red, label=L"$modeled$ $mass$ $balance$", linewidth=3)
    vlines!(ax, df2019[8];color=:gray, linestyle=:dash, label=L"$ELA$", linewidth=3)
    hlines!(ax, b_max; color=:red, linestyle=:dash, label=L"$b_{max}$", linewidth=3)

    axislegend(ax; position=:rb, labelsize=16)
    display(fig)

    b_max_Alet = b_max / (365*24*3600)
    ELA_Alet = ELA 
    β_Alet = slope / (365*24*3600)

    return b_max_Alet, ELA_Alet, β_Alet
end
