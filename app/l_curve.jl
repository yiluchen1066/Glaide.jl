using JLD2, CairoMakie, Printf

function lcurve()
    resolution = 50
    setup      = "aletsch"
    γ_rng      = 10 .^ LinRange(-9, -7, 20)

    Js_reg = Float64[]
    Js_sol = Float64[]

    irng = 9:20
    iγ   = 7

    @show γ_rng[irng]

    for γ in γ_rng[irng]
        γ = round(γ; sigdigits=2)
        local outdir = "../output/time_dependent_$(setup)_$(resolution)m_reg_$(γ)"

        istep = 100
        fname = @sprintf("step_%04d.jld2", istep)

        (X, j_hist) = load(joinpath(outdir, fname), "X", "j_hist")

        J_reg = sum(@. ((X[2:end, :] - X[1:end-1, :]))^2) +
                sum(@. ((X[:, 2:end] - X[:, 1:end-1]))^2)

        J_min = j_hist[end][2]
        J_sol = J_min - 0.5 * γ * J_reg

        push!(Js_reg, J_reg / sum(X .^ 2))
        push!(Js_sol, J_sol)
    end

    Js_sol ./= 1e-4
    Js_reg ./= 1e-3

    fig = Figure(; size=(400, 300))
    ax  = Axis(fig[1, 1]; xlabel="R", ylabel="J")

    Label(fig[1, 1, Top()], L"\times 10^{-4}"; halign=:left)
    Label(fig[1, 1, Bottom()], L"\times 10^{-3}"; halign=:right, valign=:bottom)
    lines!(ax, Js_reg, Js_sol; color=:gray, linestyle=:dash)
    scatter!(ax, Js_reg, Js_sol; color=log10.(γ_rng[irng]), colormap=:blues, marker=:circle, markersize=10, strokecolor=:black, strokewidth=1)
    scatter!(ax, Js_reg[iγ], Js_sol[iγ]; marker=:circle, strokewidth=1.5, strokecolor=:red, markersize=10, color=:transparent)
    # foreach(i -> text!(ax, position=(Js_reg[i], Js_sol[i]), γ_rng[irng[i]]), 1:3)

    text!(ax, L"\gamma = 3 \times 10^{-8}"; position=(Js_reg[iγ] + 0.1, Js_sol[iγ] + 0.01), align=(:left, :bottom))

    @show γ_rng[irng[iγ]]

    display(fig)
    return
end

lcurve()
