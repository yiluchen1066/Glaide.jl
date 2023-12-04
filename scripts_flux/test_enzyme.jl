using Enzyme
using GLMakie

using LinearAlgebra

# Enzyme shortcuts
const DupNN = DuplicatedNoNeed
const Dup   = Duplicated

# macros
#! format: off
# averages
macro av_xy(A)
    esc(:(0.25 * ($A[ix, iy    ] + $A[ix + 1, iy    ] +
                  $A[ix, iy + 1] + $A[ix + 1, iy + 1])))
end
macro av_xa(A) esc(:(0.5 * ($A[ix, iy] + $A[ix + 1, iy]))) end
macro av_ya(A) esc(:(0.5 * ($A[ix, iy] + $A[ix, iy + 1]))) end
# derivatives
macro d_xa(A) esc(:($A[ix + 1, iy] - $A[ix, iy])) end
macro d_ya(A) esc(:($A[ix, iy + 1] - $A[ix, iy])) end
macro d_xi(A) esc(:($A[ix + 1, iy + 1] - $A[ix, iy + 1])) end
macro d_yi(A) esc(:($A[ix + 1, iy + 1] - $A[ix + 1, iy])) end
#! format: on

@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])

# diffusion coefficient
function compute_D!(D, H, B, As, A, n, dx, dy)
    for iy in axes(D, 2)
        for ix in axes(D, 1)
            ∇Sx = 0.5 *
                  (@d_xi(B) / dx + @d_xi(H) / dx +
                   @d_xa(B) / dx + @d_xa(H) / dx)
            ∇Sy = 0.5 *
                  (@d_yi(B) / dy + @d_yi(H) / dy +
                   @d_ya(B) / dy + @d_ya(H) / dy)
            ∇S = sqrt(∇Sx^2 + ∇Sy^2)
            D[ix, iy] = (A * @av_xy(H)^(n + 2) + As[ix, iy] * @av_xy(H)^n) * ∇S^(n - 1)
        end
    end
    return
end

# ice flow flux
function compute_q!(qx, qy, H, B, D, dx, dy)
    nx, ny = size(H)
    for iy in 1:(ny-1)
        for ix in 1:(nx-1)
            if ix <= nx - 1 && iy <= ny - 2
                qx[ix, iy] = -@av_ya(D) * (@d_xi(H) + @d_xi(B)) / dx
            end
            if ix <= nx - 2 && iy <= ny - 1
                qy[ix, iy] = -@av_xa(D) * (@d_yi(H) + @d_yi(B)) / dy
            end
        end
    end
    return
end

function compute_qmag!(qmag, qx, qy, H)
    nx, ny = size(H)
    for iy in 1:(ny-2)
        for ix in 1:(nx-2)
            qmag[ix, iy] = sqrt(@av_xa(qx)^2 + @av_ya(qy)^2)
        end
    end
    return
end

function model!(qx, qy, qmag, H, B, D, As, A, n, dx, dy)
    compute_D!(D, H, B, As, A, n, dx, dy)
    compute_q!(qx, qy, H, B, D, dx, dy)
    compute_qmag!(qmag, qx, qy, H)
    return
end

function ∂J_∂qx_vec!(q̄x, qmag, q_obs_mag, qx)
    q̄x                .= 0
    @. q̄x[1:end-1, :] += (qmag - q_obs_mag) * $avx(qx) / (2 * qmag)
    @. q̄x[2:end, :]   += (qmag - q_obs_mag) * $avx(qx) / (2 * qmag)
    return
end

function ∂J_∂qy_vec!(q̄y, qmag, q_obs_mag, qy)
    q̄y                .= 0
    @. q̄y[:, 1:end-1] += (qmag - q_obs_mag) * $avy(qy) / (2 * qmag)
    @. q̄y[:, 2:end]   += (qmag - q_obs_mag) * $avy(qy) / (2 * qmag)
    return
end

# loss function (L2-norm squared of difference between observed and modelled flux magnitude)
function loss(qx, qy, qmag, qmag_o, D, H, B, As, A, n, dx, dy)
    compute_D!(D, H, B, As, A, n, dx, dy)
    compute_q!(qx, qy, H, B, D, dx, dy)
    # J = ∑(|q| - |q_o|)^2
    J = 0.0
    for iy in axes(qmag, 2)
        for ix in axes(qmag, 1)
            qmag[ix, iy] = sqrt(@av_xa(qx)^2 + @av_ya(qy)^2)
            J += 0.5 * (qmag[ix, iy] - qmag_o[ix, iy])^2
        end
    end
    return J
end

function grad_loss_semi!(Ās, q̄x, q̄y, D̄, qx, qy, qmag, qmag_o, H, B, D, As, A, n, dx, dy)
    q̄x .= 0
    q̄y .= 0
    D̄  .= 0

    model!(qx, qy, qmag, H, B, D, As, A, n, dx, dy)
    ∂J_∂qx_vec!(q̄x, qmag, qmag_o, qx)
    ∂J_∂qy_vec!(q̄y, qmag, qmag_o, qy)

    #compute_q!(qx, qy, H, B, D, dx, dy)
    Enzyme.autodiff(Enzyme.Reverse, compute_q!,
                    DupNN(qx, q̄x),
                    DupNN(qy, q̄y),
                    Const(H),
                    Const(B),
                    DupNN(D, D̄),
                    Const(dx), Const(dy))

    #compute_D!(D, H, B, As, A, n, dx, dy)
    Enzyme.autodiff(Enzyme.Reverse, compute_D!,
                    DupNN(D, D̄),
                    Const(H), Const(B),
                    DupNN(As, Ās),
                    Const(A), Const(n), Const(dx), Const(dy))

    return
end

function grad_loss_full!(Ās, q̄x, q̄y, q̄mag, D̄, qx, qy, qmag, qmag_o, D, H, B, As, A, n, dx, dy)
    loss(qx, qy, qmag, qmag_o, D, H, B, As, A, n, dx, dy)
    q̄x   .= 0
    q̄y   .= 0
    q̄mag .= 0
    D̄    .= 0
    Enzyme.autodiff(Enzyme.Reverse, loss, Active,
                    DupNN(qx, q̄x),
                    DupNN(qy, q̄y),
                    DupNN(qmag, q̄mag),
                    Const(qmag_o),
                    DupNN(D, D̄),
                    Const(H),
                    Const(B),
                    DupNN(As, Ās),
                    Const(A),
                    Const(n), Const(dx), Const(dy))
    return
end

@views function main()
    # physics
    lx, ly = 2.0, 1.0
    As_s0  = 1e0    # synthetic As
    As_i0  = 1e-2   # initial As
    A      = 1.0    # ice flow parameter
    n      = 3      # power law exponent
    # numerics
    nx, ny = 200, 100
    # preprocessing
    dx, dy = lx / nx, ly / ny
    xc, yc = range(-lx / 2, lx / 2, nx), range(-ly / 2, ly / 2, ny)
    xv, yv = 0.5 .* (xc[1:end-1] .+ xc[2:end]), 0.5 .* (yc[1:end-1] .+ yc[2:end])
    # fields
    H    = zeros(nx, ny)
    B    = zeros(nx, ny)
    qx   = zeros(nx - 1, ny - 2)
    qy   = zeros(nx - 2, ny - 1)
    D    = zeros(nx - 1, ny - 1)
    qmag = zeros(nx - 2, ny - 2)
    As_s = zeros(nx - 1, ny - 1)
    As_i = zeros(nx - 1, ny - 1)
    # init
    B    .= 0.1 .* xc .+ 0.2 .* yc' .+ 0.05 .* sin.(4π .* xc) .* cos.(3π .* yc')
    H    .= 0.3 .* exp.(-(xc ./ 0.5) .^ 2 .- (yc' ./ 0.3) .^ 2)
    As_s .= As_s0
    As_i .= As_i0
    As   = copy(As_i)
    # synthetic solve with As_s
    compute_D!(D, H, B, As_s, A, n, dx, dy)
    compute_q!(qx, qy, H, B, D, dx, dy)
    qmag_o = sqrt.(0.5 .* (qx[1:end-1, :] .+ qx[2:end, :]) .^ 2 .+
                   0.5 .* (qy[:, 1:end-1] .+ qy[:, 2:end]) .^ 2)
    # forward model run with As_i to compute initial loss
    J0 = loss(qx, qy, qmag, qmag_o, D, H, B, As_i, A, n, dx, dy)
    # adjoint sensitivities
    q̄x      = zeros(size(qx))
    q̄y      = zeros(size(qy))
    Ās_full = zeros(size(As_i))
    Ās_semi = zeros(size(As_i))
    D̄       = zeros(size(D))
    q̄mag    = zeros(size(qmag))
    # convergence progess
    iters = Int[]
    J_J0  = Float64[]

    # visualisation
    # fig = Figure(; size=(1000, 400))

    # ax = (S=Axis3(fig[1, 1]; aspect=:data, title="Surface"),
    #       A=Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="log10(As)"),
    #       Ā=Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="Sensitivity"),
    #       A_sl=Axis(fig[2, 2]; title="A_s at y = 0", yscale=log10),
    #       J=Axis(fig[1, 3]; title="Convergence (J/J0)", yscale=log10),
    #       Δq=Axis(fig[2, 3][1, 1]; aspect=DataAspect(), title="|q|-|q_obs|"))

    # plt = (S=surface!(ax.S, xc, yc, B .+ H; colormap=:turbo),
    #        Ā=heatmap!(ax.Ā, xv, yv, Ās; colormap=:turbo),
    #        A=heatmap!(ax.A, xv, yv, log10.(As_i); colormap=:turbo),
    #        A_sl=(lines!(ax.A_sl, xv, As_i[:, ny ÷ 2]),
    #              lines!(ax.A_sl, xv, As_s[:, ny ÷ 2]),
    #              lines!(ax.A_sl, xv, As[:, ny ÷ 2])),
    #        Δq=heatmap!(ax.Δq, xv, yv, qmag .- qmag_o; colormap=:turbo),
    #        J=lines!(ax.J, Point2.(iters, J_J0)))

    # Colorbar(fig[1, 2][1, 2], plt.A)
    # Colorbar(fig[2, 1][1, 2], plt.Ā)
    # Colorbar(fig[2, 3][1, 2], plt.Δq)

    # display(fig)
<<<<<<< HEAD
    grad_loss_semi!(Ās_semi, q̄x, q̄y, H̄, D̄, qx, qy, qmag, qmag_o, H, B, D, As, A, n, dx, dy)
    grad_loss_full!(Ās_full, qx, qy, q̄x, q̄y, q̄mag, D̄, qmag, qmag_o, D, H, B, As, A, n, dx, dy)

    

    @show Ās_semi 
    @show Ās_full

    @show extrema(Ās_full)
    @show extrema(Ās_semi)

    @show norm(Ās_full .- Ās_semi, Inf)

    @assert Ās_full ≈ Ās_semi

    @show norm(Ās_full .- Ās_semi, Inf)
    @show extrema(Ās_full)

    error("check")

    # gradient descent
    for igd in 1:5000
        q̄x .= 0
        q̄y .= 0
        Ās .= 0
        # d(loss)/(dAs) with reverse mode
        Enzyme.autodiff(Enzyme.Reverse, loss, Active,
                        DupNN(qx, q̄x),
                        DupNN(qy, q̄y),
                        DupNN(qmag, q̄mag),
                        Const(qmag_o),
                        DupNN(D, D̄),
                        Const(H),
                        Const(B),
                        DupNN(As, Ās),
                        Const(A), Const(n), Const(dx), Const(dy))
        # tune learning rate to never change As too much
        γ = 5e-2 / maximum(abs.(Ās))
        As .= As .- γ .* Ās
        # smooth As (Tikhonov regularisation)
        for _ in 1:1
            As[2:(end - 1), 2:(end - 1)] .+= 0.01 .* (As[1:(end - 2), 2:(end - 1)] .+
                                                      As[3:end, 2:(end - 1)] .+
                                                      As[2:(end - 1), 1:(end - 2)] .+
                                                      As[2:(end - 1), 3:end] .-
                                                      4.0 .* As[2:(end - 1), 2:(end - 1)])
        end
        # check loss (computes fluxes and magnitudes as side effect)
        J = loss(qx, qy, qmag, qmag_o, D, H, B, As, A, n, dx, dy)

        # report intermediate results
        if igd % 50 == 0
            push!(J_J0, J / J0)
            push!(iters, igd)

            plt.Ā[3] = Ās
            plt.A[3] = log10.(As)
            plt.A_sl[3][2] = As[:, ny ÷ 2]
            plt.Δq[3] = qmag .- qmag_o
            plt.J[1] = Point2.(iters, J_J0)
            autolimits!(ax.J)
            autolimits!(ax.A_sl)
            yield()
        end
    end
=======
    grad_loss_semi!(Ās_semi, q̄x, q̄y, D̄, qx, qy, qmag, qmag_o, H, B, D, As, A, n, dx, dy)

    grad_loss_full!(Ās_full, q̄x, q̄y, q̄mag, D̄, qx, qy, qmag, qmag_o, D, H, B, As, A, n, dx, dy)

    Ās_diff = Ās_semi .- Ās_full

    fig = Figure(; size=(600, 800))
    ax = (Ās_semi=Axis(fig[1, 1][1, 1]; aspect=DataAspect()),
          Ās_full=Axis(fig[2, 1][1, 1]; aspect=DataAspect()),
          Ās_diff=Axis(fig[3, 1][1, 1]; aspect=DataAspect()))
    plt = (Ās_semi=heatmap!(ax.Ās_semi, xv, yv, Ās_semi; colormap=:turbo),
           Ās_full=heatmap!(ax.Ās_full, xv, yv, Ās_full; colormap=:turbo),
           Ās_diff=heatmap!(ax.Ās_diff, xv, yv, Ās_diff; colormap=:turbo))

    Colorbar(fig[1, 1][1, 2], plt.Ās_semi)
    Colorbar(fig[2, 1][1, 2], plt.Ās_full)
    Colorbar(fig[3, 1][1, 2], plt.Ās_diff)

    display(fig)

    # @show Ās_semi
    # @show Ās_full

    # @assert Ās_full ≈ Ās_semi

    # @show norm(Ās_full .- Ās_semi, Inf)
    # @show extrema(Ās_full)

    # error("check")

    # # gradient descent
    # for igd in 1:5000
    #     q̄x .= 0
    #     q̄y .= 0
    #     Ās .= 0
    #     # d(loss)/(dAs) with reverse mode
    #     Enzyme.autodiff(Enzyme.Reverse, loss, Active,
    #                     DupNN(qx, q̄x),
    #                     DupNN(qy, q̄y),
    #                     DupNN(qmag, q̄mag),
    #                     Const(qmag_o),
    #                     DupNN(D, D̄),
    #                     Const(H),
    #                     Const(B),
    #                     DupNN(As, Ās),
    #                     Const(A), Const(n), Const(dx), Const(dy))
    #     # tune learning rate to never change As too much
    #     γ = 5e-2 / maximum(abs.(Ās))
    #     As .= As .- γ .* Ās
    #     # smooth As (Tikhonov regularisation)
    #     for _ in 1:1
    #         As[2:end-1, 2:end-1] .+= 0.01 .* (As[1:end-2, 2:end-1] .+
    #                                           As[3:end, 2:end-1] .+
    #                                           As[2:end-1, 1:end-2] .+
    #                                           As[2:end-1, 3:end] .-
    #                                           4.0 .* As[2:end-1, 2:end-1])
    #     end
    #     # check loss (computes fluxes and magnitudes as side effect)
    #     J = loss(qx, qy, qmag, qmag_o, D, H, B, As, A, n, dx, dy)

    #     # report intermediate results
    #     if igd % 50 == 0
    #         push!(J_J0, J / J0)
    #         push!(iters, igd)

    #         plt.Ā[3] = Ās
    #         plt.A[3] = log10.(As)
    #         plt.A_sl[3][2] = As[:, ny÷2]
    #         plt.Δq[3] = qmag .- qmag_o
    #         plt.J[1] = Point2.(iters, J_J0)
    #         autolimits!(ax.J)
    #         autolimits!(ax.A_sl)
    #         yield()
    #     end
    # end
>>>>>>> 6f1bc869aa8fff88d54ef26c7fbd801dec67c9dc

    return
end

main()
