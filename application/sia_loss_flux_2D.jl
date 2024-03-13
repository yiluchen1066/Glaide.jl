include("macros.jl")
include("sia_forward_flux_2D.jl")

function laplacian!(As, As2)
    @get_indices
    if ix >= 2 && ix <= size(As, 1) - 1 && iy >= 2 && iy <= size(As, 2) - 1
        ΔAs = As[ix-1, iy] + As[ix+1, iy] + As[ix, iy-1] + As[ix, iy+1] - 4.0 * As[ix, iy]
        As2[ix, iy] = As[ix, iy] + 1 / 8 * ΔAs
    end
    return
end

function smooth!(As, As2, nsm, nthreads, nblocks)
    for _ in 1:nsm
        @cuda threads = nthreads blocks = nblocks laplacian!(As, As2)
        As2[[1, end], :] .= As2[[2, end - 1], :]
        As2[:, [1, end]] .= As2[:, [2, end - 1]]
        As, As2 = As2, As
    end
    return
end

function ∂J_∂qx_vec!(q̄Hx, qmag, qmag_obs, qx)
    q̄Hx                .= 0
    @. q̄Hx[1:end-1, :] += (qmag - qmag_obs) * $avx(qx) / (2 * qmag + (qmag == 0))
    @. q̄Hx[2:end, :]   += (qmag - qmag_obs) * $avx(qx) / (2 * qmag + (qmag == 0))
    return
end

function ∂J_∂qy_vec!(q̄Hy, qmag, qmag_obs, qy)
    q̄Hy                .= 0
    @. q̄Hy[:, 1:end-1] += (qmag - qmag_obs) * $avy(qy) / (2 * qmag + (qmag == 0))
    @. q̄Hy[:, 2:end]   += (qmag - qmag_obs) * $avy(qy) / (2 * qmag + (qmag == 0))
    return
end

#compute the cost function 

function loss(logAs, fwd_params, loss_params; kwags...)
    (; H_obs, qmag_obs) = loss_params.fields
    (; w_H, w_q) = loss_params.scalars
    @info "Forward solve"
    solve_sia!(logAs, fwd_params...; kwags...)
    H = fwd_params.fields.H
    qmag = fwd_params.fields.qmag
    return 0.5 * (w_H * sum((H .- H_obs) .^ 2) + w_q * sum((qmag .- qmag_obs) .^ 2))
end

#compute the sensitivity: hradient of the loss function
function ∇loss!(logĀs, logAs, fwd_params, adj_params, loss_params; reg=nothing, visu=nothing)
    #unpack
    (; RH, qx, qy, β, H, B, S, H_ini, D, ELA, As, qmag, mask) = fwd_params.fields
    (; dx, dy) = fwd_params.numerical_params
    (; aρgn0, b_max, npow) = fwd_params.scalars
    (; nthreads, nblocks) = fwd_params.launch_config
    (; R̄H, q̄x, q̄y, H̄, D̄, Ās, ψ_H) = adj_params.fields
    (; H_obs, qmag_obs, Lap_As) = loss_params.fields

    @info "Forward solve"
    solve_sia!(logAs, fwd_params...; visu=visu)

    @info "Adjoint solve"
    solve_adjoint_sia!(fwd_params, adj_params, loss_params)

    ∂J_∂qx_vec!(q̄x, qmag, qmag_obs, qx)
    ∂J_∂qy_vec!(q̄y, qmag, qmag_obs, qy)

    logĀs .= 0.0
    R̄H .= .-ψ_H
    H̄ .= 0.0
    D̄ .= 0.0

    #residual!(RH, qx, qy, β, H, B, H_ini, ELA, b_max, mask, dx, dy)
    @cuda threads = nthreads blocks = nblocks ∇(residual!,
                                                DupNN(RH, R̄H),
                                                DupNN(qx, q̄x),
                                                DupNN(qy, q̄y),
                                                Const(β),
                                                DupNN(H, H̄),
                                                Const(B), Const(H_ini), Const(ELA), Const(b_max), Const(mask), Const(dx), Const(dy))
    #compute_q!(qx, qy, D, H, B, dx, dy)
    @cuda threads = nthreads blocks = nblocks ∇(compute_q!,
                                                DupNN(qx, q̄x),
                                                DupNN(qy, q̄y),
                                                DupNN(D, D̄),
                                                DupNN(H, H̄),
                                                Const(B), Const(dx), Const(dy))
    #compute_D!(D, H, B, As, aρgn0, n, dx, dy)
    @cuda threads = nthreads blocks = nblocks ∇(compute_D!,
                                                DupNN(D, D̄),
                                                Const(H), Const(B),
                                                DupNN(As, logĀs), Const(aρgn0), Const(npow), Const(dx), Const(dy))

    logĀs[[1, end], :] = logĀs[[2, end - 1], :]
    logĀs[:, [1, end]] = logĀs[:, [2, end - 1]]

    # convert to dJ/dlogAs
    logĀs .*= As

    if !isnothing(reg)
        (; nsm, Tmp, α) = reg
        Lap_As .= logAs
        smooth!(Lap_As, Tmp, nsm, nthreads, nblocks)
        logĀs .-= α .* (Lap_As .- logAs)
    end 

    return
end
