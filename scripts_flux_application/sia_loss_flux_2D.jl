
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

function ∂J_∂qx_vec!(q̄Hx, qmag, qmag_obs, qHx, w_q)
    q̄Hx                .= 0
    @. q̄Hx[1:end-1, :] += w_q*(qmag - qmag_obs) * $avx(qHx) / (2 * qmag + (qmag == 0))
    @. q̄Hx[2:end, :]   += w_q*(qmag - qmag_obs) * $avx(qHx) / (2 * qmag + (qmag == 0))
    return
end

function ∂J_∂qy_vec!(q̄Hy, qmag, qmag_obs, qHy, w_q)
    q̄Hy                .= 0
    @. q̄Hy[:, 1:end-1] += w_q*(qmag - qmag_obs) * $avy(qHy) / (2 * qmag + (qmag == 0))
    @. q̄Hy[:, 2:end]   += w_q*(qmag - qmag_obs) * $avy(qHy) / (2 * qmag + (qmag == 0))
    return
end

#compute the cost function 
function loss(logAs, fwd_params, loss_params; kwags...)
    (; H, qmag) = fwd_params.fields
    (; H_obs, qmag_obs) = loss_params.fields
    (; w_H, w_q) = loss_params.scalars
    @info "Forward solve"
    solve_sia_implicit!(logAs, fwd_params...; kwags...)
    # H = fwd_params.fields.H
    # qmag = fwd_params.fields.qmag
    return 0.5 * (w_H * sum((H .- H_obs) .^ 2) + w_q * sum((qmag .- qmag_obs) .^ 2))
end

#compute the sensitivity: hradient of the loss function
function ∇loss!(logĀs, logAs, fwd_params, adj_params, loss_params; reg=nothing, kwags...)
    #unpack
    (; RH, qHx, qHy, β, H, H_old, B, D, ELA, As, qmag, mask) = fwd_params.fields
    (; dx, dy, dt) = fwd_params.numerical_params
    (; aρgn0, b_max, npow) = fwd_params.scalars
    (; nthreads, nblocks) = fwd_params.launch_config
    (; R̄H, q̄Hx, q̄Hy, H̄, D̄, Ās, ψ_H) = adj_params.fields
    (; H_obs, qmag_obs, Lap_As) = loss_params.fields
    (; w_H, w_q) = loss_params.scalars

    @info "Forward solve"
    solve_sia_implicit!(logAs, fwd_params...; kwags...)

    @info "Adjoint solve"
    solve_adjoint_sia!(fwd_params, adj_params, loss_params)

    ∂J_∂qx_vec!(q̄Hx, qmag, qmag_obs, qHx, w_q)
    ∂J_∂qy_vec!(q̄Hy, qmag, qmag_obs, qHy, w_q)

    logĀs .= 0.0
    R̄H .= .-ψ_H
    H̄ .= 0.0
    D̄ .= 0.0

    #residual!(RH, qHx, qHy, β, H, B, ELA, b_max, H_old, mask, dx, dy, dt)
    @cuda threads = nthreads blocks = nblocks ∇(residual!,
                                                DupNN(RH, R̄H),
                                                DupNN(qHx, q̄Hx),
                                                DupNN(qHy, q̄Hy),
                                                Const(β),
                                                DupNN(H, H̄),
                                                Const(B), Const(ELA), Const(b_max), Const(H_old), Const(mask), Const(dx), Const(dy), Const(dt))
    #compute_q!(qHx, qHy, D, H, B, dx, dy)
    @cuda threads = nthreads blocks = nblocks ∇(compute_q!,
                                                DupNN(qHx, q̄Hx),
                                                DupNN(qHy, q̄Hy),
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