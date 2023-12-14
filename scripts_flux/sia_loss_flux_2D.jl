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

function ∂J_∂qx_vec!(q̄Hx, qmag, qobs_mag, qHx)
    q̄Hx                .= 0
    @. q̄Hx[1:end-1, :] += (qmag - qobs_mag) * $avx(qHx) / (2 * qmag + (qmag == 0))
    @. q̄Hx[2:end, :]   += (qmag - qobs_mag) * $avx(qHx) / (2 * qmag + (qmag == 0))
    return
end

function ∂J_∂qy_vec!(q̄Hy, qmag, qobs_mag, qHy)
    q̄Hy                .= 0
    @. q̄Hy[:, 1:end-1] += (qmag - qobs_mag) * $avy(qHy) / (2 * qmag + (qmag == 0))
    @. q̄Hy[:, 2:end]   += (qmag - qobs_mag) * $avy(qHy) / (2 * qmag + (qmag == 0))
    return
end

#compute the cost function 

function loss(logAs, fwd_params, loss_params; kwags...)
    (; H_obs, qobs_mag) = loss_params.fields
    (; w_H, w_q) = loss_params.scalars
    @info "Forward solve"
    solve_sia!(logAs, fwd_params...; kwags...)
    H = fwd_params.fields.H
    qmag = fwd_params.fields.qmag
    return 0.5 * (w_H * sum((H .- H_obs) .^ 2) + w_q * sum((qmag .- qobs_mag) .^ 2))
end

#compute the sensitivity: hradient of the loss function
function ∇loss!(logĀs, logAs, fwd_params, adj_params, loss_params; reg=nothing, kwags...)
    #unpack
    (; RH, qHx, qHy, β, H, B, D, ELA, As, qmag) = fwd_params.fields
    (; dx, dy) = fwd_params.numerical_params
    (; aρgn0, b_max, npow) = fwd_params.scalars
    (; nthreads, nblocks) = fwd_params.launch_config
    (; R̄H, q̄Hx, q̄Hy, H̄, D̄, Ās, ψ_H) = adj_params.fields
    (; H_obs, qobs_mag) = loss_params.fields

    @info "Forward solve"
    solve_sia!(logAs, fwd_params...; kwags...)

    @info "Adjoint solve"
    solve_adjoint_sia!(fwd_params, adj_params, loss_params)

    ∂J_∂qx_vec!(q̄Hx, qmag, qobs_mag, qHx)
    ∂J_∂qy_vec!(q̄Hy, qmag, qobs_mag, qHy)

    logĀs .= 0.0
    R̄H .= .-ψ_H
    H̄ .= 0.0
    D̄ .= 0.0

    #residual!(RH, qHx, qHy, β, H, B, ELA, b_max, dx, dy)
    @cuda threads = nthreads blocks = nblocks ∇(residual!,
                                                DupNN(RH, R̄H),
                                                DupNN(qHx, q̄Hx),
                                                DupNN(qHy, q̄Hy),
                                                Const(β),
                                                DupNN(H, H̄),
                                                Const(B), Const(ELA), Const(b_max), Const(dx), Const(dy))
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

    #smoothing 
    if !isnothing(reg)
        (; nsm, Tmp) = reg
        Tmp .= logĀs
        smooth!(logĀs, Tmp, nsm, nthreads, nblocks)
    end
    # so what is reg used for: 
    # in the example code, reg is for 
    # convert to dJ/dlogAs
    logĀs .*= As

    return
end
