using CairoMakie
using Enzyme

include("macros.jl")
include("sia_forward_flux_2D.jl")

@inline ∇(fun, args...) = (Enzyme.autodiff_deferred(Enzyme.Reverse, fun, Const, args...); return)
const DupNN = DuplicatedNoNeed

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

function forward_model!(logAs, fwd_params)
    (; D, H, B, As, qx, qy, qmag) = fwd_params.fields
    (; aρgn0, npow) = fwd_params.scalars
    (; dx, dy) = fwd_params.numerical_params
    (; nthreads, nblocks) = fwd_params.launch_config

    As .= exp10.(logAs)
    @cuda threads = nthreads blocks = nblocks compute_D!(D, H, B, As, aρgn0, npow, dx, dy)
    @cuda threads = nthreads blocks = nblocks compute_q!(qx, qy, D, H, B, dx, dy)
    @. qmag = sqrt($avx(qx)^2 + $avy(qy)^2)
    return
end

function ∂J_∂qx_vec!(q̄x, qmag, qobs_mag, qx)
    q̄x                .= 0
    @. q̄x[1:end-1, :] += (qmag - qobs_mag) * $avx(qx) / (2 * qmag + (qmag == 0))
    @. q̄x[2:end, :]   += (qmag - qobs_mag) * $avx(qx) / (2 * qmag + (qmag == 0))
    return
end

function ∂J_∂qy_vec!(q̄y, qmag, qobs_mag, qy)
    q̄y                .= 0
    @. q̄y[:, 1:end-1] += (qmag - qobs_mag) * $avy(qy) / (2 * qmag + (qmag == 0))
    @. q̄y[:, 2:end]   += (qmag - qobs_mag) * $avy(qy) / (2 * qmag + (qmag == 0))
    return
end

function loss_snapshot(logAs, fwd_params, loss_params)
    (; qobs_mag) = loss_params.fields
    (; qmag)     = fwd_params.fields
    forward_model!(logAs, fwd_params)
    return 0.5 * sum((qmag .- qobs_mag) .^ 2)
end

function ∇loss_snapshot!(logĀs, logAs, fwd_params, adj_params, loss_params; reg=nothing)
    #unpack forward parameters 
    (; H, B, D, qx, qy, As, qmag) = fwd_params.fields
    (; aρgn0, npow) = fwd_params.scalars
    (; dx, dy) = fwd_params.numerical_params
    (; nthreads, nblocks) = fwd_params.launch_config
    #unpack adjoint parameters
    (; q̄x, q̄y, D̄, H̄) = adj_params.fields
    #unpack loss parameters
    (; qobs_mag) = loss_params.fields

    forward_model!(logAs, fwd_params)

    ∂J_∂qx_vec!(q̄x, qmag, qobs_mag, qx)
    ∂J_∂qy_vec!(q̄y, qmag, qobs_mag, qy)

    logĀs .= 0.0
    D̄ .= 0.0
    H̄ .= 0.0

    @cuda threads = nthreads blocks = nblocks ∇(compute_q!,
                                                DupNN(qx, q̄x),
                                                DupNN(qy, q̄y),
                                                DupNN(D, D̄),
                                                Const(H),
                                                Const(B), Const(dx), Const(dy))
    @cuda threads = nthreads blocks = nblocks ∇(compute_D!,
                                                DupNN(D, D̄),
                                                Const(H), Const(B),
                                                DupNN(As, logĀs), Const(aρgn0), Const(npow), Const(dx), Const(dy))

    logĀs[[1, end], :] = logĀs[[2, end - 1], :]
    logĀs[:, [1, end]] = logĀs[:, [2, end - 1]]

    #smoothing 

    # convert to dJ/dlogAs
    logĀs .*= As

    return
end
