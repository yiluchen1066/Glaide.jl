using Enzyme

include("macros.jl")

@inline ∇(fun, args...) = (Enzyme.autodiff_deferred(Enzyme.Reverse, fun, Const, args...); return)
const DupNN = DuplicatedNoNeed

function compute_D!(D, H, B, As, A, n)
    @get_indices
    if ix <= size(D, 1) && iy <= size(D, 2)
        ∇Sx = 0.5 *
              ((B[ix + 1, iy + 1] - B[ix, iy + 1]) / dx +
               (H[ix + 1, iy + 1] - H[ix, iy + 1]) / dx +
               (B[ix + 1, iy] - B[ix, iy]) / dx +
               (H[ix + 1, iy] - H[ix, iy]) / dx)
        ∇Sy = 0.5 *
              ((B[ix + 1, iy + 1] - B[ix + 1, iy]) / dy +
               (H[ix + 1, iy + 1] - H[ix + 1, iy]) / dy +
               (B[ix, iy + 1] - B[ix, iy]) / dy +
               (H[ix, iy + 1] - H[ix, iy]) / dy)
        D[ix, iy] = (aρgn0 * @av_xy(H)^(n + 2) + As[ix, iy] * @av_xy(H)^n) * sqrt(∇Sx^2 + ∇Sy^2)^(n - 1)
    end
    return
end

#
function flux!(qx, qy, H, B, D)
    @get_indices
    if ix <= size(qHx, 1) && iy <= size(qHx, 2)
        qx[ix, iy] = -@av_ya(D) * (@d_xi(H) + @d_xi(B)) / dx
    end
    if ix <= size(qHy, 1) && iy <= size(qHy, 2)
        qy[ix, iy] = -@av_xa(D) * (@d_yi(H) + @d_yi(B)) / dy
    end
    return
end



#host function, 
function model!(qx, qy, q_mag, H, B, D, As, A, n)
    compute_D!(D, H, B, As, A, n)
    flux!(qx, qy, H, B, D)
    q_mag .= sqrt.(avx(qx).^2 .+ avy(qy).^2)
    #manual kernel or broadcast
    if ix <= size(q_mag, 1) && iy <= size(q_mag, 2)
        q_mag[ix, iy] = sqrt((0.5 * (qx[ix, iy] + qx[ix + 1, iy]))^2 + (0.5 * (qy[ix, iy] + qy[ix, iy + 1]))^2)
    end
    return
end

function loss(qx, qy, q_mag, q_obs_mag, H, B, D, As)
    model!(qx, qy, q_mag, H, B, D, As, A, n)
    J = sum(0.5 .* (q_mag .- q_obs_mag).^2)
    return J
end

function ∂J_∂qx_vec!(q̄x, q_mag, q_obs_mag, qx)
    q̄x                    .= 0.0
    @. q̄x[1:(end - 1), :] += (q_mag - q_obs_mag) * (qx[1:(end - 1), :] + qx[2:end, :]) / 2 / (2 * qmag)
    @. q̄x[2:end, :]       += (q_mag - q_obs_mag) * (qx[1:(end - 1), :] + qx[2:end, :]) / 2 / (2 * qmag)
    return
end

function ∂J_∂qy_vec!(q̄y, q_mag, q_obs_mag, qy)
    q̄y                    .= 0.0
    @. q̄y[:, 1:(end - 1)] += (q_mag - q_obs_mag) * (qy[:, 1:(end - 1)] + qy[:, 2:end]) / 2 / (2 * qmag)
    @. q̄y[:, 2:end]       += (q_mag - q_obs_mag) * (qy[:, 1:(end - 1)] + qy[:, 2:end]) / 2 / (2 * qmag)
    return
end

function grad_loss_semi!(Ās, q̄x, q̄y, D̄, H̄, qx, qy, qx_obs, qy_obs, q_mag, q_obs_mag, H, B, D, As, As_syn, A, npow,
                         dx, dy)
    model!(qx, qy, q_mag, H, B, D, As, A, n)
    model!(qx_obs, qy_obs, q_obs_mag, H, B, D, As_syn, A, n)

    ∂J_∂qx_vec!(q̄x, q_mag, q_obs_mag, qx)
    ∂J_∂qy_vec!(q̄y, q_mag, q_obs_mag, qy)

    ∇(compute_q!,
      DupNN(qx, q̄x),
      DupNN(qy, q̄y),
      DupNN(D, D̄),
      DupNN(H, H̄),
      Const(B), Const(dx), Const(dy))
    ∇(compute_D!,
      DupNN(D, D̄),
      Const(H), Const(B),
      DupNN(As, Ās), Const(aρgn0), Const(npow), Const(dx), Const(dy))

    return
end

function grad_loss_full!()
    loss!(J, qx, qy, qx_obs, qy_obs, q_mag, q_obs_mag, H, B, D, As)
    ∇(loss!,
      DupNN(),
      DupNN(),
      DupNN())

    return Enzyme(loss!)
end

function main()
    B = exp()
    H = exp()
    A = 1.0
    As_s = 1.0
    As = 0.1
    Ās_full = zeros(nx, ny)
    Ās_semi = zeros(nx, ny)
    model!(qx_obs, qy_obs, q_obs_mag, H, B, D, As_s, A, n)
    # q_obs_mag = 
    grad_loss_semi!(Ās, q̄x, q̄y, D̄, H̄, qx, qy, qx_obs, qy_obs, q_mag, q_obs_mag, H, B, D, As, As_syn, A, npow,
    dx, dy)


    grad_loss_full!(Ās_full)
    grad_loss_semi!(Ās_semi)

    @assert Ās_full ≈ Ās_semi

    @show norm(Ās_full .- Ās_semi, Inf)
    @show extrema(Ās_full)
end