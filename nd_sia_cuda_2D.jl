using CUDA,BenchmarkTools
using Plots,Plots.Measures,Printf
using DelimitedFiles
using Enzyme 

Plots.default(size=(800,600))

const DO_VISU = true 
macro get_thread_idx(A)  esc(:( begin ix =(blockIdx().x-1) * blockDim().x + threadIdx().x; iy = (blockIdx().y-1) * blockDim().y+threadIdx().y;end )) end 
macro av_xy(A)    esc(:(0.25*($A[ix,iy]+$A[ix+1,iy]+$A[ix, iy+1]+$A[ix+1,iy+1]))) end
macro av_xa(A)    esc(:( 0.5*($A[ix, iy]+$A[ix+1, iy]) )) end
macro av_ya(A)    esc(:(0.5*($A[ix, iy]+$A[ix, iy+1]))) end 
macro d_xa(A)     esc(:( $A[ix+1, iy]-$A[ix, iy])) end 
macro d_ya(A)     esc(:($A[ix, iy+1]-$A[ix,iy])) end
macro d_xi(A)     esc(:($A[ix+1, iy+1]-$A[ix, iy+1])) end 
macro d_yi(A)     esc(:($A[ix+1, iy+1]-$A[ix+1, iy])) end

function compute_rel_error_1!(Err_rel, H)
    @get_thread_idx(H)
    if ix <= size(H)[1] && iy <= size(H)[2] 
        Err_rel[ix,iy] = H[ix,iy]
    end 
    return 
end 

function compute_rel_error_2!(Err_rel, H)
    @get_thread_idx(H)
    if ix <= size(H)[1] && iy <= size(H)[2] 
        Err_rel[ix,iy] = Err_rel[ix,iy] - H[ix,iy]
    end
    return 
end 

cost(H, H_obs) = 0.5*sum((H.-H_obs).^2)

function compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, a, as, n, nx, ny, dx, dy)
    @get_thread_idx(H)
    #D in size (nx-1, ny-1)
    if ix <= nx-1 && iy <= ny-1
        av_ya_∇Sx[ix,iy] = 0.5*((B[ix+1, iy+1]-B[ix,iy+1])/dx + (H[ix+1, iy+1]-H[ix,iy+1])/dx + (B[ix+1,iy]-B[ix,iy])/dx + (H[ix+1,iy]-H[ix,iy])/dx)
        av_xa_∇Sy[ix,iy] = 0.5*((B[ix+1, iy+1]-B[ix+1,iy])/dy + (H[ix+1, iy+1]-H[ix+1,iy])/dy + (B[ix,iy+1]-B[ix,iy])/dy + (H[ix,iy+1]-H[ix,iy])/dy)
        gradS[ix,iy]     = sqrt(av_ya_∇Sx[ix,iy]^2 + av_xa_∇Sy[ix,iy]^2)
        D[ix,iy]         = (a*@av_xy(H)^(n+2)+as[ix,iy]*av_xy(H)^n)*gradS[ix,iy]^(n-1)
    end 
    return 
end 

function compute_q!(qHx, qHy, D, H, B, nx, ny, dx, dy)
    @get_thread_idx(H)
    # qHx in size(nx-1, ny-2) 
    # qHy in size(nx-2, ny-1)
    if ix <= nx-1 && iy <= ny-2 
        qHx[ix,iy] = -@av_ya(D)*(@d_xi(H)+@d_xi(B))/dx 
    end 
    if ix <= nx-2 && iy <= ny-1
        qHy[ix,iy] = -@av_xa(D)*(@d_yi(H)+@d_yi(B))/dy
    end 
    return 
end 

function timestep!(dτ, H, D, cfl, epsi, nx, ny)
    @get_thread_idx(H)
    if ix <= nx-2 && iy <= ny -2 
        dτ[ix, iy]  = min(20.0, cfl/(epsi+@av_xy(D)))
        #dτ[ix, iy]  = cfl/(@av_xy(D))
    end 
    return 
end 

function residual!(RH, qHx, qHy, β, H, B, z_ELA, b_max, nx, ny, dx, dy)
    @get_thread_idx(H)
    if ix <= nx-2 && iy <= ny-2 
        RH[ix+1,iy+1] =  -(@d_xa(qHx)/dx + @d_ya(qHy)/dy) + min(β[ix+1,iy+1]*(H[ix+1, iy+1]+B[ix+1, iy+1]-z_ELA), b_max)
    end 
    return 
end 

function compute_abs_error!(Err_abs, qHx, qHy, β, H, B, z_ELA, b_max, dx, dy, nx, ny)
    @get_thread_idx(RH) 
    if ix <= nx && iy <= ny 
        Err_abs[ix,iy] = -(@d_xa(qHx)/dx + @d_ya(qHy)/dy) + min(β[ix+1,iy+1]*(H[ix+1, iy+1]+B[ix+1, iy+1]-z_ELA), b_max)
        if H[ix,iy] ≈ 0.0 
            Err_abs[ix,iy] = 0.0 
        end 
    end 
    return 
end


function update_H!(H, dτ, RH, nx, ny)
    @get_thread_idx(H)
    if ix <= nx-2 && iy <= ny -2 
        # no damping
        H[ix+1, iy+1] = max(0.0, H[ix+1,iy+1]+dτ[ix,iy]*RH[ix+1,iy+1])
    end 
    return 
end 


function set_BC!(H)
    @get_thread_idx(H)
    if ix ==1 && iy <= size(H)[2]
        H[ix,iy] = H[ix+1, iy]
    end 
    if ix == size(H)[1] && iy <= size(H)[2]
        H[ix,iy] = H[ix-1, iy]
    end 
    if ix <= size(H)[1] && iy ==1 
        H[ix,iy] = H[ix,iy+1]
    end
    if  ix <= size(H)[1] && iy == size(H)[2] 
        H[ix,iy] = H[ix, iy-1]
    end 
    return 
end 

function update_S!(S, H, B)
    @get_thread_idx(H)
    if ix <= size(S)[1] && iy <= size(S)[2] 
        S[ix,iy] = H[ix, iy] + B[ix, iy]
    end 
    return 
end 


mutable struct Forwardproblem{T<:Real, A<:AbstractArray{T}}
    H::A; D::A; gradS::A; av_ya_∇Sx::A; av_xa_∇Sy::A; qHx::A; qHy::A; B::A
    RH::A; dτ::A; Err_abs::A; Err_rel::A; 
    cfl::T; epsi::T; β::A; z_ELA::Int; b_max::T; a::T; as::A; n::Int; nx::Int; ny::Int; dx::T; dy::T; maxiter::Int; ncheck::Int; threads::Tuple{Int, Int}; blocks::Tuple{Int, Int}; ϵtol::Tuple{Float64, Float64}
end 


function Forwardproblem(H, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, B, cfl, epsi, β, z_ELA, b_max, a, as, n, nx, ny, dx, dy, maxiter,ncheck, threads, blocks, ϵtol)
    RH = similar(H, nx, ny)
    dτ = similar(H, nx-2, ny-2)
    Err_abs = similar(H, nx, ny)
    Err_rel = similar(H, nx, ny)
    return Forwardproblem(H, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, B, RH, dτ, Err_abs, Err_rel,cfl, epsi, β, z_ELA, b_max, a, as, n, nx, ny, dx, dy, maxiter, ncheck, threads, blocks, ϵtol)
end 

function solve!(problem::Forwardproblem)
    (; H, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, B, RH, dτ, Err_abs, Err_rel, cfl, epsi, β, z_ELA, b_max, a, as, n, nx, ny, dx, dy, maxiter, ncheck, threads, blocks, ϵtol) = problem 
    RH .= 0; dτ .= 0; Err_abs .= 0; Err_rel .= 0
    #iteration loop 
    err_abs0 = 0.0 
    for iter in 1:maxiter 
        if iter % ncheck ==0 
            CUDA.@sync @cuda threads=threads blocks=blocks compute_rel_error_1!(Err_rel, H)
        end 
        CUDA.@sync @cuda threads=threads blocks=blocks compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, a, as, n, nx, ny, dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks compute_q!(qHx, qHy, D, H, B, nx, ny, dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks timestep!(dτ, H, D, cfl, epsi, nx, ny)
        CUDA.@sync @cuda threads=threads blocks=blocks residual!(RH, qHx, qHy, β, H, B, z_ELA, b_max, nx, ny, dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks update_H!(H, dτ, RH, nx, ny)
        if iter ==1 || iter % ncheck ==0 
            CUDA.@sync @cuda threads=threads blocks=blocks compute_abs_error!(Err_abs, qHx, qHy, β, H, B, z_ELA, b_max, dx, dy, nx, ny)
            CUDA.@sync @cuda threads=threads blocks=blocks compute_rel_error_2!(Err_rel, H)
            if iter ==1 
                err_abs0 = maximum(abs.(Err_abs))
            end 
            err_abs = maximum(abs.(Err_abs))/err_abs0
            err_rel = maximum(abs.(Err_rel))/maximum(H)
            @printf("iter/nx^2 = %.3e, err = [abs = %.3e, rel = %.3e] \n", iter/nx^2, err_abs, err_rel)
            if err_abs < ϵtol.abs || err_rel < ϵtol.rel
                break 
            end 
        end 
    end 
    return 
end 
        
#compute (dR/dH)^T*r
function grad_R!(tmp1, tmp2, qHx, qHy, dR_qHx, dR_qHy, β, H, dR_H, B, z_ELA, b_max, nx, ny, dx, dy)
    #residual!(RH, qHx, qHy, β, H, B, z_ELA, b_max, nx, ny, dx, dy)
    Enzyme.autodiff_deferred(residual!, Duplicated(tmp1, tmp2), Duplicated(qHx, dR_qHx), Duplicated(qHy, dR_qHy), Const(β), Duplicated(H, dR_H), Const(B), Const(z_ELA), Const(b_max), Const(nx), Const(ny), Const(dx), Const(dy))
    return 
end 

function grad_q!(qHx, dR_qHx, dR_qHy, D, dq_D, H, dq_H, B, nx, ny, dx, dy)
    #compute_q!(qHx, qHy, D, H, B, nx, ny, dx, dy)
    Enzyme.autodiff_deferred(compute_q!, Duplicated(qHx, dR_qHx), Duplicated(qHy, dR_qHy), Duplicated(D, dq_D), Duplicated(H, dq_H), Const(B), Const(nx), Const(ny), Const(dx), Const(dy))
    return 
end 

function grad_D!(D, dq_D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, dD_H, B, a, as, n, nx, ny, dx, dy)
    #compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, a, as, n, nx, ny, dx, dy)
    Enzyme.autodiff_deferred(compute_D!, Duplicated(D, dq_D), Const(gradS), Const(av_ya_∇Sx), Const(av_xa_∇Sy), Duplicated(H, dD_H), Const(B), Const(a), Const(as), Const(n), Const(nx), Const(ny), Const(dx), Const(dy))
    return 
end 

function grad_residual_H!(dR, dD_H, dq_H, dR_H, nx, ny)
    @get_thread_idx(dR)
    if ix <= nx && iy <= ny 
        dR[ix,iy] += dD_H[ix,iy] + dq_H[ix,iy] +  dR_H[ix,iy]
    end 
    return 
end 

function update_r!(r, dR, dt, H, nx, ny)
    @get_thread_idx(H)
    if ix <= nx && iy <= ny 
        if ix >1 && ix < nx && iy >1 && iy < ny 
            if H[ix, iy] <= 1.0e-2  ||
               H[ix-1,iy] <= 1.0e-2 ||
               H[ix+1,iy] <= 1.0e-2 ||
               H[ix,iy-1] <= 1.0e-2 ||
               H[ix,iy+1] <= 1.0e-2
                r[ix,iy] = 0.0 
            else
                r[ix,iy] = r[ix,iy] + dt*dR[ix,iy]
            end 
        end 
        if ix ==1 || ix == nx 
            r[ix,iy] = 0.0 
        end 
        if iy ==1 || iy == ny 
            r[ix,iy] = 0.0 
        end 
    end 
    return 
end

mutable struct AdjointProblem{T<:Real, A<:AbstractArray{T}}
end 

function AdjointProblem()
    return AdjointProblem()
end

function solve!(problem::AdjointProblem)
    (; ) = problem 

    dt = min(dx^2, dy^2)/maximum(D)/200.1
    @. 𝞉J_∂H = H - H_obs 
    for iter in 1:maxiter 
        CUDA.@sync @cuda threads = threads blocks=blocks grad_R!(tmp1, tmp2, qHx, qHy, dR_qHx, dR_qHy, β, H, dR_H, B, z_ELA, b_max, nx, ny, dx, dy)
        CUDA.@sync @cuda threads = threads blocks=blocks grad_q!(qHx, dR_qHx, dR_qHy, D, dq_D, H, dq_H, B, nx, ny, dx, dy)
        CUDA.@sync @cuda threads = threads blocks=blocks grad_D!(D, dq_D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, dD_H, B, a, as, n, nx, ny, dx, dy)
        CUDA.@sync @cuda threads = threads blocks=blocks grad_residual_H!(dR, dD_H, dq_H, dR_H, nx, ny)
        CUDA.@sync @cuda threads = threads blocks=blocks update_r!(r, dR, dt, H, nx, ny)
    end 

end 


@views function main()
    # power law exponents
    n        = 3   # Glen's flow law power exponent
    # dimensionally independent physics
    lx       = 1.0 # [m]
    aρgn0    = 1.0 # [1/s/m^n]
    # scales
    tsc      = 1/aρgn0/lx^n # [s]
    # non-dimensional numbers
    s_f      = 3e-4    # sliding to ice flow ratio: s_f = asρgn0/aρgn0/lx^2
    w_b_lx   = 0.2     # mountain width to domain length ratio
    a_b_lx   = 0.0175  # mountain height to domain length ratio
    z_ela_lx = 0.016   # ela to domain length ratio
    βtsc     = 1e-14   # ratio between characteristic time scales of ice flow and accumulation/ablation
    m_max_nd = 0.5e-16 # maximum accumulation
    tanθ     = 0.2     # slope
    # dimensionally dependent physics
    asρgn0   = s_f*aρgn0*lx^2
    w_b      = w_b_lx*lx
    a_b      = a_b_lx*lx
    z_ela    = z_ela_lx*lx
    β        = βtsc/tsc
    m_max    = m_max_nd*lx/tsc
    # numerics
    nx       = 501
    ϵtol     = (abs = 1e-8, rel = 1e-14)
    maxiter  = 10nx^2
    ncheck   = ceil(Int,0.1nx^2)
    # preprocessing
    dx       = lx/nx
    xv       = LinRange(-lx/2,lx/2,nx+1)
    xc       = avx(xv)
    # array allocation
    H        = Vector{Float64}(undef,nx  )
    B        = Vector{Float64}(undef,nx  )
    M        = Vector{Float64}(undef,nx  )
    qx       = Vector{Float64}(undef,nx+1)
    D        = Vector{Float64}(undef,nx-1)
    ∇S       = Vector{Float64}(undef,nx-1)
    r_abs    = Vector{Float64}(undef,nx  )
    r_rel    = Vector{Float64}(undef,nx  )
    # initialisation
    @. B     = a_b*(exp(-(xc/w_b)^2) + tanθ*xc/lx); B .-= minimum(B)
    @. H     = 0
    @. qx    = 0
    # iteration loop
    err_abs0 = 0.0
    for iter in 1:maxiter
        if iter % ncheck == 0; r_rel .= H; end
        ∇S .= d_x(B.+H)./dx
        D  .= avx(aρgn0.*H.^(n+2) .+ asρgn0.*H.^n).*abs.(∇S).^(n-1)
        @. qx[2:end-1] = -D*∇S
        @. M = min(β*(B + H - z_ela),m_max)
        dτ = 1.0/(6.1*maximum(D)/dx^2 + β)
        H .= max.(H .+ dτ.*(.-d_x(qx)./dx .+ M),0)
        if iter == 1 || iter % ncheck == 0
            r_abs .= .-d_x(qx)./dx .+ M; @. r_abs[H ≈ 0] = 0
            r_rel .-= H
            if iter == 1; err_abs0 = maximum(abs.(r_abs)); end
            err_abs = maximum(abs.(r_abs))/err_abs0
            err_rel = maximum(abs.(r_rel))/maximum(H)
            @printf("  iter/nx^2 = %.3e, err = [abs = %.3e, rel = %.3e]\n",iter/nx^2, err_abs, err_rel)
            areaplot(xc,[B 10.0.*H];aspect_ratio=4,show=true,xlims=(-lx/2,lx/2),ylims=(0,2*(a_b)),xlabel="x",ylabel="y")
            if err_abs < ϵtol.abs || err_rel < ϵtol.rel; break; end
        end
    end
    @show maximum(H)
    return
end

main()