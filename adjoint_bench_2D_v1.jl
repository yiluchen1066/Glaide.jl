using CUDA,BenchmarkTools
using Plots,Plots.Measures,Printf
using DelimitedFiles
using Enzyme 
#default(size=(800,600),framestyle=:box,label=false,grid=false,margin=10mm,lw=4,labelfontsize=9,tickfontsize=9,titlefontsize=12)

const DO_VISU = true 
macro get_thread_idx(A)  esc(:( begin ix =(blockIdx().x-1) * blockDim().x + threadIdx().x; iy = (blockIdx().y-1) * blockDim().y+threadIdx().y;end )) end 
macro av_xy(A)    esc(:(0.25*($A[ix,iy]+$A[ix+1,iy]+$A[ix, iy+1]+$A[ix+1,iy+1]))) end
macro av_xa(A)    esc(:( 0.5*($A[ix, iy]+$A[ix+1, iy]) )) end
macro av_ya(A)    esc(:(0.5*($A[ix, iy]+$A[ix, iy+1]))) end 
macro d_xa(A)     esc(:( $A[ix+1, iy]-$A[ix, iy])) end 
macro d_ya(A)     esc(:($A[ix, iy+1]-$A[ix,iy])) end
macro d_xi(A)     esc(:($A[ix+1, iy+1]-$A[ix, iy+1])) end 
macro d_yi(A)     esc(:($A[ix+1, iy+1]-$A[ix+1, iy])) end

CUDA.device!(6) # GPU selection

function compute_rel_error_1!(Err_rel, H, nx, ny)
    @get_thread_idx(H)
    if ix <= nx && iy <= ny 
        Err_rel[ix, iy] = H[ix, iy]
    end 
    return 
end 

function compute_rel_error_2!(Err_rel, H, nx, ny)
    @get_thread_idx(H) 
    if ix <= nx && iy <= ny
        Err_rel[ix, iy] = Err_rel[ix, iy] - H[ix, iy]
    end 
    return 
end

function cost!(H, H_obs, J, nx, ny)
    @get_thread_idx(H)
    if ix <= nx && iy <= ny
        J += (H[ix,iy]-H_obs[ix,iy])^2
    end 
    J *= 0.5 
    return 
end

cost(H, H_obs) = 0.5*sum((H.-H_obs).^2)

function compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, a, as, n, nx, ny, dx, dy)
    @get_thread_idx(H)
    if ix <= nx-1 && iy <= ny-1 
        av_ya_∇Sx[ix,iy] = 0.5*((B[ix+1, iy+1]-B[ix,iy+1])/dx + (H[ix+1, iy+1]-H[ix,iy+1])/dx + (B[ix+1,iy]-B[ix,iy])/dx + (H[ix+1,iy]-H[ix,iy])/dx)
        av_xa_∇Sy[ix,iy] = 0.5*((B[ix+1, iy+1]-B[ix+1,iy])/dy + (H[ix+1, iy+1]-H[ix+1,iy])/dy + (B[ix,iy+1]-B[ix,iy])/dy + (H[ix,iy+1]-H[ix,iy])/dy)
        gradS[ix, iy] = sqrt(av_ya_∇Sx[ix,iy]^2+av_xa_∇Sy[ix,iy]^2)
        D[ix, iy] = (a*@av_xy(H)^(n+2)+as[ix,iy]*@av_xy(H)^n)*gradS[ix,iy]^(n-1)
    end 
    return 
end 

function compute_q!(qHx, qHy, D, H, B, nx, ny, dx, dy)
    @get_thread_idx(H)
    if ix <= nx-1 && iy <= ny-2
        qHx[ix,iy] = -@av_ya(D)*(@d_xi(H)+@d_xi(B))/dx 
    end 
    if ix <= nx-2 && iy <= ny-1 
        qHy[ix,iy] = -@av_xa(D)*(@d_yi(H)+@d_yi(B))/dy 
    end 
    return
end



function residual!(RH, qHx, qHy, β, H, B, z_ELA, b_max, nx, ny , dx, dy)
    @get_thread_idx(H)
    if ix <= nx-2 && iy <= ny -2 
        RH[ix+1, iy+1] = -(@d_xa(qHx)/dx + @d_ya(qHy)/dy) + min(β[ix+1,iy+1]*(H[ix+1, iy+1]+B[ix+1, iy+1]-z_ELA[ix+1, iy+1]), b_max)
    end 
    return 
end 

function compute_abs_error!(Err_abs, qHx, qHy, β, H, B, z_ELA, b_max, nx, ny , dx, dy)
    @get_thread_idx(H)
    if ix <= nx-2 && iy <= ny-2 
        Err_abs[ix+1,iy+1] = -(@d_xa(qHx)/dx + @d_ya(qHy)/dy) + min(β[ix+1,iy+1]*(H[ix+1, iy+1]+B[ix+1, iy+1]-z_ELA[ix+1, iy+1]), b_max)
        if H[ix+1, iy+1] ≈ 0.0 
            Err_abs[ix+1,iy+1] = 0.0
        end 
    end 
    return 
end 

function update_H!(H, RH, dτ, nx, ny)
    @get_thread_idx(H)
    if ix <= nx-2 && iy <= ny -2 
        #update the inner point of H 
        H[ix+1,iy+1] = max(0.0, H[ix+1,iy+1]+dτ*RH[ix+1,iy+1])
    end 
    return 
end



function set_BC!(H, nx, ny)
    @get_thread_idx(H)
    if ix == 1 && iy <= ny 
        H[ix,iy] = H[ix+1, iy]
    end 
    if ix == nx && iy <= ny 
        H[ix,iy] = H[ix-1, iy]
    end 
    if ix <= nx && iy == 1 
        H[ix,iy] = H[ix, iy+1]
    end 
    if ix <= nx && iy == ny 
        H[ix, iy] = H[ix, iy-1]
    end 
    return 
end

function update_S!(S, H, B, nx, ny)
    @get_thread_idx(H)
    if ix <= nx && iy <= ny 
        S[ix,iy] = H[ix,iy] + B[ix,iy]
    end 
    return 
end 


mutable struct Forwardproblem{T<:Real, A<:AbstractArray{T}}
    H::A; B::A; D::A; gradS::A; av_ya_∇Sx::A; av_xa_∇Sy::A; qHx::A; qHy::A; β::A; as::A
    RH::A; Err_rel::A; Err_abs::A
    n::Int; a::T; b_max::T; z_ELA::A; dx::T; dy::T; nx::Int; ny::Int; epsi::T; ϵtol::NamedTuple{(:abs, :rel), Tuple{Float64, Float64}}; maxiter::Int; ncheck::Int; threads::Tuple{Int, Int}; blocks::Tuple{Int, Int}
end 

function Forwardproblem(H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β, as, n, a, b_max, z_ELA, dx, dy, nx, ny, epsi, ϵtol, maxiter, ncheck, threads, blocks)
    RH      = similar(H,nx, ny)
    Err_rel = similar(H, nx, ny)
    Err_abs = similar(H, nx, ny)
    return Forwardproblem(H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β, as, RH, Err_rel, Err_abs, n, a, b_max, z_ELA, dx, dy, nx, ny, epsi, ϵtol, maxiter, ncheck, threads, blocks) 
end 

#solve for H using pseudo-trasient method 
function solve!(problem::Forwardproblem)
    (;H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β, as, RH, Err_rel, Err_abs, n, a, b_max, z_ELA, dx,dy, nx, ny, epsi, ϵtol, maxiter, ncheck, threads, blocks) = problem
    RH .= 0; Err_rel .= 0; Err_abs .= 0
    err_abs0 = 0.0 

    for iter in 1:maxiter
        if iter % ncheck == 0 
            CUDA.@sync @cuda threads = threads blocks = blocks compute_rel_error_1!(Err_rel, H, nx, ny)
        end 
        CUDA.@sync @cuda threads=threads blocks=blocks compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, a, as, n, nx, ny, dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks compute_q!(qHx, qHy, D, H, B, nx, ny, dx, dy)
        #CUDA.@sync @cuda threads=threads blocks=blocks timestep!(dτ, H, D, cfl, epsi, nx, ny)
        dτ=1.0/(8.1*maximum(D)/dx^2 + maximum(β))
        CUDA.@sync @cuda threads=threads blocks=blocks residual!(RH, qHx, qHy, β, H, B, z_ELA, b_max, nx, ny , dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks update_H!(H, RH, dτ, nx, ny)
        CUDA.@sync @cuda threads=threads blocks=blocks set_BC!(H, nx,ny)
        if iter ==1 || iter % ncheck == 0 
            CUDA.@sync @cuda threads=threads blocks=blocks compute_abs_error!(Err_abs, qHx, qHy, β, H, B, z_ELA, b_max, nx, ny , dx, dy)
            CUDA.@sync @cuda threads=threads blocks=blocks compute_rel_error_2!(Err_rel, H, nx, ny)
            if iter ==1 
                err_abs0 = maximum(abs.(Err_abs))
            end 
            err_abs = maximum(abs.(Err_abs))/err_abs0 
            err_rel = maximum(abs.(Err_rel))/maximum(H)
            @printf("iter/nx^2 = %.3e, err = [abs =%.3e, rel= %.3e] \n", iter/nx^2, err_abs, err_rel)
            #p1 = heatmap(Array(H'); title = "S (forward problem)")
            #p2 = heatmap(Array(Err_abs'); title = "Err_abs")
            #p3 = heatmap(Array(Err_rel'); title = "Err_rel")
            #display(plot(p1,p2,p3))
            #if debug


            #end
            if err_abs < ϵtol.abs || err_rel < ϵtol.rel 
                break 
            end 
        end 
    end 
    return 
end 
#compute dR_qHx dR_qHy dR_H
#residual!(RH, qHx, qHy, β, H, B, z_ELA, b_max, nx, ny, dx, dy)
function grad_residual_H_1!(tmp1, tmp2, qHx, dR_qHx, qHy, dR_qHy, β, H, dR_H, B, z_ELA, b_max, nx, ny, dx, dy)
    #tmp2 .= r 
    Enzyme.autodiff_deferred(residual!, Duplicated(tmp1, tmp2), Duplicated(qHx, dR_qHx), Duplicated(qHy, dR_qHy), Const(β), Duplicated(H, dR_H), Const(B), Const(z_ELA), Const(b_max), Const(nx), Const(ny), Const(dx), Const(dy))
    return 
end 

#compute_q!(qHx, qHy, D, H, B, nx, ny, dx, dy)
#compute dq_D, dq_H
function grad_residual_H_2!(qHx, dR_qHx, qHy, dR_qHy, D, dq_D, H, dq_H, B, nx, ny, dx, dy)
    Enzyme.autodiff_deferred(compute_q!, Duplicated(qHx, dR_qHx), Duplicated(qHy, dR_qHy), Duplicated(D, dq_D), Duplicated(H, dq_H), Const(B), Const(nx), Const(ny), Const(dx), Const(dy))
    return 
end 

#compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, a, as, n, nx, ny, dx, dy)
#compute dD_H
function grad_residual_H_3!(D, dq_D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, dD_H, B, a, as, n, nx, ny, dx, dy) 
    Enzyme.autodiff_deferred(compute_D!, Duplicated(D, dq_D), Const(gradS), Const(av_ya_∇Sx), Const(av_xa_∇Sy), Duplicated(H, dD_H), Const(B), Const(a), Const(as), Const(n), Const(nx), Const(ny), Const(dx), Const(dy))
    return 
end

function grad_residual_H!(dR, dD_H, dq_H, dR_H, nx, ny)
    @get_thread_idx(dR)
    if ix <= nx && iy <= ny 
        dR[ix, iy] += dD_H[ix,iy] + dq_H[ix,iy] + dR_H[ix,iy] 
    end 
    return 
end 

#Enzyme.autodiff_deferred(compute_q!, Duplicated(qHx, dR_qHx), Duplicated(qHy, dR_qHy), Duplicated(D, dq_D), Duplicated(H, dq_H), Const(B), Const(nx), Const(ny), Const(dx), Const(dy))

function update_r!(r, R, dR, dt, H, H_cut, dmp, nx, ny)
    @get_thread_idx(R) 
    if ix <= nx && iy <= ny
        if ix > 1 && ix < nx && iy > 1 && iy < ny
            # H_cut = 1.0e-2
            if H[ix  , iy] <= H_cut ||
               H[ix-1, iy] <= H_cut ||
               H[ix+1, iy] <= H_cut ||
               H[ix, iy-1] <= H_cut ||
               H[ix, iy+1] <= H_cut
                R[ix,iy] = 0.0 
                r[ix,iy] = 0.0 
            else 
                R[ix,iy] = 0.0*R[ix,iy]*(1.0 - dmp/min(nx,ny)) + dt*dR[ix,iy] 
                #R[ix,iy] = R[ix,iy]*(dmp/min(nx,ny)) + dt*dR[ix,iy] 
                r[ix,iy] = r[ix,iy] + dt*R[ix,iy]
            end
        end
        if ix ==1 || ix == nx 
            r[ix,iy] = 0.0 
        end 
        if iy == 1 || iy == ny 
            r[ix,iy] = 0.0 
        end 
    end 
    return 
end  

mutable struct AdjointProblem{T<:Real, A<:AbstractArray{T}}
    H::A; H_obs::A; B::A; D::A; qHx::A; qHy::A; β::A; as::A; gradS::A; av_ya_∇Sx::A; av_xa_∇Sy::A
    r::A; dR::A; R::A; Err::A; dR_qHx::A; dR_qHy::A; dR_H::A; dq_D::A; dq_H::A; dD_H::A 
    tmp1::A; tmp2::A; ∂J_∂H::A
    z_ELA::A; b_max::T; nx::Int; ny::Int; dx::T; dy::T; a::T; n::Int; dmp::T; H_cut::T; ϵtol::T; niter::Int; ncheck::Int; threads::Tuple{Int, Int}; blocks::Tuple{Int, Int}
end 


function AdjointProblem(H, H_obs,B, D, qHx, qHy, β, as, gradS, av_ya_∇Sx, av_xa_∇Sy, z_ELA, b_max, nx, ny, dx, dy, a, n, dmp, H_cut, ϵtol, niter, ncheck, threads, blocks)
    dR_qHx = similar(H, (nx-1, ny-2))
    dR_qHy = similar(H,(nx-2, ny-1))
    dR_H = similar(H, (nx, ny))
    dq_D = similar(H, (nx-1, ny-1))
    dq_H = similar(H, (nx, ny))
    dD_H = similar(H, (nx, ny))
    r = similar(H, (nx,ny)) 
    R = similar(H, (nx,ny))
    dR = similar(H, (nx, ny)) 
    Err = similar(H,(nx,ny))
    tmp1 = similar(H, (nx, ny))
    tmp2 = similar(H,(nx,ny))
    ∂J_∂H = similar(H,(nx,ny)) 
    return AdjointProblem(H, H_obs,B, D, qHx, qHy, β, as, gradS, av_ya_∇Sx, av_xa_∇Sy, r, dR, R, Err, dR_qHx, dR_qHy, dR_H, dq_D, dq_H, dD_H, tmp1, tmp2, ∂J_∂H, z_ELA, b_max, nx, ny, dx, dy, a, n, dmp, H_cut, ϵtol, niter, ncheck, threads, blocks)
end 


# solve for r using pseudo-trasient method to compute sensitvity afterwards 
function solve!(problem::AdjointProblem) 
    (; H, H_obs, B, D, qHx, qHy, β, as, gradS, av_ya_∇Sx, av_xa_∇Sy, r, dR, R, Err, dR_qHx, dR_qHy, dR_H, dq_D, dq_H, dD_H, tmp1, tmp2, ∂J_∂H, z_ELA, b_max, nx, ny, dx, dy, a, n, dmp, H_cut, ϵtol, niter, ncheck, threads, blocks) = problem
    r.= 0;  R.= 0; Err .= 0; dR .= 0; dR_qHx .= 0; dR_qHy .= 0; dR_H .= 0; dq_D .= 0; dq_H .= 0; dD_H .= 0

    #dt = min(dx^2, dy^2)/maximum(D)/60.1/2
    #dt = min(dx^2, dy^2)/maximum(D)/60.1
    dt = 1.0/(8.1*maximum(D)/min(dx,dy)^2 + maximum(β))/7000.0
    ∂J_∂H .= (H .- H_obs)#./sqrt(length(H))
    # initialization for tmp1 tmp2 tmp3 tmp4 
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol  && iter < niter 
        #Err .= r 
        # compute dQ/dH 
        dR_qHx .= 0; dR_qHy .= 0 ; dR_H .= 0; dq_D .= 0.0; dq_H .= 0; dD_H .= 0
        dR .= .-∂J_∂H; tmp2 .= r 

        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H_1!(tmp1, tmp2, qHx, dR_qHx, qHy, dR_qHy, β, H, dR_H, B, z_ELA, b_max, nx, ny, dx, dy)
        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H_2!(qHx, dR_qHx, qHy, dR_qHy, D, dq_D, H, dq_H, B, nx, ny, dx, dy)
        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H_3!(D, dq_D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, dD_H, B, a, as, n, nx, ny, dx, dy)
        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H!(dR, dD_H, dq_H, dR_H, nx, ny)
        CUDA.@sync @cuda threads = threads blocks = blocks update_r!(r, R, dR, dt, H, H_cut, dmp, nx, ny)


        # if iter > 10 error("Stop") end
        if iter % ncheck == 0 
            #@. Err -= r 
            #@show(size(dR_qHx))
            merr = maximum(abs.(dt.*R[2:end-1,2:end-1]))/maximum(abs.(r))
            # p1 = heatmap(Array(r'); aspect_ratio=1, title = "r")
            # p2 = heatmap(Array(R'); aspect_ratio=1, title = "R")
            # p3 = heatmap(Array(dR'); aspect_ratio=1, title="dR")
            # # savefig(p1, "adjoint_debug/adjoint_R_$(iter).png")
            # display(plot(p1,p2,p3))
            @printf("error = %.1e\n", merr)
            (isfinite(merr) && merr >0 ) || error("adoint solve failed")
        end 
        iter += 1

    end 
    if iter == niter && merr >= ϵtol 
        error("adjoint solve not converged")
    end 
    @printf("adjoint solve converged: #iter/nx = %.1f, err = %.1e\n", iter/nx, merr)
    return 
end 

# compute 
#compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, a, as, n, nx, ny, dx, dy)
function grad_residual_as_1!(D,tmp2, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, a, as, Jn, n, nx, ny, dx, dy)
    Enzyme.autodiff_deferred(compute_D!, Duplicated(D, tmp2), Const(gradS), Const(av_ya_∇Sx), Const(av_xa_∇Sy),Const(H), Const(B), Const(a), Duplicated(as, Jn), Const(n), Const(nx), Const(ny), Const(dx), Const(dy))
    return
end 

function cost_gradient!(Jn, problem::AdjointProblem)
    (; H, H_obs, B, D, qHx, qHy, β, as, gradS, av_ya_∇Sx, av_xa_∇Sy, r, dR, R, Err, dR_qHx, dR_qHy, dR_H, dq_D, dq_H, dD_H, tmp1, tmp2, ∂J_∂H, z_ELA, b_max, nx, ny, dx, dy, a, n, dmp, ϵtol, niter, ncheck, threads, blocks) = problem
    # dimensionmismatch: array could not be broadcast to match destination 
    Jn .= 0.0
    tmp2 .= r
    dR_qHx .= 0; dR_qHy .= 0; dR_H .= 0; dq_D .= 0; dq_H .= 0; 
    CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H_1!(tmp1, tmp2, qHx, dR_qHx, qHy, dR_qHy, β, H, dR_H, B, z_ELA, b_max, nx, ny, dx, dy)

    CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H_2!(qHx, dR_qHx, qHy, dR_qHy, D, dq_D, H, dq_H, B, nx, ny, dx, dy)
    CUDA.@sync @cuda threads=threads blocks=blocks grad_residual_as_1!(D, -dq_D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, a, as, Jn, n, nx, ny, dx, dy)

    Jn[[1,end],:] = Jn[[2,end-1],:]
    Jn[:,[1,end]] = Jn[:,[2,end-1]]
    @show(maximum(Jn))
    return 
end 


function laplacian!(as, as2, H, nx, ny) 
    @get_thread_idx(H) 
    if ix >= 2 && ix <= size(as,1)-1 && iy >= 2 && iy <= size(as,2)-1
        Δas  = as[ix-1,iy]+as[ix+1, iy]+as[ix,iy-1]+as[ix,iy+1] -4.0*as[ix,iy]
        as2[ix,iy] = as[ix,iy]+1/8*Δas
    end 
    return 
end 



function smooth!(as, as2, H, nx, ny, nsm, threads, blocks)
    for _ = 1:nsm 
        CUDA.@sync @cuda threads=threads blocks=blocks laplacian!(as, as2, H, nx, ny) 
        as2[[1,end],:] .= as2[[2,end-1],:]
        as2[:,[1,end]] .= as2[:,[2,end-1]] 
        as, as2 = as2, as 
    end 
    return 
end 

function adjoint_2D()
    # power law components 
    n        =  3 
    # dimensionally independet physics 
    l        =  1.0#1e4 # length scale lx = 1e3 (natural value)
    aρgn0    =  1.0#1.35e-12 #1.0 # A*(ρg)^n # = 1.3475844936008e-12 (natural value)
    #scales 
    tsc      =  1/aρgn0/l^n # also calculated from natural values tsc = 0.7420684971878533
    #non-dimensional numbers (calculated from natural values)
    s_f_syn  = 0.03#0.0003 # sliding to ice flow ratio: s_f_syn = asρgn0_syn/aρgn0/lx^2
    s_f      = 0.026315789473684213 # sliding to ice flow ratio: s_f   = asρgn0/aρgn0/lx^2
    m_max_nd = 4.706167536706325e-12#m_max/lx*tsc m_max = 2.0 m/a lx = 1e3 tsc = 
    βtsc     = 2.353083768353162e-10#ratio between characteristic time scales of ice flow and accumulation/ablation βtsc = β0*tsc 
    β1tsc    = 3.5296256525297436e-10
    # geometry
    lx_l     = 25.0 #horizontal length to characteristic length ratio
    ly_l     = 20.0 #horizontal length to characteristic length ratio 
    lz_l     = 1.0  #vertical length to charactertistic length ratio
    w1_l     = 100.0 #width to charactertistic length ratio 
    w2_l     = 10.0 # width to characteristic length ratio
    B0_l     = 0.35 # maximum bed rock elevation to characteristic length ratio
    z_ela_l  = 0.215 # ela to domain length ratio z_ela_l = 
    z_ela_1_l = 0.09
    # numerics
    H_cut_l  = 1.0e-6
    # dimensional  dependent physics parameters
    lx          = lx_l*l #250000
    ly          = ly_l*l #200000
    lz          = lz_l*l  #1e3
    w1          = w1_l*l^2 #1e10
    w2          = w2_l*l^2 #1e9
    z_ELA_0     = z_ela_l*l # 2150
    z_ELA_1     = z_ela_1_l*l #900
    B0          = B0_l*l # 3500
    H_cut       = H_cut_l*l # 1.0e-2
    asρgn0_syn  = s_f_syn*aρgn0*l^2 #5.7e-20*(ρg)^n = 5.7e-20*(910*9.81)^3 = 4.055141889402214e-8
    asρgn0      = s_f*aρgn0*l^2 #5.0e-18*(ρg)^n = 3.54627498316e-6
    m_max       = m_max_nd*l/tsc  #2.0 m/a = 6.341958396752917e-8 m/s
    β0          = βtsc/tsc    #0.01 /a = 3.1709791983764586e-10
    β1          = β1tsc/tsc #0.015/3600/24/365 = 4.756468797564688e-10

    @show(asρgn0)
    @show(lx)
    @show(ly)
    @show(lz)
    @show(w1)
    @show(w2)
    @show(B0)
    @show(z_ELA_0)
    @show(m_max)
    @show(β0)
    @show(z_ELA_1)
    @show(β1)
    @show(H_cut)

    # numerics 
    gd_niter    = 100
    bt_niter     = 3 
    nx          = 128
    ny          = 128
    epsi        = 1e-4 
    ϵtol        = (abs = 1e-8, rel = 1e-14)
    dmp_adj     = 2*1.7 
    ϵtol_adj    = 1e-8 
    gd_ϵtol     = 1e-3 
    γ0          = 1.0e-2
    maxiter     = 5*nx^2
    ncheck      = ceil(Int,0.25*nx^2)
    ncheck_adj  = 1000
    threads     = (16,16)
    blocks      = ceil.(Int, (nx,ny)./threads)


    # derived numerics
    dx          = lx/nx 
    dy          = ly/ny
    xc          = LinRange(-lx/2+dx/2, lx/2-dx/2, nx)
    yc          = LinRange(-ly/2+dy/2, ly/2-dy/2, ny)

    # initialization 
    S = zeros(Float64, nx, ny)
    H = zeros(Float64, nx, ny)
    B = zeros(Float64, nx, ny)
    β = β0*ones(Float64, nx, ny)
    ela = z_ELA_0*ones(Float64, nx, ny)

    # β   .+= β1 .*atan.(xc./lx)
    # ela .+=  z_ELA_1.*atan.(yc'./ly .+ 0 .*xc)

    H_obs = copy(H)
    H_ini = copy(H)
    S_obs = copy(S)

    ω = 8
    B = @. B0*(exp(-xc^2/w1 - yc'^2/w2) + exp(-xc^2/w2-(yc'-ly/ω)^2/w1))



    #B = @. B0*(exp(-((xc-x0)/w)^2-((yc'-y0)/w)^2))*sin(ω*pi*(xc+yc'))
    #smoother

    #p1 = plot(xc,yc,B'; st=:surface, camera =(15,30), grid=true, aspect_ratio=1, labelfontsize=9,tickfontsize=7, xlabel="X in (m)", ylabel="Y in (m)", zlabel="Height in (m)", title="Synthetic Glacier bedrock")

    #B[2:end-1, 2:end-1] .= B[2:end-1, 2:end-1] .+ 1.0/4.1.*(diff(diff(B[:, 2:end-1], dims=1),dims=1) .+ diff(diff(B[2:end-1,:], dims=2), dims=2)) 
    #B[[1,end],:] .= B[[2,end-1],:]
    #B[:,[1,end]] .= B[:,[2,end-1]]
    S = CuArray{Float64}(S)
    B = CuArray{Float64}(B)
    H = CuArray{Float64}(H)
    S_obs = CuArray{Float64}(S_obs)
    H_obs = CuArray{Float64}(H_obs)
    H_ini = CuArray{Float64}(H_ini)
    β     = CuArray{Float64}(β)
    ela   = CuArray{Float64}(ela)

    #@show(extrema(B))
    # p1 = heatmap(xc, yc, Array((B)'); title = "B (forward problem initial)")
    # p2 = heatmap(xc, yc, Array(β'); title = "β")
    # p3 = heatmap(xc, yc, Array(ela'); title="ela")
    # #p2 = plot(xc, yc, Array(B'); levels=20, aspect_ratio =1) 
    # #p3 = plot(xc, Array(B[:,ceil(Int, ny/2)]))
    # display(plot(p1, p2, p3))

    D = CUDA.zeros(Float64,nx-1, ny-1)
    av_ya_∇Sx = CUDA.zeros(Float64, nx-1, ny-1)
    av_xa_∇Sy = CUDA.zeros(Float64, nx-1, ny-1) 
    gradS = CUDA.zeros(Float64, nx-1, ny-1)
    qHx = CUDA.zeros(Float64, nx-1,ny-2)
    qHy = CUDA.zeros(Float64, nx-2,ny-1)



    as = asρgn0*CUDA.ones(nx-1, ny-1) 
    as_ini = copy(as)
    as_syn = asρgn0_syn*CUDA.ones(nx-1,ny-1)
    as2 = similar(as) 

    Jn = CUDA.zeros(Float64,nx-1, ny-1)

    #Forwardproblem(H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β, as, n, a, b_max, z_ELA, dx, dy, nx, ny, epsi, cfl, ϵtol, maxiter, ncheck, threads, blocks)
    synthetic_problem = Forwardproblem(H_obs, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β,as_syn, n, aρgn0, m_max, ela, dx, dy, nx, ny, epsi, ϵtol, maxiter, ncheck, threads, blocks)
    # forward_problem = Forwardproblem(H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β,as,n, aρgn0, m_max, ela, dx, dy, nx, ny, epsi, cfl, ϵtol, maxiter, ncheck, threads, blocks)
    # adjoint_problem = AdjointProblem(H, H_obs,B, D, qHx, qHy, β, as, gradS, av_ya_∇Sx, av_xa_∇Sy, ela, m_max, nx, ny, dx, dy, aρgn0, n, dmp_adj, H_cut, ϵtol_adj, maxiter, ncheck_adj, threads, blocks)

    println("generating synthetic data (nx = $nx, ny = $ny)...")
    solve!(synthetic_problem)

    H .= H_obs

    le = 1e-5l

    p1=heatmap(xc, yc, Array((H.+B)'); levels=20, color =:turbo, label="H observation", aspect_ratio = 1)
    contour!(xc, yc, Array(H'); levels=le:le, lw=2.0, color=:black, line=:solid,label="Current state of H")
    contour!(xc, yc, Array(H_obs'); levels=le:le, lw=2.0, color=:red, line=:dash,label="Current state of H")
    p2 = plot(yc, Array(H[nx÷2,:]); xlabel = "y", ylabel = "H")
    display(plot(p1,p2; layout=(2,1), size=(600,980)))


    error("Here")
    println("done.")
    solve!(forward_problem)
    println("gradient descent")

    #S_obs .= B .+ H_obs 
    #S .= B .+ H 
    #CUDA.@sync @cuda threads=threads blocks=blocks update_S!(H_obs, S_obs, B, nx)
    #CUDA.@sync @cuda threads=threads blocks=blocks update_S!(H, S, B, nx)
    γ = γ0
    J_old = 0.0; J_new = 0.0 
    J_old = sqrt(cost(H, H_obs)*dx*dy)
    @show(maximum(abs.(H.-H_obs)))
    J_ini = J_old

    J_evo = Float64[]; iter_evo = Int[]
    # gradient descent iterations to update n 
    for gd_iter = 1:gd_niter 
        as_ini .= as
        solve!(adjoint_problem)
        println("compute cost gradient")
        cost_gradient!(Jn, adjoint_problem)
        println("compute cost gradient done")
        @show(extrema(Jn))

        δ = 0.1*1
        γ = min(γ, δ/maximum(abs.(Jn)))
        @show γ
        #check Jn
        # line search 
        for bt_iter = 1:bt_niter 
            @. as = clamp(as-γ*Jn, 0.0, Inf)
             #mask = (Jn .!= 0)
             #@. as[mask] = exp.(log.(as[mask]) - γ*sign(Jn[mask])*log.(abs(Jn[mask])))
            # p1 = heatmap(xc[1:end-1], yc[1:end-1], Array(as); title="as")
            # display(p1)
            # error("stop")
            smooth!(as, as2, H, nx, ny, 10, threads, blocks)
            forward_problem.H .= H_ini
            # update H 
            solve!(forward_problem)
            #update cost functiion 

            @show(maximum(abs.(H.-H_obs)))
            J_new = sqrt(cost(H, H_obs)*dx*dy)
            if J_new < J_old 
                # we accept the current value of γ
                γ *= 1.1 
                J_old = J_new

                @printf("new solution accepted\n")
                break
            else
                as .= as_ini
                γ = γ*0.5
            end 
        # end of the line search loops 
        end 

        push!(iter_evo, gd_iter); push!(J_evo, J_old/J_ini)
        CUDA.@sync @cuda threads=threads blocks=blocks update_S!(S, H, B, nx, ny)
        if DO_VISU 
             p1=heatmap(xc, yc, Array((H.+B)'); levels=20, color =:turbo, label="H observation", aspect_ratio = 1)
             contour!(xc, yc, Array(H'); levels=0.01:0.01, lw=2.0, color=:black, line=:solid,label="Current state of H")
             contour!(xc, yc, Array(H_obs'); levels=0.01:0.01, lw=2.0, color=:red, line=:dash,label="Current state of H")
             p2 = heatmap(xc, yc, Array(log10.(as)'); xlabel = "x", ylabel = "y", label="as", title = "as", aspect_ratio=1)
             p3 = heatmap(xc[1:end-1], yc[1:end-1], Array(Jn'); xlabel="x", ylabel="y",title = "Gradient of cost function Jn", aspect_ratio=1)
             p4 = plot(iter_evo, J_evo; shape = :circle, xlabel="Iterations", ylabel="J_old/J_ini", title="misfit", yaxis=:log10)
             display(plot(p1,p2,p3,p4; layout=(2,2), size=(980,980)))
             #display(plot(p5,p6; layout=(1,2), size=(980, 980)))
        end 

        # check convergence?
        if J_old/J_ini < gd_ϵtol 
            @printf("gradient descient converge, misfit = %.1e\n", J_old/J_ini)
            break 
        else 
            @printf("#iter = %d, misfit = %.1e\n", gd_iter, J_old/J_ini)
        end 

    end 

    return 
end 

adjoint_2D() 