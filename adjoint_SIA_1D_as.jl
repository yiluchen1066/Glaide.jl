using CUDA,BenchmarkTools
using Plots,Plots.Measures,Printf
using DelimitedFiles
using Enzyme 
default(size=(800,600),framestyle=:box,label=false,grid=false,margin=10mm,lw=4,labelfontsize=9,tickfontsize=9,titlefontsize=12)

const DO_VISU = true 
macro get_thread_idx(A)  esc(:( begin ix =(blockIdx().x-1) * blockDim().x + threadIdx().x; end )) end 
macro av_xa(A)    esc(:( 0.5*($A[ix]+$A[ix+1]) )) end
macro d_xa(A)     esc(:( $A[ix+1]-$A[ix])) end 

CUDA.device!(7) # GPU selection

function compute_error_1!(Err, H, nx)
    @get_thread_idx(H)
    if ix <= nx 
        Err[ix] = H[ix]
    end 
    return 
end 

function compute_error_2!(Err, H, nx)
    @get_thread_idx(H) 
    if ix <= nx 
        Err[ix] = Err[ix] - H[ix]
    end 
    return 
end

function cost!(H, H_obs, J, nx)
    @get_thread_idx(H)
    if ix <= nx 
        J += (H[ix]-H_obs[ix])^2
    end 
    J *= 0.5 
    return 
end

cost(H, H_obs) = 0.5*sum((H.-H_obs).^2)

# function compute_diffusion!(B, H, D, ∇Sx, nx, dx, a, as, n)
#     @get_thread_idx(H)
#     if ix <= nx-1 
#         ∇Sx[ix] = @d_xa(B)/dx + @d_xa(H)/dx
#         D[ix] = (a*@av_xa(H)^(n+2)+as*@av_xa(H)^n)*∇Sx[ix]^(n-1)
#     end 
#     return 
# end 

function compute_D!(D,B,H, ∇Sx,nx, dx, a, as, n)
    @get_thread_idx(H)
    if ix <=nx-1 
        ∇Sx[ix] = @d_xa(B)/dx + @d_xa(H)/dx
        D[ix] = (a*@av_xa(H)^(n+2)+as*@av_xa(H)^n)*∇Sx[ix]^(n-1)
    end 
    return 
end 


function flux_q!(qHx, B, H, D, ∇Sx, nx, dx, a, as, n)
    @get_thread_idx(H)
    if ix <= nx-1
        #∇Sx[ix] = @d_xa(B)/dx + @d_xa(H)/dx
        #D[ix] = (a*@av_xa(H)^(n+2)+as*@av_xa(H)^n)*∇Sx[ix]^(n-1)
        #qHx[ix] = -D[ix]*(@d_xa(B)/dx + @d_xa(H)/dx)
        qHx[ix]  = -(a*@av_xa(H)^(n+2)+as*@av_xa(H)^n)*(@d_xa(B)/dx + @d_xa(H)/dx)^(n-1)*(@d_xa(B)/dx + @d_xa(H)/dx)
    end 
    return 
end 


#function compute_M!(M, H, B, β, b_max, z_ELA, nx)
#    @get_thread_idx(S) 
#    if ix <= nx
#        M[ix] = min(β*(H[ix]+B[ix]-z_ELA), b_max)
#    end 
#    return 
#end 

#cost(H, H_obs) 

#cost(H, H_obs) = 0.5*sum((H.-H_obs).^2)

#M[ix+1] = min(β*(H[ix+1]+B[ix+1]-z_ELA), b_max)
function residual!(RH, H, B, qHx, β, b_max, z_ELA, dx, nx) 
    @get_thread_idx(H)
    if ix <= nx-2 
        #RH[ix] = -@d_xa(qHx)/dx
        RH[ix+1] = -@d_xa(qHx)/dx+min(β[ix+1]*(H[ix+1]+B[ix+1]-z_ELA),b_max)
        # RH[ix+1] = -@d_xa(qHx)/dx+β*(H[ix+1]+B[ix+1]-z_ELA)
    end 
    return 
end

function timestep!(D, dτ, nx, epsi, cfl)
    @get_thread_idx(D)
    if ix <= nx-2 
        dτ[ix] = 0.5*min(1.0, cfl/(epsi+@av_xa(D)))
    end 
    return 
end 


function update_H!(H,  dτ, dHdτ, RH, nx, damp)
    @get_thread_idx(H) 
    if ix <= nx-2 
        dHdτ[ix] = dHdτ[ix]*damp + RH[ix+1]
        H[ix+1]  = max(0.0, H[ix+1] + dτ[ix]*dHdτ[ix])
    end 
    return 
end 

function set_BC!(H, nx)
    @get_thread_idx(H)
    if ix == 1 
        H[ix] = H[ix+1]
    end 
    if ix == nx 
        H[ix] = H[ix-1]
    end 
    return 
end 


function update_S!(H, S, B, nx)
    @get_thread_idx(H)
    if ix <= nx 
        S[ix] = H[ix] + B[ix]
    end 
    return 
end


mutable struct Forwardproblem{T<:Real, A<:AbstractArray{T}}
    H::A; B::A; D::A; ∇Sx::A; qHx::A; β::A
    dτ::A; dHdτ::A; RH::A; Err::A
    n::Int; a::T; as::T; b_max::T; z_ELA::Int; dx::T; nx::Int; epsi::T; cfl::T; ϵtol::T; niter::Int; ncheck::Int; threads::Int; blocks::Int; dmp::T
end 

function Forwardproblem(H, B, D, ∇Sx, qHx, β, n, a, as, b_max, z_ELA, dx, nx, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp)
    RH = similar(H,nx)
    dHdτ = similar(H, nx-2) 
    dτ   = similar(H, nx-2) 
    Err  = similar(H, nx)
    return Forwardproblem(H, B, D, ∇Sx, qHx, β,dτ, dHdτ, RH, Err, n, a, as, b_max, z_ELA, dx, nx, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp) 
end 

#solve for H using pseudo-trasient method 
function solve!(problem::Forwardproblem)
    (;H, B, D, ∇Sx, qHx, β, dτ, dHdτ,RH, Err, n, a, as, b_max, z_ELA, dx, nx, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp) = problem
    dHdτ .= 0; RH .= 0; dτ .= 0; Err .= 0
    merr = 2ϵtol; iter = 1 
    lx = 30e3
    @show(size(RH))
    #xc = LinRange(dx/2, lx-dx/2, nx)
    #p1 = plot(xc, Array(H); title = "H_init (forward problem)")
    #display(plot(p1))
    while merr >= ϵtol && iter < niter 
        #Err .= H 
        #CUDA.@sync @cuda threads=threads blocks=blocks residual!(RH, H, B, qHx, β, b_max, z_ELA, dx, nx)
        CUDA.@sync @cuda threads=threads blocks=blocks compute_error_1!(Err, H, nx)
        CUDA.@sync @cuda threads=threads blocks=blocks compute_D!(D,B,H, ∇Sx,nx, dx, a, as, n)
        CUDA.@sync @cuda threads=threads blocks=blocks flux_q!(qHx, B, H, D, ∇Sx, nx, dx, a, as, n)
        CUDA.@sync @cuda threads=threads blocks=blocks residual!(RH, H, B, qHx, β, b_max, z_ELA, dx, nx)   # compute RH 
        CUDA.@sync @cuda threads=threads blocks=blocks timestep!(D, dτ, nx, epsi, cfl)
        #CUDA.@sync @cuda threads=threads blocks=blocks update_residual!(S, H, D, ∇Sx, qHx, dτ, dHdτ, RH, nx, dx, a, as, n, epsi, cfl, damp)
        CUDA.@sync @cuda threads=threads blocks=blocks update_H!(H,  dτ, dHdτ, RH, nx, dmp) # update H
        CUDA.@sync @cuda threads=threads blocks=blocks set_BC!(H, nx) 

        if iter  % ncheck == 0 
            #@. Err -= H 
            CUDA.@sync @cuda threads=threads blocks =blocks compute_error_2!(Err, H, nx)
            #merr = maximum(abs.(Err[:]))
            merr = (sum(abs.(Err[:]))./nx)
            (isfinite(merr) && merr >0) || error("forward solve failed")
            #@printf("error = %1.e\n", merr)
            #p1 = plot(xc, Array(H+B); title = "S (forward problem)", ylims=(0,1000))
            #plot!(xc, Array(B); title = "S (forward problem)", ylims=(0,1000))
            #display(plot(p1))
        end 
        iter += 1 
    end 
    if iter == niter && merr >= ϵtol
        error("forward solve not converged")
    end 
    @printf("forward solve converged: #iter/nx = %.1f, err = %.1e\n", iter/nx, merr)
    return 
end 

#residual!(RH, H, B, qHx, β, b_max, z_ELA, dx, nx) 
#compute (dR/dq)^T
function grad_residual_H_1!(tmp1, tmp2, H, B, qHx, dR_q, β, b_max, z_ELA, dx, nx)
    Enzyme.autodiff_deferred(residual!, Duplicated(tmp1, tmp2), Const(H), Const(B), Duplicated(qHx, dR_q), Const(β), Const(b_max), Const(z_ELA), Const(dx), Const(nx))
    return  
end 
#flux_q!(qHx, B, H, D, ∇Sx, nx, dx, a, as, n)
function grad_residual_H_2!(tmp3, dR_q, B, H, dQ_h, D, dQ_d, dQ_S, qHx, ∇Sx, nx, dx, a, as, n)
    Enzyme.autodiff_deferred(flux_q!,Duplicated(tmp3, dR_q),Const(B), Duplicated(H, dQ_h), Const(D), Const(∇Sx),Const(nx), Const(dx), Const(a), Const(as), Const(n))
    return 
end
    
function grad_residual_H_3!(tmp4, tmp2, H, dR_h, B, qHx, β, b_max, z_ELA, dx, nx)
    Enzyme.autodiff_deferred(residual!, Duplicated(tmp4, tmp2), Duplicated(H, dR_h),Const(B), Const(qHx), Const(β), Const(b_max), Const(z_ELA), Const(dx), Const(nx))
    return 
end 


function grad_residual_H_4!(dR, dQ_h, dR_h, nx) 
    @get_thread_idx(dR) 
    if ix <= nx 
        dR[ix] += dQ_h[ix] + dR_h[ix]
    end 
    return 
end 

function update_r!(r, dR, R, H, dt, dmp, nx) 
    @get_thread_idx(H)
    if ix <= nx 
        if H[ix] <= 1e-2
            R[ix] = 0.0 
            r[ix] = 0.0 
        else 
            R[ix] = R[ix]*(1.0 - dmp/nx) + dt*dR[ix] 
            r[ix] += dt*R[ix]
        end 
        if ix == 1 || ix == nx 
            r[ix] = 0.0 # boundary conditions of r 
        end 
    end 
    return 
end

function update_r_1!(r, dR, R, H, dt, dmp, nx)
    @get_thread_idx(H)
    if ix <= nx 
        R[ix] = R[ix]*(1.0 - dmp/nx) + dt*dR[ix]
        r[ix] += dt*R[ix]
    end 
    if ix == 1 || ix == nx 
        r[ix] = 0.0 
    end 

    return 
end 


mutable struct AdjointProblem{T<:Real, A<:AbstractArray{T}}
    H::A; H_obs::A; B::A; D::A; ∇Sx::A; qHx::A;β::A
    r::A; dR::A; R::A; dR_q::A; dQ_h::A; dR_h::A; dQ_d::A; dQ_S::A; Err::A; tmp1::A; tmp2::A; tmp3::A; tmp4::A; ∂J_∂H::A
    nx::Int; dx::T; as::T; a::T; n::Int; b_max::T; z_ELA::Int; dmp::T; ϵtol::T; niter::Int; ncheck::Int; threads::Int; blocks::Int
end 

function AdjointProblem(H, H_obs,B, D, ∇Sx, qHx, β,nx, dx, as, a, n, b_max, z_ELA, dmp, ϵtol, niter, ncheck, threads, blocks)
    dR_q = similar(H, nx-1) 
    dQ_h = similar(H, nx)
    dR_h = similar(H, nx)
    dQ_d  = similar(H, nx-1)
    dQ_S = similar(H, nx-1)
    r = similar(H, nx) 
    R = similar(H, nx)
    dR = similar(H, nx) 
    Err = similar(H)
    tmp1 = similar(H) 
    tmp2 = similar(H)
    tmp3 = similar(H) 
    tmp4 = similar(H)
    ∂J_∂H = similar(H) 
    return AdjointProblem(H, H_obs,B, D, ∇Sx, qHx, β, r, dR, R, dR_q, dQ_h, dR_h, dQ_d, dQ_S, Err, tmp1, tmp2, tmp3, tmp4, ∂J_∂H, nx, dx, as, a, n, b_max, z_ELA, dmp, ϵtol, niter, ncheck, threads, blocks)
end 


# solve for r using pseudo-trasient method to compute sensitvity afterwards 
function solve!(problem::AdjointProblem) 
    (; H, H_obs, B, D, ∇Sx, qHx, β,r, dR, R, dR_q, dQ_h, dQ_d, dQ_S, dR_h, Err, tmp1, tmp2, tmp3, tmp4, ∂J_∂H, nx, dx, as, a, n, b_max, z_ELA, dmp, ϵtol, niter, ncheck, threads, blocks) = problem
    r.= 0;  R.= 0; Err .= 0; dR_q .= 0; dQ_h .= 0; dQ_d .= 0; dQ_S .= 0; dR_h .=0; dR .= 0 
    dt = dx^2/maximum(D)/4.1
    lx = 30e3
    dx = lx/nx
    xc = LinRange(dx/2, lx-dx/2, nx)
    @. ∂J_∂H = H - H_obs
    # initialization for tmp1 tmp2 tmp3 tmp4 
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol  && iter < niter 
        #Err .= r 
        # compute dQ/dH 
        dR_q .= 0; dQ_h .= 0; dR_h.= 0
        dR .= .-∂J_∂H; tmp2 .= r 
        #println("auto differentiation procedure")
        #@show(size(dR_q))
        #@show(size(qHx))ß
        #@show(size(r))
        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H_1!(tmp1, tmp2, H, B, qHx, dR_q, β, b_max, z_ELA, dx, nx)
        #println("grad_reidual_1")
        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H_2!(tmp3, dR_q, B, H, dQ_h, D, dQ_d, dQ_S, qHx, ∇Sx, nx, dx, a, as, n)
        tmp2 .= r
        # @show 
        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H_3!(tmp4, tmp2, H, dR_h, B, qHx, β, b_max, z_ELA, dx, nx)
        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H_4!(dR, dQ_h, dR_h, nx)  
        #dR .= .-∂J_∂H
        CUDA.@sync @cuda threads = threads blocks = blocks update_r!(r, dR, R, H, dt, dmp, nx)
        # if iter > 10 error("Stop") end
        if iter % ncheck == 0 
            #@. Err -= r 
            merr = maximum(abs.(R[2:end-1]))
            #p1 = plot(xc, Array(R); title = "R")
            # savefig(p1, "adjoint_debug/adjoint_R_$(iter).png")
            #display(p1)
            #@printf("error = %.1e\n", merr)
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
 
#residual!(RH, H, B, qHx, β, b_max, z_ELA, dx, nx) 
function grad_residual_β!(tmp1, tmp2, H, B, qHx, β, Jn, b_max, z_ELA, dx, nx)
    Enzyme.autodiff_deferred(residual!,Duplicated(tmp1, tmp2), Const(H), Const(B), Const(qHx), Duplicated(β, Jn), Const(b_max), Const(z_ELA), Const(dx), Const(nx))
    return 
end

function cost_gradient!(Jn, problem::AdjointProblem)
    (; H, H_obs, B, D, ∇Sx, qHx, r, dR, R, dR_q, dQ_h, dR_h, Err, tmp1, tmp2, tmp3, tmp4, ∂J_∂H, nx, dx, as, a, n, β, b_max, z_ELA, dmp, ϵtol, niter, ncheck, threads, blocks) = problem
    Jn .= 0.0; tmp2 .= -r
    CUDA.@sync @cuda threads= threads blocks=blocks grad_residual_β!(tmp1, tmp2, H, B, qHx, β, Jn, b_max, z_ELA, dx, nx)
    
    Jn[[1,end]] .= Jn[[2,end-1]]
    return 
end 


function grad_residual_as_1!(tmp1, tmp2, B, H, D, ∇Sx, nx, dx, a, as, Jn, n)
    Enzyme.autodiff_deferred(flux_q!,Duplicated(tmp1, tmp2),  Const(B), Const(H), Const(D), Const(∇Sx), Const(nx), Const(dx), Const(a), Duplicated(as, Jn), Const(n))
    return 
end 

function cost_gradient!(Jn, problem::AdjointProblem)
    (; H, H_obs, B, D, ∇Sx, qHx, r, dR, R, dR_q, dQ_h, dR_h, Err, tmp1, tmp2, tmp3, tmp4, ∂J_∂H, nx, dx, as, a, n, β, b_max, z_ELA, dmp, ϵtol, niter, ncheck, threads, blocks) = problem
    Jn .= 0; tmp2 .= -dR_q
    CUDA.@sync @cuda threads=threads blocks=blocks grad_residual_as_1!(tmp1, tmp2, B, H, D, ∇Sx, nx, dx, a, as, Jn, n)
    Jn[[1,end]] .= Jn[[2,end-1]]
    return 
end 

function laplacian!(β,β2,nx)
    @get_thread_idx(β)
    if ix >= 2 && ix <= nx-1 
        β2[ix] = β[ix] + 0.2*(β[ix-1]+β[ix+1]-2.0*β[ix])
    end
    return
end

function smooth!(β,β2,nx,nsm, threads, blocks)
    for _ = 1:nsm 
        CUDA.@sync @cuda threads=threads blocks=blocks laplacian!(β,β2,nx)
        β[[1,end]] .= β[[2,end-1]]
        β, β2 = β2, β
    end 
    return 
end 



function adjoint_1D()
    # physics parameters
    s2y = 3600*24*365.35 # seconds to years 
    lx = 30e3
    w = 0.15*lx 
    n = 3
    ρg = 970*9.8 
    a0 = 1.5e-24
    z_ELA = 300 
    b_max = 0.1
    B0 = 500 

    #numerics parameters 
    gd_niter = 100
    bt_niter = 10
    nx = 512
    epsi = 1e-2
    dmp = 0.8
    dmp_adj = 1.7 
    ϵtol = 1e-8
    gd_ϵtol = 1e-3
    γ0 = 1.0e-10
    niter = 100000
    ncheck = 100
    threads = 16 
    blocks = ceil(Int, nx/threads)

    #derived numerics 
    dx = lx/nx
    xc = LinRange(dx/2, lx-dx/2, nx)
    x0 = xc[Int(nx/2)]

    # derived physics 
    a  = 2.0*a0/(n+2)*ρg^n*s2y 
    as = 5.7e-20 
    cfl = dx^2/4.1

    # initialization 
    S = zeros(Float64, nx)
    H = zeros(Float64, nx)
    B = zeros(Float64, nx)

    #H = @. exp(-(xc - lx/4)^2)
    #H = @. exp(-(xc - x0)^2)
    H    = @. 200*exp(-((xc-x0)/5000)^2) 
    H_obs = copy(H)
    H_ini = copy(H)
    S_obs = copy(S)

    
    B = @. B0*(exp(-(xc-x0)^2/w^2))
    #smoother 
    B[2:end-1] .= B[2:end-1] .+ 1.0/4.1.*(diff(diff(B[:])))
    B[[1,end]] .= B[[2,end-1]]
    S = CuArray{Float64}(S)
    B = CuArray{Float64}(B)
    H = CuArray{Float64}(H)
    S_obs = CuArray{Float64}(S_obs)
    H_obs = CuArray{Float64}(H_obs)
    H_ini = CuArray{Float64}(H_ini)

    D = CUDA.zeros(Float64,nx-1)
    ∇Sx = CUDA.zeros(Float64,nx-1)
    qHx = CUDA.zeros(Float64, nx-1)
    Jn  = CUDA.zeros(Float64, nx)

    #S_obs .= B .+ H_obs
    #CUDA.@sync @cuda threads=threads blocks=blocks update_S!(H_obs, S_obs, B, nx)
    #S .= B .+ H 
    #CUDA.@sync @cuda threads=threads blocks=blocks update_S!(H, S, B, nx)
    
    # how to define β_init β_syn 
    β = 0.002*CUDA.ones(nx)
    β_ini = copy(β)
    β_syn  = 0.0015*CUDA.ones(nx)
    β2 = similar(β)

    Jn = CUDA.zeros(Float64,nx)
    # display
    # synthetic problem
    
    synthetic_problem = Forwardproblem(H_obs, B, D, ∇Sx, qHx, β_syn,n, a, as, b_max, z_ELA, dx, nx, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp)
    forward_problem = Forwardproblem(H, B, D, ∇Sx, qHx, β,n, a, as, b_max, z_ELA, dx, nx, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp)
    adjoint_problem = AdjointProblem(H, H_obs,B, D, ∇Sx, qHx, β, nx, dx, as, a, n, b_max, z_ELA, dmp_adj, ϵtol, niter, ncheck, threads, blocks)

    println("generating synthetic data (nx = $nx)...")
    solve!(synthetic_problem)
    println("done.")
    solve!(forward_problem)
    H_ini .= forward_problem.H


    println("gradient descent")



    #print(H)
    #print(H_obs)

    #S_obs .= B .+ H_obs 
    #S .= B .+ H 
    #CUDA.@sync @cuda threads=threads blocks=blocks update_S!(H_obs, S_obs, B, nx)
    #CUDA.@sync @cuda threads=threads blocks=blocks update_S!(H, S, B, nx)
    γ = γ0
    J_old = 0.0; J_new = 0.0 
    J_old = sqrt(cost(H, H_obs)*dx)
    #CUDA.@sync @cuda threads=threads blocks=blocks cost!(H, H_obs, J_old, nx)
    #@printf("initial cost function equals %.1e\n", J_old)
    J_ini = J_old
    
    J_evo = Float64[]; iter_evo = Int[]
    # gradient descent iterations to update n 
    for gd_iter = 1:gd_niter 
        β_ini .= β
        println("solve adjoint problem")
        solve!(adjoint_problem)
        cost_gradient!(Jn, adjoint_problem)

        # line search 
        for bt_iter = 1:bt_niter 
            # update β
            @. β = clamp(β-γ*Jn, 0.0, 100.0)
            #p1 = plot(xc, Array(β); title="β")
            #display(p1)
            β[[1,end]] .= β[[2,end-1]]
            smooth!(β,β2,nx,400,threads,blocks)
            forward_problem.H .= H_ini
            # update H 
            solve!(forward_problem)
            #update cost functiion 
            J_new = sqrt(cost(H, H_obs)*dx)
            @show(J_new)
            if J_new < J_old 
                # we accept the current value of γ
                γ *= 1.1 
                J_old = J_new
                
                @printf("new solution accepted\n")
                break
            else
                β .= β_ini
                γ = max(γ*0.5, γ0* 0.1)
                #γ = γ*0.5 
                #if γ < γ0*0.01 
                #    error("minimal value of γ is reached")
                #end 

            end 
        # end of the line search loops 
        end 

        push!(iter_evo, gd_iter); push!(J_evo, J_old/J_ini)
        CUDA.@sync @cuda threads=threads blocks=blocks update_S!(H, S, B, nx)
        if DO_VISU 
             p1 = plot(xc, Array(H_ini); xlabel="x", ylabel="H (m)", label= "Initial state of H" ,title = "Ice thickness H(m)")
             plot!(xc, Array(H_obs); label = "Synthetic value of H")
             plot!(xc, Array(H); label="Current state of H")
             p2 = plot(xc, Array(β_ini); xlabel = "x", ylabel = "β", label="Initial state of β", title = "β")
             plot!(xc, Array(β_syn); label="Syntheic value of β")
             plot!(xc,Array(β); label="Current state of β")
             p3 = plot(xc, Array(Jn); xlabel="x", ylabel="Cost function Jn",title = "Gradient of cost function Jn")
             p4 = plot(iter_evo, J_evo; shape = :circle, xlabel="Iterations", ylabel="J_old/J_ini", title="misfit", yaxis=:log10)
             display(plot(p1, p2, p3, p4; layout=(2,2), size=(980, 980)))
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

adjoint_1D() 