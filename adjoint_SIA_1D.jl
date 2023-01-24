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

function flux_q!(qHx, S, H, D, ∇Sx, nx, dx, a, as, n)
    @get_thread_idx(S)
    if ix <= nx-1 
        ∇Sx[ix] = @d_xa(S)/dx 
        D[ix] = (a*@av_xa(H)^(n+2)+as*@av_xa(H)^n)*∇Sx[ix]^(n-1)
        qHx[ix] = -D[ix]*@d_xa(S)/dx 
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



#M[ix+1] = min(β*(H[ix+1]+B[ix+1]-z_ELA), b_max)
function residual!(RH, S, H, B, D, ∇Sx, qHx, n, a, as, β, b_max, z_ELA, dx, nx) 
    @get_thread_idx(S)
    if ix <= nx-2 
        #RH[ix] = -@d_xa(qHx)/dx
        RH[ix] = -@d_xa(qHx)/dx+min(β*(H[ix+1]+B[ix+1]-z_ELA),b_max)
    end 
    return 
end 


function timestep!(S, H, D, ∇Sx, qHx, dτ, nx, dx, a, as, n, epsi, cfl)
    @get_thread_idx(S)
    if ix <= nx-2 
        dτ[ix] = 0.5*min(1.0, cfl/(epsi+@av_xa(D)))
    end 
    return 
end 

function update_H!(S, H, D, ∇Sx, qHx, dτ, dHdτ, RH, nx, dx, a, as, n, epsi, cfl, damp)
    @get_thread_idx(H) 
    if ix <= nx-2 
        dHdτ[ix] = dHdτ[ix]*damp+RH[ix]
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
    S::A; H::A; B::A; D::A; ∇Sx::A; qHx::A
    dτ::A; dHdτ::A; RH::A; Err::A
    n::Int; a::T; as::T; β::T; b_max::T; z_ELA::Int; dx::T; nx::Int; epsi::T; cfl::T; ϵtol::T; niter::Int; ncheck::Int; threads::Int; blocks::Int; dmp::T
end 

function Forwardproblem(S, H, B, D, ∇Sx, qHx, n, a, as, β, b_max, z_ELA, dx, nx, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp)
    RH = similar(H, nx-2)
    dHdτ = similar(H, nx-2) 
    dτ   = similar(H, nx-2) 
    Err  = similar(H)
    return Forwardproblem(S, H, B, D, ∇Sx, qHx, RH, dHdτ, dτ, Err, n, a, as, β, b_max, z_ELA, dx, nx, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp)
end 

#solve for H using pseudo-trasient method 
function solve!(problem::Forwardproblem)
    (;S, H, B, D, ∇Sx, qHx, RH, dHdτ, dτ, Err, n, a, as, β, b_max, z_ELA, dx, nx, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp) = problem
    dHdτ .= 0; RH .= 0; dτ .= 0; Err .= 0
    merr = 2ϵtol; iter = 1 
    lx = 30e3
    #xc = LinRange(dx/2, lx-dx/2, nx)
    #p1 = plot(xc, Array(H); title = "H_init (forward problem)")
    #display(plot(p1))
    while merr >= ϵtol && iter < niter 
        #Err .= H 
        CUDA.@sync @cuda threads=threads blocks=blocks update_S!(H, S, B, nx)
        CUDA.@sync @cuda threads=threads blocks=blocks compute_error_1!(Err, H, nx)
        CUDA.@sync @cuda threads=threads blocks=blocks flux_q!(qHx, S, H, D, ∇Sx, nx, dx, a, as, n)
        CUDA.@sync @cuda threads=threads blocks=blocks residual!(RH, S, H, B, D, ∇Sx, qHx, n, a, as, β, b_max, z_ELA, dx, nx)   # compute RH 
        CUDA.@sync @cuda threads=threads blocks=blocks timestep!(S, H, D, ∇Sx, qHx, dτ, nx, dx, a, as, n, epsi, cfl)
        CUDA.@sync @cuda threads=threads blocks=blocks update_H!(S, H, D, ∇Sx, qHx, dτ, dHdτ, RH, nx, dx, a, as, n, epsi, cfl, dmp) # update H
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


#compute (dR/dH)^T
function grad_residual_H!(tmp1, tmp2, tmp3, tmp4, S, H, B, D, ∇Sx, qHx, dR_q, dQ_h, dR_h, dR, n, a, as, β, b_max, z_ELA, dx, nx)
    @get_thread_idx(H)
    Enzyme.autodiff_deferred(residual!, Duplicated(tmp1, tmp2), Const(S), Const(H), Const(B), Const(D), Const(∇Sx), Duplicated(qHx, dR_q), Const(n), Const(a), Const(as), Const(β), Const(b_max), Const(z_ELA), Const(dx), Const(nx))
    Enzyme.autodiff_deferred(flux_q!,Duplicated(tmp3, dR_q),Const(S), Duplicated(H, dQ_h), Const(D), Const(∇Sx), Const(nx), Const(dx), Const(a), Const(as), Const(n))
    Enzyme.autodiff_deferred(residual!, Duplicated(tmp4, tmp2), Const(S), Duplicated(H, dR_h),Const(D), Const(∇Sx), Const(qHx), Const(n), Const(a), Const(as), Const(β), Const(b_max), Const(z_ELA), Const(dx), Const(nx))
    #dR .= dQ_h + dR_h
    if ix <= nx 
        dR[ix] = dQ_h[ix] + dR_h[ix]
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

mutable struct AdjointProblem{T<:Real, A<:AbstractArray{T}}
    S::A; H::A; H_obs::A; B::A; D::A; ∇Sx::A; qHx::A;
    r::A; dR::A; R::A; dR_q::A; dQ_h::A; dR_h::A; Err::A; tmp1::A; tmp2::A; tmp3::A; tmp4::A; ∂J_∂H::A
    nx::Int; dx::T; as::T; a::T; n::Int; β::T; b_max::T; z_ELA::Int; dmp::T; ϵtol::T; niter::Int; ncheck::Int; threads::Int; blocks::Int
end 

function AdjointProblem(S, H, H_obs,B, D, ∇Sx, qHx, nx, dx, as, a, n, β, b_max, z_ELA, dmp, ϵtol, niter, ncheck, threads, blocks)
    dR_q = similar(H) 
    dQ_h = similar(H)
    dR_h = similar(H)
    r = similar(H) 
    R = similar(H)
    dR = similar(H) 
    Err = similar(H)
    tmp1 = similar(H) 
    tmp2 = similar(H)
    tmp3 = similar(H) 
    tmp4 = similar(H)
    ∂J_∂H = similar(H) 
    return AdjointProblem(S, H, H_obs,B, D, ∇Sx, qHx, r, dR, R, dR_q, dQ_h, dR_h, Err, tmp1, tmp2, tmp3, tmp4, ∂J_∂H, nx, dx, as, a, n, β, b_max, z_ELA, dmp, ϵtol, niter, ncheck, threads, blocks)
end 


# solve for r using pseudo-trasient method to compute sensitvity afterwards 
function solve!(problem::AdjointProblem) 
    (; S, H, H_obs, B, D, ∇Sx, qHx, r, R, dR, dR_q, dQ_h, dR_h, Err, tmp1, tmp2, tmp3, tmp4, ∂J_∂H, nx, dx, as, a, n, β, b_max, z_ELA, dmp, ϵtol, niter, ncheck, threads, blocks) = problem
    r.= 0; dR .= 0; R.= 0; Err .= 0; dR_q .= 0; dQ_h .= 0; dR_h.= 0
    dt = dx/ 3.0 
    @. ∂J_∂H = H - H_obs
    # initialization for tmp1 tmp2 tmp3 tmp4 
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol  && iter < niter 
        Err .= r 
        # compute dQ/dH 
        tmp2 .= r 
        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H!(tmp1, tmp2, tmp3, tmp4, S, H, B, D, ∇Sx, qHx, dR_q, dQ_h, dR_h, dR, n, a, as, β, b_max, z_ELA, dx, nx)
        dR .= .-∂J_∂H
        CUDA.@sync @cuda threads = threads blocks = blocks update_r!(r, dR, R, H, dt, dmp, nx)
        if iter % ncheck == 0 
            @. Err -= r 
            merr = maximum(abs.(Err))
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

function grad_residual_β(tmp1, tmp2, S, H, B, D, ∇Sx, qHx, n, a,as, β, dR_β, b_max, z_ELA, dx, nx)
    Enzyme.autodiff_deferred(residual!,Duplicated(tmp1, tmp2), Const(S), Const(H), Const(B), Const(D), Const(∇Sx), Const(qHx), Const(n), Const(a), Const(as), Duplicated(β, dR_β), Const(b_max), Const(z_ELA), Const(dx), Const(nx))
    return 
end

function cost_gradient!(Jn, problem::AdjointProblem)
    (;r,tmp1, tmp2, S, H, B, D, ∇Sx, qHx, n, a, as, β, dR_β, b_max, z_ELA, dx, nx)=problem
    Jn .= 0.0; tmp2 .= -r
    CUDA.@sync @cuda threads= threads blocks=blocks grad_residual_β!(tmp1, tmp2, S, H, B, D, ∇Sx, qHx, n, a,as, β, dR_β, b_max, z_ELA, dx, nx)
    Jn[[1,end]] .= Jn[[2,end-1]]
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
    bt_niter = 3
    nx = 512
    epsi = 1e-2
    dmp = 0.8
    dmp_adj = 1.7 
    ϵtol = 1e-8
    γ0 = 1.0
    niter = 100000
    ncheck = 1000
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
    CUDA.@sync @cuda threads=threads blocks=blocks update_S!(H_obs, S_obs, B, nx)
    #S .= B .+ H 
    CUDA.@sync @cuda threads=threads blocks=blocks update_S!(H, S, B, nx)
    
    # how to define β_init β_syn 
    β = 0.002 
    β_init = β
    β_syn  = β

    # display
    # synthetic problem
    synthetic_problem = Forwardproblem(S_obs, H_obs, B, D, ∇Sx, qHx, n, a, as, β_syn, b_max, z_ELA, dx, nx, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp)
    forward_problem = Forwardproblem(S, H, B, D, ∇Sx, qHx, n, a, as, β, b_max, z_ELA, dx, nx, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp)
    adjoint_problem = AdjointProblem(S, H, H_obs,B, D, ∇Sx, qHx, nx, dx, as, a, n, β, b_max, z_ELA, dmp_adj, ϵtol, niter, ncheck, threads, blocks)

    println("generating synthetic data (nx = $nx)...")
    solve!(synthetic_problem)
    println("done.")
    solve!(forward_problem)
    println("gradient descent")


    #S_obs .= B .+ H_obs 
    #S .= B .+ H 
    #CUDA.@sync @cuda threads=threads blocks=blocks update_S!(H_obs, S_obs, B, nx)
    #CUDA.@sync @cuda threads=threads blocks=blocks update_S!(H, S, B, nx)

    J_old = 0.0; J_new = 0.0 
    CUDA.@sync @cuda threads=threads blocks=blocks cost!(H, H_obs, J_old, nx)
    J_ini = J_old
    γ = γ0
    J_evo = Float64[]; iter_evo = Int[]
    # gradient descent iterations to update n 
    for gd_iter = 1:gd_niter 
        β_init = β
        solve!(adjoint_problem)
        cost_gradient!(Jn, adjoint_problem)
        # line search 
        for bt_iter = 1:bt_niter 
            # update β
            β = min(β-γ*Jn, 100.0)
            forward_problem.H = H_ini
            # update H 
            solve!(forward_problem)
            #update cost functiion 
            cost!(H, H_obs, J_new,nx)
            if J_new < J_old 
                # we accept the current value of γ
                γ *= 1.1 
                J_old = J_new
                break
            else
                β = β_init
                γ = max(γ*0.5, γ0* 0.1)
            end 
        # end of the line search loops 
        end 
        push!(iter_evo, gd_iter); push!(J_evo, J_old/J_ini)
        CUDA.@sync @cuda threads=threads blocks=blocks update_S!(H, S, B, nx)
        if DO_VISU 
            p1 = heatmap(xc, Array(S'); title = "S", aspect_ratio =1)
            p2 = heatmap(xc, Array(β'); title = "β", aspect_ratio =1)
            p3 = heatmap(xc, Array(S_obs'); title = "S_obs", aspect_ratio = 1)
            p4 = plot(iter_evo, J_evo; title="misfit", yaxis=:log10)
            display(plot(p1, p2, p3, p4; layout=(2,2), size=(980, 980)))
        end 

        # check convergence?
        if J_old/J_ini < gd_ϵtol 
            @printf("gradient descient converge, misfit = %.1e\n", J_old)
            break 
        else 
            @printf("#iter = %d, misfit = %.1e\n", gd_iter, J_old)
        end 

    end 

    return 
end 

adjoint_1D() 