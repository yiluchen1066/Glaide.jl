using CUDA,BenchmarkTools
using Plots,Plots.Measures,Printf
using DelimitedFiles
using Enzyme 
default(size=(800,600),framestyle=:box,label=false,grid=false,margin=10mm,lw=4,labelfontsize=9,tickfontsize=9,titlefontsize=12)

macro get_thread_idx()  esc(:( begin ix =(blockIdx().x-1) * blockDim().x + threadIdx().x; end )) end 
macro av_xa(A)    esc(:( 0.5*($A[ix]+$A[ix+1]) )) end
macro d_xa(A)     esc(:( $A[ix+1]-$A[ix])) end 

CUDA.device!(7) # GPU selection

function flux_q!(qHx, S, H, D, ∇Sx, nx, dx, a, as, n)
    @get_thread_idx(S)
    if ix <= nx-1 
        ∇Sx[ix] = @d_xa(S)/dx 
        D[ix] = (a*@av_xa(H)^(n+2)+as*@av_xa(H)^n)*∇Sx[ix]^(n-1)
        qHx[ix] = -D[ix]*@d_xa(S)/dx 
    end 
    return 
end 


#function compute_M!(M, H, B, grad_b, b_max, z_ELA, nx)
#    @get_thread_idx(S) 
#    if ix <= nx
#        M[ix] = min(grad_b*(H[ix]+B[ix]-z_ELA), b_max)
#    end 
#    return 
#end 

#M[ix+1] = min(grad_b*(H[ix+1]+B[ix+1]-z_ELA), b_max)
function residual!(RH, S, H, B, D, ∇Sx, qHx, n, a, as, grad_b, b_max, z_ELA, dx, nx) 
    @get_thread_idx(S)
    CUDA.@sync @cuda threads=threads blocks=blocks flux_q!(qHx, S, H, D, ∇Sx, nx, dx, a, as, n)
    if ix <= nx-2 
        RH[ix] = -@d_xa(qHx)/dx+min(grad_b*(H[ix+1]+B[ix+1]-z_ELA),b_max)
    end 
    return 
end 

function timestep!(S, H, D, ∇Sx, qHx, dτ, nx, dx, a, as, n, epsi, cfl)
    @get_thread_idx(S)
    CUDA.@sync @cuda threads=threads blocks=blocks flux_q!(qHx, S, H, D, ∇Sx, nx, dx, a, as, n)
    if ix <= nx-2 
        dτ[ix] = 0.5*min(1.0, cfl/(epsi+@av_xa(D)))
    end 
    return 
end 

function update_H!(S, H, D, ∇Sx, qHx, dτ, dHdτ, RH, nx, dx, a, as, n, epsi, cfl, damp)
    @get_thread_idx(H) 
    CUDA.@sync @cuda threads=threads blocks=blocks timestep!(S, H, D, ∇Sx, qHx, dτ, nx, dx, a, as, n, epsi, cfl)
    if ix <= nx-2 
        dHdτ[ix] = dHdτ[ix]*damp+RH[ix+1]
        H[ix+1]  = max(0.0, H[ix+1] + dτ[ix]*dHdτ[ix])
    end 
    return 
end 


mutable struct Forwardproblem{T<:Real, A<:AbstractArray{T}}
    S::A; H::A; B::A; D::A; ∇Sx::A; qHx::A
    dτ::A; dHdτ::A; RH::A; Err::A
    n::Int; a::T; as::T; grad_b::T; b_max::T; z_ELA::T; dx::T; nx::Int; epsi::T; cfl::T; ϵtol::T; niter::T; threads::Int; blocks::Int; dmp::T

end 

function Forwardproblem(S, H, B, D, ∇Sx, qHx, n, a, as, grad_b, b_max, z_ELA, dx, nx, epsi, cfl, ϵtol, niter, threads, blocks, damp)
    RH = similar(H, nx-2)
    dHdτ = similar(H, nx-2) 
    dτ   = similar(H, nx-2) 
    Err  = similar(H)
    return Forwardproblem(S, H, B, D, ∇Sx, qHx, RH, dHdτ, dτ, Err, n, a, as, grad_b, b_max, z_ELA, dx, nx, epsi, cfl, ϵtol, niter, threads, blocks, damp)
end 

#solve for H using pseudo-trasient method 
function solve!(problem::Forwardproblem)
    (;S, H, B, D, ∇Sx, qHx, RH, dHdτ, dτ, Err, n, a, as, grad_b, b_max, z_ELA, dx, nx, epsi, cfl, ϵtol, niter, threads, blocks, damp) = problem
    dHdτ .= 0; RH .= 0; dτ .= 0; Err .= 0
    merr = 2ϵtol; iter = 1 
    while merr >= ϵtol && iter < niter 
        Err .= H 
        CUDA.@sync @cuda threads=threads blocks=blocks residual!(RH, S, H, B, D, ∇Sx, qHx, n, a, as, grad_b, b_max, z_ELA, dx, nx)   # compute RH 
        CUDA.@sync @cuda threads=threads blocks=blocks update_H!(S, H, D, ∇Sx, qHx, dτ, dHdτ, RH, nx, dx, a, as, n, epsi, cfl, damp) # update H
        if iter  % ncheck == 0 
            @. Err -= H 
            merr = maximum(abs.(Err))
            (isfinite(merr) && merr >0) || error("forward solve failed")
        end 
        iter += 1 
    end 
    if iter == niter && merr >= ϵtol
        error("forward solve not converged")
    end 
    @printf("forward solve converged: #iter/nx = %.1f, err = %.1e\n, iter/nx, merr")
    return 
end 

# compute dQ/dH
function grad_residual_H_1_1!(tmp1, tmp2, S, H, dQ, D, ∇Sx, nx, dx, as, a, n)
    # compute (dQ/dH)^T*r
    Enzyme.autodiff_deferred(flux_q!, Duplicated(tmp1, tmp2), Const(S), Duplicated(H, dQ), Const(D), Const(∇Sx), Const(nx), Const(dx), Const(a), Const(as), Const(n))
    return 
end 

# compute dR/H = ()
function grad_residual_H_1!(tmp1, tmp2, tmp3, tmp4, S, H, B, dQ, D, ∇Sx, qHx, dR, M, nx, dx, as, a, n, grad_b, b_max, z_ELA)
    grad_residual_H_1_1!(tmp1, tmp2, S, H, dQ, D, ∇Sx, nx, dx, as, a, n)
    tmp4 .= dQ
    # compute (dR/dQ)^T*dQ
    Enzyme.autodiff_deferred(residual!, Duplicated(tmp3, tmp4), Const(S), Const(H), Const(B), Const(D), Const(∇Sx), Duplicated(qHx, dR), Cost(M), Const(n), Const(a), Const(as), Const(grad_b), Const(b_max), Const(z_ELA), Const(dx), Const(nx))
    # the result is stored in dR 
    return 
end 

function grad_residual_H_2_1!(tmp5, tmp2, dM, H, B, grad_b, b_max, z_ELA, nx)
    # compute (dM/dH)^T*r
    # tmp2 should be initialized as r, stored in dM 
    Enzyme.autodiff_deferred(compute_M!, Duplicated(tmp5, tmp2), Duplicated(H, dM),Const(B), Const(grad_b), Const(b_max), Const(z_ELA), Const(nx))
    return 
end 

function grad_residual_H_2!(tmp2, tmp5, tmp6, tmp7, S, dM, H, B, D, ∇Sx, qHx, M, dR, n, a, as, grad_b, b_max, z_ELA, dx, nx)
    grad_residual_H_2_1!(tmp5, tmp2, dM, H, B, grad_b, b_max, z_ELA, nx)
    tmp7 .= dM 
    # compute (dR/dM)^T*dM
    Enzyme.autodiff_deferred(residual!, Duplicated(tmp6, tmp7), Const(S), Const(H), Const(B), Const(D), Const(∇Sx), Const(qHx), Duplicated(M, dR), Const(n), Const(a), Const(as),
    Const(grad_b), Const(b_max), Const(z_ELA), Const(dx), Const(nx))
    # the result stored in dR
    return 
end 

function grad_residual_H!(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, S, H, B, dQ, dM, D, ∇Sx, qHx, dR_Q, dR_M, M, nx, dx, as, a, n, grad_b, b_max, z_ELA)
    grad_residual_H_1!(tmp1, tmp2, tmp3, tmp4, S, H, B, dQ, D, ∇Sx, qHx, dR_Q, M, nx, dx, as, a, n, grad_b, b_max, z_ELA)
    grad_residual_H_2!(tmp2, tmp5, tmp6, tmp7, S, dM, H, B, D, ∇Sx, qHx, M, dR_M, n, a, as, grad_b, b_max, z_ELA, dx, nx)
    dR .= dR_Q + dR_M
    return 
end 

function update_r!(r, dR, R, H, dt, dmp, nx) 
    @get_threads_idx(H)
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
    S::A; H::A; B::A; D::A; ∇Sx::A; qHx::A; M::A; 
    dQ::A; dM::A; dR_Q::A; dR_M::A; r::A; R::A; dR::A; Err::A; tmp1::A; tmp2::A; tmp3::A; tmp4::A; tmp5::A; tmp6::A; tmp7::A;∂J_∂H::A
    nx::Int; dx::T; as::T; a::T; n::T; grad_b::T; b_max::T; z_ELA::T; dmp::T; dt::T; ϵtol::T; niter::Int; ncheck::Int; threads::Int; blocks::Int
end 

function AdjointProblem(S, H, B, D, ∇Sx, qHx, M, nx, dx, as, a, n, grad_b, b_max, z_ELA, dmp, dt, ϵtol, niter, ncheck, threads, blocks)
    dQ = similar(H) 
    dM = similar(H)
    dR_Q = similar(H) 
    dR_M = similar(H)
    r = similar(H) 
    R = similar(H)
    dR = similar(H) 
    Err = similar(H)
    tmp1 = similar(H) 
    tmp2 = similar(H)
    tmp3 = similar(H) 
    tmp4 = similar(H)
    tmp5 = similar(H)
    tmp6 = similar(H) 
    tmp7 = similar(H)
    ∂J_∂H = similar(H) 
    return AdjointProblem(S, H, B, D, dQ, dM, ∇Sx, qHx, dR_Q, dR_M, M, r, R, dR, Err, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, ∂J_∂H, nx, dx, as, a, n, grad_b, b_max, z_ELA, dmp, dt, ϵtol, niter, ncheck, threads, blocks)
end 


# solve for r using pseudo-trasient method to compute sensitvity afterwards 
function solve!(problem::AdjointProblem) 
    (; S, H, B, D, dQ, dM, ∇Sx, qHx, dR_Q, dR_M, M, r, R, dR, Err, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, ∂J_∂H, nx, dx, as, a, n, grad_b, b_max, z_ELA, dmp, dt, ϵtol, niter, ncheck, threads, blocks) = problem
    r.= 0; dR .= 0; R.= 0; Err .= 0; dQ .= 0, dM .= 0; dR_Q .= 0; dR_M .= 0
    dt = dx/ 3.0 
    @. ∂J_∂H = H - H_obs
    # initialization for tmp1 tmp2 tmp3 tmp4 
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol  && iter < niter 
        Err .= r 
        # compute dQ/dH 
        tmp2 .= r 
        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H!(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, S, H, B, dQ, dM, D, ∇Sx, qHx, dR_Q, dR_M, M, nx, dx, as, a, n, grad_b, b_max, z_ELA)
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

function grad_residual_β_1!(tmp1, tmp2, H, B, grad_b, dM, b_max, z_ELA, nx)
    # compute (dM/dbeta)^T*r 
    Enzyme.autodiff_deferred(compute_M!,Duplicated(tmp1, tmp2), Const(H), Const(B), Duplicated(grad_b, dM), Const(b_max), Const(z_ELA), Const(nx))
    return 
end 

function grad_residual_β!()
    grad_residual_β_1!(tmp1, tmp2, H, B, grad_b, dM, b_max, z_ELA, nx)



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
    @get_thread_idx(S) 
    if ix <= nx 
        S[ix] = B[ix] + H[ix]
    end 
    return 
end  

function adjoint_1D()
    # physics parameters

    #numerics parameters 
    gd_niter =      

    #preprocessing parameters 

    # initialization 

    # display 
    adjoint_problem = 


    
    # gradient descent iterations to update n 
    for gd_iter = 1:gd_niter 
        n_init = n 
        solve!(adjoint_problem)






    return 
end 

# adjoint_1D() 


function update_without_limiter!(S,H,B,M,D,∇Sx,qHx,dHdt,dx,dt,n,a,as,threads,blocks, nx)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_∇S_without_limiter!(∇Sx, S, dx, nx)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_D!(D,∇Sx,H, n, a, as,nx)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_flux_without_limiter!(S, qHx, D, ∇Sx, dx, nx)
    CUDA.@sync @cuda threads=threads blocks=blocks update_1!(dHdt, qHx, M, dx, nx)
    CUDA.@sync @cuda threads=threads blocks=blocks update_2!(dHdt, H, dt, nx)
    CUDA.@sync @cuda threads=threads blocks=blocks set_BC!(H,nx)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_S!(H, B, S,nx)
    return
end

function update_with_limiter!(S,H,B,M,D,∇Sx,qHx,B_avg, H_avg,dHdt,dx,dt,n,a,as,threads,blocks, nx)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_limiter!(B, B_avg, S, H_avg, nx)
    CUDA.@sync @cuda threads=threads blocks=blocks computer_∇S_with_limiter!(∇Sx, S, B_avg, dx, nx)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_D!(D,∇Sx,H_avg, n, a, as, nx)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_flux_with_limiter!(S, qHx, B_avg, D, dx, nx)
    CUDA.@sync @cuda threads=threads blocks=blocks update_1!(dHdt, qHx, M, dx, nx)
    CUDA.@sync @cuda threads=threads blocks=blocks update_2!(dHdt, H, dt, nx)
    CUDA.@sync @cuda threads=threads blocks=blocks set_BC!(H, nx)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_S!(H, B, S, nx)
    return 
end 

function sia_1D()
    # physics
    s2y     = 3600*24*365.25 # seconds to years
    lx      = 30e3     # lx, ly = 30 km
    n       = 3
    ρg      = 910*9.8
    ttot    = 50e3
    a0      = 1.5e-24
    # numerics
    nx      = 151
    nout    = 5000    # error check frequency
    ndt     = 100      # dt check/update
    threads = 16 # n threads
    dtsc    = 0.7     # iterative dt scaling 
    epsi    = 1e-4
    ϵtol    = 1e-8
    # derived numerics
    dx      = lx/nx
    xc      = LinRange(dx/2,lx-dx/2,nx)
    cfl     = dx^2/4.1
    blocks  = ceil(Int,nx/threads)
    # derived physics 
    a       = 2.0*a0/(n+2)*ρg^n*s2y
    as      = 5.7e-20
    # read benchmark data
    bed = readdlm("/scratch-1/yilchen/Msc-Inverse-SIA/Msc-Inverse-SIA/benchmark_data/bed.txt", ' ', Float64, '\n')
    M2  = readdlm("/scratch-1/yilchen/Msc-Inverse-SIA/Msc-Inverse-SIA/benchmark_data/M2.txt", ' ', Float64, '\n')
    MUSCL = readdlm("/scratch-1/yilchen/Msc-Inverse-SIA/Msc-Inverse-SIA/benchmark_data/MUSCL.txt", ' ', Float64, '\n')
    upstream = readdlm("/scratch-1/yilchen/Msc-Inverse-SIA/Msc-Inverse-SIA/benchmark_data/upstream.txt", ' ', Float64, '\n')
    benchmark = readdlm("/scratch-1/yilchen/Msc-Inverse-SIA/Msc-Inverse-SIA/benchmark_data/benchmark.txt", ' ', Float64, '\n')
    # array initialisation
    B       = zeros(Float64, nx)
    M       = zeros(Float64, nx)
    H       = zeros(Float64, nx)

    @printf("Size of bed is %d, size of M2 is %d, size of MUSCL is %d, size of upstream is %d \n", size(bed,1), size(M2,1), size(MUSCL,1), size(upstream,1))
    # define bed vector
    xm,xmB  = 20e3,7e3
    M .= (((n.*2.0./xm.^(2*n-1)).*xc.^(n-1)).*abs.(xm.-xc).^(n-1)).*(xm.-2.0*xc).*0.6
    M[xc.>xm ] .= 0.0 
    B[xc.<xmB] .= 500
    for ism=1:2
         B[2:end-1] .= B[2:end-1] .+ 1.0/4.1.*(diff(diff(B)))
         B[[1,end]] .= B[[2,end-1]] # BCs
     end
    # visu
    #opts = (aspect_ratio=1,xlims=(xc[1],xc[end]),c=:turbo)
    #p1 = heatmap(xc,B',title="H"; opts...)
    #p2 = heatmap(xc,M',title="B"; opts...)
    #display(plot(p1, p2))
    # array initialization 
    H     = CuArray{Float64}(H) 
    B     = CuArray{Float64}(B)
    M     = CuArray{Float64}(M)
    ∇Sx   = CUDA.zeros(Float64,nx-1)
    S     = CUDA.zeros(Float64,nx)
    D     = CUDA.zeros(Float64,nx-1)
    B_avg = CUDA.zeros(Float64,nx-1)
    H_avg = CUDA.zeros(Float64,nx-1)
    qHx   = CUDA.zeros(Float64,nx-1)
    dHdt  = CUDA.zeros(Float64,nx-2)
    S    .= B .+ H # init bed
    #t = 0.0; it = 1; dt = 1.0
    t = 0.0; it = 1; dt = dtsc * min(1.0, cfl/(epsi+maximum(D)))
    anim = @animate while t <= ttot
        if (it==1 || it%ndt==0) dt=dtsc*min(1.0, cfl/(epsi+maximum(D))) end
        update_without_limiter!(S,H,B,M,D,∇Sx,qHx,dHdt,dx,dt,n,a,as,threads,blocks, nx)
        #update_with_limiter!(S,H,B,M,D,∇Sx,qHx,B_avg, H_avg,dHdt,dx,dt,n,a,as,threads,blocks, nx)
        if it%nout == 0
            @printf("it = %d, t = %1.2f, max(dHdt) = %1.2e \n", it, t, maximum(dHdt[2:end-1]))
            #p1 = heatmap(xc,Array(S'), title="S, it=$(it)"; opts...)
            #p2 = heatmap(xc,Array(H'), title="H"; opts...)
            #p3 = plot(xc, [Array(S),Array(B)])
            #title = plot(title = "SIA 1D")
            # p3 = plot(xc, Array(S), label = "Implementation", xlabel="X in m", ylabel="Height in m", color=:firebrick1)
            # p3 = plot!(xc, Array(B), label = "bedrock", xlabel="X in m", ylabel="Height in m", color=:grey)
            # #p3 = plot(xc, [Array(S),Array(B)], label = ["Implementation" "bedrock"], xlabel="X in m", ylabel="Height in m")
            # p3 = plot!(M2[:,1]*1e3, [M2[:,2], MUSCL[:,2], upstream[:,2],benchmark[:,2]], label = [ "M2" "MUSCL superbee" "upstream" "benchmark"],xlabel="X in m", ylabel="Height in m")
            # p4 = plot(xc, Array(H),xlabel="X in m", ylabel="Height in m",color=:red)
            # display(plot(p3, title = "SIA 1D cliff benchmark"))
        end
        it += 1
        t += dt
    end
    p3 = plot(xc, Array(S), label = "Implementation", xlabel="X in m", ylabel="Height in m", color=:firebrick1)
    p3 = plot!(xc, Array(B), label = "bedrock", xlabel="X in m", ylabel="Height in m", color=:grey)
    #p3 = plot(xc, [Array(S),Array(B)], label = ["Implementation" "bedrock"], xlabel="X in m", ylabel="Height in m")
    p3 = plot!(M2[:,1]*1e3, [M2[:,2], MUSCL[:,2], upstream[:,2],benchmark[:,2]], label = [ "M2" "MUSCL superbee" "upstream" "benchmark"],xlabel="X in m", ylabel="Height in m")
    p4 = plot(xc, Array(H),xlabel="X in m", ylabel="Height in m",color=:red)
    display(plot(p3, title = "SIA 1D cliff benchmark"))
    savefig("1D_without_limiter.png")
    #gif(anim, "1D_without_limiter.gif", fps=15)
end

sia_1D()
