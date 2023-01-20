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

function residual!(RH, S, H, D, ∇Sx, qHx, M, n, a, as, dx, nx) 
    @get_thread_idx(S)
    CUDA.@sync @cuda threads=threads blocks=blocks flux_q!(qHx, S, H, D, ∇Sx, nx, dx, a, as, n)
    if ix <= nx-2 
        RH[ix] = -@d_xa(qHx)/dx+M[ix+1]
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

mutable struct Forwardproblem{T<:Real, A<:AbstractArray{T}}
    S::A; H::A; D::A; ∇Sx::A; qHx::A; M::A;
    dτ::A; dHdτ::A; RH::A; Err::A
    n::Int; a::A; as::A; dx::A; nx::Int; epsi::A; cfl::A; ϵtol::A; niter::Int; threads::Int; blocks::Int; dmp::A

end 

function Forwardproblem(S, H, D, ∇Sx, qHx, M, n, a, as, dx, nx, epsi, cfl, ϵtol, niter, threads, blocks, dmp)
    RH = similar(H, nx-2)
    dHdτ = similar(H, nx-2) 
    dτ   = similar(H, nx-2) 
    Err  = similar(H)
    return Forwardproblem(S, H, D, ∇Sx, qHx, M, RH, dHdτ, dτ, Err, n, a, as, dx, nx, epsi, cfl, ϵtol, niter, threads, blocks, dmp)
end 

#solve for H using pseudo-trasient method 
function solve!(problem::Forwardproblem)
    (;S, H, D, ∇Sx, qHx, M, RH, dHdτ, dτ, Err, n, a, as, dx, nx, epsi, cfl, ϵtol, niter, threads, blocks, dmp) = problem
    dHdτ .= 0; RH .= 0; dτ .= 0; Err .= 0
    merr = 2ϵtol; iter = 1 
    while merr >= ϵtol && iter < niter 
        Err .= H 
        CUDA.@sync @cuda threads=threads blocks=blocks residual!(S, H, D, ∇Sx, qHx, M, RH, n, a, as, dx, nx) # compute RH 
        CUDA.@sync @cuda threads=threads blocks=blocks timestep!(S, H, D, ∇Sx, qHx, dτ, nx, dx, a, as, n, epsi, cfl) # compute dτ
        @. dHdτ = dHdτ*dmp + RH 
        @. H[2:end-1] = max(0.0, H+dτ*dHdτ)
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

mutable struct AdjointProblem{T<:Real, A<:AbstractArray{T}}
    S::A; H::A; D::A; ∇Sx::A; qHx::A; M::A
    r::A; R::A; dR::A; Err::A; tmp1::A; tmp2::A; tmp3::A; tmp4::A; ∂J_∂H::A; dQ::A
    nx::Int; dx::T; as::T; a::T; n::T; dmp::T; dt::T; ϵtol::T; niter::Int; ncheck::Int
    

end 

function AdjointProblem(S, H, D, ∇Sx, qHx, M, nx, dx, as, a, n, dmp, dt, ϵtol, niter, ncheck)
    r = similar(H) 
    R = similar(H)
    dR = similar(H) 
    Err = similar(H)
    tmp1 = similar(H) 
    tmp2 = similar(H)
    tmp3 = similar(H) 
    tmp4 = similar(H)
    ∂J_∂H = similar(H) 
    dQ = similar(H)     
    return AdjointProblem(S, H, D, ∇Sx, qHx, M, r, R, dR, Err, tmp1, tmp2, tmp3, tmp4, ∂J_∂H, dQ, nx, dx, as, a, n, dmp, dt, ϵtol, niter, ncheck)
end 

# compute dQ/dH
function residual_grad_1!(tmp1, tmp2, S, H, dQ, D, ∇Sx, nx, dx, as, a, n)
    # compute (dQ/dH)^T*r
    Enzyme.autodiff_deferred(flux_q!, Duplicated(tmp1, tmp2), Const(S), Duplicated(H, dQ), Const(D), Const(∇Sx), Const(nx), Const(dx), Const(as), Const(a), Const(n))
    return 
end 

function residual_grad!(tmp1, tmp2, tmp3, tmp4, S, H, dQ, D, ∇Sx, qHx, dR, M, nx, dx, as, a, n)
    residual_grad_1!(tmp1, tmp2, S, H, dQ, D, ∇Sx, nx, dx, as, a, n)
    tmp3 .= dQ
    Enzyme.autodiff_deferred(residual!, Duplicated(tmp3, tmp4), Const(S), Const(H), Const(D), Const(∇Sx), Duplicated(qHx, dR), Cost(M), Const(n), Const(a), Const(as), Const(dx), Const(nx))
    # the result is stored in dR 
    return 
end 

# solve for r using pseudo-trasient method to compute sensitvity afterwards 
function solve!(problem::AdjointProblem) 
    r.= 0; dR .= 0; R.= 0; Err .= 0; dQ .= 0 
    # definition of dt/dτ in the adjoint problem 
    @. ∂J_∂H = H - H_obs
    # initialization for tmp1 tmp2 tmp3 tmp4 
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol  && iter < niter 
        Err .= r 
        # compute dQ/dH 
        tmp2 .= r 
        CUDA.@sync @cuda threads = threads blocks = blocks residual_grad!(tmp1, tmp2, tmp3, tmp4, S, H, dQ, D, ∇Sx, qHx, dR, M, nx, dx, as, a, n)
        dR .= .-∂J_∂H
        @. R = (1.0 - dmp/nx)*R + dt*dR 
        @. r += dt*R 
        r[1:1] .= 0.0; r[end:end] .= 0.0 
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
    # physics 


    
    # gradient descent iterations to update n 
    for gd_iter = 1:gd_niter 




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
