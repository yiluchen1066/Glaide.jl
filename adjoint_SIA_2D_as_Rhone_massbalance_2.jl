using CUDA,BenchmarkTools
using Plots,Plots.Measures,Printf
using DelimitedFiles, DataFrames
using Enzyme 
using HDF5
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

CUDA.device!(7) # GPU selection

function compute_error_1!(Err, H, nx, ny)
    @get_thread_idx(H)
    if ix <= nx && iy <= ny 
        Err[ix, iy] = H[ix, iy]
    end 
    return 
end 

function compute_error_2!(Err, H, nx, ny)
    @get_thread_idx(H) 
    if ix <= nx && iy <= ny
        Err[ix, iy] = Err[ix, iy] - H[ix, iy]
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

# function compute_diffusion!(B, H, D, ∇Sx, nx, dx, a, as, n)
#     @get_thread_idx(H)
#     if ix <= nx-1 
#         ∇Sx[ix] = @d_xa(B)/dx + @d_xa(H)/dx
#         D[ix] = (a*@av_xa(H)^(n+2)+as*@av_xa(H)^n)*∇Sx[ix]^(n-1)
#     end 
#     return 
# end 

function compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, a, as, n, nx, ny, dx, dy)
    @get_thread_idx(H)
    if ix <= nx-1 && iy <= ny-1 
        #av_ya_∇Sx(nx-1, ny-1) av_xa_∇Sy(nx-1, ny-1)
        # new macro needed 
        av_ya_∇Sx[ix,iy] = 0.5*((B[ix+1, iy+1]-B[ix,iy+1])/dx + (H[ix+1, iy+1]-H[ix,iy+1])/dx + (B[ix+1,iy]-B[ix,iy])/dx + (H[ix+1,iy]-H[ix,iy])/dx)
        av_xa_∇Sy[ix,iy] = 0.5*((B[ix+1, iy+1]-B[ix+1,iy])/dy + (H[ix+1, iy+1]-H[ix+1,iy])/dy + (B[ix,iy+1]-B[ix,iy])/dy + (H[ix,iy+1]-H[ix,iy])/dy)
        gradS[ix, iy] = sqrt(av_ya_∇Sx[ix,iy]^2+av_xa_∇Sy[ix,iy]^2)
        D[ix, iy] = (a*@av_xy(H)^(n+2)+as[ix,iy]*@av_xy(H)^n)*gradS[ix,iy]^(n-1)
    end 
    return 
end 

#function compute_qHx!(qHx, D, H, B, dx, nx, ny)
#    @get_thread_idx(H)
#    if ix <= nx-1 && iy <= ny-2 
#        qHx[ix,iy] = -@av_ya(D)*(@d_xi(H)+@d_xi(B))/dx 
#    end 
#    return 
#end 

#function compute_qHy!(qHy, D, H, B, dy, nx, ny)
#    @get_thread_idx(H) 
#    if ix <= nx-2 && iy <= ny-1 
#        qHy[ix,iy] = -@av_xa(D)*(@d_yi(H)+@d_yi(B))/dy 
#    end 
#    return
#end 



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


function compute_M!(M, β, H, B, z_ELA, b_max, nx,ny)
    @get_thread_idx(M)
    if ix <= nx && iy <= ny 
        M[ix,iy] = min(β[ix,iy]*(H[ix,iy]+B[ix,iy]-z_ELA), b_max)
    end 
    return 
end 

function residual!(RH, qHx, qHy, H, B, zs_sample, ms_sample, dz_sample, mb, nx, ny, dx, dy)
    @get_thread_idx(H)
    if ix <= nx-2 && iy <= ny-2 
        iz_f = clamp((H[ix+1,iy+1]+B[ix+1,iy+1]-zs_sample[1])/dz_sample, 0.0, Float64(length(ms_sample)-2))
        iz   = floor(Int64, iz_f)+1
        f    = iz_f - (iz-1)
        mb[ix+1,iy+1]   = ms_sample[iz]*(1.0-f) + ms_sample[iz+1]*f
        RH[ix+1,iy+1] =  -(@d_xa(qHx)/dx + @d_ya(qHy)/dy) + mb[ix+1,iy+1]
    end 
    return 
end 

# function mb_data_driven(z)
#     iz_f = (z-zs_sample[1])/dz_sample
#     iz = clamp(floor(Int,iz_f) + 1,1,length(zs_sample)-1)
#     f = iz_f - (iz - 1)
#     return ms_sample[iz]*(1-f) + ms_sample[iz+1]*f
# end

function timestep!(dτ, H, D, cfl, epsi, nx, ny)
    @get_thread_idx(H)
    if ix <= nx-2 && iy <= ny-2 
        dτ[ix,iy] = 0.5*min(1.0, cfl/(epsi+@av_xy(D)))
    end 
    return 
end

function update_H!(H, dHdτ, RH, dτ, damp, nx, ny)
    @get_thread_idx(H)
    if ix <= nx-2 && iy <= ny -2 
        dHdτ[ix, iy] = dHdτ[ix,iy]*damp + RH[ix+1,iy+1]
        #update the inner point of H 
        H[ix+1,iy+1] = max(0.0, H[ix+1,iy+1]+dτ[ix,iy]*dHdτ[ix,iy])
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


mutable struct Forwardproblem{T<:Real, A<:AbstractArray{T}}
    H::A; B::A; D::A; gradS::A; av_ya_∇Sx::A; av_xa_∇Sy::A; qHx::A; qHy::A; M::A; β::A; as::A; zs_sample::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}; ms_sample::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}; mb::A
    dτ::A; dHdτ::A; RH::A; Err::A
    n::Int; a::T; b_max::T; z_ELA::Int; dz_sample::T; dx::T; dy::T; nx::Int; ny::Int; epsi::T; cfl::T; ϵtol::T; niter::Int; ncheck::Int; threads::Tuple{Int, Int}; blocks::Tuple{Int, Int}; dmp::T
end 

function Forwardproblem(H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, M, β, as, zs_sample, ms_sample, mb, n, a, b_max, z_ELA, dz_sample, dx, dy, nx, ny, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp)
    RH = similar(H,nx, ny)
    dHdτ = similar(H, nx-2, ny-2) 
    dτ   = similar(H, nx-2, ny-2) 
    Err  = similar(H, nx, ny)
    return Forwardproblem(H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, M, β, as, zs_sample, ms_sample, mb, dτ, dHdτ, RH, Err, n, a, b_max, z_ELA, dz_sample, dx, dy, nx, ny, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp) 
end 

#solve for H using pseudo-trasient method 
function solve!(problem::Forwardproblem)
    (;H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, M, β, as, zs_sample, ms_sample, mb, dτ, dHdτ,RH, Err, n, a, b_max, z_ELA, dz_sample, dx,dy, nx, ny, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp) = problem
    dHdτ .= 0; RH .= 0; dτ .= 0; Err .= 0
    merr = 2ϵtol; iter = 1 
    lx, ly = 30e3, 30e3
    xc = LinRange(dx/2, lx-dx/2, nx)
    yc = LinRange(dy/2, ly-dy/2, ny)
    #M  = CuArray{Float64}(H)
    #p1 = plot(xc, Array(H); title = "H_init (forward problem)")
    #display(plot(p1))
    while merr >= ϵtol && iter < niter 
        #Err .= H 
        #CUDA.@sync @cuda threads=threads blocks=blocks residual!(RH, H, B, qHx, β, b_max, z_ELA, dx, nx)
        CUDA.@sync @cuda threads=threads blocks=blocks compute_error_1!(Err, H, nx, ny)
        CUDA.@sync @cuda threads=threads blocks=blocks compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, a, as, n, nx, ny, dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks compute_q!(qHx, qHy, D, H, B, nx, ny, dx, dy)
        #CUDA.@sync @cuda threads=threads blocks=blocks compute_M!(M, β, H, B, z_ELA, b_max, nx,ny)
        CUDA.@sync @cuda threads=threads blocks=blocks residual!(RH, qHx, qHy, H, B, zs_sample, ms_sample, dz_sample, mb, nx, ny, dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks timestep!(dτ, H, D, cfl, epsi, nx, ny)
        CUDA.@sync @cuda threads=threads blocks=blocks update_H!(H, dHdτ, RH, dτ, dmp, nx, ny)# update H
        CUDA.@sync @cuda threads=threads blocks=blocks set_BC!(H, nx, ny)

        if iter  % ncheck == 0 
            #@. Err -= H 
            CUDA.@sync @cuda threads=threads blocks =blocks compute_error_2!(Err, H, nx, ny)
            #merr = maximum(abs.(Err[:]))
            #@show(merr)
            merr = (sum(abs.(Err[:,:]))./nx./ny)
            #M .= min.(β.*(H.+B.-z_ELA), b_max)
            (isfinite(merr) && merr >0) || error("forward solve failed")
            #@printf("error = %1.e\n", merr)
            # p1 = heatmap(xc, yc, Array((H.+B)'); title = "S (forward problem)")
            # p2 = Plots.plot(xc, Array(H[:,ceil(Int,ny/2)]);title="Ice thickness")
            # Plots.plot!(xc, Array(B[:,ceil(Int,ny/2)]);label="bedrock")
            # p3 = heatmap(xc, yc, Array(M'); title="Mass balance term")
            # p4 = Plots.plot(xc, Array(M[:,ceil(Int,ny/2)]); title="Cross section of the mass balance term")
            # display(plot(p1,p2,p3,p4;layout=(2,2)))
            #plot!(xc, Array(B); title = "S (forward problem)", ylims=(0,1000))
            #display(plot(p1))
            #error("check forward model")
        end 
        iter += 1 
    end 
    if iter == niter && merr >= ϵtol
        error("forward solve not converged")
    end 
    @printf("forward solve converged: #iter/nx = %.1f, err = %.1e\n", iter/nx, merr)
    return 
end 
#compute dR_qHx dR_qHy dR_H
#residual!(RH, qHx, qHy, H, B, zs_sample, ms_sample, dz_sample, nx, ny, dx, dy)
function grad_residual_H_1!(tmp1, tmp2, qHx, dR_qHx, qHy, dR_qHy, β, H, dR_H, B, zs_sample, ms_sample, z_ELA, b_max, dz_sample, nx, ny, dx, dy)
    #tmp2 .= r 
    Enzyme.autodiff_deferred(residual!, Duplicated(tmp1, tmp2), Duplicated(qHx, dR_qHx), Duplicated(qHy, dR_qHy), Duplicated(H, dR_H), Const(B), Const(zs_sample), Const(ms_sample), Const(dz_sample), Const(nx), Const(ny), Const(dx), Const(dy))
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

function update_r!(r, R, dR, dt, H, dmp, nx, ny)
    @get_thread_idx(R) 
    if ix <= nx && iy <= ny 
        if H[ix, iy] <= 1e-2#2.0e1
            R[ix,iy] = 0.0 
            r[ix,iy] = 0.0 
        else 
            R[ix,iy] = R[ix,iy]*(1.0 - dmp/min(nx,ny)) + dt*dR[ix,iy] 
            r[ix,iy] = r[ix,iy] + dt*R[ix,iy]
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
    H::A; H_obs::A; B::A; D::A; qHx::A; qHy::A; β::A; as::A; gradS::A; av_ya_∇Sx::A; av_xa_∇Sy::A; zs_sample::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}; ms_sample::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    r::A; dR::A; R::A; Err::A; dR_qHx::A; dR_qHy::A; dR_H::A; dq_D::A; dq_H::A; dD_H::A 
    tmp1::A; tmp2::A; ∂J_∂H::A
    z_ELA::Int; b_max::T; dz_sample::T; nx::Int; ny::Int; dx::T; dy::T; a::T; n::Int; dmp::T; ϵtol::T; niter::Int; ncheck::Int; threads::Tuple{Int, Int}; blocks::Tuple{Int, Int}
end 


function AdjointProblem(H, H_obs,B, D, qHx, qHy, β, as, gradS, av_ya_∇Sx, av_xa_∇Sy, zs_sample, ms_sample, z_ELA, b_max, dz_sample, nx, ny, dx, dy, a, n, dmp, ϵtol, niter, ncheck, threads, blocks)
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
    return AdjointProblem(H, H_obs,B, D, qHx, qHy, β, as, gradS, av_ya_∇Sx, av_xa_∇Sy, zs_sample, ms_sample, r, dR, R, Err, dR_qHx, dR_qHy, dR_H, dq_D, dq_H, dD_H, tmp1, tmp2, ∂J_∂H, z_ELA, b_max, dz_sample,  nx, ny, dx, dy, a, n, dmp, ϵtol, niter, ncheck, threads, blocks)
end 


# solve for r using pseudo-trasient method to compute sensitvity afterwards 
function solve!(problem::AdjointProblem) 
    (; H, H_obs, B, D, qHx, qHy, β, as, gradS, av_ya_∇Sx, av_xa_∇Sy, zs_sample, ms_sample, r, dR, R, Err, dR_qHx, dR_qHy, dR_H, dq_D, dq_H, dD_H, tmp1, tmp2, ∂J_∂H, z_ELA, b_max, dz_sample, nx, ny, dx, dy, a, n, dmp, ϵtol, niter, ncheck, threads, blocks) = problem
    r.= 0;  R.= 0; Err .= 0; dR .= 0; dR_qHx .= 0; dR_qHy .= 0; dR_H .= 0; dq_D .= 0; dq_H .= 0; dD_H .= 0
    lx, ly = 30e3, 30e3
    dx = lx/nx
    dy = ly/ny
    xc = LinRange(dx/2, lx-dx/2, nx) 
    yc = LinRange(dy/2, ly-dy/2, ny) 
    xc1 = LinRange(dx/2,lx-3*dx/2, nx-1)
    yc1 = LinRange(3*dy/2, ly-3*dy/2, ny-2)

    #dt = min(dx^2, dy^2)/maximum(D)/60.1/2
    dt = min(dx^2, dy^2)/maximum(D)/60.1
    @. ∂J_∂H = H - H_obs
    # initialization for tmp1 tmp2 tmp3 tmp4 
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol  && iter < niter 
        #Err .= r 
        # compute dQ/dH 
        dR_qHx .= 0; dR_qHy .= 0 ; dR_H .= 0; dq_D .= 0.0; dq_H .= 0; dD_H .= 0
        dR .= .-∂J_∂H; tmp2 .= r 
        
        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H_1!(tmp1, tmp2, qHx, dR_qHx, qHy, dR_qHy, β, H, dR_H, B, zs_sample, ms_sample, z_ELA, b_max, dz_sample, nx, ny, dx, dy)
        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H_2!(qHx, dR_qHx, qHy, dR_qHy, D, dq_D, H, dq_H, B, nx, ny, dx, dy)
        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H_3!(D, dq_D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, dD_H, B, a, as, n, nx, ny, dx, dy)
        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H!(dR, dD_H, dq_H, dR_H, nx, ny)
        CUDA.@sync @cuda threads = threads blocks = blocks update_r!(r, R, dR, dt, H, dmp, nx, ny)
        
        # if iter > 10 error("Stop") end
        if iter % ncheck == 0 
            #@. Err -= r 
            #@show(size(dR_qHx))
            merr = maximum(abs.(R[2:end-1,2:end-1]))
            # @show(maximum(dR))
            # @show(maximum(r))
            # @show(maximum(R))
            # @show(maximum(dD_H))
            # @show(maximum(dR_H))
            # @show(maximum(dq_H))
        
            p1 = heatmap(xc, yc, Array(dR_H'); title = "dR_H")
            p2 = heatmap(xc, yc, Array(dq_H'); title = "dq_H")
            p3 = heatmap(xc, yc, Array(dD_H'); title = "dD_H")
            p4 = heatmap(xc, yc, Array(dR'); title ="dR")
            p5 = heatmap(xc, yc, Array(r'); title="r")
            p6 = heatmap(xc, yc, Array(R'); title="R")

            # savefig(p1, "adjoint_debug/adjoint_R_$(iter).png")
            display(plot(p1,p2,p3,p4,p5,p6; layout=(3,2)))
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
    (; H, H_obs, B, D, qHx, qHy, β, as, gradS, av_ya_∇Sx, av_xa_∇Sy, zs_sample, ms_sample, r, dR, R, Err, dR_qHx, dR_qHy, dR_H, dq_D, dq_H, dD_H, tmp1, tmp2, ∂J_∂H, z_ELA, b_max, dz_sample, nx, ny, dx, dy, a, n, dmp, ϵtol, niter, ncheck, threads, blocks) = problem
    # dimensionmismatch: array could not be broadcast to match destination 
    Jn .= 0.0
    tmp2 .= r
    dR_qHx .= 0; dR_qHy .= 0; dR_H .= 0; dq_D .= 0; dq_H .= 0; 
    CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H_1!(tmp1, tmp2, qHx, dR_qHx, qHy, dR_qHy, β, H, dR_H, B, zs_sample, ms_sample, z_ELA, b_max, dz_sample, nx, ny, dx, dy)
    
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
        as[[1,end],:] .= as[[2,end-1],:]
        as[:,[1,end]] .= as[:,[2,end-1]] 
        as, as2 = as2, as 
    end 
    return 
end 

function adjoint_2D()
    # load Rhone data: surface elevation and bedrock 
    rhone_data = h5open("Rhone_data_padding/alps/data_Rhone.h5","r")
    xc = rhone_data["glacier/x"][:,1]
    yc = rhone_data["glacier/y"][1,:]
    B  = rhone_data["glacier/z_bed"][]
    S  = rhone_data["glacier/z_surf"][]
    @show size(xc) size(yc)
    @show(size(B))
    @show(extrema(B))

    close(rhone_data)
    # load Rhone data: mass balance elevation band
    df = readdlm("glacier_data/belev_01238.dat")
    zs_sample = convert(Vector{Float64},df[2:end,1])
    ms_sample = convert(Vector{Float64},df[2:end,2])
    dz_sample = zs_sample[2]-zs_sample[1]
    zs_sample = CuArray{Float64}(zs_sample)
    ms_sample = CuArray{Float64}(ms_sample)

    p1= Plots.plot(zs_sample, ms_sample)
    display(plot(p1))
    
    @show(typeof(df))
    @show(size(df))
    @show(size(zs_sample))
    @show(size(ms_sample))
    @show(typeof(zs_sample))
    @show(typeof(ms_sample))

    # also convert the units 


    # physics parameters
    s2y = 3600*24*365.35 # seconds to years 
    n = 3
    ρg = 970*9.8 
    a0 = 1.5e-24
    z_ELA = 980#1200#670#125#200
    b_max = 0.08#0.1
    B0 = 200
    sm = 50

    #numerics parameters 
    gd_niter = 100
    bt_niter = 3
    nx,ny=size(B)
    epsi = 1e-2
    dmp = 0.3#0.7
    # dmp_adj = 50*1.7
    dmp_adj = 2*1.7#0*1.7#200*1.7#50*1.7#2*1.7
    ϵtol = 1e-4
    ϵtol_adj = 1e-8
    gd_ϵtol =1e-3
    γ0 = 1.0e-10#1.0e-9#5.0e-9#1.0e-10
    niter = 500000
    ncheck = 1000
    ncheck_adj = 100
    threads = (16,16) 
    blocks = ceil.(Int,(nx,ny)./threads)


    #derived numerics 
    dx = xc[2]-xc[1]
    dy = yc[2]-yc[1]
    lx, ly = dx*nx, dy*ny
    xc = LinRange(dx/2, lx-dx/2, nx)
    yc = LinRange(dy/2, ly-dy/2, ny)
    @show(size(xc))
    @show(size(yc))
    x0 = xc[round(Int,nx/2)]
    y0 = yc[round(Int,ny/2)]

    # derived physics 
    a  = 2.0*a0/(n+2)*ρg^n*s2y 
    as = 5.7e-20 
    cfl = max(dx^2,dy^2)/4.1

    # convert the units of the mass balance term from mw/year to mi/second 
    @. ms_sample = ms_sample/s2y*1000/917

    # smoother 
    for is = 1:sm
        B[2:end-1, 2:end-1] .= B[2:end-1,2:end-1] .+ 1.0/4.0.*(diff(diff(B[:,2:end-1], dims=1), dims=1) .+ diff(diff(B[2:end-1,:], dims=2), dims=2))
        B[[1,end],:] .= B[[2,end-1],:]
        B[:,[1,end]] .= B[:,[2,end-1]]
    end 
    for is = 1:sm 
        # Array could not be broadcast to a common size; got a dimension with length 381 and 380 
        S[2:end-1, 2:end-1] .= S[2:end-1,2:end-1] .+ 1.0/4.0.*(diff(diff(S[:,2:end-1], dims=1), dims=1) .+ diff(diff(S[2:end-1,:], dims=2), dims=2))
        S[[1,end],:] .= S[[2,end-1],:]
        S[:,[1,end]] .= S[:,[2,end-1]]
    end 
    # initialization 
    H = zeros(Float64, nx, ny)

    H  = max.(0.0, S.-B)
    H_obs = copy(H)
    H_ini = copy(H)
    S_obs = copy(S)

    p1= Plots.contour(xc, yc, H'; levels=20, aspect_ratio=1)
    p2 = Plots.contour(xc, yc, H_ini'; levels=20, aspect_ratio=1)
    p3 = Plots.contour(xc, yc, H_obs'; levels=20, aspect_ratio=1)
    display(plot(p1,p2,p2; layout=(1,3),size=(980,980)))

    
    #B = @. B0*(exp(-((xc-x0)/w)^2-((yc'-y0)/w)^2))*sin(ω*pi*(xc+yc'))
    #smoother 
    p1 = plot(xc,yc,B'; st=:surface, camera =(20,25), aspect_ratio=1)
    p2 = Plots.contour(xc, yc, B'; levels =20, aspect_ratio=1)
    p3 = Plots.contour(xc,yc, H'; levels=20, aspect_ratio=1)
    p4 = Plots.contourf(xc, yc, (B.+H)'; levels=20, aspect_ratio=1)
    display(plot(p1,p2,p3,p4; layout=(2,2), size=(980,980)))
    #error("initial display")


    #B[2:end-1, 2:end-1] .= B[2:end-1, 2:end-1] .+ 1.0/4.1.*(diff(diff(B[:, 2:end-1], dims=1),dims=1) .+ diff(diff(B[2:end-1,:], dims=2), dims=2)) 
    #B[[1,end],:] .= B[[2,end-1],:]
    #B[:,[1,end]] .= B[:,[2,end-1]]
    S = CuArray{Float64}(S)
    B = CuArray{Float64}(B)
    H = CuArray{Float64}(H)
    S_obs = CuArray{Float64}(S_obs)
    H_obs = CuArray{Float64}(H_obs)
    H_ini = CuArray{Float64}(H_ini)

    #@show(extrema(B))
    #p1 = heatmap(xc, yc, Array((H.+B)'); title = "S (forward problem initial)")
    #plot!(xc, Array(B); title = "S (forward problem)", ylims=(0,1000))
    #display(plot(p1))
    #error("check forward model")

    D = CUDA.zeros(Float64,nx-1, ny-1)
    av_ya_∇Sx = CUDA.zeros(Float64, nx-1, ny-1)
    av_xa_∇Sy = CUDA.zeros(Float64, nx-1, ny-1) 
    gradS = CUDA.zeros(Float64, nx-1, ny-1)
    qHx = CUDA.zeros(Float64, nx-1,ny-2)
    qHy = CUDA.zeros(Float64, nx-2,ny-1)
    M  = CUDA.zeros(Float64, nx,ny)
    mb = CUDA.zeros(Float64, nx,ny)
    

    #S_obs .= B .+ H_obs
    #CUDA.@sync @cuda threads=threads blocks=blocks update_S!(H_obs, S_obs, B, nx)
    #S .= B .+ H 
    #CUDA.@sync @cuda threads=threads blocks=blocks update_S!(H, S, B, nx)
    
    # how to define β_init β_syn 
    β = 0.00040*CUDA.ones(nx, ny)
    #0.00025# 0.00010 
    
    as = 2.0e-2*CUDA.ones(nx-1, ny-1) 
    as_ini = copy(as)
    as_syn = 5.7e-20*CUDA.ones(nx-1,ny-1)
    as2 = similar(as) 

    Jn = CUDA.zeros(Float64,nx-1, ny-1)
    dqH_D = CUDA.zeros(Float64, nx-1, ny-1)
    # display
    # synthetic problem
    
    #Forwardproblem(H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, M, β, as, zs_sample, ms_sample, mb, n, a, b_max, z_ELA, dz_sample, dx, dy, nx, ny, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp)
    #AdjointProblem(H, H_obs,B, D, qHx, qHy, β, as, gradS, av_ya_∇Sx, av_xa_∇Sy, zs_sample, ms_sample, z_ELA, b_max, dz_sample, nx, ny, dx, dy, a, n, dmp, ϵtol, niter, ncheck, threads, blocks)
    synthetic_problem = Forwardproblem(H_obs, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, M, β,as_syn, zs_sample, ms_sample, mb, n, a, b_max, z_ELA, dz_sample, dx, dy, nx, ny, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp)
    forward_problem = Forwardproblem(H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, M, β,as, zs_sample, ms_sample, mb, n, a, b_max, z_ELA, dz_sample, dx, dy, nx, ny, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp)
    adjoint_problem = AdjointProblem(H, H_obs,B, D, qHx, qHy, β, as, gradS, av_ya_∇Sx, av_xa_∇Sy, zs_sample, ms_sample, z_ELA, b_max, dz_sample, nx, ny, dx, dy, a, n, dmp_adj, ϵtol_adj, niter, ncheck_adj, threads, blocks)

    println("generating synthetic data (nx = $nx, ny = $ny)...")
    solve!(synthetic_problem)
    p1= Plots.contour(xc, yc, Array(H_obs'); levels=20, aspect_ratio=1)
    display(plot(p1))
    error("check")
    println("done.")
    solve!(forward_problem)
    #H_ini .= forward_problem.H
    println("gradient descent")

    #print(H)
    #print(H_obs)

    #S_obs .= B .+ H_obs 
    #S .= B .+ H 
    #CUDA.@sync @cuda threads=threads blocks=blocks update_S!(H_obs, S_obs, B, nx)
    #CUDA.@sync @cuda threads=threads blocks=blocks update_S!(H, S, B, nx)
    γ = γ0
    J_old = 0.0; J_new = 0.0 
    J_old = sqrt(cost(H, H_obs)*dx*dy)
    @show(maximum(abs.(H.-H_obs)))
    #@show(maximum(J_old))
    #CUDA.@sync @cuda threads=threads blocks=blocks cost!(H, H_obs, J_old, nx)
    #@printf("initial cost function equals %.1e\n", J_old)
    J_ini = J_old
    
    J_evo = Float64[]; iter_evo = Int[]
    # gradient descent iterations to update n 
    for gd_iter = 1:gd_niter 
        as_ini .= as
        solve!(adjoint_problem)
        println("compute cost gradient")
        cost_gradient!(Jn, adjoint_problem)
        println("compute cost gradient done")
        #check Jn
        # line search 
        for bt_iter = 1:bt_niter 
            @. as = clamp(as-γ*Jn, 0.0, 100.0)
            # mask = (Jn .!= 0)
            # @. as[mask] = exp.(log.(as[mask]) - γ*sign(Jn[mask])*log.(abs(Jn[mask])))
            # p1 = heatmap(xc[1:end-1], yc[1:end-1], Array(as); title="as")
            # display(p1)
            # error("stop")
            as[[1,end],:] .= as[[2,end-1],:]
            as[:,[1,end]] .= as[:, [2,end-1]]
            smooth!(as, as2, H, nx, ny, 10, threads, blocks)
            forward_problem.H .= H_ini
            # update H 
            solve!(forward_problem)
            #update cost functiion 
            
            @show(maximum(abs.(H.-H_obs)))
            #@show(H_obs)
            J_new = sqrt(cost(H, H_obs)*dx*dy)
            #p1 = heatmap(xc, yc,  Array(H_obs'); xlabel="x", ylabel="y", label= "H_obs" ,title = "Ice thickness H(m)")
            #p2 = heatmap!(xc, yc, Array(H'); xlabel="x", ylabel="y", label= "H" ,title = "Ice thickness H(m)")
            #display(plot(p1, p2))
            #error("zero J_new")
            #@show(J_new)
            if J_new < J_old 
                # we accept the current value of γ
                γ *= 1.1 
                J_old = J_new
                
                @printf("new solution accepted\n")
                break
            else
                as .= as_ini
                γ = max(γ*0.5, γ0* 0.1)
            end 
        # end of the line search loops 
        end 



        push!(iter_evo, gd_iter); push!(J_evo, J_old/J_ini)
        CUDA.@sync @cuda threads=threads blocks=blocks update_S!(S, H, B, nx, ny)
        if DO_VISU 
             #p1 = heatmap(xc, yc,  Array(H_ini'); xlabel="x", ylabel="y", label= "Initial state of H" ,title = "Ice thickness H(m)")
             #heatmap!(xc, yc, Array(H_obs'); label = "Synthetic value of H")
             #heatmap!(xc, yc,  Array(B'); label="Bed rock")
             #p1=contour(xc,yc, Array((H.-H_obs)'), levels=20, color =:turb)
             p1=heatmap(xc, yc, Array((H.+B)'); levels=20, color =:turbo, label="H observation", aspect_ratio = 1)
             contour!(xc, yc, Array(H'); levels=0.01:0.01, lw=2.0, color=:black, line=:solid,label="Current state of H")
             contour!(xc, yc, Array(H_obs'); levels=0.01:0.01, lw=2.0, color=:red, line=:dash,label="Current state of H")
             p2 = heatmap(xc, yc, Array(as'); xlabel = "x", ylabel = "y", label="as", title = "as", aspect_ratio=1)
             #heatmap!(xc[1:end-1], yc[1:end-1],Array(as_syn); label="Syntheic value of as")
             #heatmap!(xc[1:end-1], yc[1:end-1], Array(log10.(as)); label="Current state of as")
             p3 = heatmap(xc[1:end-1], yc[1:end-1], Array(Jn'); xlabel="x", ylabel="y",title = "Gradient of cost function Jn", aspect_ratio=1)
             p4 = plot(iter_evo, J_evo; shape = :circle, xlabel="Iterations", ylabel="J_old/J_ini", title="misfit", yaxis=:log10)
             p5 = contourf(xc,yc, Array(H_ini'),levels = 20, color=:turbo)
             xlabel!("X")
             ylabel!("Y")
             title!("Initial state of H (m)")
             p6 = contourf!(xc, yc, Array(H'), levels=20, color=:turbo)
             xlabel!("X")
             ylabel!("Y")
             title!("Current State of H (m)")
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