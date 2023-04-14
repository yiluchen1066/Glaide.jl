using CUDA,BenchmarkTools
using Plots,Plots.Measures,Printf
using Enzyme 
using HDF5, DelimitedFiles, Rasters
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

function residual!(RH, qHx, qHy, β, H, B, mask, z_ELA, b_max, nx, ny , dx, dy)
    @get_thread_idx(H)
    if ix <= nx-2 && iy <= ny -2 
        RH[ix+1, iy+1] = -(@d_xa(qHx)/dx + @d_ya(qHy)/dy) + mask[ix+1,iy+1]*min(β[ix+1,iy+1]*(H[ix+1, iy+1]+B[ix+1, iy+1]-z_ELA[ix+1, iy+1]), b_max)
    end 
    return 
end 


function compute_abs_error!(Err_abs, qHx, qHy, β, H, B, mask, z_ELA, b_max, nx, ny , dx, dy)
    @get_thread_idx(H)
    if ix <= nx-2 && iy <= ny-2 
        Err_abs[ix+1,iy+1] = -(@d_xa(qHx)/dx + @d_ya(qHy)/dy) + mask[ix+1, iy+1]*min(β[ix+1,iy+1]*(H[ix+1, iy+1]+B[ix+1, iy+1]-z_ELA[ix+1, iy+1]), b_max)
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
    H::A; B::A; D::A; gradS::A; av_ya_∇Sx::A; av_xa_∇Sy::A; qHx::A; qHy::A; β::A; as::A; mask::A
    RH::A; Err_rel::A; Err_abs::A
    n::Int; a::T; b_max::T; z_ELA::A; dx::T; dy::T; nx::Int; ny::Int; epsi::T; cfl::T; ϵtol::NamedTuple{(:abs, :rel), Tuple{Float64, Float64}}; maxiter::Int; ncheck::Int; threads::Tuple{Int, Int}; blocks::Tuple{Int, Int}
end 

function Forwardproblem(H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β, as, mask, n, a, b_max, z_ELA, dx, dy, nx, ny, epsi, cfl, ϵtol, maxiter, ncheck, threads, blocks)
    RH      = similar(H,nx, ny)
    Err_rel = similar(H, nx, ny)
    Err_abs = similar(H, nx, ny)
    return Forwardproblem(H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β, as, mask, RH, Err_rel, Err_abs, n, a, b_max, z_ELA, dx, dy, nx, ny, epsi, cfl, ϵtol, maxiter, ncheck, threads, blocks) 
end 

#solve for H using pseudo-trasient method 
function solve!(problem::Forwardproblem)
    (;H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β, as, mask, RH, Err_rel, Err_abs, n, a, b_max, z_ELA, dx,dy, nx, ny, epsi, cfl, ϵtol, maxiter, ncheck, threads, blocks) = problem
    RH .= 0; Err_rel .= 0; Err_abs .= 0
    lx, ly = 30e3, 30e3
    err_abs0 = 0.0 
    
    for iter in 1:maxiter
        if iter % ncheck == 0 
            CUDA.@sync @cuda threads = threads blocks = blocks compute_rel_error_1!(Err_rel, H, nx, ny)
        end 
        CUDA.@sync @cuda threads=threads blocks=blocks compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, a, as, n, nx, ny, dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks compute_q!(qHx, qHy, D, H, B, nx, ny, dx, dy)
        #CUDA.@sync @cuda threads=threads blocks=blocks timestep!(dτ, H, D, cfl, epsi, nx, ny)
        dτ=1.0/(16.1*maximum(D)/dx^2 + maximum(β))
        CUDA.@sync @cuda threads=threads blocks=blocks residual!(RH, qHx, qHy, β, H, B, mask, z_ELA, b_max, nx, ny , dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks update_H!(H, RH, dτ, nx, ny)
        CUDA.@sync @cuda threads=threads blocks=blocks set_BC!(H, nx,ny)
        if iter ==1 || iter % ncheck == 0 
            CUDA.@sync @cuda threads=threads blocks=blocks compute_abs_error!(Err_abs, qHx, qHy, β, H, B, mask, z_ELA, b_max, nx, ny , dx, dy)
            CUDA.@sync @cuda threads=threads blocks=blocks compute_rel_error_2!(Err_rel, H, nx, ny)
            if iter ==1 
                err_abs0 = maximum(abs.(Err_abs))
            end 
            err_abs = maximum(abs.(Err_abs))/err_abs0 
            err_rel = maximum(abs.(Err_rel))/maximum(H)
            @printf("iter/nx^2 = %.3e, err = [abs =%.3e, rel= %.3e] \n", iter/nx^2, err_abs, err_rel)
            #p1 = heatmap(Array(H'); title = "H (forward problem)")
            #p2 = plot(Array(B[160,:] .+ H[160,:]), linecolor=:blue)
            #p2 = plot!(Array(B[160,:]))
            #p2 = heatmap(Array(Err_abs'); title = "Err_abs")
            #p3 = heatmap(Array(Err_rel'); title = "Err_rel")
            #display(plot(p1,p2))
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
#residual!(RH, qHx, qHy, β, H, B, mask, z_ELA, b_max, nx, ny , dx, dy)
function grad_residual_H_1!(tmp1, tmp2, qHx, dR_qHx, qHy, dR_qHy, β, H, dR_H, B, mask, z_ELA, b_max, nx, ny, dx, dy)
    #tmp2 .= r 
    Enzyme.autodiff_deferred(residual!, Duplicated(tmp1, tmp2), Duplicated(qHx, dR_qHx), Duplicated(qHy, dR_qHy), Const(β), Duplicated(H, dR_H), Const(B), Const(mask),Const(z_ELA), Const(b_max), Const(nx), Const(ny), Const(dx), Const(dy))
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
    H::A; H_obs::A; B::A; D::A; qHx::A; qHy::A; β::A; as::A; gradS::A; av_ya_∇Sx::A; av_xa_∇Sy::A; mask::A
    r::A; dR::A; R::A; Err::A; dR_qHx::A; dR_qHy::A; dR_H::A; dq_D::A; dq_H::A; dD_H::A 
    tmp1::A; tmp2::A; ∂J_∂H::A
    z_ELA::A; b_max::T; nx::Int; ny::Int; dx::T; dy::T; a::T; n::Int; dmp::T; H_cut::T; ϵtol::T; niter::Int; ncheck::Int; threads::Tuple{Int, Int}; blocks::Tuple{Int, Int}
end 


function AdjointProblem(H, H_obs,B, D, qHx, qHy, β, as, gradS, av_ya_∇Sx, av_xa_∇Sy, mask, z_ELA, b_max, nx, ny, dx, dy, a, n, dmp, H_cut, ϵtol, niter, ncheck, threads, blocks)
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
    return AdjointProblem(H, H_obs,B, D, qHx, qHy, β, as, gradS, av_ya_∇Sx, av_xa_∇Sy, mask, r, dR, R, Err, dR_qHx, dR_qHy, dR_H, dq_D, dq_H, dD_H, tmp1, tmp2, ∂J_∂H, z_ELA, b_max, nx, ny, dx, dy, a, n, dmp, H_cut, ϵtol, niter, ncheck, threads, blocks)
end 


# solve for r using pseudo-trasient method to compute sensitvity afterwards 
function solve!(problem::AdjointProblem) 
    (; H, H_obs, B, D, qHx, qHy, β, as, gradS, av_ya_∇Sx, av_xa_∇Sy, mask, r, dR, R, Err, dR_qHx, dR_qHy, dR_H, dq_D, dq_H, dD_H, tmp1, tmp2, ∂J_∂H, z_ELA, b_max, nx, ny, dx, dy, a, n, dmp, H_cut, ϵtol, niter, ncheck, threads, blocks) = problem
    r.= 0;  R.= 0; Err .= 0; dR .= 0; dR_qHx .= 0; dR_qHy .= 0; dR_H .= 0; dq_D .= 0; dq_H .= 0; dD_H .= 0

    #dt = min(dx^2, dy^2)/maximum(D)/60.1/2
    #dt = min(dx^2, dy^2)/maximum(D)/60.1
    dt = 1.0/(8.1*maximum(D)/min(dx,dy)^2 + maximum(β))/10.0
    ∂J_∂H .= (H .- H_obs)#./sqrt(length(H))
    # initialization for tmp1 tmp2 tmp3 tmp4 
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol  && iter < niter 
        #Err .= r 
        # compute dQ/dH 
        dR_qHx .= 0; dR_qHy .= 0 ; dR_H .= 0; dq_D .= 0.0; dq_H .= 0; dD_H .= 0
        dR .= .-∂J_∂H; tmp2 .= r 
        
        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H_1!(tmp1, tmp2, qHx, dR_qHx, qHy, dR_qHy, β, H, dR_H, B, mask, z_ELA, b_max, nx, ny, dx, dy)
        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H_2!(qHx, dR_qHx, qHy, dR_qHy, D, dq_D, H, dq_H, B, nx, ny, dx, dy)
        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H_3!(D, dq_D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, dD_H, B, a, as, n, nx, ny, dx, dy)
        CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H!(dR, dD_H, dq_H, dR_H, nx, ny)
        CUDA.@sync @cuda threads = threads blocks = blocks update_r!(r, R, dR, dt, H, H_cut, dmp, nx, ny)

        
        # if iter > 10 error("Stop") end
        if iter % ncheck == 0 
            #@. Err -= r 
            #@show(size(dR_qHx))
            merr = maximum(abs.(dt.*R[2:end-1,2:end-1]))/maximum(abs.(r))
            #p1 = heatmap(Array(r'); aspect_ratio=1, title = "r")
            #p2 = heatmap(Array(R'); aspect_ratio=1, title = "R")
            #p3 = heatmap(Array(dR'); aspect_ratio=1, title="dR")
            # # savefig(p1, "adjoint_debug/adjoint_R_$(iter).png")
            #display(plot(p1,p2,p3))
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
    (; H, H_obs, B, D, qHx, qHy, β, as, gradS, av_ya_∇Sx, av_xa_∇Sy, mask, r, dR, R, Err, dR_qHx, dR_qHy, dR_H, dq_D, dq_H, dD_H, tmp1, tmp2, ∂J_∂H, z_ELA, b_max, nx, ny, dx, dy, a, n, dmp, H_cut, ϵtol, niter, ncheck, threads, blocks) = problem
    # dimensionmismatch: array could not be broadcast to match destination 
    Jn .= 0.0
    tmp2 .= r
    dR_qHx .= 0; dR_qHy .= 0; dR_H .= 0; dq_D .= 0; dq_H .= 0; 
    CUDA.@sync @cuda threads = threads blocks = blocks grad_residual_H_1!(tmp1, tmp2, qHx, dR_qHx, qHy, dR_qHy, β, H, dR_H, B, mask, z_ELA, b_max, nx, ny, dx, dy)
    
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

@views function load_data(bed_dat, surf_dat)
    z_bed  = reverse(dropdims(Raster(bed_dat); dims=3), dims=2)
    z_surf = reverse(dropdims(Raster(surf_dat); dims=3), dims=2)
    xy = DimPoints(dims(z_bed, (X, Y)))
    (x,y) = (first.(xy), last.(xy))
    return z_bed.data, z_surf.data, x.data[:,1], y.data[1,:]
end

@views function smooth_2!(A, nsm)
    for _ = 1:nsm
        @inbounds A[2:end-1, 2:end-1] .= A[2:end-1, 2:end-1] .+ 1.0 / 4.1 .* (diff(diff(A[:, 2:end-1], dims=1), dims=1) .+ diff(diff(A[2:end-1, :], dims=2), dims=2))
        @inbounds A[[1, end], :] .= A[[2, end - 1], :]
        @inbounds A[:, [1, end]] .= A[:, [2, end - 1]]
    end
    return
end

function adjoint_2D()
    # load the data 
    nsm = 10
    B_rhone, S_rhone, xc, yc = load_data("Rhone_data_padding/Archive/Rhone_BedElev_cr.tif", "Rhone_data_padding/Archive/Rhone_SurfElev_cr.tif")

    
    H_rhone                  = S_rhone .- B_rhone 

    smooth_2!(B_rhone, nsm)
    smooth_2!(S_rhone, nsm)
    smooth_2!(H_rhone, nsm)
    
    #H_rhone                  = 0.5.*(S_rhone .- B_rhone)
    
    lx       = xc[end]-xc[1] #3820m
    ly       = yc[end]-yc[1] #8410m
    @show(size(B_rhone)[1])
    @show(size(S_rhone)[2])
    @show(size(H_rhone))
    # plots the loaded bedrock and surface elevation as initial condition
    p1 = plot(xc,yc,B_rhone'; st=:surface, camera =(20,25), aspect_ratio=1)
    p2 = Plots.contour(xc, yc, B_rhone'; levels =20, aspect_ratio=1)
    p3 = heatmap(xc, yc, S_rhone', aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), xrotation=45, title="elevation")
    p4 = heatmap(xc, yc, H_rhone', aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), xrotation=45, title="ice thickness")
    display(plot(p1,p2,p3,p4))

    

    # power law components 
    n        =  3 
    # dimensionally independet physics 
    l        =  1e3#1e4#1.0 # length scale lx = 1e3 (natural value)
    aρgn0    =  1.3517139631340713e-12 #1.0 # A*(ρg)^n # = 1.3475844936008e-12 (natural value)
    #scales 
    tsc      =  1/aρgn0/l^n # also calculated from natural values tsc = 739.8014870553008
    #non-dimensional numbers (calculated from natural values)
    #s_f_syn  = 0.03 # sliding to ice flow ratio: s_f_syn = asρgn0_syn/aρgn0/lx^2
    s_f      = 5.0#2.631578947368421 # sliding to ice flow ratio: s_f   = asρgn0/aρgn0/lx^2
    m_max_nd = 2.697779395337379e-8#m_max/lx*tsc m_max = 2.0 m/a lx = 1e3 tsc = 
    βtsc     = 1.12385736177553e-7#ratio between characteristic time scales of ice flow and accumulation/ablation βtsc = β0*tsc 
    # geometry
    z_ela_l  = 2.9355889409400002 # ela to domain length ratio z_ela_l = 
    # numerics
    H_cut_l  = 1.0e-6
    # dimensional  dependent physics parameters
    z_ELA_0     = z_ela_l*l # 2935.58894094
    H_cut       = H_cut_l*l # 1.0e-2
    #asρgn0_syn  = s_f_syn*aρgn0*l^2 #5.7e-20*(ρg)^n = 5.7e-20*(910*9.81)^3 = 4.055141889402214e-8
    asρgn0      = s_f*aρgn0*l^2 #5.0e-18*(ρg)^n = 3.5571420082475557e-6
    m_max       = m_max_nd*l/tsc  # 1.15 m/a = 3.646626078132927e-8 m/s
    β0          = βtsc/tsc    #0.00479074/a = 1.5191336884830034e-10


    M           = min.(β0.*(H_rhone .+ B_rhone .- z_ELA_0), m_max)
    Mask        = ones(Float64, size(H_rhone))
    Mask[M.>0.0 .&& H_rhone.<=0.0] .= 0.0

    #p1 = plot(xc,yc,B_rhone'; st=:surface, camera =(20,25), aspect_ratio=1)
    #p2 = Plots.contour(xc, yc, B_rhone'; levels =20, aspect_ratio=1)
    #p3 = heatmap(xc, yc, S_rhone', aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), xrotation=45, title="elevation")
    p4 = heatmap(xc, yc, (H_rhone.*Mask)', aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), xrotation=45, title="ice thickness")
    display(plot(p4))


    @show(z_ELA_0)
    @show(H_cut)
    @show(asρgn0)
    @show(m_max)
    @show(β0)


    # numerics 
    gd_niter    = 100
    bt_niter     = 3 
    nx          = size(B_rhone)[1]
    ny          = size(B_rhone)[2]
    epsi        = 1e-4 
    ϵtol        = (abs = 1e-8, rel = 1e-5)
    dmp_adj     = 2*1.7 
    ϵtol_adj    = 5e-5
    gd_ϵtol     = 1e-3 
    γ0          = 1.0e-2
    maxiter     = 500000
    ncheck      = 1000
    ncheck_adj  = 1000
    threads     = (16,16)
    blocks      = ceil.(Int, (nx,ny)./threads)


    # derived numerics
    dx          = xc[2]-xc[1]
    dy          = yc[2]-yc[1]
    x0          = xc[round(Int, nx/2)]
    y0          = yc[round(Int, ny/2)]
    cfl          = max(dx^2, dy^2)/8.1


    # initialization 
    β = β0*ones(Float64, nx, ny)
    ela = z_ELA_0*ones(Float64, nx, ny)

    H     = copy(H_rhone)
    H_obs = copy(H_rhone)
    H_ini = copy(H_rhone)
    S     = copy(S_rhone)
    S_obs = copy(S_rhone)
    B     = copy(B_rhone)

    p1 = plot(xc,yc,B'; st=:surface, camera =(20,25), aspect_ratio=1)
    p2 = Plots.contour(xc, yc, B'; levels =20, aspect_ratio=1)
    p3 = heatmap(xc, yc, S', aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), xrotation=45, title="elevation")
    p4 = heatmap(xc, yc, H', aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), xrotation=45, title="ice thickness")
    display(plot(p1,p2,p3,p4))
    
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
    mask  = CuArray{Float64}(Mask)

    D = CUDA.zeros(Float64,nx-1, ny-1)
    av_ya_∇Sx = CUDA.zeros(Float64, nx-1, ny-1)
    av_xa_∇Sy = CUDA.zeros(Float64, nx-1, ny-1) 
    gradS = CUDA.zeros(Float64, nx-1, ny-1)
    qHx = CUDA.zeros(Float64, nx-1,ny-2)
    qHy = CUDA.zeros(Float64, nx-2,ny-1)


    as = asρgn0*CUDA.ones(nx-1, ny-1) 
    as_ini = copy(as)
    as2 = similar(as) 

    Jn = CUDA.zeros(Float64,nx-1, ny-1)
    

    #Forwardproblem(H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β, as, Mask, n, a, b_max, z_ELA, dx, dy, nx, ny, epsi, cfl, ϵtol, maxiter, ncheck, threads, blocks)
    #synthetic_problem = Forwardproblem(H_obs, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β,as_syn, mask, n, aρgn0, m_max, ela, dx, dy, nx, ny, epsi, cfl, ϵtol, maxiter, ncheck, threads, blocks)
    
    forward_problem = Forwardproblem(H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β,as, mask, n, aρgn0, m_max, ela, dx, dy, nx, ny, epsi, cfl, ϵtol, maxiter, ncheck, threads, blocks)

    adjoint_problem = AdjointProblem(H, H_obs,B, D, qHx, qHy, β, as, gradS, av_ya_∇Sx, av_xa_∇Sy, mask, ela, m_max, nx, ny, dx, dy, aρgn0, n, dmp_adj, H_cut, ϵtol_adj, maxiter, ncheck_adj, threads, blocks)

    println("generating synthetic data (nx = $nx, ny = $ny)...")
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
             p1=heatmap(xc, yc, Array((H)'); levels=20, color =:turbo, aspect_ratio = 1)
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