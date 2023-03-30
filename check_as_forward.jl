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

function residual!(RH, qHx, qHy, β, H, B, z_ELA, b_max, nx, ny, dx, dy)
    @get_thread_idx(H)
    if ix <= nx-2 && iy <= ny-2 
        #RH[ix+1,iy+1] = - (@d_xa(qHx)/dx + @d_ya(qHy)/dy)
        RH[ix+1,iy+1] =  -(@d_xa(qHx)/dx + @d_ya(qHy)/dy) + min(β[ix+1,iy+1]*(H[ix+1, iy+1]+B[ix+1, iy+1]-z_ELA), b_max)
    end 
    return 
end 

function timestep!(dτ, H, D, cfl, epsi, nx, ny)
    @get_thread_idx(H)
    if ix <= nx-2 && iy <= ny-2 
        dτ[ix,iy] = min(20.0, cfl/(epsi+@av_xy(D)))
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


mutable struct Forwardproblem{T<:Real, A<:AbstractArray{T}}
    H::A; B::A; D::A; gradS::A; av_ya_∇Sx::A; av_xa_∇Sy::A; qHx::A; qHy::A; β::A; as::A
    dτ::A; dHdτ::A; RH::A; Err::A
    n::Int; a::T; b_max::T; z_ELA::Int; dx::T; dy::T; nx::Int; ny::Int; epsi::T; cfl::T; ϵtol::T; niter::Int; ncheck::Int; threads::Tuple{Int, Int}; blocks::Tuple{Int, Int}; dmp::T
end 

function Forwardproblem(H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β, as, n, a, b_max, z_ELA, dx, dy, nx, ny, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp)
    RH = similar(H,nx, ny)
    dHdτ = similar(H, nx-2, ny-2) 
    dτ   = similar(H, nx-2, ny-2) 
    Err  = similar(H, nx, ny)
    return Forwardproblem(H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β, as,dτ, dHdτ, RH, Err, n, a, b_max, z_ELA, dx, dy, nx, ny, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp) 
end 

#solve for H using pseudo-trasient method 
function solve!(problem::Forwardproblem)
    (;H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β, as, dτ, dHdτ,RH, Err, n, a, b_max, z_ELA, dx,dy, nx, ny, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp) = problem
    dHdτ .= 0; RH .= 0; dτ .= 0; Err .= 0
    merr = 2ϵtol; iter = 1 
    lx, ly = 30e3, 30e3
    xc = LinRange(dx/2, lx-dx/2, nx)
    yc = LinRange(dy/2, ly-dy/2, ny)
    #p1 = plot(xc, Array(H); title = "H_init (forward problem)")
    #display(plot(p1))
    while merr >= ϵtol && iter < niter 
        #Err .= H 
        #CUDA.@sync @cuda threads=threads blocks=blocks residual!(RH, H, B, qHx, β, b_max, z_ELA, dx, nx)
        CUDA.@sync @cuda threads=threads blocks=blocks compute_error_1!(Err, H, nx, ny)
        CUDA.@sync @cuda threads=threads blocks=blocks compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, a, as, n, nx, ny, dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks compute_q!(qHx, qHy, D, H, B, nx, ny, dx, dy)
        #CUDA.@sync @cuda threads=threads blocks=blocks compute_qHx!(qHx, D, H, B, dx, nx, ny)
        #CUDA.@sync @cuda threads=threads blocks=blocks compute_qHy!(qHy, D, H, B, dy, nx, ny)
        CUDA.@sync @cuda threads=threads blocks=blocks residual!(RH, qHx, qHy, β, H, B, z_ELA, b_max, nx, ny, dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks timestep!(dτ, H, D, cfl, epsi, nx, ny)
        CUDA.@sync @cuda threads=threads blocks=blocks update_H!(H, dHdτ, RH, dτ, dmp, nx, ny)# update H
        CUDA.@sync @cuda threads=threads blocks=blocks set_BC!(H, nx, ny)

        if iter  % ncheck == 0 
            #@. Err -= H 
            CUDA.@sync @cuda threads=threads blocks =blocks compute_error_2!(Err, H, nx, ny)
            #merr = maximum(abs.(Err[:]))
            #@show(merr)
            merr = (sum(abs.(Err[:,:]))./nx./ny)
            (isfinite(merr) && merr >0) || error("forward solve failed")
            #@printf("error = %1.e\n", merr)
            
            
            #p1 = heatmap(xc, yc, Array((H.+B)'); title = "S (forward problem)")
            #p2 = Plots.plot(xc, Array(H[ceil(Int,nx/2),:]);title="Ice thickness")
            #display(plot(p1,p2;layout=(2,2)))
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

function adjoint_2D()
    # physics parameters
    lx, ly, lz = 30e3, 30e3, 1e3
    ox, oy, oz = -lx/2, -ly/2, 0.0 
    ω = 0.0001
    w = 0.4*lx 
    w1 = 0.45*lx
    n = 3
    ρg = 970*9.8
    z_ELA = 700#125#200
    b_max = 0.0005#0.1
    B0 = 200

    #numerics parameters 
    gd_niter = 100
    bt_niter = 3
    nx = 512#512#63
    ny = 512#512#63
    epsi = 1e-4
    dmp = 0.8#0.7
    # dmp_adj = 50*1.7
    dmp_adj = 2*1.7
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
    dx = lx/nx
    dy = ly/ny
    xv = LinRange(ox, ox+lx, nx+1)
    yv = LinRange(oy, oy+ly, ny+1)
    xc = 0.5*(xv[1:end-1]+xv[2:end])
    yc = 0.5*(yv[1:end-1]+yv[2:end])
    #xc = LinRange(dx/2, lx-dx/2, nx)
    #yc = LinRange(dy/2, ly-dy/2, ny)
    @show(size(xc))
    @show(size(yc))
    x0 = xc[round(Int,nx/2)]
    y0 = yc[round(Int,ny/2)]

    # derived physics 
    aρgn = 1.9e-24*ρg^n
    asρgn0 = 5.7e-20*ρg^n
    cfl = max(dx^2,dy^2)/8.1

    #bedrock properties
    wx1, wy1 = 0.4lx, 0.2ly 
    wx2, wy2 = 0.15lx, 0.15ly 
    ω1 = 4 
    ω2 = 6

    # initialization 
    S = zeros(Float64, nx, ny)
    H = zeros(Float64, nx, ny)
    B = zeros(Float64, nx, ny)
    #H = @. exp(-(xc - lx/4)^2)
    #H = @. exp(-(xc - x0)^2)
    #H    = @. 200*exp(-((xc-x0)/5000)^2-((yc'-y0)/5000)^2) 
    #H = @. (B0+100)*(exp(-((xc-x0)/w1)^2-((yc'-y0)/w1)^2))*sin(ω*pi*(xc+yc'))+(B0+100)
    #@. H_slope =  1.0*(xc-x0)+1.5*(yc-y0)'
    #H    = H_gaussian .+ H_slope
    H_obs = copy(H)
    H_ini = copy(H)
    S_obs = copy(S)

    # sinwave (elevation and valley baedrock initialization)
    fun = @. 0.4exp(-(xc/wx1)^2-((yc-0.2ly)'/wy1)^2) + 0.2*exp(-(xc/wx2)^2-((yc-0.1ly)'/wy2)^2)*(cos(ω1*π*(xc/lx + 0*yc'/ly-0.25)) + 1)
    @. fun += 0.025exp(-(xc/wx2)^2-(yc'/wy2)^2)*cos(ω2*π*(0*xc/lx + yc'/ly)) 
    @. fun += 0xc + 0.15*(yc'/ly)
    zmin,zmax = extrema(fun)
    @. fun = (fun - zmin)/(zmax-zmin)
    @. B += oz + (lz-oz)*fun 



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

    # how to define β_init β_syn 
    β = 0.0001*CUDA.ones(nx, ny)

    # make figure
    p1 = heatmap(xc,yc,Array(B'))

    as_rng = [0.0, 5e-20]#, 5e-18, 5e-17]
    colors = [:white, :red, :yellow, :blue]
    labels = ["a0.0", "5e-24",  "5e-21", "5e-20", "5e-18"]
    for (ias, as0) in enumerate(as_rng)
        asρgn0 = as0*ρg^n
        asρgn  = asρgn0*CUDA.ones(nx-1,ny-1)
        @show asρgn0
        synthetic_problem = Forwardproblem(H_obs, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β,asρgn, n, aρgn, b_max, z_ELA, dx, dy, nx, ny, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp)

        println("generating synthetic data (nx = $nx, ny = $ny)...")
        solve!(synthetic_problem)
        contour!(xc, yc, Array(H_obs');levels=0.01:0.01, lw=2.0, color=colors[ias], line=:solid,label=labels[ias])
    end 
    display(plot(p1))
    savefig("synthetic_test_as.png")



    return 
end 

adjoint_2D() 