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
            merr = (sum(abs.(Err[:,:]))./nx./ny)
        end 
        iter += 1 
    end 
    if iter == niter && merr >= ϵtol
        error("forward solve not converged")
    end 
    @printf("forward solve converged: #iter/nx = %.1f, err = %.1e\n", iter/nx, merr)
    return 
end 
# compute 


function adjoint_2D()
    # load Rhone data: surface elevation and bedrock 
    rhone_data = h5open("Rhone_data_padding/alps/data_Rhone.h5","r")
    xc = rhone_data["glacier/x"][:,1]
    yc = rhone_data["glacier/y"][1,:]
    B  = rhone_data["glacier/z_bed"][]
    S  = rhone_data["glacier/z_surf"][]
    close(rhone_data)
    # load Rhone data: mass balance elevation band
    df = readdlm("glacier_data/belev_01238.dat")
    zs_sample = convert(Vector{Float64},df[2:end,1])
    ms_sample = convert(Vector{Float64},df[2:end,2])
    dz_sample = zs_sample[2]-zs_sample[1]
    zs_sample = CuArray{Float64}(zs_sample)
    ms_sample = CuArray{Float64}(ms_sample)

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
    nx,ny=size(B)
    epsi = 1e-2
    dmp = 0.3#0.7
    ϵtol = 1e-4
    niter = 500000
    ncheck = 1000
    threads = (16,16) 
    blocks = ceil.(Int,(nx,ny)./threads)


    #derived numerics 
    dx = xc[2]-xc[1]
    dy = yc[2]-yc[1]
    lx, ly = dx*nx, dy*ny
    xc = LinRange(dx/2, lx-dx/2, nx)
    yc = LinRange(dy/2, ly-dy/2, ny)

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

    S = CuArray{Float64}(S)
    B = CuArray{Float64}(B)
    H = CuArray{Float64}(H)
    S_obs = CuArray{Float64}(S_obs)
    H_obs = CuArray{Float64}(H_obs)
    H_ini = CuArray{Float64}(H_ini)


    D = CUDA.zeros(Float64,nx-1, ny-1)
    av_ya_∇Sx = CUDA.zeros(Float64, nx-1, ny-1)
    av_xa_∇Sy = CUDA.zeros(Float64, nx-1, ny-1) 
    gradS = CUDA.zeros(Float64, nx-1, ny-1)
    qHx = CUDA.zeros(Float64, nx-1,ny-2)
    qHy = CUDA.zeros(Float64, nx-2,ny-1)
    M  = CUDA.zeros(Float64, nx,ny)
    mb = CUDA.zeros(Float64, nx,ny)
    
    # how to define β_init β_syn 
    β = 0.00040*CUDA.ones(nx, ny)
    #0.00025# 0.00010 
    
    as = 2.0e-2*CUDA.ones(nx-1, ny-1) 
    as_ini = copy(as)
    as_syn = 5.7e-20*CUDA.ones(nx-1,ny-1)
    as2 = similar(as) 
    
    synthetic_problem = Forwardproblem(H_obs, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, M, β,as_syn, zs_sample, ms_sample, mb, n, a, b_max, z_ELA, dz_sample, dx, dy, nx, ny, epsi, cfl, ϵtol, niter, ncheck, threads, blocks, dmp)

    println("generating synthetic data (nx = $nx, ny = $ny)...")
    solve!(synthetic_problem)
    p1= Plots.contour(xc, yc, Array(H_obs'); levels=20, aspect_ratio=1)
    display(plot(p1))
    println("done.")
   
    return 
end 

adjoint_2D() 