using CUDA,BenchmarkTools
using Plots,Plots.Measures,Printf
default(size=(800,600),framestyle=:box,label=false,grid=false,margin=10mm,lw=4,labelfontsize=9,tickfontsize=9,titlefontsize=12)

macro get_thread_idx(A)  esc(:( begin nx = size($A,1); ix =(blockIdx().x-1) * blockDim().x + threadIdx().x; end )) end 
macro av_xa(A)    esc(:( 0.5*($A[ix]+$A[ix+1]) )) end
macro d_xa(A)     esc(:( $A[ix+1]-$A[ix])) end 

CUDA.device!(7) # GPU selection

function compute_limiter!(B, B_avg, S, H_avg) 
    @get_thread_idx(B) 
    if ix <= nx-1
        B_avg[ix] = max(B[ix], B[ix+1])
        H_avg[ix] = 0.5*( max(0.0, max( S[ix]-B_avg[ix], S[ix+1]-B_avg[ix]) ) )
    end 
    return 
end 

function compute_∇S_without_limiter!(∇Sx, S, dx)
    @get_thread_idx(S) 
    if ix <= nx-1 
        ∇Sx[ix+1] = @d_xa(S)/dx
    end 
    return 
end 

function computer_∇S_with_limiter!(∇Sx, S, B_avg, dx)
    @get_thread_idx(S) 
    if ix<= nx-1 
        ∇Sx[ix+1] = (max(B_avg[ix], S[ix+1])-max(B_avg[ix], S[ix]))/dx 
    end 
    return 
end 

function compute_D!(D,∇Sx,H, n, a, as)
    @get_thread_idx(D) 
    if ix<= nx
        D[ix] = (a*H[ix]^(n+2))*@av_xa(∇Sx)^(n-1)
        #D[ix] = (a*H[ix]^(n+2)+as*H[ix]^n)*@av_xa(∇Sx)^(n-1)
    end 
    return 
end 

function cumpute_flux_without_limiter!(S, qHx, D, ∇Sx, dx) 
    @get_thread_idx(S)
    if ix <= nx-1
        qHx[ix+1] = -@av_xa(D)*@d_xa(S)/dx
    end 
    return 
end 

function compute_flux_with_limiter!(S, qHx, B_avg, D, dx)
    @get_thread_idx(S) 
    if ix <= nx-1 
        qHx[ix+1] = -@av_xa(D)*( max(B_avg[ix], S[ix+1])- max(B_avg[ix], S[ix]))/dx
    end 
    return 
end 

function update_H_S!(dHdt, H, S, B, qHx, M, dt, dx)
    @get_thread_idx(H) 
    if ix <= nx 
        dHdt[ix]  = M[ix] - @d_xa(qHx)/dx 
        H[ix]     = max(0.0, H[ix] + dt*dHdt[ix])
        S[ix]     = B[ix] + H[ix] 
    end 

    return 
end 


function update_without_limiter!(S,H,B,M,D,∇Sx,qHx,dHdt,dx,dt,n,a,as,threads,blocks)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_∇S_without_limiter!(∇Sx, S, dx)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_D!(D,∇Sx,H, n, a, as)
    CUDA.@sync @cuda threads=threads blocks=blocks cumpute_flux_without_limiter!(S, qHx, D, ∇Sx, dx)
    CUDA.@sync @cuda threads=threads blocks=blocks update_H_S!(dHdt, H, S, B, qHx, M, dt, dx)
    return
end

function update_with_limiter!(S,H,B,M,D,∇Sx,qHx,B_avg, H_avg,dHdt,dx,dt,n,a,as,threads,blocks)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_limiter!(B, B_avg, S, H_avg)
    CUDA.@sync @cuda threads=threads blocks=blocks computer_∇S_with_limiter!(∇Sx, S, B_avg, dx)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_D!(D,∇Sx,H, n, a, as)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_flux_with_limiter!(S, qHx, B_avg, D, dx)
    CUDA.@sync @cuda threads=threads blocks=blocks update_H_S!(dHdt, H, S, B, qHx, M, dt, dx)
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
    nx      = 127
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
    # array initialisation
    B       = zeros(Float64, nx)
    M       = zeros(Float64, nx)
    H       = zeros(Float64, nx)
    # define bed vector
    xm,xmB  = 20e3,7e3
    M .= (((n.*2.0./xm.^(2*n-1)).*xc.^(n-1)).*abs.(xm.-xc).^(n-1)).*(xm.-2.0*xc)
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
    ∇Sx   = CUDA.zeros(Float64,nx+1)
    S     = CUDA.zeros(Float64,nx)
    D     = CUDA.zeros(Float64,nx)
    B_avg = CUDA.zeros(Float64,nx-1)
    H_avg = CUDA.zeros(Float64,nx-1)
    qHx   = CUDA.zeros(Float64,nx+1)
    dHdt  = CUDA.zeros(Float64,nx)
    S    .= B .+ H # init bed
    t = 0.0; it = 1; dt = dtsc * min(1.0, cfl/(epsi+maximum(D)))
    while t <= ttot
        if (it==1 || it%ndt==0) dt=dtsc*min(1.0, cfl/(epsi+maximum(D))) end
        #update_without_limiter!(S,H,B,M,D,∇Sx,qHx,dHdt,dx,dt,n,a,as,threads,blocks)
        update_with_limiter!(S,H,B,M,D,∇Sx,qHx,B_avg, H_avg,dHdt,dx,dt,n,a,as,threads,blocks)
        if it%nout == 0
            @printf("it = %d, t = %1.2f, max(dHdt) = %1.2e \n", it, t, maximum(dHdt[2:end-1,2:end-1]))
            #p1 = heatmap(xc,Array(S'), title="S, it=$(it)"; opts...)
            #p2 = heatmap(xc,Array(H'), title="H"; opts...)
            p3 = plot(xc, [Array(S),Array(B)])
            p4 = plot(xc, Array(H))
            display(plot(p3,p4))
        end
        it += 1
        t += dt
    end
end

sia_1D()
