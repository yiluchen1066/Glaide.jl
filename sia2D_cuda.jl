using CUDA,BenchmarkTools
using Plots,Plots.Measures,Printf
default(size=(800,600),framestyle=:box,label=false,grid=false,margin=10mm,lw=4,labelfontsize=9,tickfontsize=9,titlefontsize=12)

macro get_thread_idx(A)  esc(:( begin nx,ny = size($A); ix = (blockIdx().x-1) * blockDim().x+threadIdx().x; iy = (blockIdx().y-1) * blockDim().y+threadIdx().y; end )) end
macro av_xa(A)    esc(:( 0.5*($A[ix,iy]+$A[ix+1,iy]) )) end
macro av_ya(A)    esc(:( 0.5*($A[ix,iy]+$A[ix,iy+1]) )) end
macro d_xa(A)     esc(:( $A[ix+1,iy]-$A[ix,iy] )) end
macro d_ya(A)     esc(:( $A[ix,iy+1]-$A[ix,iy] )) end

CUDA.device!(7) # GPU selection

function compute_∇S!(∇Sx,∇Sy,S,dx,dy)
    @get_thread_idx(S)
    if ix<=nx-1 && iy<=ny
        ∇Sx[ix+1,iy] = @d_xa(S)/dx
    end
    if ix<=nx && iy<=ny-1
        ∇Sy[ix,iy+1] = @d_ya(S)/dy
    end
    return
end

function compute_D!(D,∇Sx,∇Sy,H,n,a)
    @get_thread_idx(H)
    if ix<=nx && iy<=ny
        D[ix,iy] = a*H[ix,iy]^(n+2)*(sqrt(@av_xa(∇Sx)^2 + @av_ya(∇Sy)^2))^(n-1)
    end
    return
end

function compute_flux!(S,qHx,qHy,D,dx,dy)
    @get_thread_idx(S)
    if (ix<=nx-1 && iy<=ny  ) qHx[ix+1,iy  ] = -@av_xa(D)*@d_xa(S)/dx end
    if (ix<=nx   && iy<=ny-1) qHy[ix  ,iy+1] = -@av_ya(D)*@d_ya(S)/dy end
    return
end 

function update_H_S!(dHdt,H,S,B,qHx,qHy,M,dt,dx,dy)
    @get_thread_idx(S)
    if (ix<=nx && iy<=ny)
        dHdt[ix,iy] = -(@d_xa(qHx)/dx + @d_ya(qHy)/dy) + M[ix,iy]
        H[ix,iy]    = max(0.0, H[ix,iy] + dt*dHdt[ix,iy])
        S[ix,iy]    = B[ix,iy] + H[ix,iy]
    end
    return
end

function update!(S,H,B,M,D,∇Sx,∇Sy,qHx,qHy,dHdt,dx,dy,dt,n,a,threads,blocks)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_∇S!(∇Sx,∇Sy,S,dx,dy)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_D!(D,∇Sx,∇Sy,H,n,a)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_flux!(S,qHx,qHy,D,dx,dy)
    CUDA.@sync @cuda threads=threads blocks=blocks update_H_S!(dHdt,H,S,B,qHx,qHy,M,dt,dx,dy)
    return
end

function sia_2D()
    # physics
    s2y     = 3600*24*365.25 # seconds to years
    lx,ly   = 30e3,30e3      # lx, ly = 30 km
    n       = 3
    ρg      = 910*9.8
    ttot    = 50e3
    a0      = 1.5e-24
    # numerics
    nx,ny   = 150,150
    nout    = 5000    # error check frequency
    ndt     = 100      # dt check/update
    threads = (16,16) # n threads
    dtsc    = 0.7     # iterative dt scaling 
    epsi    = 1e-4
    ϵtol    = 1e-8
    # derived numerics
    dx      = lx/nx
    dy      = ly/ny 
    xc      = LinRange(dx/2,lx-dx/2,nx)
    yc      = LinRange(dy/2,ly-dy/2,ny)
    cfl     = max(dx^2,dy^2)/4.1
    blocks  = ceil.(Int,(nx,ny)./threads)
    # derived physics 
    a       = 2.0*a0/(n+2)*ρg^n*s2y
    # array initialisation
    B       = zeros(nx,ny)
    M       = zeros(nx,ny)
    H       = zeros(nx,ny)
    # define bed vector
    xm,xmB  = 20e3,7e3
    M .= (((n.*2.0./xm.^(2*n-1)).*xc.^(n-1)).*abs.(xm.-xc).^(n-1)).*(xm.-2.0*xc)
    M[xc.>xm ,:] .= 0.0 
    B[xc.<xmB,:] .= 500
    B[2:end-1,2:end-1] .= B[2:end-1,2:end-1] .+ 1.0./4.1.*(diff(diff(B[:,2:end-1], dims=1), dims=1) .+ diff(diff(B[2:end-1,:], dims=2), dims=2))
    B[:,[1,end]] .= B[:,[2,end-1]] # BCs
    # visu
    opts = (aspect_ratio=1,xlims=(xc[1],xc[end]),ylim=(yc[1],yc[end]),c=:turbo)
    p1 = heatmap(xc,yc,B',title="H"; opts...)
    p2 = heatmap(xc,yc,M',title="B"; opts...)
    display(plot(p1, p2))
    # array initialization 
    H     = CuArray{Float64}(H) 
    B     = CuArray{Float64}(B)
    M     = CuArray{Float64}(M)
    ∇Sx   = CUDA.zeros(Float64,nx+1,ny  )
    ∇Sy   = CUDA.zeros(Float64,nx  ,ny+1)
    S     = CUDA.zeros(Float64,nx  ,ny  )
    D     = CUDA.zeros(Float64,nx  ,ny  )
    qHx   = CUDA.zeros(Float64,nx+1,ny  )
    qHy   = CUDA.zeros(Float64,nx  ,ny+1)
    dHdt  = CUDA.zeros(Float64,nx  ,ny  )
    S    .= B .+ H # init bed
    t = 0.0; it = 1; dt = dtsc * min(1.0, cfl/(epsi+maximum(D)))
    while t <= ttot
        if (it==1 || it%ndt==0) dt=dtsc*min(1.0, cfl/(epsi+maximum(D))) end
        update!(S,H,B,M,D,∇Sx,∇Sy,qHx,qHy,dHdt,dx,dy,dt,n,a,threads,blocks)
        if it%nout == 0
            @printf("it = %d, t = %1.2f, max(dHdt) = %1.2e \n", it, t, maximum(dHdt[2:end-1,2:end-1]))
            p1 = heatmap(xc,yc,Array(S'), title="SIA 2D S, it=$(it)", tickfontsize=6; opts...)
            p2 = heatmap(xc,yc,Array(H'), title="SIA 2D H", tickfontsize=6; opts...)
            p3 = plot(xc, [Array(S[:,ceil(Int,ny/2)]),Array(B[:,ceil(Int,ny/2)])], xlabel="X in m", ylabel = "Height in m")
            p4 = plot(xc, Array(H[:,ceil(Int,ny/2)]),xlabel="X in m", ylabel = "Height in m")
            display(plot(p1,p2,p3,p4))
        end
        it += 1
        t += dt
    end
end

sia_2D()