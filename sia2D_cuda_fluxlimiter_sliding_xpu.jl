using CUDA,BenchmarkTools
using Plots,Plots.Measures,Printf
using ParallelStencil 
using ParallelStencil.FiniteDifferences2D 
default(size=(800,600),framestyle=:box,label=false,grid=false,margin=10mm,lw=4,labelfontsize=9,tickfontsize=9,titlefontsize=12)

const USE_GPU = false

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2) 
else
    @init_parallel_stencil(Threads, Float64, 2) 
end 

macro get_thread_idx(A)  esc(:( begin nx,ny = size($A); ix = (blockIdx().x-1) * blockDim().x+threadIdx().x; iy = (blockIdx().y-1) * blockDim().y+threadIdx().y; end )) end

macro av_xy(A)    esc(:( 0.25*($A[ix,iy]+$A[ix+1,iy]+$A[ix,iy+1]+$A[ix+1,iy+1]) )) end
macro av_xa(A)    esc(:( 0.5*($A[ix,iy]+$A[ix+1,iy]) )) end
macro av_ya(A)    esc(:( 0.5*($A[ix,iy]+$A[ix,iy+1]) )) end
macro d_xa(A)     esc(:( $A[ix+1,iy]-$A[ix,iy] )) end
macro d_ya(A)     esc(:( $A[ix,iy+1]-$A[ix,iy] )) end
macro d_xi(A)     esc(:( $A[ix+1,iy+1]-$A[ix,iy+1] )) end
macro d_yi(A)     esc(:( $A[ix+1,iy+1]-$A[ix+1,iy] )) end
macro d_xa_avy(A) esc(:( 0.5*(($A[ix+1,iy]-$A[ix,iy]) + ($A[ix+1,iy+1]-$A[ix,iy+1])) )) end
macro d_ya_avx(A) esc(:( 0.5*(($A[ix,iy+1]-$A[ix,iy]) + ($A[ix+1,iy+1]-$A[ix+1,iy])) )) end

macro ne(A) esc(:($A[ix+1, iy])) end 
macro sw(A) esc(:($A[ix, iy+1])) end 
macro se(A) esc(:($A[ix+1, iy+1])) end 

CUDA.device!(7) # GPU selection


@parallel_indices (ix, iy) function bc!(qHx::Data.Array, qHy::Data.Array, H)
    nx, ny = size(H)
    if (ix<=nx+1 && iy==1   )  qHx[ix,iy] = qHx[ix,iy+1] end
    if (ix<=nx+1 && iy==ny  )  qHx[ix,iy] = qHx[ix,iy-1] end
    if (ix==1    && iy<=ny+1)  qHy[ix,iy] = qHy[ix+1,iy] end
    if (ix==nx   && iy<=ny+1)  qHy[ix,iy] = qHy[ix-1,iy] end
    return 
end 

@parallel_indices (ix, iy) function bc_flux!(qHx::Data.Array, qHy::Data.Array, H)
    nx, ny = size(H)
    if (ix<=nx+1 && iy==1   )  qHx[ix,iy] = qHx[ix,iy+1] end
    if (ix<=nx+1 && iy==ny  )  qHx[ix,iy] = qHx[ix,iy-1] end
    if (ix==1    && iy<=ny+1)  qHy[ix,iy] = qHy[ix+1,iy] end
    if (ix==nx   && iy<=ny+1)  qHy[ix,iy] = qHy[ix-1,iy] end
    return 
end 

@parallel_indices (ix,iy) function compute_1!(B::Data.Array, B_avg::Data.Array, Bx::Data.Array, By::Data.Array)
    nx, ny = size(B)
    if (ix <= nx-1 && iy <= ny-1) @all(B_avg) = max(max(@all(B), @sw(B)), max(@ne(B), @se(B))) end 
    if (ix <= nx-1 && iy <= ny-2) @all(Bx) = max(@se(B), @sw(B)) end 
    if (ix <= nx-2 && iy <= ny-1) @all(By)    = max(@se(B), @ne(B)) end 
    return 
end 

@parallel_indices (ix, iy) function compute_diffus!(S::Data.Array, H_avg::Data.Array, B_avg::Data.Array, dSdx::Data.Array, dSdy::Data.Array, gradS::Data.Array, 
D::Data.Array, H::Data.Array, n::Data.Int64, a::Data.Number, as::Data.Number, dx::Data.Number, dy::Data.Number)
    nx,ny = size(H)
    if ix<= nx-1 && iy <= ny-1
        @all(H_avg) = 0.25*(max(0.0, @all(S)-@all(B_avg), @ne(S)-@all(B_avg), @sw(S)-@all(B_avg), @se(S)-@all(B_avg)))
        @all(dSdx)  = 0.5 *(max(@all(B_avg), @ne(S)) - max(@all(B_avg), @all(S)) + max(@all(B_avg), @se(S))-max(@all(B_avg), @sw(S)))/dx
        @all(dSdy)  = 0.5 *(max(@all(B_avg), @sw(S)) - max(@all(B_avg), @all(S)) + max(@all(B_avg), @se(S))-max(@all(B_avg), @ne(S)))/dy
        @all(gradS) = sqrt(@all(dSdx)^2+@all(dSdy)^2)
        @all(D)     = (a*@all(H_avg)^(n+2)+as*@all(H_avg)^n)*@all(gradS)^(n-1)
    end 
    return 
end 


@parallel_indices (ix,iy) function compute_flux!(S::Data.Array, qHx::Data.Array, qHy::Data.Array, D::Data.Array, Bx::Data.Array, By::Data.Array, dx::Data.Number, dy::Data.Number) 
    nx, ny = size(S) 
    if (ix <= nx-1 && iy<= ny-2) @se(qHx) = -@av_ya(D)*(max(@all(Bx),@se(S))-max(@all(Bx), @sw(S)))/dx end 
    if (ix <= nx-2 && iy<= ny-1) @se(qHy) = -@av_xa(D)*(max(@all(By), @se(S))-max(@all(By),@ne(S)))/dy end 
    return 
end 



@parallel_indices (ix, iy) function update_H_S!(dHdt::Data.Array,H::Data.Array,S::Data.Array,B::Data.Array,qHx::Data.Array,qHy::Data.Array,M::Data.Array,dt::Data.Number,dx::Data.Number,dy::Data.Number)
    # @all(dHdt) = -(@d_xa(qHx)/dx +@d_ya(qHy)/dy) + @all(M) 
    # @all(H)    = max(0.0, @all(H)+dt*@all(dHdt))
    # @all(S)    = @all(B) + @all(H)
    nx, ny = size(S)
    if (ix <= nx-1 && iy <= ny-1) dHdt[ix,iy] =  -(@d_xa(qHx)/dx + @d_ya(qHy)/dy) + M[ix,iy] end 
    if (ix <= nx-1 && iy <= ny-1) H[ix,iy]    = max(0.0, H[ix,iy] + dt*dHdt[ix,iy]) end
    if (ix <= nx-1 && iy <= ny-1) S[ix,iy]    = B[ix,iy] + H[ix,iy] end 

    return 
end 



# function update!(S::Data.Array, H::Data.Array, B::Data.Array, M::Data.Array, B_avg::Data.Array, H_avg::Data.Array, qHx::Data.Array, qHy::Data.Array, 
#     dSdx::Data.Array, dSdy::Data.Array, gradS::Data.Array, D::Data.Array, Bx::Data.Array, By::Data.Array, dHdt::Data.Array, dx::Data.Number, dy::Data.Number, dt::Data.Number, n::Data.Number, a::Data.Number, as::Data.Number)
#     @parallel compute_1!(B, B_avg, Bx, By)
#     @parallel compute_diffus!(S, H_avg, B_avg, dSdx, dSdy, gradS, D, H, n, a, as, dx, dy)
#     @parallel compute_flux!(S, qHx, qHy, D, Bx, By, dx, dy)
#     @parallel bc!(qHx,qHy,H)
#     @parallel update_H_S!(dHdt,H,S,B,qHx,qHy,M,dt,dx,dy)
#     return 
# end 

function sia_2D()
    # physics
    s2y     = 3600*24*365.25 # seconds to years
    lx,ly   = 30e3,30e3      # lx, ly = 30 km
    n       = 3
    ρg      = 910*9.8
    ttot    = 50e3
    a0      = 1.5e-24
    # numerics
    nx,ny   = 255,255
    nout    = 5000    # error check frequency
    ndt     = 100      # dt check/update
    #threads = (16,16) # n threads
    dtsc    = 0.7     # iterative dt scaling 
    epsi    = 1e-4
    ϵtol    = 1e-8
    # derived numerics
    dx      = lx/nx
    dy      = ly/ny 
    xc      = LinRange(dx/2,lx-dx/2,nx)
    yc      = LinRange(dy/2,ly-dy/2,ny)
    cfl     = max(dx^2,dy^2)/4.1
    #threads and blocks are no longer needed; the kernel launch parameters being automatically adapted 
    #blocks  = ceil.(Int,(nx,ny)./threads)
    # derived physics 
    a       = 2.0*a0/(n+2)*ρg^n*s2y
    as      = 5.7e-20 # sliding flow parameter 
    # array initialisation
    B       = zeros(nx,ny)
    M       = zeros(nx,ny)
    H       = zeros(nx,ny)
    # define bed vector
    xm,xmB  = 20e3,7e3
    M .= (((n.*2.0./xm.^(2*n-1)).*xc.^(n-1)).*abs.(xm.-xc).^(n-1)).*(xm.-2.0*xc)
    #M   = Data.Array(@. (((n*2.0/xm^(2*n-1))*xc^(n-1))*abs.(xm.-xc)^(n-1))*(xm-2.0*xc))
    M[xc.>xm ,:] .= 0.0 
    B[xc.<xmB,:] .= 500
    B[2:end-1,2:end-1] .= B[2:end-1,2:end-1] .+ 1.0./4.1.*(diff(diff(B[:,2:end-1], dims=1), dims=1) .+ diff(diff(B[2:end-1,:], dims=2), dims=2))
    B[:,[1,end]] .= B[:,[2,end-1]] # BCs
    # visu
    opts = (aspect_ratio=1,xlims=(xc[1],xc[end]),ylim=(yc[1],yc[end]),c=:turbo)
    p1 = heatmap(xc,yc,Array(B'),title="H"; opts...)
    p2 = heatmap(xc,yc,Array(M'),title="B"; opts...)
    display(plot(p1, p2))
    # array initialization
    B     = Data.Array(B)
    M     = Data.Array(M)
    H     = Data.Array(H) 
    S     = @zeros(nx  ,ny  )
    gradS = @zeros(nx-1,ny-1)
    D     = @zeros(nx-1,ny-1)
    qHx   = @zeros(nx+1,ny  )
    qHy   = @zeros(nx  ,ny+1)
    dHdt  = @zeros(nx  ,ny  )
    B_avg = @zeros(nx-1,ny-1) 
    H_avg = @zeros(nx-1,ny-1)
    dSdx  = @zeros(nx-1,ny-1)
    dSdy  = @zeros(nx-1,ny-1) 
    Bx    = @zeros(nx-1,ny-2)
    By    = @zeros(nx-2,ny-1)
    S    .= B .+ H # init bed
    t = 0.0; it = 1; dt = dtsc * min(1.0, cfl/(epsi+maximum(D)))
    while t <= ttot
        if (it==1 || it%ndt==0) dt=dtsc*min(1.0, cfl/(epsi+maximum(D))) end
        @parallel compute_1!(B, B_avg, Bx, By)
        @parallel compute_diffus!(S, H_avg, B_avg, dSdx, dSdy, gradS, D, H, n, a, as, dx, dy)
        @parallel compute_flux!(S, qHx, qHy, D, Bx, By, dx, dy)
        @parallel bc!(qHx,qHy,H)
        @parallel update_H_S!(dHdt,H,S,B,qHx,qHy,M,dt,dx,dy)
        if it%nout == 0
            @printf("it = %d, t = %1.2f, max(dHdt) = %1.2e \n", it, t, maximum(dHdt[2:end-1,2:end-1]))
            p1 = heatmap(xc,yc,Array(S'), title="S, it=$(it)"; opts...)
            p2 = heatmap(xc,yc,Array(H'), title="H"; opts...)
            p3 = plot(xc, [Array(S[:,ceil(Int,ny/2)]),Array(B[:,ceil(Int,ny/2)])])
            p4 = plot(xc, Array(H[:,ceil(Int,ny/2)]))
            display(plot(p1,p2,p3,p4))
        end
        it += 1
        t += dt
    end
end

sia_2D()
