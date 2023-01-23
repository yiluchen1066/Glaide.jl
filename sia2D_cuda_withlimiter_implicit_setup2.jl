using CUDA,BenchmarkTools
using Plots,Plots.Measures,Printf
default(size=(800,600),framestyle=:box,label=false,grid=false,margin=10mm,lw=4,labelfontsize=9,tickfontsize=9,titlefontsize=12)

macro get_thread_idx(A)  esc(:( begin ix = (blockIdx().x-1) * blockDim().x+threadIdx().x; iy = (blockIdx().y-1) * blockDim().y+threadIdx().y; end )) end
macro av_xy(A)  esc(:( 0.25*($A[ix,iy]+$A[ix+1,iy]+$A[ix,iy+1]+$A[ix+1,iy+1]) )) end
macro av_xa(A)  esc(:( 0.5*($A[ix,iy]+$A[ix+1,iy]) )) end
macro av_ya(A)  esc(:( 0.5*($A[ix,iy]+$A[ix,iy+1]) )) end
macro d_xa(A)   esc(:( $A[ix+1,iy]-$A[ix,iy] )) end
macro d_ya(A)   esc(:( $A[ix,iy+1]-$A[ix,iy] )) end
macro d_xi(A)   esc(:( $A[ix+1,iy+1]-$A[ix,iy+1] )) end
macro d_yi(A)   esc(:( $A[ix+1,iy+1]-$A[ix+1,iy] )) end

CUDA.device!(7) # GPU selection

function compute_error_1!(Err, H, nx, ny) 
    @get_thread_idx(H) 
    if ix <= nx && iy <= ny 
        Err[ix,iy] = H[ix,iy] 
    end 
    return 
end 

function compute_error_2!(Err,H, nx, ny) 
    @get_thread_idx(H) 
    if ix <= nx && iy <= ny 
        Err[ix,iy] = Err[ix,iy] - H[ix,iy] 
    end 
    return 
end 

function compute_limiter_1!(B,S,B_avg, H_avg, Bx, By, nx, ny) 
    @get_thread_idx(B) 
    if ix<= nx-1 && iy<= ny-1 
        B_avg[ix,iy] = max(B[ix,iy], B[ix+1, iy], B[ix,iy+1], B[ix+1, iy+1])
        H_avg[ix,iy] = 0.25*(max(0.0,S[ix,iy]-B_avg[ix,iy])+max(0.0,S[ix+1,iy]-B_avg[ix,iy])+max(0.0,S[ix,iy+1]-B_avg[ix,iy])+max(0.0,S[ix+1,iy+1]-B_avg[ix,iy]))
    end 

    if ix<= nx-1 && iy <= ny-2 
        Bx[ix,iy]    = max(B[ix,iy+1], B[ix+1, iy+1])
    end 

    if ix <= nx-2 && iy<= ny-1 
        By[ix,iy]    = max(B[ix+1, iy+1], B[ix+1, iy])
    end 

    return 
end 


function compute_∇S_with_limiter!(S, B_avg, ∇Sx_lim,∇Sy_lim,dx,dy, nx, ny)
    @get_thread_idx(S)

    #∇Sx_lim = zeros(nx-1, ny-1) ∇Sy_lim = zeros(nx-1, ny-1)
    if ix <= nx-1 && iy <= ny-1 
        ∇Sx_lim[ix,iy] = 0.5*(max(B_avg[ix,iy],S[ix+1,iy])-max(B_avg[ix,iy],S[ix,iy])+max(B_avg[ix,iy],S[ix+1,iy+1])-max(B_avg[ix,iy],S[ix,iy+1]))/dx
        ∇Sy_lim[ix,iy] = 0.5*(max(B_avg[ix,iy],S[ix,iy+1])-max(B_avg[ix,iy],S[ix,iy])+max(B_avg[ix,iy],S[ix+1,iy+1])-max(B_avg[ix,iy],S[ix+1,iy]))/dy 
    end 
    return 
end 


function compute_D_with_limiter!(S,∇Sx_lim,∇Sy_lim,D,H_avg,n,nx,ny,a,as)
    @get_thread_idx(S)
    if ix <= nx-1 && iy<= ny-1 
        #gradS[ix,iy]   = sqrt(∇Sx_lim[ix,iy]^2+∇Sy_lim[ix,iy]^2)
        #D[ix,iy]       = (a*H_avg[ix,iy]^(n+2)+as*H_avg[ix,iy]^n)*(sqrt(∇Sx_lim[ix,iy]^2+∇Sy_lim[ix,iy]^2))^(n-1)
        D[ix,iy]       = (a*H_avg[ix,iy]^(n+2)+as*H_avg[ix,iy]^n)*(sqrt(∇Sx_lim[ix,iy]^2+∇Sy_lim[ix,iy]^2))^(n-1)
    end 
    return 
end 



function compute_flux_with_limiter!(S,qHx,qHy,D,Bx,By,dx,dy, nx,ny)
    # qHx = zeros(nx-1, ny-2) qHy = zeros(nx-2, ny-1)
    # Bx = zeros(nx-1, ny-2) By = zeros(nx-2, ny-1)
    @get_thread_idx(S)
    if ix <= nx-1 && iy <= ny-2 
        qHx[ix,iy] = -@av_ya(D)*(max(Bx[ix,iy], S[ix+1,iy+1])-max(Bx[ix,iy],S[ix,iy+1]))/dx 
    end 
    if ix <= nx-2 && iy <= ny-1 
        qHy[ix,iy] = -@av_xa(D)*(max(By[ix,iy], S[ix+1,iy+1])-max(By[ix,iy],S[ix+1,iy]))/dy 
    end 
    return 
end

function compute_icethickness!(S,M,RH,dHdτ,D,qHx,qHy,dτ,damp,cfl,epsi,dx,dy, nx,ny)
    @get_thread_idx(S)
    if ix<=nx-2 && iy<=ny-2
        RH[ix,iy]   = -(@d_xa(qHx)/dx + @d_ya(qHy)/dy) + M[ix+1,iy+1]
        dHdτ[ix,iy] = dHdτ[ix,iy]*damp+RH[ix,iy] 
        dτ[ix,iy]   = 0.5*min(1.0, cfl/(epsi+@av_xy(D)))
    end
    return
end 

function update_H!(H,dHdτ,dτ,nx,ny)
    @get_thread_idx(H)
    if ix<=nx-2 && iy<=ny-2
        H[ix+1,iy+1] = max(0.0, H[ix+1,iy+1] + dτ[ix,iy]*dHdτ[ix,iy])
    end
    return
end 

function set_BC!(H,nx,ny)
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
        H[ix,iy] = H[ix, iy-1] 
    end 

    return 
end 

function update_S!(S,H,B,nx,ny)
    @get_thread_idx(S)
    if ix<=nx && iy<=ny
        S[ix,iy] = B[ix,iy] + H[ix,iy]
    end
    return
end

#
function compute_M!(S,M, grad_b,b_max,z_ELA,nx,ny) 
    @get_thread_idx(S) 
    if ix <= nx && iy <= ny 
        M[ix,iy] = min(grad_b*(S[ix,iy]-z_ELA),b_max)
    end 
    return 
end 


function update_with_limiter!(S, H, B, M, D, B_avg, H_avg, Bx, By, ∇Sx_lim,∇Sy_lim, gradS, qHx, qHy, dHdτ, RH,dx, dy, dτ,damp,cfl,epsi,n, a, as, nx,ny, threads, blocks)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_limiter_1!(B,S,B_avg, H_avg, Bx, By, nx, ny) 
    CUDA.@sync @cuda threads=threads blocks=blocks compute_∇S_with_limiter!(S, B_avg, ∇Sx_lim,∇Sy_lim,dx,dy, nx, ny)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_D_with_limiter!(S,∇Sx_lim,∇Sy_lim,D,H_avg,n,nx,ny,a,as)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_flux_with_limiter!(S,qHx,qHy,D,Bx,By,dx,dy, nx,ny)
    CUDA.@sync @cuda threads=threads blocks=blocks compute_icethickness!(S,M,RH,dHdτ,D,qHx,qHy,dτ,damp,cfl,epsi,dx,dy, nx,ny) # compute ice thickness
    CUDA.@sync @cuda threads=threads blocks=blocks update_H!(H,dHdτ,dτ,nx,ny) # update ice thickness
    CUDA.@sync @cuda threads=threads blocks=blocks set_BC!(H,nx,ny) # update ice thickness
    CUDA.@sync @cuda threads=threads blocks=blocks update_S!(S,H,B,nx,ny)# update surface 
    return
end


function sia_2D()
    # physics
    s2y     = 3600*24*365.25 # seconds to years
    lx,ly   = 30e3, 30e3     # lx, ly = 30 km
    w       = 0.15*lx
    n       = 3
    ρg      = 970*9.8
    ttot    = 10e4 #5000 #10e4
    a0      = 1.5e-24
    grad_b  = 0.001 #0.01 # mass balance gradient 
    z_ela   = 300
    b_max   = 0.1 # 2.0
    B0      = 500 #3500 
    # numerics
    nx,ny   = 512,512
    nout    = 1000    # error check frequency
    ndt     = 20      # dt check/update
    threads = (16,16) # n threads
    dtsc    = 0.9     # iterative stau scaling 
    epsi    = 1e-2
    ϵtol    = 1e-8
    damp    = 0.7 
    itMax   = 100000
    # derived numerics
    dx      = lx/nx
    dy      = ly/ny 
    xc      = LinRange(dx/2,lx-dx/2,nx)
    yc      = LinRange(dy/2,ly-dy/2,ny)
    x0      = xc[Int(nx/2)] 
    y0      = yc[Int(ny/2)]
    cfl     = max(dx^2,dy^2)/4.1
    # derived physics 
    a       = 2.0*a0/(n+2)*ρg^n*s2y
    as      = 5.7e-20
    # array initialisation
    B       = zeros(nx,ny)
    M       = zeros(nx,ny)
    #H       = zeros(nx,ny)
    H       = ones(nx,ny).*100
    # define bed vector
    xm,xmB  = 20e3,7e3
    #M .= (((n.*2.0./xm.^(2*n-1)).*xc.^(n-1)).*abs.(xm.-xc).^(n-1)).*(xm.-2.0*xc)
    #M[xc.>xm ,:] .= 0.0 
    
    # cylinder bedrock setup 
    #B[sqrt.((xc.-x0).^2 .+ (yc'.-y0).^2) .< xmB] .= 500
    #B      = @. B0*(exp(-(xc-x0)^2/10^10-yc'^2/10^9)+exp(-(xc-x0)^2/10^9-(yc'-ly/8)^2/10^10)) 
    B      = @. B0*(exp(-(xc-x0)^2/w^2-(yc'-y0)^2/w^2))

    # smoother 
    B[2:end-1,2:end-1] .= B[2:end-1,2:end-1] .+ 1.0./4.1.*(diff(diff(B[:,2:end-1], dims=1), dims=1) .+ diff(diff(B[2:end-1,:], dims=2), dims=2))
    B[[1,end],:] .= B[[2,end-1],:]
    B[:,[1,end]] .= B[:,[2,end-1]]
    # plot
    p1 = heatmap(xc, yc, B', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:turbo, title= "H")
    #p2 = heatmap(xc, yc, M', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:turbo, title = "B")
    #display(plot(p1, p2))
    display(plot(p1))
    # array initialization 
    H     = CuArray{Float64}(H) 
    B     = CuArray{Float64}(B)
    M     = CuArray{Float64}(M)
    S     = CUDA.zeros(Float64,nx ,ny  )
    B_avg = CUDA.zeros(Float64, nx-1,ny-1)
    H_avg = CUDA.zeros(Float64, nx-1,ny-1)
    Bx    = CUDA.zeros(Float64,nx-1,ny-2)
    By    = CUDA.zeros(Float64,nx-2,ny-1)
    ∇Sx   = CUDA.zeros(Float64,nx-1,ny  )
    ∇Sy   = CUDA.zeros(Float64,nx  ,ny-1)
    ∇Sx_lim  = CUDA.zeros(Float64,nx-1,ny-1)
    ∇Sy_lim  = CUDA.zeros(Float64,nx-1,ny-1)
    gradS = CUDA.zeros(Float64,nx-1,ny-1)
    D     = CUDA.zeros(Float64,nx-1,ny-1)
    qHx   = CUDA.zeros(Float64,nx-1,ny-2)
    qHy   = CUDA.zeros(Float64,nx-2,ny-1)
    dHdτ  = CUDA.zeros(Float64,nx-2,ny-2)
    RH    = CUDA.zeros(Float64,nx-2,ny-2) 
    dτ    = CUDA.zeros(Float64,nx-2,ny-2) 
    Err   = CUDA.zeros(Float64,nx,ny) 
    blocks = ceil.(Int,(nx,ny)./threads)
    opts = (aspect_ratio=1,xlims=(xc[1],xc[end]),ylim=(yc[1],yc[end]),c=:turbo)
    # init bed
    CUDA.@sync @cuda threads=threads blocks=blocks update_S!(S,H,B,nx,ny)

    it = 1; err = 2*ϵtol
    while (err> ϵtol && it<itMax) 
        CUDA.@sync @cuda threads=threads blocks=blocks compute_error_1!(Err, H, nx, ny)
        CUDA.@sync @cuda threads=threads blocks=blocks compute_M!(S,M, grad_b,b_max,z_ela,nx,ny) 
        update_with_limiter!(S, H, B, M, D, B_avg, H_avg, Bx, By, ∇Sx_lim,∇Sy_lim, gradS, qHx, qHy, dHdτ, RH,dx, dy, dτ,damp,cfl,epsi,n, a, as, nx,ny, threads, blocks)
        it = it+1
        if it%nout == 0 
            CUDA.@sync @cuda threads=threads blocks=blocks compute_error_2!(Err,H, nx, ny) 
            err = (sum(abs.(Err[:,:]))./nx./ny) 
            @printf("iter = %d, max resid = %1.3e \n", it, err) 
            # p1 = heatmap(xc,yc,Array(S'), title="S, it=$(it)"; opts...)
            # p2 = heatmap(xc,yc,Array(H'), title="H"; opts...)
            # p3 = plot(xc, [Array(S[:,ceil(Int,ny/2)]),Array(B[:,ceil(Int,ny/2)])])
            # p4 = plot(xc, Array(H[:,ceil(Int,ny/2)]))
            # display(plot(p1,p3, title="SIA 2D"))
            if (err < ϵtol) break; end 
        end 
    end 
    p1 = heatmap(xc,yc,Array(S'), title="S",xlabel="X direction in m", ylabel="Y direction in m"; opts...)
    p2 = heatmap(xc,yc,Array(H'), title="H"; opts...)
    p3 = plot(xc, [Array(S[:,ceil(Int,ny/2)]),Array(B[:,ceil(Int,ny/2)])],xlabel="X in m", ylabel="Height in m")
    p4 = plot(xc, Array(H[:,ceil(Int,ny/2)]))
    display(plot(p3, title="SIA 2D"))
    savefig("2D_setup2_cross_section_100.png")
end

sia_2D()