const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA,  Float64, 2)
else
    @init_parallel_stencil(Threads,  Float64, 2)
end
using CUDA,BenchmarkTools
using Plots,Plots.Measures,Printf
using DelimitedFiles
default(size=(800,600),framestyle=:box,label=false,grid=false,margin=10mm,lw=4,labelfontsize=9,tickfontsize=9,titlefontsize=12)

# ParallelStencil's FiniteDifferences2D submodule provides macros we need @inn_x(), @inn_y(), @d_xa(), @d_ya() 
# @all(A) = A[ix,iy] 
# @d_xa(A) = A[ix+1,iy] - A[ix,iy] 
# @d_ya(A) = A[ix,iy+1] - A[ix,iy]
# @inn_x(A) = A[ix+1, iy]
# @inn_y(A) = A[ix,iy+1] 
# @inn(A) = A[ix+1, iy+1] 

import ParallelStencil: INDICES
ix,iy       = INDICES[1], INDICES[2] 

macro get_thread_idx(A)  esc(:( begin ix = (blockIdx().x-1) * blockDim().x+threadIdx().x; iy = (blockIdx().y-1) * blockDim().y+threadIdx().y; end )) end
macro nw(A)     esc(:($A[$ix,$iy])) end
macro ne(A)     esc(:($A[$ix+1,$iy])) end 
macro sw(A)     esc(:($A[$ix,$iy+1])) end 
macro se(A)     esc(:($A[$ix+1,$iy+1])) end  
macro w_inn(A)  esc(:($A[$ix,$iy+1])) end 
macro e_inn(A)  esc(:($A[$ix+1,$iy+1])) end 
macro n_inn(A)  esc(:($A[$ix+1,$iy])) end 
macro s_inn(A)  esc(:($A[$ix+1,$iy+1])) end 
macro av_xy(A)  esc(:( 0.25*($A[$ix,$iy]+$A[$ix+1,$iy]+$A[$ix,$iy+1]+$A[$ix+1,$iy+1]) )) end
macro av_xa(A)  esc(:( 0.5*($A[$ix,$iy]+$A[$ix+1,$iy]) )) end
macro av_ya(A)  esc(:( 0.5*($A[$ix,$iy]+$A[$ix,$iy+1]) )) end
macro d_xi(A)   esc(:( $A[$ix+1,$iy+1]-$A[$ix,$iy+1] )) end
macro d_yi(A)   esc(:( $A[$ix+1,$iy+1]-$A[$ix+1,$iy] )) end

CUDA.device!(7) # GPU selection

@parallel function compute_error_1!(Err, H) 
    @all(Err) = @all(H) 
    return 
end 

@parallel function computer_error_2!(Err, H) 
    @all(Err) = @all(Err) - @all(H) 
    return 
end 

@parallel function compute_limiter_1!(B, S, B_avg, H_avg, Bx, By) 
    @all(B_avg) = max(@nw(B), @ne(B), @sw(B), @se(B)) 
    @all(H_avg) = 0.25*(max(0.0,@nw(S)-@all(B_avg))+max(0.0,@ne(S)-@all(B_avg))+max(0.0,@sw(S)-@all(B_avg))+max(0.0,@se(S)-@all(B_avg)))
    @all(Bx)    = max(@w_inn(B), @e_inn(B))
    @all(By)    = max(@s_inn(B), @n_inn(B))
    return 
end 

@parallel function compute_∇S_with_limiter!(S, B_avg, ∇Sx_lim,∇Sy_lim,dx,dy)
    @all(∇Sx_lim) = 0.5*(max(@all(B_avg),@ne(S))-max(@all(B_avg),@nw(S))+max(@all(B_avg),@se(S))-max(@all(B_avg),@sw(S)))/dx
    @all(∇Sy_lim) = 0.5*(max(@all(B_avg),@sw(S))-max(@all(B_avg),@nw(S))+max(@all(B_avg),@se(S))-max(@all(B_avg),@ne(S)))/dy
    return 
end 


@parallel function compute_D_with_limiter!(∇Sx_lim,∇Sy_lim,D,H_avg,n,a,as)
    @all(D)       = (a*@all(H_avg)^(n+2)+as*@all(H_avg)^n)*(sqrt(@all(∇Sx_lim)^2+@all(∇Sy_lim)^2))^(n-1)
    return 
end 

@parallel function compute_flux_with_limiter!(S,qHx,qHy,D,Bx,By,dx,dy)
    @all(qHx)     =  -@av_ya(D)*(max(@all(Bx),@e_inn(S))-max(@all(Bx),@w_inn(S)))/dx 
    @all(qHy)     =  -@av_xa(D)*(max(@all(By), @s_inn(S))-max(@all(By),@n_inn(S)))/dy
    return 
end 


@parallel function compute_icethickness!(S,M,RH,dHdτ,D,qHx,qHy,dτ,damp,cfl,epsi,dx,dy)
    @all(RH)      = -(@d_xa(qHx)/dx + @d_ya(qHy)/dy) + @inn(M) 
    @all(dHdτ)    = @all(dHdτ)*damp + @all(RH) 
    @all(dτ)      = 0.5*min(1.0, cfl/(epsi+@av_xy(D)))
    return 
end 

@parallel function update_H!(H,dHdτ,dτ)
    @inn(H)       = max(0.0, @inn(H)+@all(dτ)*@all(dHdτ))
    return 
end 

@parallel_indices (ix,iy) function set_BC!(H,nx,ny)
    if ix == 1 && iy <= ny 
        H[ix,iy] = H[ix+1,iy]
    end 
    if ix == nx && iy<= ny 
        H[ix,iy] = H[ix-1, iy]
    end 
    if ix >= 2 && ix <= nx-1 && iy == 1 
        H[ix,iy] = H[ix,iy+1] 
    end 
    if ix >= 2 && ix <= nx-1 && iy == ny 
        H[ix,iy] = H[ix,iy-1] 
    end 
    return 
end 

@parallel function update_S!(S, H, B) 
    @all(S) = @all(B) + @all(H) 
    return 
end 



# function update_with_limiter!(S, H, B, M, D, B_avg, H_avg, Bx, By, ∇Sx_lim,∇Sy_lim, gradS, qHx, qHy, dHdτ, RH,dx, dy, dτ,damp,cfl,epsi,n, a, as, nx,ny, threads, blocks)
#     CUDA.@sync @cuda threads=threads blocks=blocks compute_limiter_1!(B,S,B_avg, H_avg, Bx, By, nx, ny) 
#     CUDA.@sync @cuda threads=threads blocks=blocks compute_∇S_with_limiter!(S, B_avg, ∇Sx_lim,∇Sy_lim,dx,dy, nx, ny)
#     CUDA.@sync @cuda threads=threads blocks=blocks compute_D_with_limiter!(S,∇Sx_lim,∇Sy_lim,D,H_avg,n,nx,ny,a,as)
#     CUDA.@sync @cuda threads=threads blocks=blocks compute_flux_with_limiter!(S,qHx,qHy,D,Bx,By,dx,dy, nx,ny)
#     CUDA.@sync @cuda threads=threads blocks=blocks compute_icethickness!(S,M,RH,dHdτ,D,qHx,qHy,dτ,damp,cfl,epsi,dx,dy, nx,ny) # compute ice thickness
#     CUDA.@sync @cuda threads=threads blocks=blocks update_H!(H,dHdτ,dτ,nx,ny) # update ice thickness
#     CUDA.@sync @cuda threads=threads blocks=blocks set_BC!(H,nx,ny) # update ice thickness
#     CUDA.@sync @cuda threads=threads blocks=blocks update_S!(S,H,B,nx,ny)# update surface 
#     return
# end


function sia_2D()
    # physics
    s2y     = 3600*24*365.25 # seconds to years
    lx,ly   = 30e3, 30e3     # lx, ly = 30 km
    n       = 3
    ρg      = 970*9.8
    ttot    = 10e4 #5000 #10e4
    a0      = 1.5e-24
    # numerics
    nx,ny   = 128,128
    nout    = 1000    # error check frequency
    ndt     = 20      # dt check/update
    #threads = (16,16) # n threads
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
    cfl     = min(dx^2,dy^2)/4.1
    # derived physics 
    a       = 2.0*a0/(n+2)*ρg^n*s2y
    as      = 5.7e-20
    # array initialisation
    B       = zeros(nx,ny)
    M       = zeros(nx,ny)
    #H       = ones(nx,ny).*10
    # read no limiter data
    #S_without_limiter = readdlm("/scratch-1/yilchen/Msc-Inverse-SIA/Msc-Inverse-SIA/S_without_limiter.txt")
    #S_without_limiter_implicit = readdlm("/scratch-1/yilchen/Msc-Inverse-SIA/Msc-Inverse-SIA/S_without_limiter_implicit.txt")
    # define bed vector
    xm,xmB  = 20e3,7e3
    #M       = @. (((n*2.0/xm^(2*n-1))*xc^(n-1))*abs(xm-xc)^(n-1))*(xm-2.0*xc)
    M .= (((n.*2.0./xm.^(2*n-1)).*xc.^(n-1)).*abs.(xm.-xc).^(n-1)).*(xm.-2.0*xc)
    M[xc.>xm ,:] .= 0.0 
    # intial condition of B: cliff benchmark 
    B[xc.<xmB,:] .= 500
    # smoother
    B[2:end-1,2:end-1] .= B[2:end-1,2:end-1] .+ 1.0./4.1.*(diff(diff(B[:,2:end-1], dims=1), dims=1) .+ diff(diff(B[2:end-1,:], dims=2), dims=2))
    # boundary condition 
    B[[1,end],:] .= B[[2,end-1],:]
    B[:,[1,end]] .= B[:,[2,end-1]]
    # plot
    print(size(B))
    print(size(M))
    print(size(nx))
    print(size(ny))
    p1 = heatmap(xc, yc, B', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:turbo, title= "B")
    p2 = heatmap(xc, yc, M', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:turbo, title = "M")
    display(plot(p1, p2))
    # array initialization 
    B     = Data.Array(B) 
    M     = Data.Array(M) 
    H     = @zeros(nx,ny)
    S     = @zeros(nx,ny)
    B_avg = @zeros( nx-1,ny-1)
    H_avg = @zeros( nx-1,ny-1)
    Bx    = @zeros(nx-1,ny-2)
    By    = @zeros(nx-2,ny-1)
    ∇Sx   = @zeros(nx-1,ny  )
    ∇Sy   = @zeros(nx  ,ny-1)
    ∇Sx_lim  = @zeros(nx-1,ny-1)
    ∇Sy_lim  = @zeros(nx-1,ny-1)
    gradS = @zeros(nx-1,ny-1)
    D     = @zeros(nx-1,ny-1)
    qHx   = @zeros(nx-1,ny-2)
    qHy   = @zeros(nx-2,ny-1)
    dHdτ  = @zeros(nx-2,ny-2)
    RH    = @zeros(nx-2,ny-2) 
    dτ    = @zeros(nx-2,ny-2) 
    Err   = @zeros(nx,ny) 
    #blocks = ceil.(Int,(nx,ny)./threads)
    opts = (aspect_ratio=1,xlims=(xc[1],xc[end]),ylim=(yc[1],yc[end]),c=:turbo)
    # init bed
    @parallel update_S!(S, H, B)
    it = 1; err = 2*ϵtol
    while (err> ϵtol && it<itMax) 
        @parallel compute_error_1!(Err, H)
        #update_with_limiter!(S, H, B, M, D, B_avg, H_avg, Bx, By, ∇Sx_lim,∇Sy_lim, gradS, qHx, qHy, dHdτ, RH,dx, dy, dτ,damp,cfl,epsi,n, a, as, nx,ny, threads, blocks)
        @parallel compute_limiter_1!(B,S,B_avg, H_avg, Bx, By) 
        @parallel compute_∇S_with_limiter!(S, B_avg, ∇Sx_lim,∇Sy_lim,dx,dy)
        @parallel compute_D_with_limiter!(∇Sx_lim,∇Sy_lim,D,H_avg,n,a,as)
        @parallel compute_flux_with_limiter!(S,qHx,qHy,D,Bx,By,dx,dy)
        @parallel compute_icethickness!(S,M,RH,dHdτ,D,qHx,qHy,dτ,damp,cfl,epsi,dx,dy) # compute ice thickness
        @parallel update_H!(H,dHdτ,dτ) # update ice thickness
        @parallel set_BC!(H,nx,ny) # update ice thickness
        @parallel update_S!(S,H,B)# update surface 
        it = it+1
        if it%nout == 0 
            @parallel computer_error_2!(Err, H)
            err = (sum(abs.(Err[:,:]))./nx./ny) 
            @printf("iter = %d, max resid = %1.3e \n", it, err) 
            p1 = heatmap(xc,yc,Array(S'), title="S, it=$(it)"; opts...)
            p2 = heatmap(xc,yc,Array(H'), title="H"; opts...)
            p3 = plot(xc, [Array(S[:,ceil(Int,ny/2)]),Array(B[:,ceil(Int,ny/2)])])
            p4 = plot(xc, Array(H[:,ceil(Int,ny/2)]))
            display(plot(p1,p2,p3,p4, title="SIA 2D"))
            if (err < ϵtol) break; end 
        end 
    end 
    # p1 = heatmap(xc,yc,Array(S'), title="S, it=$(it)"; opts...)
    # p2 = heatmap(xc,yc,Array(H'), title="H"; opts...)
    # p3 = plot(xc, S_without_limiter[:,ceil(Int,ny/2)], label="S without limiter (explicit)",xlabel="X in m", ylabel="Height in m")
    # p3 = plot!(xc, S_without_limiter_implicit[:,ceil(Int,ny/2)], label="S without limiter (implicit)",xlabel="X in m", ylabel="Height in m")
    # p3 = plot!(xc, [Array(S[:,ceil(Int,ny/2)]),Array(B[:,ceil(Int,ny/2)])],label=["S with limiter (implicit)" "bedrock"],xlabel="X in m", ylabel="Height in m")
    # p4 = plot(xc, Array(H[:,ceil(Int,ny/2)]))
    # display(plot(p3, title="SIA 2D"))
    # savefig("2D_with_limiter_cross_section.png")

end 
sia_2D()

# execute the code having the USE_GPU = false flag set, we are running on multi-threading CPU backend with multi-threading enabled 
# changing the USE_GPU flag to true (having first relauched a julia session) will make the application running on a GPU. On the GPU, you can reduce ttot and increase nx, ny in order to achieve higher 
# effective memory througput 
