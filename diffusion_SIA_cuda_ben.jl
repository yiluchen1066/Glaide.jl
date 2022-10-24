using Plots,Plots.Measures,Printf, CUDA, BenchmarkTools
default(size=(2500,1000),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)
@views avx(A) = 0.5.*(A[1:end-1,:].+A[2:end,:])
@views avy(A) = 0.5.*(A[:,1:end-1].+A[:,2:end])

function update_flux_v1!(qx, qy, H, rock, dx, dy, ρg, μ, n)
    nx, ny = size(H)

    ix = (blockIdx().x-1) * blockDim().x +threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y +threadIdx().y
    if ix <= nx-1 && iy <= ny
        qx[ix+1, iy] = ρg/3μ * ((H[ix+1,iy]+H[ix,iy])/2)^n * (H[ix+1,iy]+rock[ix+1, iy]-H[ix,iy]-rock[ix,iy])/dx
    end 
    if iy <= ny-1 && ix <= nx
        qy[ix, iy+1] = ρg/3μ * ((H[ix,iy+1]+H[ix,iy])/2)^n * (H[ix,iy+1]+rock[ix, iy+1]-H[ix,iy]-rock[ix,iy])/dy
    end 

    return 
    #qx[2:end-1,:] .= ρg/3μ .* avx(H.^n) .* diff(H.+rock, dims =1 )./dx # size mismatch 
    #qy[:,2:end-1] .= ρg/3μ .* avy(H.^n) .* diff(H.+rock, dims =2 )./dy
end 

function update_flux_v2!(qx, qy, H, rock, dx, dy, ρg, μ, n)
    #qx[2:end-1,:] .= ρg/3μ .* (1.0./(1+n).*diff(H.^(n+1), dims=1)./dx .+ avx(H.^n) .* diff(rock, dims =1 )./dx) # size mismatch 
    #qy[:,2:end-1] .= ρg/3μ .* (1.0./(1+n).*diff(H.^(n+1), dims=2)./dy .+ avy(H.^n) .* diff(rock, dims =2 )./dy)
    nx, ny = size(H) 

    ix = (blockIdx().x-1) * blockDim().x +threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y +threadIdx().y 

    if ix <= nx-1 && iy <= ny
        qx[ix+1, iy] = ρg/3μ * (1.0/(1+n)*(H[ix+1,iy]^(n+1)-H[ix,iy]^(n+1))/dx + ((H[ix+1,iy]+H[ix,iy])/2)^n*(rock[ix+1,iy]-rock[ix,iy])/dx)
    end 
    if iy <= ny -1 && ix <= nx
        qy[ix,iy+1] = ρg/3μ * (1.0/(1+n)*(H[ix,iy+1]^(n+1)-H[ix,iy]^(n+1))/dy + ((H[ix,iy+1]+H[ix,iy])/2)^n*(rock[ix,iy+1]-rock[ix,iy])/dy)
    end 

    return 

end 

function update_height!(qx, qy, H, dx, dy, dt)
    #H .+= dt.*(diff(qx, dims=1)./dx .+ diff(qy, dims=2)./dy)
    nx, ny = size(H) 

    ix = (blockIdx().x-1)*blockDim().x+threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y+threadIdx().y

    if ix <= nx && iy <= ny
        H[ix, iy] += dt*((qx[ix+1,iy]-qx[ix,iy])/dx+(qy[ix,iy+1]-qy[ix,iy])/dy)
    end 

    return 
end 

function update_H!(H, qx, qy, rock, dx, dy, dt, ρg, μ, n, mode, threads, blocks)
    if mode == :v1
        CUDA.@sync @cuda threads = threads blocks = blocks update_flux_v1!(qx, qy, H, rock, dx, dy, ρg, μ, n)
    else 
        CUDA.@sync @cuda threads = threads blocks = blocks  update_flux_v2!(qx, qy, H, rock, dx, dy, ρg, μ, n)
    end              
    CUDA.@sync @cuda threads = threads blocks = blocks update_height!(qx, qy, H, dx, dy, dt)

    return 
end 


@views function nonlinear_diffusion_1D()
    # physics
    lx,ly  = 30.0, 30.0 #lx, ly = 30 km
    n    = 3
    ρg   = 970*9.8
    μ    = 1e13   # viscousity of ice
    ttot = 10e10
    b_0 = 500 # 500m 
    # numerics
    nx,ny   = 1000, 1000
    threads = (32,32)
    nvis = 1000
    mode = :v1
    # derived numerics
    dx   = lx/nx
    dy   = ly/ny 
    xc   = LinRange(dx/2,lx-dx/2,nx)
    yc   = LinRange(dy/2,ly-dy/2,ny)
    print(xc)
    # array initialisation: Gaussian distribution 
    #H    = @. exp(-(xc-lx/2)^2 - (yc'-ly/2)^2); H_i = copy(H)
    H = ones(nx,ny); H_i = copy(H)
    # define bed vector. 
    B = zeros(size(H))
    B[xc.<7, yc.<7] .= b_0
    B[2:end-1,2:end-1] .= B[2:end-1,2:end-1] .+ 1.0./4.1.*(diff(diff(B[:,2:end-1], dims=1), dims=1) .+ diff(diff(B[2:end-1,:], dims=2), dims=2))

    p1 = heatmap(xc, yc, H', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:turbo, title= "H")
    p2 = heatmap(xc, yc, B', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:turbo, title = "B")
    display(plot(p1, p2))

    H = CuArray(H) 
    B = CuArray(B)

    init_mass= sum(H_i)*dx*dy
    #qx   = zeros(Float64, nx+1,ny)
    #qy   = zeros(Float64, nx, ny+1)
    qx   = CUDA.zeros(Float64, nx+1,ny)
    qy   = CUDA.zeros(Float64, nx, ny+1)

    blocks =@.  ceil(Int, (nx,ny)/threads)

    

    t = 0.0
    it = 1 
    #ttot = -1.0
    while t <= ttot
    # time loop
    
        dt = min(dx^2, dy^2) ./(ρg/3μ.*maximum(H.^n))./4.1
        
        update_H!(H, qx, qy, B, dx, dy, dt, ρg, μ, n, mode, threads, blocks)

        
        if (it % nvis)==0
            mass = sum(H)*dx*dy
            p1 = heatmap(xc, yc, Array((H.+B)'); xlims = (xc[1], xc[end]), ylims = (yc[1],yc[end]), aspect_ratio = 1.0, xlabel = "lx", ylabel = "ly", title = "time = $(round(t,digits=1))", c=:turbo)
            p2 = plot(yc, Array(H[round(Int,nx/2),:]);  xlims = (xc[1], xc[end]), ylims = (yc[1], yc[end]), xlabel = "y", ylabel = "H", title = "mass balance is $(mass - init_mass)")
            display(plot(p1,p2,layout=(1,2)))
        end
        t += dt
        it += 1
    end 

    dt = min(dx^2, dy^2) ./(ρg/3μ.*maximum(H.^n))./4.1

    t_it = @belapsed update_H!($H, $qx, $qy, $B, $dx, $dy, $dt, $ρg, $μ, $n, $mode, $threads, $blocks)

    A_eff = (1+2+4)*nx*ny*8/1e9

    T_eff = A_eff/t_it

    @printf("effective memory throuput = %1.3f GB/s\n", T_eff)

end

nonlinear_diffusion_1D()
