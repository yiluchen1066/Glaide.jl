using Plots,Plots.Measures,Printf, CUDA, BenchmarkTools
default(size=(2500,1000),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)
@views av(A) = 0.25.*(A[1:end-1, 1:end-1].+A[2:end, 1:end-1].+A[1:end, 2:end].+A[2:end, 2:end])
@views avx(A) = 0.5.*(A[1:end-1,:].+A[2:end,:])
@views avy(A) = 0.5.*(A[:,1:end-1].+A[:,2:end])
@views inn(A) = A[2:end-1, 2:end-1]

function compute_err_init!(Err, H) 
    nx, ny = size(H) 

    ix = (blockIdx().x-1) * blockDim().x+threadIdx().x 
    iy = (blockIdx().y-1) * blockDim().y+threadIdx().y 

    if ix <= nx && iy<= ny 
        Err[ix, iy] = H[ix,iy]
    end 
    return 
end 

function update_err!(Err, H) 
    nx, ny = size(H) 
    ix = (blockIdx().x-1) * blockDim().x+threadIdx().x 
    iy = (blockIdx().y-1) * blockDim().y+threadIdx().y 
    
    if ix <= nx && iy <= ny 
        Err[ix,iy] = Err[ix,iy] - H[ix,iy]
    end 
    return 
end 

function compute_diffusivity_1!(S, dSdx, dSdy,  dx, dy)
    #dSdx .= diff(S, dims=1)./dx
    #dSdy .= diff(S, dims=2)./dy 
    #gradS .= sqrt.(avy(dSdx).^2 .+ avx(dSdy).^2)

    nx, ny = size(S) 

    ix = (blockIdx().x-1) * blockDim().x+threadIdx().x 
    iy = (blockIdx().y-1) * blockDim().y+threadIdx().y 

    if ix <= nx-1 && iy <= ny 
        dSdx[ix,iy] = (S[ix+1, iy]-S[ix,iy])/dx
    end 
    if iy <= ny-1 && ix <= nx
        dSdy[ix, iy] = (S[ix, iy+1]-S[ix,iy])/dy 
    end 
    
    return 

end 

function compute_diffusitivity_2!(S, gradS, dSdx, dSdy, D, H, n, a) 

    nx, ny = size(S) 

    ix = (blockIdx().x-1) * blockDim().x+threadIdx().x 
    iy = (blockIdx().y-1) * blockDim().y+threadIdx().y 

    if ix <= nx-1 && iy <= ny-1
        gradS[ix,iy] = sqrt(((dSdx[ix, iy+1]+dSdx[ix,iy])/2)^2+((dSdy[ix+1, iy]+dSdy[ix,iy])/2)^2)
        D[ix, iy] = a*0.25*(H[ix,iy]+H[ix+1,iy]+H[ix, iy+1]+H[ix+1,iy+1])^(n+2)*gradS[ix,iy]^(n-1)
    end 
    return 
end 


function compute_flux!(S, qHx, qHy, D, dx, dy) 
    #qHx .= .-avy(D).* diff(S[:,2:end-1], dims=1)./dx
    #qHy .= .-avx(D).* diff(S[2:end-1,:], dims=2)./dy 

    nx, ny = size(S) 
    
    ix = (blockIdx().x-1) * blockDim().x+threadIdx().x 
    iy = (blockIdx().y-1) * blockDim().y+threadIdx().y 

    if ix <= nx-1 && iy <= ny-2
        qHx[ix,iy] = -(D[ix, iy+1]+D[ix,iy])/2*(S[ix+1,iy+1]-S[ix, iy+1])/dx
    end 
    if ix <= nx-2 && iy <= ny-1 
        qHy[ix,iy] = -(D[ix+1,iy]+D[ix,iy])/2*(S[ix+1, iy+1]-S[ix+1,iy])/dy
    end 
    return  

end 

function compute_icethickness!(S, M, ResH, qHx, qHy, dx, dy)

    #dtau = dtausc.*min.(1.0, cfl./(epsi.+av(D)))

    nx, ny = size(S) 

    ix = (blockIdx().x-1) * blockDim().x+threadIdx().x 
    iy = (blockIdx().y-1) * blockDim().y+threadIdx().y 

    if ix <= nx-2 && iy <= ny-2 
        #avD = 0.25*(D[ix,iy]+D[ix,iy+1]+D[ix+1, iy]+D[ix+1, iy+1])
        #dtau[ix, iy] = dtausc*min(1.0, cfl/(epsi+avD))
        ResH[ix, iy] = -((qHx[ix+1, iy]-qHx[ix,iy])/dx+(qHy[ix, iy+1]-qHy[ix,iy])/dy) + M[ix+1, iy+1]
    end 
    return 

end 

function update_H!(H, ResH, dt)
    #H[2:end-1, 2:end-1] .= max.(0.0, H[2:end-1, 2:end-1].+ dtau.*dHdt)
    nx, ny = size(H) 

    ix = (blockIdx().x-1) * blockDim().x+threadIdx().x 
    iy = (blockIdx().y-1) * blockDim().y+threadIdx().y 

    if ix <= nx-2 && iy <= ny-2
        H[ix+1,iy+1] = max(0.0, H[ix+1, iy+1]+dt*ResH[ix,iy])
    end 

    return 
end 

function update_S!(S, H, B)
    #update surface S = B + H 
    nx, ny = size(S) 

    ix = (blockIdx().x-1) * blockDim().x+threadIdx().x 
    iy = (blockIdx().y-1) * blockDim().y+threadIdx().y 

    if ix<= nx && iy <= ny 
        S[ix,iy] = B[ix,iy] + H[ix,iy] 
    end 

    return 
end 

function update!(S, H, B, M, D, dSdx, dSdy, gradS, qHx, qHy, ResH, dx, dy, dt, n, a, threads, blocks)
    CUDA.@sync @cuda threads = threads blocks = blocks compute_diffusivity_1!(S, dSdx, dSdy,  dx, dy) #compute diffusitivity 
    CUDA.@sync @cuda threads = threads blocks = blocks compute_diffusitivity_2!(S, gradS, dSdx, dSdy, D, H, n, a)
    CUDA.@sync @cuda threads = threads blocks = blocks compute_flux!(S, qHx, qHy, D, dx, dy) # compute flux 
    CUDA.@sync @cuda threads = threads blocks = blocks compute_icethickness!(S, M, ResH, qHx, qHy, dx, dy) # compute ice thickness
    CUDA.@sync @cuda threads = threads blocks = blocks update_H!(H, ResH, dt) # update ice thickness
    CUDA.@sync @cuda threads = threads blocks = blocks update_S!(S, H, B) # update surface 

    return 
end 


@views function nonlinear_diffusion_1D()
    # physics
    s2y    = 3600*24*365.25 # seconds to years 
    lx,ly  = 30.0, 30.0 #lx, ly = 30 km
    n    = 3
    ρg   = 970*9.8
    μ    = 1e13   # viscousity of ice
    ttot = 10e4
    a0   = 1.5e-24
    # numerics
    nx,ny   = 127, 127
    itMax   = 1e5 # number of iteration steps 
    nout    = 200 # error check frequency 
    threads = (32,32)
    dtsc = 1.0/10.0# iterative stau scaling 
    damp = 0.85 # convergence acceleration
    tolnl = 1e-6
    epsi  = 1e-4 
    # derived numerics
    dx   = lx/nx
    dy   = ly/ny 
    xc   = LinRange(dx/2,lx-dx/2,nx)
    yc   = LinRange(dy/2,ly-dy/2,ny)
    cfl   = max(dx^2, dy^2)/4.1 
    # derived physics 
    a    = 2.0*a0/(n+2)*ρg^n*s2y
    # array initialisation
    B = zeros(nx, ny)
    M = zeros(nx, ny) 
    # initial condition 
    H = ones(nx,ny); H_i = copy(H)
    # define bed vector. 
    xm, xmB = 20.0, 7.0 
    M .= (((n.*2.0./xm.^(2*n-1)).*xc.^(n-1)).*abs.(xm.-xc).^(n-1)).*(xm.-2.0*xc)
    M[xc.>xm, :] .= 0.0 
    B[xc.<xmB, :] .= 500
    B[2:end-1,2:end-1] .= B[2:end-1,2:end-1] .+ 1.0./4.1.*(diff(diff(B[:,2:end-1], dims=1), dims=1) .+ diff(diff(B[2:end-1,:], dims=2), dims=2))
    #S .= B .+ H 

    p1 = heatmap(xc, yc, B', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title= "H")
    p2 = heatmap(xc, yc, M', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title = "B")
    display(plot(p1, p2))

    H = CuArray{Float64}(H) 
    B = CuArray{Float64}(B)
    M = CuArray{Float64}(M)

    #init_mass= sum(H_i)*dx*dy

    # array initialization 
    S = CUDA.zeros(Float64, nx,ny)
    Err = CUDA.zeros(Float64, nx, ny)
    dSdx = CUDA.zeros(Float64, nx-1, ny) 
    dSdy = CUDA.zeros(Float64, nx, ny-1) 
    gradS = CUDA.zeros(Float64, nx-1, ny-1) 
    D = CUDA.zeros(Float64, nx-1, ny-1) # diffusion coefficients 
    qHx = CUDA.zeros(Float64, nx-1, ny-2) 
    qHy = CUDA.zeros(Float64, nx-2, ny-1) 
    dtau = CUDA.zeros(Float64, nx-2, ny-2) 
    ResH = CUDA.zeros(Float64, nx-2, ny-2) 
    dHdt = CUDA.zeros(Float64, nx-2, ny-2) 

    blocks =@.  ceil(Int, (nx,ny)/threads)

    CUDA.@sync @cuda threads=threads blocks = blocks update_S!(S, H, B)
    
    t = 0.0; it = 1
    while t <= ttot 

        #dt = min(dx^2, dy^2) ./(ρg/3μ.*maximum(H.^n))./4.1
        #dtau[ix, iy] = dtausc*min(1.0, cfl/(epsi+avD))

        dt = dtsc*min(1.0, cfl/(epsi+maximum(D)))
        CUDA.@sync @cuda threads = threads blocks = blocks compute_err_init!(Err, H) # array operation for cuda array 
        update!(S, H, B, M, D, dSdx, dSdy, gradS, qHx, qHy, ResH, dx, dy, dt, n, a, threads, blocks)

        if mod(it, nout) == 0
            CUDA.@sync @cuda threads = threads blocks = blocks update_err!(Err, H)
            err = (sum(abs.(Err))./nx./ny)
            @printf("it = %d, error = %1.2e \n", it, err)
            p3 = heatmap(xc, yc, Array(S'), aspect_ratio = 1, xlims=(xc[1], xc[end]), ylim=(yc[1], yc[end]), c=:viridis, title = "S")
            p4 = heatmap(xc, yc, Array(H'), aspect_ratio = 1, xlims=(xc[1], xc[end]), ylim=(yc[1], yc[end]), c=:viridis, title = "H")
            display(plot(p3, p4))
        end 
        it += 1 
        t  += dt 

    end 


    # it = 1; err = 2*tolnl 
    # while err > tolnl && it< itMax
    #     CUDA.@sync @cuda threads = threads blocks = blocks compute_err_init!(Err, H) # array operation for cuda array 
    #     update!(S, H, B, M, D, dSdx, dSdy, gradS, qHx, qHy, dtau, ResH, dHdt, dx, dy, n, a, dtausc, cfl, epsi, damp, threads, blocks)

    #     if mod(it, nout) == 0
    #         CUDA.@sync @cuda threads = threads blocks = blocks update_err!(Err, H)
    #         err = (sum(abs.(Err))./nx./ny)
    #         @printf("it = %d, error = %1.2e \n", it, err)
    #         p1 = heatmap(xc, yc, Array(S), aspect_ratio = 1, xlims=(xc[1], xc[end]), ylim=(yc[1], yc[end]), c=:viridis, title = "S")
    #         p1 = heatmap(xc, yc, Array(H), aspect_ratio = 1, xlims=(xc[1], xc[end]), ylim=(yc[1], yc[end]), c=:viridis, title = "H")
    #         display(plot(p1, p2))
    #     end 
    #     it += 1 
    # end 

    #dt = min(dx^2, dy^2) ./(ρg/3μ.*maximum(H.^n))./4.1

    #t_it = @belapsed update_H!($H, $qx, $qy, $B, $dx, $dy, $dt, $ρg, $μ, $n, $mode, $threads, $blocks)

    #A_eff = (1+2+4)*nx*ny*8/1e9

    #T_eff = A_eff/t_it

    #@printf("effective memory throuput = %1.3f GB/s\n", T_eff)

end

nonlinear_diffusion_1D()
