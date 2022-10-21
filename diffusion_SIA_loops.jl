using Plots,Plots.Measures,Printf, BenchmarkTools
default(size=(2500,1000),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)
@views avx(A) = 0.5.*(A[1:end-1,:].+A[2:end,:])
@views avy(A) = 0.5.*(A[:,1:end-1].+A[:,2:end])

function update_flux_v1!(qx, qy, H, rock, dx, dy, ρg, μ, n)
    nx, ny = size(H)

    Threads.@threads for j = 1:ny
        for i = 1:nx
            if i <= nx-1
                qx[i+1,j] = ρg/3μ * ((H[i+1,j]+H[i,j])/2)^n * (H[i+1,j]+rock[i+1, j]-H[i,j]-rock[i,j])/dx
            end 
            if j <= ny-1
                qy[i,j+1] = ρg/3μ * ((H[i,j+1]+H[i,j])/2)^n * (H[i,j+1]+rock[i, j+1]-H[i,j]-rock[i,j])/dy
            end 
        end 
    end 

    return 
    #qx[2:end-1,:] .= ρg/3μ .* avx(H.^n) .* diff(H.+rock, dims =1 )./dx # size mismatch 
    #qy[:,2:end-1] .= ρg/3μ .* avy(H.^n) .* diff(H.+rock, dims =2 )./dy
end 

function update_flux_v2!(qx, qy, H, rock, dx, dy, ρg, μ, n)
    #qx[2:end-1,:] .= ρg/3μ .* (1.0./(1+n).*diff(H.^(n+1), dims=1)./dx .+ avx(H.^n) .* diff(rock, dims =1 )./dx) # size mismatch 
    #qy[:,2:end-1] .= ρg/3μ .* (1.0./(1+n).*diff(H.^(n+1), dims=2)./dy .+ avy(H.^n) .* diff(rock, dims =2 )./dy)
    nx, ny = size(H) 

    Threads.@threads for j = 1:ny
        for i = 1:nx
           if i <= nx-1
                qx[i+1, j] = ρg/3μ * (1.0/(1+n)*(H.^(n+1)[i+1,j]-H.^(n+1)[i,j])/dx + ((H[i+1,j]+H[i,j])/2)^n*(rock[i+1,j]-rock[i,j])/dx)
           end 
           if j <= ny -1 
                qy[i,j+1] = ρg/3μ * (1.0/(1+n)*(H.^(n+1)[i,j+1]-H.^(n+1)[i,j])/dy + ((H[i,j+1]+H[i,j])/2)^n*(rock[i,j+1]-rock[i,j])/dy)
           end 
        end 
    end 

    return 

end 

function update_height!(qx, qy, H, dx, dy, dt)
    #H .+= dt.*(diff(qx, dims=1)./dx .+ diff(qy, dims=2)./dy)
    nx, ny = size(H) 

    Threads.@threads for j = 1:ny 
        for i = 1:nx
            H[i,j] += dt*((qx[i+1,j]-qx[i,j])/dx+(qy[i,j+1]-qy[i,j])/dy)
        end 
    end 

    return 
end 

function update_H!(H, qx, qy, rock, dx, dy, dt, ρg, μ, n, mode)

    #dt = min(dx^2, dy^2) ./(ρg/3μ.*maximum(H.^n))./4.1
    if mode == :v1
        update_flux_v1!(qx, qy, H, rock, dx, dy, ρg, μ, n)
    else 
        update_flux_v2!(qx, qy, H, rock, dx, dy, ρg, μ, n)
    end              
    update_height!(qx, qy, H, dx, dy, dt)

    return 
end

@views function nonlinear_diffusion_1D()
    # physics
    lx,ly  = 10.0, 10.0
    n    = 3
    ρg   = 970*9.8
    μ    = 1e13   # viscousity of ice
    # numerics
    nx   = 1000
    ny   = 1000
    nvis = 50
    mode = :v1
    # derived numerics
    dx   = lx/nx
    dy   = ly/ny 
    nt   = 1000
    xc   = LinRange(dx/2,lx-dx/2,nx)
    yc   = LinRange(dy/2,ly-dy/2,ny)
    # array initialisation: Gaussian distribution 
    H    = @. exp(-(xc-lx/2)^2 - (yc'-ly/2)^2); H_i = copy(H)
    rock    = @. (0.5*xc+1.0) + (0.3*yc'+1.0); rock_i = copy(rock)
    init_mass= sum(H_i)*dx*dy
    qx   = zeros(Float64, nx+1,ny)
    qy   = zeros(Float64, nx, ny+1)

    dt = 0.0; nt = 0
    # time loop
    for it = 1:nt
        dt = min(dx^2, dy^2) ./(ρg/(3μ).*maximum(H.^n))./4.1
        update_H!(H, qx, qy, rock, dx, dy, dt, ρg, μ, n, mode)
        
        if (it % nvis)==0
            mass = sum(H)*dx*dy
            p1 = heatmap(xc, yc, H'; xlims = (0,lx), ylims = (0,ly), aspect_ratio = 1.0, xlabel = "lx", ylabel = "ly", title = "time = $(round(it*dt,digits=1))", c=:turbo)
            p2 = plot(yc, H[round(Int,nx/2),:];  ylims = (0,1.0), xlabel = "y", ylabel = "H", title = "mass balance is $(mass - init_mass)")
            display(plot(p1,p2,layout=(1,2)))
        end
    end 
    dt = min(dx^2, dy^2) ./(ρg/(3μ).*maximum(H.^n))./4.1

    t_it = @belapsed update_H!($H, $qx, $qy, $rock, $dx, $dy, $dt, $ρg, $μ, $n, $mode)


    A_eff = (1+2+4)*nx*ny*8/1e9

    T_eff = A_eff/t_it

    @printf("effective memory throuput = %1.3f GB/s\n", T_eff)
end

nonlinear_diffusion_1D()

