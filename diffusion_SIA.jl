using Plots,Plots.Measures,Printf
default(size=(2500,1000),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)
@views avx(A) = 0.5.*(A[1:end-1,:].+A[2:end,:])
@views avy(A) = 0.5.*(A[:,1:end-1].+A[:,2:end])

@views function nonlinear_diffusion_1D()
    # physics
    lx,ly  = 10.0, 10.0
    n    = 3
    ρg   = 970*9.8 #density of ice
    μ    = 1e13   # viscousity of ice, constant ? 
    # numerics
    nx   = 100
    ny   = 100 
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
    print(typeof(rock))
    init_mass= sum(H_i)*dx*dy
    qx   = zeros(nx+1,ny)
    qy   = zeros(nx, ny+1)

    # time loop
    for it = 1:nt
        dt = min(dx^2, dy^2) ./(ρg/3μ.*maximum(H.^n))./4.1
        if mode == :v1
            qx[2:end-1,:] .= ρg/3μ .* avx(H.^n) .* diff(H.+rock, dims =1 )./dx # size mismatch 
            qy[:,2:end-1] .= ρg/3μ .* avy(H.^n) .* diff(H.+rock, dims =2 )./dy
        else 
            qx[2:end-1,:] .= ρg/3μ .* (1.0./(1+n).*diff(H.^(n+1), dims=1)./dx .+ avx(H.^n) .* diff(rock, dims =1 )./dx) # size mismatch 
            qy[:,2:end-1] .= ρg/3μ .* (1.0./(1+n).*diff(H.^(n+1), dims=2)./dy .+ avy(H.^n) .* diff(rock, dims =2 )./dy)  
        end              
        H .+= dt.*(diff(qx, dims=1)./dx .+ diff(qy, dims=2)./dy)
        if (it % nvis)==0
            mass = sum(H)*dx*dy
            p1 = heatmap(xc, yc, H'; xlims = (0,lx), ylims = (0,ly), aspect_ratio = 1.0, xlabel = "lx", ylabel = "ly", title = "time = $(round(it*dt,digits=1))", c=:turbo)
            p2 = plot(yc, H[round(Int,nx/2),:];  ylims = (0,1.0), xlabel = "y", ylabel = "H", title = "mass balance is $(mass - init_mass)")
            display(plot(p1,p2,layout=(1,2)))
        end
    end 
end

nonlinear_diffusion_1D()
