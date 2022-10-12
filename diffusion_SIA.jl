using Plots,Plots.Measures,Printf
default(size=(1200,400),framestyle=:box,label=false,grid=false,margin=10mm,lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)
@views av(A) = 0.5.*(A[1:end-1].+A[2:end])
@views function nonlinear_diffusion_1D()
    # physics
    lx   = 20.0
    n    = 3
    ρg   = 970*9.8 #density of ice
    μ    = 1e13   # viscousity of ice 
    # numerics
    nx   = 200
    nvis = 5
    # derived numerics
    dx   = lx/nx
    nt   = 15
    xc   = LinRange(dx/2,lx-dx/2,nx)
    # array initialisation
    H    = @. 0.5cos(9π*xc/lx)+0.5; H_i = copy(H)
    qx   = zeros(Float64,nx-1)
    # time loop
    for it = 1:nt
        dt = dx^2 ./(ρg/3μ.*maximum(H.^n))./2.1
        qx .= ρg/3μ .* av(H.^n) .* diff(H)./dx # size mismatch 
        H[2:end-1] .-= dt.*diff(qx)./dx
        # qx .= 1/3*ρg/μ/(n+1)*diff(H.^(n+1))./dx
        if (it % nvis)==0
            display(plot(xc,[H_i,H];xlims=(0,lx), ylims=(-0.1,1.1), xlabel="lx", ylabel="ice thickness", title="time = $(round(it*dt,digits=1))"))
        end
    end
end

nonlinear_diffusion_1D()
