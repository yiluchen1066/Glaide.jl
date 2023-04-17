using Plots
Plots.default(size=(1200,200))

using Printf

@views avx(A) = 0.5.*(A[1:end-1].+A[2:end])
@views d_x(A) = A[2:end].-A[1:end-1]

@views function main()
    # power law exponents
    n        = 3   # Glen's flow law power exponent
    # dimensionally independent physics
    lsc      = 1e4 # [m]
    aρgn0    = 1.35e-12 #1.9e-24*(970*9.81)^3#1.0 # [1/s/m^n]
    # scales
    tsc      = 1/aρgn0/lsc^n # [s]
    # non-dimensional numbers
    lx_lsc    = 25.0
    s_f       = 0.02    # sliding to ice flow ratio: s_f = asρgn0/aρgn0/lx^2
    w_b_lsc   = 5     # mountain width to length scale ratio
    a_b_lsc   = 0.35  # mountain height to length scale ratio
    z_ela_lsc = 0.215  # ela to length scale ratio
    βtsc      = 2e-10   # ratio between characteristic time scales of ice flow and accumulation/ablation
    m_max_nd  = 5e-12 # maximum accumulation
    tanθ      = 0.2     # slope
    # dimensionally dependent physics
    lx       = lx_lsc*lsc
    asρgn0   = s_f*aρgn0*lsc^2
    w_b      = w_b_lsc*lsc
    a_b      = a_b_lsc*lsc
    z_ela    = z_ela_lsc*lsc
    β        = βtsc/tsc # 0.01/3600/24/365 = 3.1709791983764586e-10
    m_max    = m_max_nd*lsc/tsc # 2.0/3600/365 = 6.341958396752917e-8
    @show(asρgn0)
    @show(w_b)
    @show(a_b)
    @show(z_ela)
    @show(β)
    @show(m_max)
    # numerics
    nx       = 501
    ϵtol     = (abs = 1e-8, rel = 1e-14)
    maxiter  = 10nx^2
    ncheck   = ceil(Int,0.1nx^2)
    # preprocessing
    dx       = lx/nx
    xv       = LinRange(-lx/2,lx/2,nx+1)
    xc       = avx(xv)
    # array allocation
    H        = Vector{Float64}(undef,nx  )
    B        = Vector{Float64}(undef,nx  )
    M        = Vector{Float64}(undef,nx  )
    qx       = Vector{Float64}(undef,nx+1)
    D        = Vector{Float64}(undef,nx-1)
    ∇S       = Vector{Float64}(undef,nx-1)
    r_abs    = Vector{Float64}(undef,nx  )
    r_rel    = Vector{Float64}(undef,nx  )
    # initialisation
    @. B     = a_b*(exp(-(xc/w_b)^2) + tanθ*xc/lx); B .-= minimum(B)
    @. H     = 0
    @. qx    = 0
    # iteration loop
    err_abs0 = 0.0
    for iter in 1:maxiter
        if iter % ncheck == 0; r_rel .= H; end
        ∇S .= d_x(B.+H)./dx
        D  .= avx(aρgn0.*H.^(n+2) .+ asρgn0.*H.^n).*abs.(∇S).^(n-1)
        @. qx[2:end-1] = -D*∇S
        @. M = min(β*(B + H - z_ela),m_max)
        dτ = 1.0/(6.1*maximum(D)/dx^2 + β)
        H .= max.(H .+ dτ.*(.-d_x(qx)./dx .+ M),0)
        if iter == 1 || iter % ncheck == 0
            r_abs .= .-d_x(qx)./dx .+ M; @. r_abs[H ≈ 0] = 0
            r_rel .-= H
            if iter == 1; err_abs0 = maximum(abs.(r_abs)); end
            err_abs = maximum(abs.(r_abs))/err_abs0
            err_rel = maximum(abs.(r_rel))/maximum(H)
            @printf("  iter/nx^2 = %.3e, err = [abs = %.3e, rel = %.3e]\n",iter/nx^2, err_abs, err_rel)
            display(areaplot(xc,[B 10.0.*H];aspect_ratio=4,show=true,xlims=(-lx/2,lx/2),ylims=(0,2*(a_b)),xlabel="x",ylabel="y"))
            if err_abs < ϵtol.abs || err_rel < ϵtol.rel; break; end
        end
    end
    @show maximum(H)
    return
end

main()