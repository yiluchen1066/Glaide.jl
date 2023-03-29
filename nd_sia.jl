using Plots
Plots.default(size=(1200,200))

using Printf

@views avx(A) = 0.5.*(A[1:end-1].+A[2:end])
@views d_x(A) = A[2:end].-A[1:end-1]

@views function main()
    # power law exponents
    n        = 3   # Glen's flow law power exponent
    # dimensionally independent physics
    lx       = 1.0 # [m]
    aρgn0    = 1.0 # [1/s/m^n]
    # scales
    tsc      = 1/aρgn0/lx^n # [s]
    # non-dimensional numbers
    s_f      = 3e-4    # sliding to ice flow ratio: s_f = asρgn0/aρgn0/lx^2
    w_b_lx   = 0.2     # mountain width to domain length ratio
    a_b_lx   = 0.0175  # mountain height to domain length ratio
    z_ela_lx = 0.016   # ela to domain length ratio
    βtsc     = 1e-14   # ratio between characteristic time scales of ice flow and accumulation/ablation
    m_max_nd = 0.5e-16 # maximum accumulation
    tanθ     = 0.2     # slope
    # dimensionally dependent physics
    asρgn0   = s_f*aρgn0*lx^2
    w_b      = w_b_lx*lx
    a_b      = a_b_lx*lx
    z_ela    = z_ela_lx*lx
    β        = βtsc/tsc
    m_max    = m_max_nd*lx/tsc
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
            areaplot(xc,[B 10.0.*H];aspect_ratio=4,show=true,xlims=(-lx/2,lx/2),ylims=(0,2*(a_b)),xlabel="x",ylabel="y")
            if err_abs < ϵtol.abs || err_rel < ϵtol.rel; break; end
        end
    end
    @show maximum(H)
    return
end

main()