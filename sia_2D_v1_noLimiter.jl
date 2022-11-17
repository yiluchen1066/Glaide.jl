const USE_GPU  = false  # Use GPU? If this is set false, then the CUDA packages do not need to be installed! :)
const GPU_ID   = 0
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(GPU_ID) # select GPU
    macro pow(args...)  esc(:(CUDA.pow($(args...)))) end
else
    @init_parallel_stencil(Threads, Float64, 2)
    pow(x,y) = x^y
    macro pow(args...)  esc(:(pow($(args...)))) end
end
using Test, Plots, Printf, Statistics, LinearAlgebra

################################ Macros from cuda_scientific
import ParallelStencil: INDICES
ix,  iy   = INDICES[1], INDICES[2]
ixi, iyi  = :($ix+1), :($iy+1)

macro av_xya(A) esc(:( ($A[$ix, $iy] + $A[($ix+1), $iy] + $A[$ix,($iy+1)] + $A[($ix+1),($iy+1)])*0.25 )) end
# macro  av_xa(A) esc(:( ($A[$ix, $iy ] + $A[($ix+1), $iy   ])*0.5 )) end
# macro  av_ya(A) esc(:( ($A[$ix, $iy ] + $A[ $ix   ,($iy+1)])*0.5 )) end
macro     sw(A) esc(:( $A[ $ix   ,  $iy   ] )) end
macro     se(A) esc(:( $A[ $ix   , ($iy+1)] )) end
macro     nw(A) esc(:( $A[($ix+1),  $iy   ] )) end
macro     ne(A) esc(:( $A[($ix+1), ($iy+1)] )) end
macro  e_inn(A) esc(:( $A[($ix+1),  $iyi  ] )) end
macro  w_inn(A) esc(:( $A[ $ix   ,  $iyi  ] )) end
macro  n_inn(A) esc(:( $A[ $ixi  , ($iy+1)] )) end
macro  s_inn(A) esc(:( $A[ $ixi  ,  $iy   ] )) end
################################

@parallel function err_1!(Err::Data.Array, H::Data.Array)

    @all(Err) = @all(H)

    return
end 

@parallel function err_2!(Err::Data.Array, H::Data.Array)

    @all(Err) = @all(Err) - @all(H)

    return
end 

# @parallel function compute_1!(B_avg::Data.Array, Bx::Data.Array, By::Data.Array, B::Data.Array)

#     @all(B_avg) = max(max(@sw(B), @se(B)), max(@nw(B), @ne(B)))
#     @all(Bx)    = max(@e_inn(B), @w_inn(B))
#     @all(By)    = max(@n_inn(B), @s_inn(B))

#     return
# end

@parallel function compute_2!(H_avg::Data.Array, dSdx::Data.Array, dSdy::Data.Array, grdS2::Data.Array, D::Data.Array, H::Data.Array, S::Data.Array, _dx::Data.Number, _dy::Data.Number, a1::Data.Number, a2::Data.Number, npow::Int)

    # @all(H_avg) = 0.25*(max(0.0, @sw(S)-@all(B_avg)) + max(0.0, @se(S)-@all(B_avg)) + max(0.0, @nw(S)-@all(B_avg)) + max(0.0, @ne(S)-@all(B_avg)))
    @all(H_avg) = @av_xya(H)
    # @all(dSdx)  = 0.5*_dx*(max(@all(B_avg), @se(S)) - max(@all(B_avg), @sw(S)) + max(@all(B_avg), @ne(S)) - max(@all(B_avg), @nw(S)))
    # @all(dSdy)  = 0.5*_dy*(max(@all(B_avg), @nw(S)) - max(@all(B_avg), @sw(S)) + max(@all(B_avg), @ne(S)) - max(@all(B_avg), @se(S)))
    @all(dSdx)  = _dx*@d_xa(S)
    @all(dSdy)  = _dy*@d_ya(S)
    @all(grdS2) = @av_ya(dSdx)*@av_ya(dSdx) + @av_xa(dSdy)*@av_xa(dSdy)  # DEBUG: |∇S|^2 from Visnjevic et al. 2018 eq. (A5) should be |∇S|^(n-1). In the code |∇S| = sqrt(dS/dx^2 + dS/dy^2), and |∇S|^(n-1) = (sqrt(dS/dx^2 + dS/dy^2))^(npow-1)
    @all(D)     = (a1*@pow(@all(H_avg), (npow+2)) + a2*@pow(@all(H_avg), npow))*@all(grdS2)

    return
end

@parallel function compute_3!(qHx::Data.Array, qHy::Data.Array, D::Data.Array, S::Data.Array, _dx::Data.Number, _dy::Data.Number)

    # @all(qHx)  = -@av_ya(D)*( max(@all(Bx), @e_inn(S)) - max(@all(Bx), @w_inn(S)) )*_dx
    # @all(qHy)  = -@av_xa(D)*( max(@all(By), @n_inn(S)) - max(@all(By), @s_inn(S)) )*_dy

    @all(qHx)  = -@av_ya(D)*@d_xi(S)*_dx
    @all(qHy)  = -@av_xa(D)*@d_yi(S)*_dy

    return
end


@parallel function compute_4!(dHdτ::Data.Array, RH::Data.Array, dτ::Data.Array, qHx::Data.Array, qHy::Data.Array, A::Data.Array, D::Data.Array, _dx::Data.Number, _dy::Data.Number, damp::Data.Number, cfl::Data.Number, ε::Data.Number)

    @all(RH)   = -_dx*@d_xa(qHx) -_dy*@d_ya(qHy) + @inn(A)
    @all(dHdτ) = @all(dHdτ)*damp + @all(RH)
    @all(dτ)   = 0.5*min(1.0, cfl/(ε+@av_xya(D)))

    return
end

@parallel function compute_5!(H::Data.Array, dHdτ::Data.Array, dτ::Data.Array)

    @inn(H) = max(0.0, @inn(H) + @all(dτ)*@all(dHdτ))

    return
end

@parallel_indices (ix,iy) function set_BC!(H::Data.Array)

if (ix==1         && iy<=size(H,2)) H[ix,iy] = 0.0 end
if (ix==size(H,1) && iy<=size(H,2)) H[ix,iy] = 0.0 end
if (ix<=size(H,1) && iy==1        ) H[ix,iy] = 0.0 end
if (ix<=size(H,1) && iy==size(H,2)) H[ix,iy] = 0.0 end

return
end 

@parallel function compute_S!(S::Data.Array, B::Data.Array, H::Data.Array)

    @all(S) = @all(B) + @all(H)

    return
end 

##################################################
@views function sia2D()
    # physics
    ρg      = 910.0*9.81
    s2yr    = 31557600.0
    a0      = 1.5e-24
    npow    = 3
    a0      = 2.0
    Lx      = 30e3
    Ly      = 30e3
    xm      = 20e3
    t_tot   = 1e1
    # numerics
    BLOCK_X = 16
    BLOCK_Y = 16 
    GRID_X  = 8
    GRID_Y  = 8
    nx      = GRID_X*BLOCK_X
    ny      = GRID_Y*BLOCK_Y
    nout    = 1000
    ε       = 1e-2
    tolnl   = 1e-8
    damp    = 0.7
    itMax   = 100000
    # preprocess
    a1      = 2.0*a0/(npow+2)*ρg^npow*s2yr
    #a1      = 1.9e-24*ρg^npow*s2yr
    #a2      = 5.7e-20*ρg^npow*s2yr
    a2      = 5.7e-20
    dx, dy  = Lx/nx, Ly/ny
    xc      = LinRange(dx/2, Lx-dx/2, nx)
    yc      = LinRange(dy/2, Ly-dy/2, ny)
    (Xc,Yc) = ([x for x=xc,y=yc], [y for x=xc,y=yc])
    _dx, _dy = 1.0/dx, 1.0/dy
    cfl     = 1.0/8.1*max(dx*dx, dy*dy)
    cuthreads = (BLOCK_X, BLOCK_Y, 1)
    cublocks  = (GRID_X , GRID_Y , 1)
    # initial
    S       = @zeros(nx  ,ny  )
    B_avg   = @zeros(nx-1,ny-1)
    H_avg   = @zeros(nx-1,ny-1)
    dSdx    = @zeros(nx-1,ny  )
    dSdy    = @zeros(nx  ,ny-1)
    grdS2   = @zeros(nx-1,ny-1)
    D       = @zeros(nx-1,ny-1)
    # Bx      = @zeros(nx-1,ny-2)
    # By      = @zeros(nx-2,ny-1)
    qHx     = @zeros(nx-1,ny-2)
    qHy     = @zeros(nx-2,ny-1)
    RH      = @zeros(nx-2,ny-2)
    dHdτ    = @zeros(nx-2,ny-2)
    dτ      = @zeros(nx-2,ny-2)
    Err     = @zeros(nx  ,ny  )
    H       =  @ones(nx  ,ny  )
    B       =  zeros(nx  ,ny  )
    A       =  zeros(nx  ,ny  )
    # B[Xc.<7000] .= 500
    A       = (((npow.*a0./xm.^(2*npow-1)).*Xc.^(npow-1)).*abs.(xm.-Xc).^(npow-1)).*(xm.-2.0.*Xc)
    A[Xc.>xm] .= 0.0
    # ns      = 1
    # for is = 1:ns # smoothing (Mahaffy, 1976)
    #     B[2:end-1,2:end-1] .= B[2:end-1,2:end-1] .+ 1.0./4.1.*(diff(diff(B[:,2:end-1],dims=1),dims=1) + diff(diff(B[2:end-1,:],dims=2),dims=2))
    # end

    p1 = heatmap(xc,yc,B', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="B")
    p2 = heatmap(xc,yc,A', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="A")
    display(plot(p1, p2))

    B = Data.Array(B)
    A = Data.Array(A)

    @parallel cublocks cuthreads compute_S!(S, B, H)

    it = 1; err = 2*tolnl
    while (err>tolnl && it<itMax)

        @parallel cublocks cuthreads err_1!(Err, H)
        # @parallel cublocks cuthreads compute_1!(B_avg, Bx, By, B)
        @parallel cublocks cuthreads compute_2!(H_avg, dSdx, dSdy, grdS2, D, H, S, _dx, _dy, a1, a2, npow)
        @parallel cublocks cuthreads compute_3!(qHx, qHy, D, S, _dx, _dy)
        @parallel cublocks cuthreads compute_4!(dHdτ, RH, dτ, qHx, qHy, A, D, _dx, _dy, damp, cfl, ε)
        @parallel cublocks cuthreads compute_5!(H, dHdτ, dτ)
        @parallel cublocks cuthreads set_BC!(H)
        @parallel cublocks cuthreads compute_S!(S, B, H)

        # time_p = time_p + dτ
        it = it+1

        if mod(it, nout) == 0

            @parallel cublocks cuthreads err_2!(Err, H)

            err = (sum(abs.(Err[:]))./nx./ny)
            @printf("iter = %d, max resid = %1.3e \n", it, err)

            p1 = heatmap(xc,yc,S', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="S")
            p2 = heatmap(xc,yc,H', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="H")
            display(plot( p1, p2 ))

            if (err < tolnl) break; end
        end
    end

    return
end

sia2D()
