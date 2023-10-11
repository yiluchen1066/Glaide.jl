using CUDA,BenchmarkTools
using Printf
using DelimitedFiles
using Enzyme 
using JLD2
using CairoMakie

# TODO:
# 1. Fix the benchmark from Visjevic
# 2. Remove old scripts
# 3. Move model outputs to one folder
# 4. Add this folder to .gitignore
# 5. Code refactoring: split things into separate files
# 6. Upgrade the adjoint formulation (compute adj. fluxes)
# 7. ???
 
macro get_thread_idx()  esc(:( begin ix =(blockIdx().x-1) * blockDim().x + threadIdx().x; iy = (blockIdx().y-1) * blockDim().y+threadIdx().y;end )) end 
macro av_xy(A)    esc(:(0.25*($A[ix,iy]+$A[ix+1,iy]+$A[ix, iy+1]+$A[ix+1,iy+1]))) end
macro av_xa(A)    esc(:( 0.5*($A[ix, iy]+$A[ix+1, iy]) )) end
macro av_ya(A)    esc(:(0.5*($A[ix, iy]+$A[ix, iy+1]))) end 
macro d_xa(A)     esc(:( $A[ix+1, iy]-$A[ix, iy])) end 
macro d_ya(A)     esc(:($A[ix, iy+1]-$A[ix,iy])) end
macro d_xi(A)     esc(:($A[ix+1, iy+1]-$A[ix, iy+1])) end 
macro d_yi(A)     esc(:($A[ix+1, iy+1]-$A[ix+1, iy])) end

CUDA.device!(5) # GPU selection

function compute_rel_error_1!(Err_rel, H)
    @get_thread_idx
    if ix <= size(H,1) && iy <= size(H,2)
        @inbounds Err_rel[ix,iy] = H[ix,iy] 
    end 
    return 
end 

function compute_rel_error_2!(Err_rel, H)
    @get_thread_idx
    @inbounds if ix <= size(H,1) && iy <= size(H,2)
        Err_rel[ix,iy] = Err_rel[ix,iy] - H[ix,iy]
    end 
    return 
end 

cost(H, H_obs) = 0.5*sum((H.-H_obs).^2)

function compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, aρgn0, As, n, dx, dy)
    @get_thread_idx
    @inbounds if ix <= size(D,1) && iy <= size(D,2)
        av_ya_∇Sx[ix,iy] = 0.5*((B[ix+1, iy+1]-B[ix, iy+1])/dx + (H[ix+1, iy+1]-H[ix,iy+1])/dx + (B[ix+1,iy]-B[ix,iy])/dx + (H[ix+1,iy]-H[ix,iy])/dx)
        av_xa_∇Sy[ix,iy] = 0.5*((B[ix+1, iy+1]-B[ix+1, iy])/dy + (H[ix+1, iy+1]-H[ix+1,iy])/dy + (B[ix,iy+1]-B[ix,iy])/dy + (H[ix,iy+1]-H[ix,iy])/dy)
        gradS[ix,iy]     = sqrt(av_ya_∇Sx[ix,iy]^2+av_xa_∇Sy[ix,iy]^2)
        D[ix,iy]         = (aρgn0*@av_xy(H)^(n+2)+As[ix,iy]*@av_xy(H)^n)*gradS[ix,iy]^(n-1)
    end 
    return 
end 


function compute_q!(qHx, qHy, D, H, B, dx, dy)
    @get_thread_idx
    if ix <= size(qHx,1) && iy <= size(qHx, 2)
        @inbounds qHx[ix,iy] = -@av_ya(D)*(@d_xi(H)+@d_xi(B))/dx 
    end 
    if ix <= size(qHy,1) && iy <= size(qHy, 2)
        @inbounds qHy[ix,iy] = -@av_xa(D)*(@d_yi(H)+@d_yi(B))/dy 
    end 
    return
end

function residual!(RH, qHx, qHy, β, H, B, ELA, b_max, dx, dy)
    @get_thread_idx
    if ix <= size(H, 1)-2 && iy <= size(H,2)-2
        @inbounds RH[ix+1, iy+1] = -(@d_xa(qHx)/dx + @d_ya(qHy)/dy) + min(β[ix+1,iy+1]*(H[ix+1, iy+1]+B[ix+1, iy+1]-ELA[ix+1, iy+1]), b_max)
    end 
    return 
end 

function compute_abs_error!(Err_abs, qHx, qHy, β, H, B, ELA, b_max, dx, dy)
    @get_thread_idx
    if ix <= size(H,1)-2 && iy <= size(H,2)-2
        @inbounds Err_abs[ix+1,iy+1] = -(@d_xa(qHx)/dx + @d_ya(qHy)/dy) + min(β[ix+1,iy+1]*(H[ix+1, iy+1]+B[ix+1, iy+1]-ELA[ix+1, iy+1]), b_max)
        if H[ix+1, iy+1] ≈ 0.0 
            @inbounds Err_abs[ix+1,iy+1] = 0.0
        end 
    end 
    return 
end 

function update_H!(H, RH, dτ)
    @get_thread_idx
    if ix <= size(H,1)-2 && iy <= size(H,2)-2
        #update the inner point of H 
        @inbounds H[ix+1,iy+1] = max(0.0, H[ix+1,iy+1]+dτ*RH[ix+1,iy+1])
    end 
    return 
end

function set_BC!(H)
    @get_thread_idx
    if (ix == 1 && iy <= size(H,2))
        @inbounds H[ix,iy] = H[ix+1, iy]
    end 
    if (ix == size(H,1) && iy <= size(H,2))
        @inbounds H[ix,iy] = H[ix-1, iy]
    end 
    if (ix <= size(H,1) && iy == 1) 
        @inbounds H[ix,iy] = H[ix, iy+1]
    end 
    if (ix <= size(H,1) && iy == size(H,2))
        @inbounds H[ix, iy] = H[ix, iy-1]
    end 
    return 
end

function update_S!(S, H, B)
    @get_thread_idx
    if ix <= size(H,1) && iy <= size(H,2) 
        @inbounds S[ix,iy] = H[ix,iy] + B[ix,iy]
    end 
    return 
end 

function laplacian!(As, As2, H)
    @get_thread_idx
    if (ix >= 2 && ix <= size(As,1)-1 && iy <= size(As,2)-1)
        @inbounds ΔAs        = As[ix-1, iy]+As[ix+1,iy]+As[ix,iy-1]+As[ix,iy+1]-4.0*As[ix,iy]
        @inbounds As2[ix,iy] = As[ix,iy]+1/8*ΔAs
    end 
    return 
end 

function smooth!(As, As2, H, nsm, threads, blocks)
    for _ in 1:nsm
        CUDA.@sync @cuda threads=threads blocks=blocks laplacian!(As, As2, H)
        As2[[1,end],:] .= As[[2,end-1],:]
        As2[:,[1,end]] .= As[:, [2,end-1]]
        As, As2        = As2, As
    end 
    return 
end

function As_clean(As, H)
    @get_thread_idx
    if ix <= size(As,1) && iy <= size(As,2)
        if H[ix,iy] == 0.0 
            As[ix,iy] = 0.0 
        end 
    end 
    return 
end 

function update_ψ!(H, H̄, ψ_H, H_cut, dt)
    @get_thread_idx
    @inbounds if ix <= size(H,1) && iy <= size(H,2)
        if ix >1 && ix < size(H,1) && iy >1 && iy < size(H,2)
            if H[ix,iy]    <= H_cut ||
                H[ix-1,iy] <= H_cut ||
                H[ix+1,iy] <= H_cut ||
                H[ix,iy-1] <= H_cut ||
                H[ix,iy+1] <= H_cut 
                    H̄[ix,iy]    = 0.0 
                    ψ_H[ix,iy]  = 0.0
            else 
                ψ_H[ix,iy] = ψ_H[ix,iy] + dt*H̄[ix,iy]
            end 
        end 
        if ix == 1 || ix == size(H,1)
            ψ_H[ix,iy] = 0.0 
        end 
        if iy == 1 || iy == size(H,2)
            ψ_H[ix,iy] = 0.0 
        end 
    end 
    return
end


@inline ∇(fun, args...) = (Enzyme.autodiff_deferred(Enzyme.Reverse, fun, args...); return)
const DupNN = DuplicatedNoNeed

@views function forward_solve!(logAs, fields, scalars, iter_params; visu=nothing)
    (;H, H_ini, B, β, ELA, D, av_ya_∇Sx, av_xa_∇Sy, gradS, qHx, qHy, As, RH, Err_rel, Err_abs) = fields 
    (;nx, ny, dx, dy, maxiter, ncheck, threads, blocks)                                        = scalars 
    (;aρgn0, b_max, ϵtol, n)                                                                   = iter_params
    isnothing(visu) || ((;fig, plt)=visu)
    err_abs0 = Inf
    RH .= 0; Err_rel .= 0; Err_abs .= 0
    As .= logAs
    H  .= H_ini 
    CUDA.synchronize()
    # iterative loop 
    iters_evo = Float64[]; err_abs_evo = Float64[]; err_rel_evo= Float64[] 
    for iter in 1:maxiter
        if iter % ncheck == 0 
            CUDA.@sync @cuda threads=threads blocks=blocks compute_rel_error_1!(Err_rel, H)
        end 
        CUDA.@sync @cuda threads=threads blocks=blocks compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, aρgn0, As, n, dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks compute_q!(qHx, qHy, D, H, B, dx, dy)
        dτ = 1/(12.1*maximum(D)/dx^2+maximum(β))
        CUDA.@sync @cuda threads=threads blocks=blocks residual!(RH, qHx, qHy, β, H, B, ELA, b_max, dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks update_H!(H, RH, dτ)
        CUDA.@sync @cuda threads=threads blocks=blocks set_BC!(H)
        if iter == 1 || iter % ncheck == 0 
            CUDA.@sync @cuda threads=threads blocks=blocks compute_abs_error!(Err_abs, qHx, qHy, β, H, B, ELA, b_max, dx, dy)
            CUDA.@sync @cuda threads=threads blocks=blocks compute_rel_error_2!(Err_rel, H)
            if iter == 1 
                err_abs0 = maximum(abs.(Err_abs))
            end 
            err_abs = maximum(abs.(Err_abs))/err_abs0
            err_rel = maximum(abs.(Err_rel))/maximum(H)
            push!(iters_evo, iter/nx); push!(err_abs_evo, err_abs); push!(err_rel_evo, err_rel)
            @printf("iter/nx^2=%.3e, err= [abs=%.3e, rel=%.3e] \n", iter/nx^2, err_abs, err_rel)
            if !isnothing(visu)
                plt.H[3]            = Array(H) 
                plt.As[3]           = Array(As)
                #plt.Err[1][2][]   = Point2.(iters_evo, err_abs_evo)
                #plt.Err[1][2][]   = Point2.(iters_evo, err_rel_evo)
                display(fig)
            end
            if err_rel < ϵtol.rel 
                break 
            end 
        end 
    end 
    return 
end 

#solve for ψ_H
@views function adjoint_solve!(logAs, fwd_params, adj_params, loss_params)
    #unpack forward 
    (;H, RH, qHx, qHy, β, H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, As, ELA) = fwd_params.fields
    (;nx, ny, dx, dy, maxiter, threads, blocks)                          = fwd_params.scalars
    (;b_max, aρgn0, n)                                                   = fwd_params.iter_params

    #unpack adjoint 
    (;R̄H, H̄, ψ_H, q̄Hx, q̄Hy, D̄, ∂J_∂H)                                    = adj_params.fields 
    (;ϵtol_adj, ncheck_adj, H_cut)                                       = adj_params.iter_params
    (;H_obs)                                                             = loss_params.fields
    #dt = 1.0/(8.1*maximum(D)/min(dx,dy)^2+maximum(β))/7000
    dt = 1.0/(8.1*maximum(D)/min(dx,dy)^2+maximum(β))
    ∂J_∂H .= H .- H_obs
    @show(maximum(∂J_∂H))
    @show(H_cut)

    merr = 2ϵtol_adj; iter = 1
    while merr >= ϵtol_adj && iter < maxiter
        #initialization 
        #residual!(RH, qHx, qHy, β, H, B, z_ELA, b_max, dx, dy)
        R̄H  .= ψ_H
        q̄Hx .= 0.0
        q̄Hy .= 0.0 
        H̄   .= .-∂J_∂H
        D̄   .= 0.0
        CUDA.@sync @cuda threads=threads blocks=blocks ∇(residual!, 
            DupNN(RH, R̄H),
            DupNN(qHx, q̄Hx), 
            DupNN(qHy, q̄Hy),
            Const(β),
            DupNN(H,H̄),
            Const(B), Const(ELA), Const(b_max), Const(dx), Const(dy))
        #compute_q!(qHx, qHy, D, H, B, dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks ∇(compute_q!,
            DupNN(qHx, q̄Hx), 
            DupNN(qHy, q̄Hy),
            DupNN(D,D̄),
            DupNN(H, H̄),
            Const(B), Const(dx), Const(dy))
        #compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, aρgn0, As, n, dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks ∇(compute_D!, 
            DupNN(D, D̄), 
            Const(gradS), Const(av_ya_∇Sx), Const(av_xa_∇Sy), 
            DupNN(H, H̄),
            Const(B), Const(aρgn0), Const(As), Const(n), Const(dx), Const(dy))
        CUDA.@sync @cuda threads=threads blocks=blocks update_ψ!(H, H̄, ψ_H, H_cut, dt)
        
        if iter % ncheck_adj == 0 
            merr = maximum(abs.(dt.*H̄[2:end-1, 2:end-1]))/maximum(abs.(ψ_H))
            @printf("error = %.1e\n", merr) 
            # visualization for adjoint solver
            @printf("H̄ = %.1e\n", maximum(abs.(H̄)))
            (isfinite(merr) && merr >0) || error("adjoint solve failed")
        end 

        iter += 1 
    end 

    if iter == maxiter && merr >= ϵtol_adj
        error("adjoint not converged")
    end 

    @printf("adjoint solve converged: #iter/nx = %.1f, err = %.1e\n", iter/nx, merr)
    return 
end 

@views function loss(logAs, fwd_params, loss_params; kwags...)
    (;H_obs)    = loss_params.fields
    @info "Forward solve"
    forward_solve!(logAs, fwd_params...; kwags...)
    H    = fwd_params.fields.H
    return 0.5*sum((H.-H_obs).^2)
end 

function ∇loss!(Ās, logAs, fwd_params, adj_params, loss_params; reg=nothing,kwargs...)
    #unpack
    (;RH, qHx, qHy, β, H, B, D, ELA, gradS, av_ya_∇Sx, av_xa_∇Sy, As)   = fwd_params.fields
    (;dx, dy, threads, blocks)                                          = fwd_params.scalars
    (;aρgn0, b_max,n)                                                   = fwd_params.iter_params
    (;R̄H, q̄Hx, q̄Hy, H̄, D̄, Ās, ψ_H)                                      = adj_params.fields 



    @info "Forward solve" 
    forward_solve!(logAs, fwd_params...; kwargs...)

    @info "Adjoint solve"
    adjoint_solve!(logAs, fwd_params, adj_params, loss_params)

    Ās  .= 0.0 
    R̄H  .= .-ψ_H
    q̄Hx .= 0.0
    q̄Hy .= 0.0 
    H̄   .= 0.0 
    D̄   .= 0.0 

    #residual!(RH, qHx, qHy, β, H, B, z_ELA, b_max, dx, dy)
    CUDA.@sync @cuda threads=threads blocks=blocks ∇(residual!, 
        DupNN(RH, R̄H), 
        DupNN(qHx, q̄Hx), 
        DupNN(qHy, q̄Hy), 
        Const(β),
        DupNN(H, H̄),
        Const(B), Const(ELA), Const(b_max), Const(dx), Const(dy))
    #compute_q!(qHx, qHy, D, H, B, dx, dy)
    CUDA.@sync @cuda threads=threads blocks=blocks ∇(compute_q!,
        DupNN(qHx, q̄Hx), 
        DupNN(qHy, q̄Hy),
        DupNN(D, D̄), 
        DupNN(H, H̄),
        Const(B), Const(dx), Const(dy))
    #compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, aρgn0, As, n, dx, dy)
    CUDA.@sync @cuda threads=threads blocks=blocks ∇(compute_D!,
        DupNN(D, D̄),
        Const(gradS), Const(av_ya_∇Sx), Const(av_xa_∇Sy), Const(H), Const(B), Const(aρgn0),
        DupNN(As, Ās), Const(n), Const(dx), Const(dy))

    Ās[[1,end],:] = Ās[[2,end-1],:]
    Ās[:,[1,end]] = Ās[:,[2,end-1]]  

    #smoothing 
    if !isnothing(reg) 
        (;nsm, Tmp) = reg
        Tmp .= Ās
        smooth!(Ās, Tmp, H, nsm, threads, blocks)
    end
    # convert to dJ/dlogAs
    #Ās .*= As 

    return 
end 

@views function main()
        #CUDA.device!(1)
    #physics
    ## power law components 
    n         = 3 
    ## dimensionally independent physics 
    l         = 1.0#1e4 #1.0 # length scale  
    aρgn0     = 1.0#1.3517139631340709e-12 #1.0 #A*(ρg)^n = 1.9*10^(-24)*(910*9.81)^3
    ## time scales 
    tsc       = 1/aρgn0/l^n 
    ## non-dimensional numbers 
    s_f_syn   = 0.0003 # sliding to ice flow ratio: s_f_syn = asρgn0_syn/aρgn0/lx^2
    s_f_syn   = 0.01
    s_f       = 0.026315789473684213 # sliding to ice flow ratio: s_f   = asρgn0/aρgn0/lx^2
    b_max_nd  = 4.706167536706325e-12#m_max/lx*tsc m_max = 2.0 m/a lx = 1e3 tsc = 
    βtsc      = 2.353083768353162e-10#ratio between characteristic time scales of ice flow and accumulation/ablation βtsc = β0*tsc 
    β1tsc     = 3.5296256525297436e-10
    γ_nd      = 1e0
    δ_nd      = 1e-1
    ## geometry 
    lx_l      = 25.0 #horizontal length to characteristic length ratio
    ly_l      = 20.0 #horizontal length to characteristic length ratio 
    lz_l      = 1.0  #vertical length to charactertistic length ratio
    w1_l      = 100.0 #width to charactertistic length ratio 
    w2_l      = 10.0 # width to characteristic length ratio
    B0_l      = 0.35 # maximum bed rock elevation to characteristic length ratio
    z_ela_l   = 0.215 # ela to domain length ratio z_ela_l = 
    z_ela_1_l = 0.09
    ## numerics 
    H_cut_l   = 1.0e-6
    ## dimensional dependent physics parameters 
    lx          = lx_l*l #250000
    ly          = ly_l*l #200000
    lz          = lz_l*l  #1e3
    w1          = w1_l*l^2 #1e10
    w2          = w2_l*l^2 #1e9
    z_ELA_0     = z_ela_l*l # 2150
    z_ELA_1     = z_ela_1_l*l #900
    B0          = B0_l*l # 3500
    H_cut       = H_cut_l*l # 1.0e-2
    asρgn0_syn  = s_f_syn*aρgn0*l^2 #5.7e-20*(ρg)^n = 5.7e-20*(910*9.81)^3 = 4.055141889402214e-8
    asρgn0      = s_f*aρgn0*l^2 #5.0e-18*(ρg)^n = 3.54627498316e-6
    b_max       = b_max_nd*l/tsc  #2.0 m/a = 6.341958396752917e-8 m/s
    β0          = βtsc/tsc    #0.01 /a = 3.1709791983764586e-10
    β1          = β1tsc/tsc #0.015/3600/24/365 = 4.756468797564688e-10
    γ0          = γ_nd*l^(2-2n)*tsc^(-2) #1.0e-2
    δ           = δ_nd*l^(4-2n)*tsc^(-2)#0.1
    le          = 1e-6#0.01 
    #observations


    #numerics
    ngd   = 30 
    bt_niter    = 3 
    nx          = 128 
    ny          = 128 
    epsi        = 1e-4 
    ϵtol        = (abs = 1e-8, rel = 1e-8)
    dmp_adj     = 2*1.7 
    ϵtol_adj    = 1e-8
    gd_ϵtol     = 1e-3 
    Δγ          = 0.2

    maxiter     = 5*nx^2 
    ncheck      = ceil(Int, 0.25*nx^2)
    ncheck_adj  = 1000 
    threads     = (16,16)
    blocks      = ceil.(Int, (nx, ny)./threads)

    #check 
    @show(asρgn0_syn)
    @show(asρgn0)
    @show(lx)
    @show(ly)
    @show(w1)
    @show(w2)
    @show(B0)
    @show(z_ELA_0)
    @show(z_ELA_1)
    @show(b_max)
    @show(β0)
    @show(β1)
    @show(H_cut)
    @show(nx)
    @show(ny)
    @show(ϵtol)
    @show(maxiter)
    @show(ncheck)
    @show(threads)
    @show(blocks)
    @show(ϵtol_adj)
    @show(ncheck_adj)


    #perprocessing
    dx          = lx/nx 
    dy          = ly/ny 
    xc          = LinRange(-lx/2+dx/2, lx/2-dx/2, nx)
    yc          = LinRange(-ly/2+dy/2, ly/2-dy/2, ny)
    cfl          = max(dx^2,dy^2)/8.1  

    #init 
    S           = CUDA.zeros(Float64, nx, ny)
    S_obs       = CUDA.zeros(Float64, nx, ny)
    H           = CUDA.zeros(Float64, nx, ny)
    H_obs       = CUDA.zeros(Float64, nx, ny)
    H_ini       = CUDA.zeros(Float64, nx, ny)
    B           = CUDA.zeros(Float64, nx, ny)
    β           = β0*CUDA.ones(Float64, nx, ny)
    ELA         = z_ELA_0*CUDA.ones(Float64, nx, ny)

    β         .+= β1 .* atan.(xc./lx)
    ELA       .+= z_ELA_1.*atan.(yc'./ly .+ 0 .* xc)

    ω           = 8 
    B           = @. B0*(exp(-xc^2/w1-yc'^2/w2)+exp(-xc^2/w2-(yc'-ly/ω)^2/w1))
    
    D           = CUDA.zeros(Float64, nx-1, ny-1)
    av_ya_∇Sx   = CUDA.zeros(Float64, nx-1, ny-1)
    av_xa_∇Sy   = CUDA.zeros(Float64, nx-1, ny-1)
    gradS       = CUDA.zeros(Float64, nx-1, ny-1)
    qHx         = CUDA.zeros(Float64, nx-1, ny-2)
    qHy         = CUDA.zeros(Float64, nx-2, ny-1)
    B           = CuArray{Float64}(B)

    As          = asρgn0*CUDA.ones(nx-1, ny-1)
    As_ini_vis  = asρgn0*CUDA.ones(nx-1, ny-1)
    As_ini      = asρgn0*CUDA.ones(nx-1, ny-1)
    As_syn      = asρgn0_syn*CUDA.ones(nx-1, ny-1)
    As2         = asρgn0*CUDA.ones(nx-1, ny-1)
    logAs       = CUDA.zeros(Float64, nx-1, ny-1)
    logAs_syn   = CUDA.zeros(Float64, nx-1, ny-1)
    logAs2      = CUDA.zeros(Float64, nx-1, ny-1)
    logAs_ini   = CUDA.zeros(Float64, nx-1, ny-1)
    logAs      .= As
    logAs_syn  .= As_syn
    logAs_ini  .= As_ini

    #As          = asρgn0_syn*CUDA.ones(nx-1,ny-1)
    #logAs       = CUDA.zeros(Float64, nx-1, ny-1)
    #logAs      .= log.(As)

    RH          = CUDA.zeros(Float64, nx, ny)
    Err_rel     = CUDA.zeros(Float64, nx, ny)
    Err_abs     = CUDA.zeros(Float64, nx, ny)

    #init adjoint storage
    q̄Hx         = CUDA.zeros(Float64,nx-1, ny-2)
    q̄Hy         = CUDA.zeros(Float64,nx-2, ny-1)
    D̄           = CUDA.zeros(Float64,nx-1, ny-1)
    H̄           = CUDA.zeros(Float64,nx,   ny)
    R̄H          = CUDA.zeros(Float64,nx,   ny)
    Ās          = CUDA.zeros(Float64,nx-1, ny-1)
    Tmp         = CUDA.zeros(Float64,nx-1, ny-1)
    ψ_H         = CUDA.zeros(Float64,nx,   ny)
    ∂J_∂H       = CUDA.zeros(Float64,nx,   ny)
    dJ_dlogAs   = CUDA.zeros(Float64,nx-1, ny-1)

    #init visualization 
    iter_evo = Float64[]; errs_abs_evo = Float64[]; errs_rel_evo = Float64[]
    J_evo = Float64[1.0]; iter_evo_gd = Int[0] 
    xc_1        = xc[1:end-1]
    yc_1        = yc[1:end-1]
    fig          = Figure(resolution=(2000, 1500), fontsize=32)
    opts        = ()

    axs         = (
        H       = Axis(fig[1,1][1,1]; aspect=DataAspect(), title="H", opts...), 
        As      = Axis(fig[1,2][1,1]; aspect=DataAspect(), title="As", opts...),
    )

    gd_error    =  Axis(fig[2,1]; yscale=log10, aspect=1, title="Error", opts...)

    #for axname in eachindex(axs)
    #     ylims!(axs[axname], -10, 10)
    # end 
    xlims!(gd_error, 0, 20)
    ylims!(gd_error, 1e-8, 2)

    plt  = (
        H     = heatmap!(axs.H,  xc, yc, Array(H); colormap=:turbo), 
        As    = heatmap!(axs.As, xc_1, yc_1, Array(As); colormap=:viridis),
    )

    plt_err = scatterlines!(gd_error, Point2.(iter_evo_gd, J_evo), linewidth=4)



    Colorbar(fig[1,1][1,2], plt.H)
    Colorbar(fig[1,2][1,2], plt.As)

    # action 
    fwd_params = (
        fields       = (;H, H_ini, B, β, ELA, D, av_ya_∇Sx, av_xa_∇Sy, gradS, qHx, qHy, As, RH, Err_rel, Err_abs),
        scalars     = (;nx, ny, dx, dy, maxiter, ncheck, threads, blocks),
        iter_params = (;aρgn0, b_max, ϵtol,n),  
    )
    fwd_visu    = (;plt, fig)

    @info "Synthetic solve" 
    #solve for synthetic forward model using As_syn for H_obs
    @show(maximum(logAs_syn))
    forward_solve!(logAs_syn, fwd_params...; visu=fwd_visu)
    H_obs .= H #the observed data is just the synthetic data

    write("output/synthetic_old_2.dat", Array(H_obs), Array(D), Array(As), Array(ELA), Array(β))
    @info "Done"
    adj_params = (
        fields       = (;q̄Hx, q̄Hy, D̄, H̄, R̄H, Ās, ψ_H, ∂J_∂H),
        iter_params = (;ϵtol_adj, ncheck_adj, H_cut),
    )

    loss_params = (
        fields       = (;H_obs),
    )

    #the 
    reg         = (;nsm=50, Tmp)

    J(_logAs)          = loss(_logAs, fwd_params, loss_params)
    ∇J!(_Ās,_logAs)    = ∇loss!(_Ās, _logAs, fwd_params, adj_params, loss_params; reg)
    @info "Inversion for As"
    @info "Forward solve"
    forward_solve!(logAs, fwd_params...; visu=fwd_visu)

    write("output/forward_old_2.dat", Array(H), Array(D), Array(As), Array(ELA), Array(β))
    @info "Done"
    #adjoint_solve!(logAs, fwd_params, adj_params, loss_params)
    @info "Adjoint solve"
    adjoint_solve!(logAs, fwd_params, adj_params, loss_params)
    write("output/adjoint_old_2.dat", Array(ψ_H), Array(H̄), Array(q̄Hx), Array(q̄Hy), Array(D̄))
    @info "Done"

    error("here")
    #initial guess

    γ = γ0
    J_old = 0.0; J_new = 0.0 
    J_old = sqrt(cost(H, H_obs)*dx*dy)
    @show(maximum(abs.(H.-H_obs)))
    J_ini = J_old
    
    
    for gd_iter = 1:ngd
        #starting from the initial guess as_ini
        As_ini      .= As
        logAs       .= As
        logAs_ini   .= As_ini
        ∇J!(dJ_dlogAs, logAs) # calculating and smoothing the sensitivity dJ_dlogAs

        # line search 
        for bt_iter = 1:bt_niter 
            @. logAs = clamp(logAs-γ*dJ_dlogAs, 0.0, Inf)
            smooth!(logAs, logAs2, H, 10, threads, blocks)
            
            @show(maximum(abs.(H.-H_obs)))
            J_new = sqrt(cost(H, H_obs)*dx*dy)
            if J_new < J_old 
                # we accept the current value of γ
                γ *= 1.1 
                J_old = J_new
                
                @printf("new solution accepted\n")
                break
            else
                logAs  .= logAs_ini
                γ = γ*0.5
            end 
        # end of the line search loops 
        end 

        push!(iter_evo_gd, gd_iter); push!(J_evo, J_old/J_ini)
        #visualization 
        plt_err[1][] = Point2.(iter_evo_gd,J_evo)
    end

    #   As      .= As_ini
    #  logAs   .= log.(As)
    #  iter_evo=Int[0]; cost_evo = Float64[]
    #  @info "Gradient descent - inversion for As"
    #  cost_evo = Float64[]
    # # for igd in 1:ngd


    # for igd in 1:ngd
    #     #evaluate gradient of the cost function
    #     ∇J!(dJ_dlogAs, logAs)
    #     # update logAs
    #     γ = Δγ / maximum(abs.(dJ_dlogAs))
    #     @. logAs -= γ*dJ_dlogAs
    #     push!(iter_evo, igd); push!(cost_evo, J(logAs))
    #     @printf "Loss J = %1.2e (γ = %1.2e) \n" last(cost_evo)/first(cost_evo) γ
        
    #     # visualization

    #  end
    return 
end 

main()

