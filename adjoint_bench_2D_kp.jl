using CUDA,BenchmarkTools
using Printf
using DelimitedFiles
using Enzyme 
using JLD2
using Optim
using CairoMakie
#default(size=(1320,980),framestyle=:box,label=false,grid=true,margin=8mm,lw=3.5, labelfontsize=11,tickfontsize=11,titlefontsize=14)

macro get_thread_idx(A)  esc(:( begin ix =(blockIdx().x-1) * blockDim().x + threadIdx().x; iy = (blockIdx().y-1) * blockDim().y+threadIdx().y;end )) end 
macro av_xy(A)    esc(:(0.25*($A[ix,iy]+$A[ix+1,iy]+$A[ix, iy+1]+$A[ix+1,iy+1]))) end
macro av_xa(A)    esc(:( 0.5*($A[ix, iy]+$A[ix+1, iy]) )) end
macro av_ya(A)    esc(:(0.5*($A[ix, iy]+$A[ix, iy+1]))) end 
macro d_xa(A)     esc(:( $A[ix+1, iy]-$A[ix, iy])) end 
macro d_ya(A)     esc(:($A[ix, iy+1]-$A[ix,iy])) end
macro d_xi(A)     esc(:($A[ix+1, iy+1]-$A[ix, iy+1])) end 
macro d_yi(A)     esc(:($A[ix+1, iy+1]-$A[ix+1, iy])) end

CUDA.device!(5) # GPU selection

function laplacian!(as, as2, H)
    @get_thread_idx(H)
    if ix >= 2 && ix <= size(as,1)-1 && iy <= size(as,2)-1
        Δas = as[ix-1, iy]+as[ix+1, iy]+as[ix,iy-1]+as[ix,iy+1]-4.0as[ix,iy]
        as2[ix,iy] = as[ix,iy]+1/8*Δas
    end 
    return 
end 

function smooth!(as, as2, H, nsm, threads, blocks)
    for _= 1:nsm 
        CUDA.@sync @cuda threads=threads blocks=blocks laplacian!(as, as2, H)
        as2[[1,end],:] .= as[[2, end-1],:]
        as2[:,[1,end]] .= as[:,[2,end-1]]
        as, as2 = as2, as
    end 
    return 
end 

function compute_rel_error_1!(Err_rel, H) 
    @get_thread_idx(H)
    if (ix>=1 && ix<=size(H,1) && iy>=1 && iy<=size(H,2))
        @inbounds Err_rel[ix,iy] = H[ix,iy]
    end 
    return 
end 

function compute_rel_error_2!(Err_rel, H)
    @get_thread_idx(H) 
    if (ix <= size(H,1) && iy <= size(H,2))
        @inbounds Err_rel[ix, iy] = Err_rel[ix, iy] - H[ix, iy]
    end 
    return 
end

#J = 0.5*sum((H-H_obs)^2+(qHx-qHx_obs)^2+(qHy-qHy_obs)^2)
cost(H, H_obs, qHx, qHx_obs, qHy, qHy_obs) = 0.5*sum((H.-H_obs).^2+(qHx.-qHx_obs).^2+(qHy.-qHy_obs).^2)

function compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, aρgn0, As, n, dx, dy)
    @get_thread_idx(H)
    if (ix<=size(H,1)-1 && iy<=size(H,2)-1)
        @inbounds av_ya_∇Sx[ix,iy] = 0.5*((B[ix+1, iy+1]-B[ix,iy+1])/dx + (H[ix+1, iy+1]-H[ix,iy+1])/dx + (B[ix+1,iy]-B[ix,iy])/dx + (H[ix+1,iy]-H[ix,iy])/dx)
        @inbounds av_xa_∇Sy[ix,iy] = 0.5*((B[ix+1, iy+1]-B[ix+1,iy])/dy + (H[ix+1, iy+1]-H[ix+1,iy])/dy + (B[ix,iy+1]-B[ix,iy])/dy + (H[ix,iy+1]-H[ix,iy])/dy)
        @inbounds gradS[ix, iy] = sqrt(av_ya_∇Sx[ix,iy]^2+av_xa_∇Sy[ix,iy]^2)
        @inbounds D[ix, iy] = (aρgn0*@av_xy(H)^(n+2)+As[ix,iy]*@av_xy(H)^n)*gradS[ix,iy]^(n-1)
    end 
    return 
end 

function compute_q!(qHx, qHy, D, H, B, dx, dy)
    @get_thread_idx(H)
    if (ix<=size(H,1)-1 && iy<=size(H,2)-2)
        @inbounds qHx[ix,iy] = -@av_ya(D)*(@d_xi(H)+@d_xi(B))/dx 
    end 
    if (ix<=size(H,1)-2 && iy<=size(H,2)-1)
        @inbounds qHy[ix,iy] = -@av_xa(D)*(@d_yi(H)+@d_yi(B))/dy
    end 
    return 
end

function residual!(RH, qHx, qHy, β, H, B, ELA, m_max, dx, dy)
    @get_thread_idx(H)
    if (ix <= size(H,1)-2 && iy <= size(H,2)-2)
        @inbounds RH[ix+1, iy+1] = -(@d_xa(qHx)/dx + @d_ya(qHy)/dy) + min(β[ix+1,iy+1]*(H[ix+1, iy+1]+B[ix+1, iy+1]-ELA[ix+1, iy+1]), m_max)
    end 
    return 
end 

function compute_abs_error!(Err_abs, qHx, qHy, β, H, B, ELA, m_max, dx, dy)
    @get_thread_idx(H) 
    if (ix <= size(H,1)-2 && iy <= size(H,2)-2)
        @inbounds Err_abs[ix+1,iy+1] = - (@d_xa(qHx)/dx+@d_ya(qHy)/dy) + min(β[ix+1, iy+1]*(H[ix+1,iy+1]+B[ix+1,iy+1]-ELA[ix+1, iy+1]),m_max)
        if H[ix+1,iy+1] ≈ 0.0 
            Err_abs[ix+1,iy+1] = 0.0 
        end 
    end 
    return 
end

function update_H!(H, RH, dτ)
    @get_thread_idx(H) 
    if (ix <= size(H,1)-2 && iy <= size(H,2)-2)
        @inbounds H[ix+1,iy+1] = max(0.0, H[ix+1, iy+1]+dτ*RH[ix+1,iy+1])
    end 
    return 
end 

function set_BC!(H)
    @get_thread_idx(H)
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
        @inbounds H[ix,iy] = H[ix, iy-1]
    end 
    return 
end 

function update_S!(S, H, B)
    @get_thread_idx(H)
    if (ix <= size(H,1) && iy <= size(H,2))
        @inbounds S[ix,iy] = H[ix,iy] + B[ix,iy]
    end 
    return 
end 
#forward

#adjoint 
@inline ∇(fun,args...) = (Enzyme.autodiff_deferred(Enzyme.Reverse, fun, args...); return)
const DupNN = DuplicatedNoNeed

#fwd_params = (
#        fields       =     (;H,B,D,gradS,av_ya_∇Sx, av_xa_∇Sy, qHx,qHy, β, As, RH, Err_rel, Err_abs, ELA), 
#        scalars     =     (;n, nx, ny, dx, dy, maxiter, ncheck, threads, blocks), 
#        iter_params =     (;aρgn0, m_max, ϵtol),
#    )

@views function forward_solve!(logAs, fields, scalars, iter_params; visu=nothing)
    (;H, H_ini, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β, As, RH, Err_rel, Err_abs, ELA)     = fields 
    (;n, nx, ny, dx, dy, maxiter, ncheck, threads, blocks)                                  = scalars
    (;aρgn0, m_max, ϵtol)                                                                   = iter_params
    #visualization paramaters
    isnothing(visu) || ((;H_forward_vis, Err_abs_forward_vis, Err_rel_forward_vis)=visu)
    err_abs0 = Inf
    As .= exp.(logAs)
    H  .= H_ini
    for iter in 1:maxiter
        if iter % ncheck == 0
            CUDA.@sync @cuda threads=threads blocks=blocks compute_rel_error_1!(Err_rel, H)
        end 
        CUDA.@sync @cuda threads=threads blocks=blocks compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, aρgn0, As, n, dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks compute_q!(qHx, qHy, D, H, B, dx, dy)
        dτ=1/(12.1*maximum(D)/dx^2+maximum(β))
        CUDA.@sync @cuda threads=threads blocks=blocks residual!(RH, qHx, qHy, β, H, B, ELA, m_max, dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks update_H!(H, RH, dτ)
        CUDA.@sync @cuda threads=threads blocks=blocks set_BC!(H)
        if iter==1 || iter %ncheck == 0
            CUDA.@sync @cuda threads=threads blocks=blocks compute_abs_error!(Err_abs, qHx, qHy, β, H, B, ELA, m_max, dx, dy)
            CUDA.@sync @cuda threads=threads blocks=blocks compute_rel_error_2!(Err_rel, H)
            if iter == 1 
                err_abs0 = maximum(abs.(Err_abs))
            end 
            err_abs= maximum(abs.(Err_abs))/err_abs0
            err_rel= maximum(abs.(Err_rel))/maximum(H)
            @printf("iter/nx^2=%.3e, err= [abs=%.3e, rel=%.3e] \n", iter/nx^2, err_abs, err_rel)
            if !isnothing(visu)
                copyto!(H_forward_vis, H')
                copyto!(Err_abs_forward_vis, Err_abs')
                copyto!(Err_rel_forward_vis, Err_rel')
                p1 = Plots.heatmap(H_forward_vis; title="Ice thickness H (forward model)")
                p2 = Plots.heatmap(Err_abs_forward_vis; title="Err_abs")
                p3 = Plots.heatmap(Err_rel_forward_vis; title="Err_rel")
                display(Plots.plot(p1,p2,p3))

            end 
            if err_rel < ϵtol.rel
                break 
            end 
        end 
    end 
    return 
end 

@views function adjoint_solve!(logAs, fwd_params, adj_params, loss_params)
    #unpack forward 
    (;H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β, As, RH, ELA) = fwd_params.fields
    (;n, nx, ny, dx, dy, maxiter, threads, blocks)                    = fwd_params.scalars
    (;aρgn0, m_max)                                                   = fwd_params.iter_params

    #unpack adjoint
    (;q̄Hx, q̄Hy,D̄, H̄, R̄H, ψ_H)                                         = adj_params.fields
    (;ϵtol_adj,ncheck_adj)                                            = adj_params.iter_params
    (;H_obs, qHx_obs, qHy_obs, dJ_dH, ∂J_∂qx, ∂J_∂qy)                 = loss_params.fields

    dJ_dH  .= H   .- H_obs
    ∂J_∂qx .= qHx .- qHx_obs
    ∂J_∂qy .= qHy .- qHy_obs

    q̄Hx    .= ∂J_∂qx
    q̄Hy    .= ∂J_∂qy

    D̄      .= 0.0
    dt = 1.0/(8.1*maximum(D)/min(dx, dy)^2+maximum(β))

    #compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, aρgn0, As, n, dx, dy)
    #compute_q!(qHx, qHy, D, H, B, dx, dy)
    #residual!(RH, qHx, qHy, β, H, B, ELA, m_max, dx, dy)
    CUDA.@sync @cuda threads=threads blocks=blocks ∇(compute_q!,
            DupNN(qHx,q̄Hx),
            DupNN(qHy,q̄Hy),
            DupNN(D,D̄),
            DupNN(H,dJ_dH),
            Const(B), Const(dx), Const(dy)) 

    CUDA.@sync @cuda threads=threads blocks=blocks ∇(compute_D!,
            DupNN(D, D̄),
            Const(gradS),
            Const(av_ya_∇Sx), Const(av_xa_∇Sy),
            DupNN(H, dJ_dH),
            Const(B), Const(aρgn0), Const(As), Const(n), Const(dx), Const(dy))

    merr = 2ϵtol_adj; iter = 1
    while merr >= ϵtol_adj && iter < maxiter 

        R̄H      .= ψ_H
        q̄Hx     .= 0.0
        q̄Hy     .= 0.0
        D̄       .= 0.0
        H̄       .= .-dJ_dH
        #residual!(RH, qHx, qHy, β, H, B, ELA, m_max, dx, dy)
        CUDA.@sync @cuda threads = threads blocks=blocks ∇(residual!,
            DupNN(RH, R̄H),
            DupNN(qHx, q̄Hx),
            DupNN(qHy, q̄Hy),
            Const(β), DupNN(H, H̄), Const(B),
            Const(ELA), Const(m_max), Const(dx), Const(dy))
        #compute_q!(qHx, qHy, D, H, B, dx, dy)
        CUDA.@sync @cuda threads = threads blocks=blocks ∇(compute_q!,
            DupNN(qHx, q̄Hx),
            DupNN(qHy, q̄Hy),
            DupNN(D, D̄),
            DupNN(H, H̄),
            Const(B), Const(dx), Const(dy))
        #compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, aρgn0, As, n, dx, dy)
        CUDA.@sync @cuda threads = threads blocks=blocks ∇(compute_D!,
            DupNN(D, D̄),
            Const(gradS), Const(av_xa_∇Sy), Const(av_ya_∇Sx),
            DupNN(H,H̄),
            Const(B), Const(aρgn0), Const(As), Const(n),Const(dx), Const(dy))

        ψ_H .+= dt.*H̄
        ψ_H[[1,end],:] .= 0.0; ψ_H[:,[1,end]] .= 0.0

        if iter % ncheck_adj == 0
            merr = maximum(abs.(H̄))
            p1   = Plots.heatmap(Array(ψ_H'); aspect_ratio = 1, title="ψ")
            p2   = Plots.heatmap(Array(H̄'); aspect_ratio =1, title="H")
            display(Plots.plot(p1,p2))
            @printf("error = %.1e\n", merr)
            (isfinite(merr) && merr >0 || error("adjoint failed"))
        end 
        iter += 1
    end 

    if iter == maxiter && merr > ϵtol_adj
        error("adjoint not converged")
    end 
    @printf("adjoint solve converged: #iter/nx = %.1f, error = %.1e\n", iter/nx, merr)
    return 
end 

@views function loss(logAs, fwd_params, loss_params;kwags...)
    
    (;H_obs, qHx_obs, qHy_obs) = loss_params.fields
    
    @info "Forward solve"
    #forward_solve!(logAs, fields, scalars, iter_params; visu=nothing)
    forward_solve!(logAs, fwd_params.fields, fwd_params.scalars, fwd_params.iter_params)
    H     = fwd_params.fields.H
    qHx   = fwd_params.fields.qHx
    qHy   = fwd_params.fields.qHy
    return 0.5*sum((H.-H_obs).^2 .+(qHx.-qHx_obs).^2 .+(qHy.-qHy_obs).^2)
end 

function ∇loss!(Ās, logAs, fwd_params,adj_params,loss_params; reg=nothing,kwargs...)
    #unpack
    (;RH,qHx,qHy,β, H, B, ELA,D,gradS,av_ya_∇Sx, av_xa_∇Sy,As)   = fwd_params.fields 
    (;dx, dy,n, threads, blocks)                                 = fwd_params.scalars 
    (;m_max,aρgn0)                                               = fwd_params.iter_params
    (;R̄H, ψ_H, q̄Hx, q̄Hy, D̄)                                      = adj_params.fields
    (;∂J_∂qx, ∂J_∂qy)                                            = loss_params.fields 

    @info "Forward solve"
    #solve for H, qHx, qHy
    forward_solve!(logAs, fwd_params...; kwargs...)
    
    @info "Adjoint solve"
    #solve for ψ_H
    adjoint_solve!(logAs, fwd_params, adj_params, loss_params)

    R̄H    .= -ψ_H
    q̄Hx   .= ∂J_∂qx
    q̄Hy   .= ∂J_∂qy
    D̄     .= 0.0 
    Ās    .= 0.0
    CUDA.@sync @cuda threads=threads blocks=blocks ∇(residual!,
        DupNN(RH, R̄H), 
        DupNN(qHx, q̄Hx), 
        DupNN(qHy, q̄Hy),
        Const(β), Const(H), Const(B), Const(ELA), Const(m_max), Const(dx), Const(dy))
    CUDA.@sync @cuda threads=threads blocks=blocks ∇(compute_q!, 
        DupNN(qHx, q̄Hx), 
        DupNN(qHy, q̄Hy),
        DupNN(D, D̄),
        Const(H), Const(B), Const(dx), Const(dy))
    CUDA.@sync @cuda threads=threads blocks=blocks ∇(compute_D!, 
        DupNN(D, D̄), 
        Const(gradS), Const(av_ya_∇Sx), Const(av_xa_∇Sy), Const(H), Const(B), Const(aρgn0), 
        DupNN(As, Ās), 
        Const(n), Const(dx), Const(dy))

    #smoothing
    if !isnothing(reg)
        (;nsm, Tmp) = reg
        Tmp .= Ās
        smooth!(Ās, Tmp, H, nsm, threads, blocks)
    end
    #conver to dJ/dlogAs
    Ās .*= As #logĀs
    return 
end


@views function main()
    #CUDA.device!(1)
    #physics
    ## power law components 
    n         = 3 
    ## dimensionally independent physics 
    l         = 1e4 #1.0 # length scale  
    aρgn0     = 1.3517139631340709e-12 #1.0 #A*(ρg)^n = 1.9*10^(-24)*(910*9.81)^3
    ## time scales 
    tsc       = 1/aρgn0/l^n 
    ## non-dimensional numbers 
    s_f_syn   = 0.0003 # sliding to ice flow ratio: s_f_syn = asρgn0_syn/aρgn0/lx^2
    s_f_syn   = 0.01
    s_f       = 0.026315789473684213 # sliding to ice flow ratio: s_f   = asρgn0/aρgn0/lx^2
    m_max_nd  = 4.706167536706325e-12#m_max/lx*tsc m_max = 2.0 m/a lx = 1e3 tsc = 
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
    m_max       = m_max_nd*l/tsc  #2.0 m/a = 6.341958396752917e-8 m/s
    β0          = βtsc/tsc    #0.01 /a = 3.1709791983764586e-10
    β1          = β1tsc/tsc #0.015/3600/24/365 = 4.756468797564688e-10
    γ0          = γ_nd*l^(2-2n)*tsc^(-2) #1.0e-2
    δ           = δ_nd*l^(4-2n)*tsc^(-2)#0.1
    le          = 1e-6#0.01 
    #observations
    @show lx
    @show ly 
    @show lz 
    @show w1
    @show w2
    @show z_ELA_0 
    @show z_ELA_1 
    @show B0 
    @show H_cut 
    @show asρgn0_syn 
    @show asρgn0
    @show m_max 
    @show β0 
    @show β1 
    @show γ0 


    #numerics
    gd_niter    = 30 
    bt_niter    = 3 
    nx          = 128 
    ny          = 128 
    epsi        = 1e-4 
    ϵtol        = (abs = 1e-4, rel = 1e-4)
    dmp_adj     = 2*1.7 
    ϵtol_adj    = 1e-8
    gd_ϵtol     = 1e-3 
    Δγ          = 0.2

    maxiter     = 5*nx^2 
    ncheck      = ceil(Int, 0.25*nx^2)
    ncheck_adj  = 1000 
    threads     = (16,16)
    blocks      = ceil.(Int, (nx, ny)./threads)

    #perprocessing
    dx          = lx/nx 
    dy          = lx/ny 
    xc          = LinRange(-lx/2+dx/2, lx/2-dx/2, nx)
    yc          = LinRange(-ly/2+dy/2, ly/2-dy/2, ny)
    cfl          = max(dx^2,dy^2)/8.1                                                         

    #init 
    S           = CUDA.zeros(Float64, nx,ny)
    S_obs       = CUDA.zeros(Float64, nx,ny)
    H           = CUDA.zeros(Float64, nx,ny)
    H_obs       = CUDA.zeros(Float64, nx,ny)
    H_ini       = CUDA.zeros(Float64, nx,ny)
    B           = CUDA.zeros(Float64, nx,ny)
    β           = β0*CUDA.ones(Float64, nx, ny)
    ELA         = z_ELA_0*CUDA.ones(Float64, nx, ny)

    β         .+= β1 .* atan.(xc./lx)
    ELA       .+= z_ELA_1.*atan.(yc'./ly .+ 0 .*xc)

    ω           = 8 
    B           = @. B0*(exp(-xc^2/w1 - yc'^2/w2) + exp(-xc^2/w2-(yc'-ly/ω)^2/w1))

    D           = CUDA.zeros(Float64, nx-1,ny-1)
    av_ya_∇Sx   = CUDA.zeros(Float64, nx-1,ny-1)
    av_xa_∇Sy   = CUDA.zeros(Float64, nx-1,ny-1)
    gradS       = CUDA.zeros(Float64, nx-1,ny-1)
    qHx         = CUDA.zeros(Float64, nx-1,ny-2)
    qHy         = CUDA.zeros(Float64, nx-2,ny-1)
    B           = CuArray{Float64}(B)

    As          = asρgn0*CUDA.ones(nx-1,ny-1)
    As_ini_vis  = asρgn0*CUDA.ones(nx-1,ny-1)
    As_ini      = asρgn0*CUDA.ones(nx-1,ny-1)
    As_syn      = asρgn0_syn*CUDA.ones(nx-1,ny-1)
    Tmp         = asρgn0*CUDA.ones(nx-1,ny-1)
    logAs       = CUDA.zeros(nx-1, ny-1)
    logAs_syn   = CUDA.zeros(nx-1, ny-1)
    logAs      .= log.(As)
    logAs_syn  .= log.(As_syn)

    @show maximum(As)
    @show maximum(As_syn)
    @show maximum(logAs)
    @show maximum(logAs_syn)


    RH          = CUDA.zeros(Float64, nx, ny)
    Err_rel     = CUDA.zeros(Float64, nx, ny)
    Err_abs     = CUDA.zeros(Float64, nx, ny)

    #init adjoint storage
    q̄Hx         = CUDA.zeros(Float64, nx-1, ny-2)
    q̄Hy         = CUDA.zeros(Float64, nx-2, ny-1)
    D̄           = CUDA.zeros(Float64, nx-1, ny-1)
    H̄           = CUDA.zeros(Float64, nx,   ny)
    R̄H          = CUDA.zeros(Float64, nx,   ny)
    Ās          = CUDA.zeros(Float64, nx-1, ny-1)
    ψ_H         = CUDA.zeros(Float64, nx,   ny)
    qHx_obs     = CUDA.zeros(Float64, nx-1, ny-2)
    qHy_obs     = CUDA.zeros(Float64, nx-2, ny-1)
    dJ_dH       = CUDA.zeros(Float64, nx,   ny)
    ∂J_∂qx      = CUDA.zeros(Float64, nx-1, ny-2)
    ∂J_∂qy      = CUDA.zeros(Float64, nx-2, ny-1)

    #init visualization 
    xc_1        = xc[1:end-1]
    yc_1        = yc[1:end-1]
    fig          = Figure(resolution=(2000, 500), fontsize=32)
    opts        = ()


    axs         = (
        H       = Axis(fig[1,1][1,1]; aspect=DataAspect(), title="H", opts...),
        As      = Axis(fig[1,2][1,1]; aspect=DataAspect(), title="As", opts...),
        Err     = Axis(fig[1,3]; aspect=1, title="Error", opts...),

    )
    
    # CairoMakie.xlims!(axs.H,   -10,   10)
    # CairoMakie.xlims!(axs.H_s, 0.003, 0.02)
    # CairoMakie.xlims!(axs.As, -10, 10)
    # CairoMakie.xlims!(axs.As_s, -2.2, -1.4)

    # for axname in eachindex(axs)
    #     CairoMakie.ylims!(axs[axname], -10, 10)
    # end 

    plt     = (
        H   = heatmap!(axs.H, xc, yc, Array(H); colormap=:turbo),
        As  = heatmap!(axs.As, xc_1, yc_1, Array(As); color=:viridis)
    )

    H_forward_vis       = zeros(nx, ny)
    Err_abs_forward_vis = zeros(nx, ny)
    Err_rel_forward_vis = zeros(nx, ny)


    #action 
    fwd_params = (
        fields       =     (;H,H_ini,B,D,gradS,av_ya_∇Sx, av_xa_∇Sy, qHx,qHy, β, As, RH, Err_rel, Err_abs, ELA), 
        scalars     =     (;n, nx, ny, dx, dy, maxiter, ncheck, threads, blocks), 
        iter_params =     (;aρgn0, m_max, ϵtol),
    )
    fwd_visu        =     (;H_forward_vis, Err_abs_forward_vis, Err_rel_forward_vis)
    #fwd_visu  = (;) forward visualization
    @info "Synthetic solve" # solve synthetic with synthetic value of As_syn
    #forward_solve!(logAs_syn, fwd_params...; visu=fwd_visu)
    # turn off the forward visualization
    forward_solve!(logAs_syn, fwd_params...; visu=fwd_visu)
    H_obs          .= H
    qHx_obs        .= qHx
    qHy_obs        .= qHy


    error("check forward model")
    #store true data 
    #now the observe data is just the synthetic data

    adj_params  =(
        fields       =     (;q̄Hx, q̄Hy, D̄, H̄, R̄H, ψ_H),
        iter_params =     (;ϵtol_adj, ncheck_adj), 
    )
    loss_params = (
        fields       =      (;H_obs, qHx_obs, qHy_obs, dJ_dH, ∂J_∂qx, ∂J_∂qy), 
    )

    reg         = (;nsm=50, Tmp)
    # loss functions 
 
    J(_logAs)  =  loss(_logAs, fwd_params, loss_params)
    ∇J!(_Ās, _logAs) =  ∇loss!(_Ās, _logAs, fwd_params, adj_params, loss_params; reg, visu=fwd_visu)

    @info "Gradient descent - inversion for As" 
    # initial guess
    As     .= As_ini
    logAs  .= log.(As)
    cost_evo = Float64[]
    
    for gd_iter = 1:gd_niter
        # evaluate the gradient of the cost function 
        ∇J!(Ās,logAs)
        γ = Δγ / maximum(abs.(Ās))
        @. logAs -= γ*Ās
        push!(cost_evo, J(logAs)) 
        @printf " --> Loss J = %1.2e (γ = %1.2e)\n" last(cost_evo)/first(cost_evo)

        #visualization
        errs_evo   = Optim.f_trace(result)
        errs_evo ./= errs_evo[1]
        iters_evo  = 1:length(errs_evo)
        H_s        = H[nx÷2,:]
        H_obs_s    = H_obs[nx÷2,:]
        As_s       = As[nx÷2,:]
        As_ini_vis_s = As_ini_vis[nx÷2,:]
        As_syn_s   = As_syn[nx÷2,:]

        plts        = (
            H       = CarioMakie.heatmap!(axs.H, xc, yc, H; colormap=:turbo, colorrange=(0.002, 0.02)), 
            H_v     = CarioMakie.vlines!(axs.H, xc[nx÷2]; color=:magenta, linewidth=4, linestyle=:dash),
            H_s     = (
                CarioMakie.lines!(axs.H_s, H_obs_s, yc; linewdith=4, color=:red, label="synthetic"),
                CarioMakie.lines!(axs.H_s, H_s, yc; linewdith=4, color=:blue, label="current"),
            ),
            As      = CarioMakie.heatmap!(axs.As, xc_1, yc_1, As; colormap=:viridis, colorrange=(-2.0, -1.55)),
            As_v    = CarioMakie.vlines!(axs.As, xc_1[nx÷2];linewdith=4, color=:magenta, linstyle=:dash),
            As_s    = (
                CarioMakie.lines!(axs.As_s, As_s, yc_1; linewidth=4, color=:blue, label="current"),
                Cariomakie.lines!(axs.As_s, As_ini_vis_s,yc_1; linewidth=4, color=:green, label="initial"),
                CarioMakie.lines!(axs.As_s, As_syn_s, yc_1; linewidth=4, color=:red, label="synthetic"),
            ),

        )

        axislegend(axs.H_s; position=:lb, labelsize=20)
        axislegend(axs.As_s;position=:lb, labelsize=20)
        
        cb = Colorbar(fig[2,1], plts.H; vertical=false, label=L"H\text{[m]}", ticksize=4.0)
        Colorbar(fig[2,3], plt.As; vertical=false, label=L"\log_10{10}(A_s)", ticksize=4.0)
        colorgap!(fig.layout, 50)

        colorsize!(fig.layout,1,axs.H.scene.px_area[].widths[1])
        colorsize!(fig.layout,2,axs.H.acene.px_area[].widths[1])
        colorsize!(fig.layout,3,axs.As.scene.px_area[].widths[1])
        colorsize1(fig.layout,4,axs.As.scene.px_area[].widths[1])

        plts.H[3][]       = H 
        plts.As[3][]      = As
        plts.H_s[2][1][]  = Point2.(H_s, yc)
        plts.As_s[1][1][] = Point2.(As_s,yc_1)
    
    end
    return 
end 

main()