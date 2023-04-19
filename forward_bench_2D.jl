using CUDA,BenchmarkTools
using Plots,Plots.Measures,Printf
using DelimitedFiles
using Enzyme 
default(size=(800,600),framestyle=:box,label=false,grid=false,margin=10mm,lw=2,labelfontsize=11,tickfontsize=11,titlefontsize=14)

const DO_VISU = true 
macro get_thread_idx(A)  esc(:( begin ix =(blockIdx().x-1) * blockDim().x + threadIdx().x; iy = (blockIdx().y-1) * blockDim().y+threadIdx().y;end )) end 
macro av_xy(A)    esc(:(0.25*($A[ix,iy]+$A[ix+1,iy]+$A[ix, iy+1]+$A[ix+1,iy+1]))) end
macro av_xa(A)    esc(:( 0.5*($A[ix, iy]+$A[ix+1, iy]) )) end
macro av_ya(A)    esc(:(0.5*($A[ix, iy]+$A[ix, iy+1]))) end 
macro d_xa(A)     esc(:( $A[ix+1, iy]-$A[ix, iy])) end 
macro d_ya(A)     esc(:($A[ix, iy+1]-$A[ix,iy])) end
macro d_xi(A)     esc(:($A[ix+1, iy+1]-$A[ix, iy+1])) end 
macro d_yi(A)     esc(:($A[ix+1, iy+1]-$A[ix+1, iy])) end

CUDA.device!(6) # GPU selection

function compute_rel_error_1!(Err_rel, H, nx, ny)
    @get_thread_idx(H)
    if ix <= nx && iy <= ny 
        Err_rel[ix, iy] = H[ix, iy]
    end 
    return 
end 

function compute_rel_error_2!(Err_rel, H, nx, ny)
    @get_thread_idx(H) 
    if ix <= nx && iy <= ny
        Err_rel[ix, iy] = Err_rel[ix, iy] - H[ix, iy]
    end 
    return 
end

function cost!(H, H_obs, J, nx, ny)
    @get_thread_idx(H)
    if ix <= nx && iy <= ny
        J += (H[ix,iy]-H_obs[ix,iy])^2
    end 
    J *= 0.5 
    return 
end

cost(H, H_obs) = 0.5*sum((H.-H_obs).^2)

function compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, a, as, n, nx, ny, dx, dy)
    @get_thread_idx(H)
    if ix <= nx-1 && iy <= ny-1 
        av_ya_∇Sx[ix,iy] = 0.5*((B[ix+1, iy+1]-B[ix,iy+1])/dx + (H[ix+1, iy+1]-H[ix,iy+1])/dx + (B[ix+1,iy]-B[ix,iy])/dx + (H[ix+1,iy]-H[ix,iy])/dx)
        av_xa_∇Sy[ix,iy] = 0.5*((B[ix+1, iy+1]-B[ix+1,iy])/dy + (H[ix+1, iy+1]-H[ix+1,iy])/dy + (B[ix,iy+1]-B[ix,iy])/dy + (H[ix,iy+1]-H[ix,iy])/dy)
        gradS[ix, iy] = sqrt(av_ya_∇Sx[ix,iy]^2+av_xa_∇Sy[ix,iy]^2)
        D[ix, iy] = (a*@av_xy(H)^(n+2)+as[ix,iy]*@av_xy(H)^n)*gradS[ix,iy]^(n-1)
    end 
    return 
end 

function compute_q!(qHx, qHy, D, H, B, nx, ny, dx, dy)
    @get_thread_idx(H)
    if ix <= nx-1 && iy <= ny-2
        qHx[ix,iy] = -@av_ya(D)*(@d_xi(H)+@d_xi(B))/dx 
    end 
    if ix <= nx-2 && iy <= ny-1 
        qHy[ix,iy] = -@av_xa(D)*(@d_yi(H)+@d_yi(B))/dy 
    end 
    return
end



function residual!(RH, qHx, qHy, β, H, B, z_ELA, b_max, nx, ny , dx, dy)
    @get_thread_idx(H)
    if ix <= nx-2 && iy <= ny -2 
        RH[ix+1, iy+1] = -(@d_xa(qHx)/dx + @d_ya(qHy)/dy) + min(β[ix+1,iy+1]*(H[ix+1, iy+1]+B[ix+1, iy+1]-z_ELA[ix+1, iy+1]), b_max)
    end 
    return 
end 

function compute_abs_error!(Err_abs, qHx, qHy, β, H, B, z_ELA, b_max, nx, ny , dx, dy)
    @get_thread_idx(H)
    if ix <= nx-2 && iy <= ny-2 
        Err_abs[ix+1,iy+1] = -(@d_xa(qHx)/dx + @d_ya(qHy)/dy) + min(β[ix+1,iy+1]*(H[ix+1, iy+1]+B[ix+1, iy+1]-z_ELA[ix+1, iy+1]), b_max)
        if H[ix+1, iy+1] ≈ 0.0 
            Err_abs[ix+1,iy+1] = 0.0
        end 
    end 
    return 
end 

function update_H!(H, RH, dτ, nx, ny)
    @get_thread_idx(H)
    if ix <= nx-2 && iy <= ny -2 
        #update the inner point of H 
        H[ix+1,iy+1] = max(0.0, H[ix+1,iy+1]+dτ*RH[ix+1,iy+1])
    end 
    return 
end



function set_BC!(H, nx, ny)
    @get_thread_idx(H)
    if ix == 1 && iy <= ny 
        H[ix,iy] = H[ix+1, iy]
    end 
    if ix == nx && iy <= ny 
        H[ix,iy] = H[ix-1, iy]
    end 
    if ix <= nx && iy == 1 
        H[ix,iy] = H[ix, iy+1]
    end 
    if ix <= nx && iy == ny 
        H[ix, iy] = H[ix, iy-1]
    end 
    return 
end

function update_S!(S, H, B, nx, ny)
    @get_thread_idx(H)
    if ix <= nx && iy <= ny 
        S[ix,iy] = H[ix,iy] + B[ix,iy]
    end 
    return 
end 


mutable struct Forwardproblem{T<:Real, A<:AbstractArray{T}}
    H::A; B::A; D::A; gradS::A; av_ya_∇Sx::A; av_xa_∇Sy::A; qHx::A; qHy::A; β::A; as::A
    RH::A; Err_rel::A; Err_abs::A
    n::Int; a::T; b_max::T; z_ELA::A; dx::T; dy::T; nx::Int; ny::Int; epsi::T; cfl::T; ϵtol::NamedTuple{(:abs, :rel), Tuple{Float64, Float64}}; maxiter::Int; ncheck::Int; threads::Tuple{Int, Int}; blocks::Tuple{Int, Int}
end 

function Forwardproblem(H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β, as, n, a, b_max, z_ELA, dx, dy, nx, ny, epsi, cfl, ϵtol, maxiter, ncheck, threads, blocks)
    RH      = similar(H,nx, ny)
    Err_rel = similar(H, nx, ny)
    Err_abs = similar(H, nx, ny)
    return Forwardproblem(H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β, as, RH, Err_rel, Err_abs, n, a, b_max, z_ELA, dx, dy, nx, ny, epsi, cfl, ϵtol, maxiter, ncheck, threads, blocks) 
end 

#solve for H using pseudo-trasient method 
function solve!(problem::Forwardproblem)
    (;H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β, as, RH, Err_rel, Err_abs, n, a, b_max, z_ELA, dx,dy, nx, ny, epsi, cfl, ϵtol, maxiter, ncheck, threads, blocks) = problem
    RH .= 0; Err_rel .= 0; Err_abs .= 0
    lx, ly = 30e3, 30e3
    err_abs0 = 0.0 
    
    for iter in 1:maxiter
        if iter % ncheck == 0 
            CUDA.@sync @cuda threads = threads blocks = blocks compute_rel_error_1!(Err_rel, H, nx, ny)
        end 
        CUDA.@sync @cuda threads=threads blocks=blocks compute_D!(D, gradS, av_ya_∇Sx, av_xa_∇Sy, H, B, a, as, n, nx, ny, dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks compute_q!(qHx, qHy, D, H, B, nx, ny, dx, dy)
        #CUDA.@sync @cuda threads=threads blocks=blocks timestep!(dτ, H, D, cfl, epsi, nx, ny)
        dτ=1.0/(8.1*maximum(D)/dx^2 + maximum(β))
        CUDA.@sync @cuda threads=threads blocks=blocks residual!(RH, qHx, qHy, β, H, B, z_ELA, b_max, nx, ny , dx, dy)
        CUDA.@sync @cuda threads=threads blocks=blocks update_H!(H, RH, dτ, nx, ny)
        CUDA.@sync @cuda threads=threads blocks=blocks set_BC!(H, nx,ny)
        if iter ==1 || iter % ncheck == 0 
            CUDA.@sync @cuda threads=threads blocks=blocks compute_abs_error!(Err_abs, qHx, qHy, β, H, B, z_ELA, b_max, nx, ny , dx, dy)
            CUDA.@sync @cuda threads=threads blocks=blocks compute_rel_error_2!(Err_rel, H, nx, ny)
            if iter ==1 
                err_abs0 = maximum(abs.(Err_abs))
            end 
            err_abs = maximum(abs.(Err_abs))/err_abs0 
            err_rel = maximum(abs.(Err_rel))/maximum(H)
            @printf("iter/nx^2 = %.3e, err = [abs =%.3e, rel= %.3e] \n", iter/nx^2, err_abs, err_rel)
            #p1 = heatmap(Array(H'); title = "S (forward problem)")
            #p2 = heatmap(Array(Err_abs'); title = "Err_abs")
            #p3 = heatmap(Array(Err_rel'); title = "Err_rel")
            #display(plot(p1,p2,p3))
            #if debug


            #end
            if err_abs < ϵtol.abs || err_rel < ϵtol.rel 
                break 
            end 
        end 
    end 
    return 
end 

function adjoint_2D()
    # power law components 
    n        =  3 
    # dimensionally independet physics 
    l        =  1e4#1.0 # length scale lx = 1e3 (natural value)
    aρgn0    =  1.3517139631340713e-12 #1.0 # A*(ρg)^n # = 1.3475844936008e-12 (natural value)
    #scales 
    tsc      =  1/aρgn0/l^n # also calculated from natural values tsc = 0.7420684971878533
    #non-dimensional numbers (calculated from natural values)
    s_f_syn  = 0.0003 # sliding to ice flow ratio: s_f_syn = asρgn0_syn/aρgn0/lx^2
    s_f_no     = 0.0 # sliding to ice flow ratio: s_f   = asρgn0/aρgn0/lx^2
    s_f_1      = 0.026315789473684206
    s_f_2      = 2.6315789473684205e-6
    s_f_3      = 0.0026315789473684214
    m_max_nd = 4.706167536706325e-12#m_max/lx*tsc m_max = 2.0 m/a lx = 1e3 tsc = 
    βtsc     = 2.353083768353162e-10#ratio between characteristic time scales of ice flow and accumulation/ablation βtsc = β0*tsc 
    β1tsc    = 3.5296256525297436e-10
    # geometry
    lx_l     = 25.0 #horizontal length to characteristic length ratio
    ly_l     = 20.0 #horizontal length to characteristic length ratio 
    lz_l     = 1.0  #vertical length to charactertistic length ratio
    w1_l     = 100.0 #width to charactertistic length ratio 
    w2_l     = 10.0 # width to characteristic length ratio
    B0_l     = 0.35 # maximum bed rock elevation to characteristic length ratio
    z_ela_l  = 0.215 # ela to domain length ratio z_ela_l = 
    z_ela_1_l = 0.09
    # numerics
    H_cut_l  = 1.0e-6
    # dimensional  dependent physics parameters
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
    asρgn0_no   = s_f_no*aρgn0*l^2 #0.0 = 0.0 
    asρgn0_1    = s_f_1*aρgn0*l^2 #5.0e-18*(ρg)^n  = 3.5571420082475557e-6
    asρgn0_2    = s_f_2*aρgn0*l^2 #5.0e-22*(ρg)^n = 3.5571420082475552e-10
    asρgn0_3    = s_f_3*aρgn0*l^2 # 5.0e-19*(ρg)^n = 3.557142008247556e-7
    m_max       = m_max_nd*l/tsc  #2.0 m/a = 6.341958396752917e-8 m/s
    β0          = βtsc/tsc    #0.01 /a = 3.1709791983764586e-10
    β1          = β1tsc/tsc #0.015/3600/24/365 = 4.756468797564688e-10


    @show(asρgn0_no)
    @show(asρgn0_1)
    @show(asρgn0_2)
    @show(asρgn0_3)
    @show(asρgn0_syn)
    @show(lx)
    @show(ly)
    @show(lz)
    @show(w1)
    @show(w2)
    @show(B0)
    @show(z_ELA_0)
    @show(m_max)
    @show(β0)
    @show(z_ELA_1)
    @show(β1)
    @show(H_cut)

    # numerics 
    nx          = 128
    ny          = 128
    epsi        = 1e-4 
    ϵtol        = (abs = 1e-4, rel = 1e-4)
    maxiter     = 5*nx^2
    ncheck      = ceil(Int,0.25*nx^2)
    ncheck_adj  = 1000
    threads     = (16,16)
    blocks      = ceil.(Int, (nx,ny)./threads)


    # derived numerics
    ox, oy, oz  = -lx/2, -ly/2, 0.0 
    dx          = lx/nx 
    dy          = ly/ny 
    xv          = LinRange(ox, ox+lx, nx+1)
    yv          = LinRange(oy, oy+ly, ny+1)
    #xc         = 0.5*(xv[1:end-1]+xv[2:end])
    #yc         = 0.5*(yv[1:end-1]+yv[2:end])
    xc          = LinRange(-lx/2+dx/2, lx/2-dx/2, nx)
    yc          = LinRange(-ly/2+dy/2, ly/2-dy/2, ny)
    x0          = xc[round(Int, nx/2)]
    y0          = yc[round(Int, ny/2)]
    cfl          = max(dx^2, dy^2)/8.1


    # initialization 
    S = zeros(Float64, nx, ny)
    H = zeros(Float64, nx, ny)
    B = zeros(Float64, nx, ny)
    β = β0*ones(Float64, nx, ny)
    ela = z_ELA_0*ones(Float64, nx, ny)

    β   .+= β1 .*atan.(xc./lx)
    ela .+=  z_ELA_1.*atan.(yc'./ly .+ 0 .*xc)

    H_obs = copy(H)
    H_ini = copy(H)
    S_obs = copy(S)

    ω = 8
    B = @. B0*(exp(-xc^2/w1 - yc'^2/w2) + exp(-xc^2/w2-(yc'-ly/ω)^2/w1))


    
    #B = @. B0*(exp(-((xc-x0)/w)^2-((yc'-y0)/w)^2))*sin(ω*pi*(xc+yc'))
    #smoother

    #p1 = plot(xc,yc,B'; st=:surface, camera =(15,30), grid=true, aspect_ratio=1, labelfontsize=9,tickfontsize=7, xlabel="X in (m)", ylabel="Y in (m)", zlabel="Height in (m)", title="Synthetic Glacier bedrock")

    #B[2:end-1, 2:end-1] .= B[2:end-1, 2:end-1] .+ 1.0/4.1.*(diff(diff(B[:, 2:end-1], dims=1),dims=1) .+ diff(diff(B[2:end-1,:], dims=2), dims=2)) 
    #B[[1,end],:] .= B[[2,end-1],:]
    #B[:,[1,end]] .= B[:,[2,end-1]]
    S = CuArray{Float64}(S)
    B = CuArray{Float64}(B)
    H = CuArray{Float64}(H)
    S_obs = CuArray{Float64}(S_obs)
    H_obs = CuArray{Float64}(H_obs)
    H_ini = CuArray{Float64}(H_ini)
    β     = CuArray{Float64}(β)
    ela   = CuArray{Float64}(ela)
    H_no  = copy(H)
    H_1   = copy(H)
    H_2   = copy(H)
    H_3   = copy(H)

    #@show(extrema(B))
    #p1 = heatmap(xc, yc, Array((B)'); title = "B (forward problem initial)")
    #p2 = heatmap(xc, yc, Array(β'); title = "β")
    #p3 = heatmap(xc, yc, Array(ela'); title="ela")
    #p2 = plot(xc, yc, Array(B'); levels=20, aspect_ratio =1) 
    #p3 = plot(xc, Array(B[:,ceil(Int, ny/2)]))
    #display(plot(p1, p2, p3))

    D = CUDA.zeros(Float64,nx-1, ny-1)
    av_ya_∇Sx = CUDA.zeros(Float64, nx-1, ny-1)
    av_xa_∇Sy = CUDA.zeros(Float64, nx-1, ny-1) 
    gradS = CUDA.zeros(Float64, nx-1, ny-1)
    qHx = CUDA.zeros(Float64, nx-1,ny-2)
    qHy = CUDA.zeros(Float64, nx-2,ny-1)



    as_no = asρgn0_no*CUDA.ones(nx-1, ny-1) 
    as_1 = asρgn0_1*CUDA.ones(nx-1, ny-1) 
    as_2 = asρgn0_2*CUDA.ones(nx-1, ny-1) 
    as_3 = asρgn0_3*CUDA.ones(nx-1, ny-1) 
    as_syn = asρgn0_syn*CUDA.ones(nx-1,ny-1)


    Jn = CUDA.zeros(Float64,nx-1, ny-1)
    
    #Forwardproblem(H, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β, as, n, a, b_max, z_ELA, dx, dy, nx, ny, epsi, cfl, ϵtol, maxiter, ncheck, threads, blocks)
    synthetic_problem = Forwardproblem(H_obs, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β,as_syn, n, aρgn0, m_max, ela, dx, dy, nx, ny, epsi, cfl, ϵtol, maxiter, ncheck, threads, blocks)
    
    forward_problem_no = Forwardproblem(H_no, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β,as_no,n, aρgn0, m_max, ela, dx, dy, nx, ny, epsi, cfl, ϵtol, maxiter, ncheck, threads, blocks)
    forward_problem_1 = Forwardproblem(H_1, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β,as_1,n, aρgn0, m_max, ela, dx, dy, nx, ny, epsi, cfl, ϵtol, maxiter, ncheck, threads, blocks)
    forward_problem_2 = Forwardproblem(H_2, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β,as_2,n, aρgn0, m_max, ela, dx, dy, nx, ny, epsi, cfl, ϵtol, maxiter, ncheck, threads, blocks)
    forward_problem_3 = Forwardproblem(H_3, B, D, gradS, av_ya_∇Sx, av_xa_∇Sy, qHx, qHy, β,as_3,n, aρgn0, m_max, ela, dx, dy, nx, ny, epsi, cfl, ϵtol, maxiter, ncheck, threads, blocks)

    println("generating synthetic data (nx = $nx, ny = $ny)...")
    solve!(synthetic_problem)
    println("done.")
    solve!(forward_problem_no)
    solve!(forward_problem_1)
    solve!(forward_problem_2)
    solve!(forward_problem_3)

    println("gradient descent")

    Hp_obs = copy(H_obs)
    Hp_no  = copy(H_no)
    Hp_1  = copy(H_1)
    Hp_2  = copy(H_2)
    Hp_3  = copy(H_3)

    Hp_obs[H_obs .==0.0] .= NaN
    Hp_no[H_no.==0.0] .= NaN
    Hp_1[H_1.==0.0] .= NaN
    Hp_2[H_2.==0.0] .= NaN
    Hp_3[H_3.==0.0] .= NaN

    p2=plot(Array(Hp_no[nx÷2,:]),yc;xlabel="Y",ylabel="H", label="no sliding", legend=:outerbottom)
    plot!(Array(Hp_1[nx÷2,:]),yc;xlabel="Y",ylabel="H",label="as=5.0e-18*(ρg)^n ", legend=:outerbottom)
    plot!(Array(Hp_2[nx÷2,:]),yc;xlabel="Y",ylabel="H", label="as=5.0e-22*(ρg)^n", legend=:outerbottom)
    plot!(Array(Hp_3[nx÷2,:]),yc;xlabel="Y",ylabel="H", label="as=5.0e-19*(ρg)^n", legend=:outerbottom)
    plot!(Array(Hp_obs[nx÷2,:]),yc; xlabel="Y",ylabel="H",label="as=5.7e-20*(ρg)^n(synthetic)",  legend=:outerbottom)
    display(plot(p2; size=(490, 490)))
    savefig("forward_as.png")

    

    return 
end 

adjoint_2D() 