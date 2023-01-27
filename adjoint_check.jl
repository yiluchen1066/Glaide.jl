using Enzyme

function residual_one_step!(R,H,D,β,dx)
    for ix in 2:length(R)-1
        R[ix] = D*(H[ix-1] - 2.0*H[ix] + H[ix+1])/dx^2 + β*H[ix]
    end
    return
end

function residual_two_step_1!(qHx,H,D,dx)
    for ix in eachindex(qHx)
        qHx[ix] = -D*(H[ix+1] - H[ix])/dx
    end
    return
end

function residual_two_step_2!(R,qHx,H,β,dx)
    for ix in 2:length(R)-1
        R[ix] = -(qHx[ix] - qHx[ix-1])/dx + β*H[ix]
    end
    return
end

function check()
    xc  = LinRange(-10,10,101)
    nx  = length(xc)
    dx  = xc[2] - xc[1]
    H   = exp.(.-xc.^2)
    D   = 10.0
    β   = 0.1
    R1  = zeros(nx)
    R2  = zeros(nx)
    qHx = zeros(nx-1)

    # one step
    residual_one_step!(R1,H,D,β,dx)

    # two steps
    residual_two_step_1!(qHx,H,D,dx)
    residual_two_step_2!(R2,qHx,H,β,dx)

    # adjoint variable
    r1 = ones(nx)
    r2 = ones(nx)
    # other temp variables...

    # Check if gradients of residual given by different
    # formulations are the same
    # your code...
    # calculating (dR/dH)^T*r 
    Enzyme.autiodiff_deferred(residual_one_step!, Duplicated(tmp1, tmp2), Duplicated(H, dR_H_one_step), Const(D), Const(β), Const(dx))
    





    @show maximum(abs.(R2 .- R1))
    return
end

check()