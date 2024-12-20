using Enzyme, CUDA, StaticArrays

function residual(H, B, n, d)
    ∇Hl  = H[2] - H[1]
    ∇Hr  = H[3] - H[2]
    ∇Bl  = B[2] - B[1]
    ∇Br  = B[3] - B[2]
    ∇Sl  = ∇Bl + ∇Hl
    ∇Sr  = ∇Br + ∇Hr
    ∇Snl = sqrt(∇Sl^2)^(n - 1)
    ∇Snr = sqrt(∇Sr^2)^(n - 1)
    ql   = ∇Snl * (H[2]^(n + 3) - H[1]^(n + 3)) + ∇Bl * H[1]^(n + 2)
    qr   = ∇Snr * (H[3]^(n + 3) - H[2]^(n + 3)) + ∇Br * H[2]^(n + 2)
    r    = d * (qr - ql) + H[2]
    return r
end

function residual!(r, H, B, n, d)
    for i in 2:length(H)-1
        ∇Hl  = H[i] - H[i-1]
        ∇Hr  = H[i+1] - H[i]
        ∇Bl  = B[i] - B[i-1]
        ∇Br  = B[i+1] - B[i]
        ∇Sl  = ∇Bl + ∇Hl
        ∇Sr  = ∇Br + ∇Hr
        ∇Snl = sqrt(∇Sl^2)^(n - 1)
        ∇Snr = sqrt(∇Sr^2)^(n - 1)
        ql   = ∇Snl * (H[i]^(n + 3) - H[i-1]^(n + 3)) + ∇Bl * H[i-1]^(n + 2)
        qr   = ∇Snr * (H[i+1]^(n + 3) - H[i]^(n + 3)) + ∇Br * H[i]^(n + 2)
        r[i] = d * (qr - ql) + H[i]
    end
    return
end

function residual2!(r, H, B, n, d)
    for i in eachindex(H)
        iW = max(i - 1, 1)
        iE = min(i + 1, length(H))
        Hl = SVector(H[iW], H[i], H[iE])
        Bl = SVector(B[iW], B[i], B[iE])
        r[i] = residual(Hl, Bl, n, d)
    end
end

function enzyme_wtf()
    nx = 32
    n  = 3
    d  = 1e0
    H  = [i / nx for i in 1:nx]
    B  = [sin(4π * i / nx) for i in 1:nx]
    r  = zeros(Float64, nx)
    r̄ = rand(Float64, nx)
    H̄ = zeros(Float64, nx)
    residual2!(r, H, B, n, d)
    r̄2 = copy(r̄)
    Enzyme.autodiff(Enzyme.Reverse, Const(residual2!), DupNN(r, r̄), DupNN(H, H̄), Const(B), Const(n), Const(d))
    
    # display(r)
    display(r̄2)
    display(H̄)

    display(r̄2 - H̄)
    return
end

enzyme_wtf()
