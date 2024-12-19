using Enzyme, CUDA

function residual!(r, H, n, d)
    for i in 2:length(H)-1
        ∇Hl  = H[i] - H[i-1]
        ∇Hr  = H[i+1] - H[i]
        ∇Hnl = sqrt(∇Hl^2)^(n - 1)
        ∇Hnr = sqrt(∇Hr^2)^(n - 1)
        ql   = ∇Hnl * (H[i]^(n + 3) - H[i-1]^(n + 3))
        qr   = ∇Hnr * (H[i+1]^(n + 3) - H[i]^(n + 3))
        r[i] = d * (qr - ql) + H[i]
    end
    return
end

function enzyme_wtf()
    nx = 10
    n  = 3
    d  = 1e0
    H  = rand(Float64, nx)
    r  = zeros(Float64, nx)
    r̄  = ones(Float64, nx)
    H̄  = zeros(Float64, nx)
    Enzyme.autodiff(Enzyme.Reverse, Const(residual!), DupNN(r, r̄), DupNN(H, H̄), Const(n), Const(d))

    display(H̄)
    return
end

enzyme_wtf()
