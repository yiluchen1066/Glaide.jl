function gradient_descent(J, ∇J!, As0, γ, maxiter, rtol; reg_params=nothing)
    As = copy(As0)
    Ās = similar(As)

    J0 = J(As)

    conv_hist = Float64[J0]
    for iter in 1:maxiter
        ∇J!(Ās, As)

        # invert in log-space to avoid negative values
        @. As = exp(log(As) - γ * (Ās / As))

        J1 = J(As)
        @printf("  iter = %d, J/J₀ = %1.3e\n", iter, J1 / J0)

        push!(conv_hist, J1)
    end

    return As, conv_hist
end
