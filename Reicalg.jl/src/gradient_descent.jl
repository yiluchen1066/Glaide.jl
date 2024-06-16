function gradient_descent(J, ∇J!, As0, γ, maxiter; reg_params=nothing, callback=nothing)
    As = copy(As0)
    Ās = similar(As)
    Ās .= 0.0

    if !isnothing(reg_params)
        Ãs = similar(As)
    end

    dAs_dγ = similar(As)
    dAs_dγ .= 0.0

    J1 = J(As)
    isnothing(callback) || callback(0, γ, J1, As, Ās)

    for iter in 1:maxiter
        ∇J!(Ās, As)

        dmp = 0.95
        @. dAs_dγ = dAs_dγ * dmp + (Ās * As)

        # invert in log-space to avoid negative values
        @. As = As0 * exp(log(As / As0) - γ * dAs_dγ)

        if !isnothing(reg_params)
            As, Ãs = regularise!(As, Ãs, γ, reg_params...)
        end

        J1 = J(As)
        isnothing(callback) || callback(iter, γ, J1, As, Ās)
    end

    return As
end
