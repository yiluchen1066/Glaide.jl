function gradient_descent(J, ∇J!, As0, γ, maxiter; reg_params=nothing, callback=nothing)
    # need an extra array for backtracking
    As = copy(As0)

    # storage for gradient
    Ās = similar(As)
    Ās .= 0.0

    # storage for damped gradient
    dAs_dγ = similar(As)
    dAs_dγ .= 0.0

    # storage for regularisation
    if !isnothing(reg_params)
        Ãs = similar(As)
    end

    # invoke callback to report initial condition
    J1 = J(As)
    isnothing(callback) || callback(0, γ, J1, As, Ās)

    # gradient descent loop
    for iter in 1:maxiter
        ∇J!(Ās, As)

        # damped gradient
        dmp = 0.8
        @. dAs_dγ = dAs_dγ * dmp + (Ās * As)

        # invert in log-space to avoid negative values
        @. As = As0 * exp(log(As / As0) - γ * dAs_dγ)

        # regularise
        if !isnothing(reg_params)
            As, Ãs = regularise!(As, Ãs, γ, reg_params...)
        end

        # report intermediate results if needed
        J1 = J(As)
        isnothing(callback) || callback(iter, γ, J1, As, Ās)
    end

    return As
end
