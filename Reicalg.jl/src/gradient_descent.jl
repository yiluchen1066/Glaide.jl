function gradient_descent(J::JF, ∇J!::GJF, X0, γ, niter; reg_params=nothing, callback=nothing) where {JF,GJF}
    # need an extra array for backtracking
    X = copy(X0)

    # storage for gradient
    X̄ = similar(X)

    # storage for search direction
    P = similar(X)

    # initialise the search direction with the direction of the steepest descent
    ∇J!(X̄, X)
    @. X̄ *= X

    @. P = -X̄

    # storage for regularisation
    if !isnothing(reg_params)
        X̃ = similar(X)
    end

    # invoke callback to report initial condition
    J1 = J(X)

    isnothing(callback) || callback(0, γ, J1, X, X̄)

    # gradient descent loop
    for iter in 1:niter
        ∇J!(X̄, X)
        @. X̄ *= X

        # gradient with momentum
        dmp = 0.5
        @. P = P * dmp - X̄

        # invert in log-space to avoid negative values
        @. X = exp(log(X) + γ * P)

        # regularise
        if !isnothing(reg_params)
            X, X̃ = regularise!(X, X̃, γ, reg_params...)
        end

        # report intermediate results if needed
        J1 = J(X)
        isnothing(callback) || callback(iter, γ, J1, X, X̄)
    end

    return X
end
