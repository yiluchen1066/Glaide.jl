function gradient_descent(model, objective, X0, γ, niter; momentum=0.0, regularisation=nothing, callback=nothing, kwargs...)
    X = X0

    # storage for gradient and search direction
    X̄ = similar(X)
    P = similar(X)

    J1 = J(X, objective, model; kwargs...)
    fill!(X̄, 0.0)
    isnothing(callback) || callback(0, γ, J1, X, X̄)

    # gradient descent loop
    for iter in 1:niter
        # compute gradient
        ∇J!(X̄, X, objective, model; kwargs...)

        # convert gradient to log-space
        @. X̄ *= X

        # initialise the search direction with the direction of the steepest descent
        if iter == 1
            @. P = -X̄
        end

        # gradient with momentum
        @. P = P * momentum - X̄

        # invert in log-space to avoid negative values
        @. X = exp(log(X) + γ * P)

        # TODO: refactor this
        @. X = clamp(X, 1e-22, 1e-17)

        # regularise
        regularise!(X, γ, regularisation)

        # report intermediate results if needed
        J1 = J(X, objective, model)
        isnothing(callback) || callback(iter, γ, J1, X, X̄)
    end

    return X
end
