Base.@kwdef struct BacktrackingLineSearch
    τ::Float64 = 0.5
    c::Float64 = 0.5
    α_min::Float64
    α_max::Float64
end

Base.@kwdef struct OptimisationOptions{LS,CB}
    line_search::LS
    callback::CB   = nothing
    maxiter::Int   = 1000
    j_tol::Float64 = 0.0
    x_tol::Float64 = 0.0
end

struct OptmisationState{T,I,A}
    iter::I
    α::T
    j_value::T
    j_change::T
    x_change::T
    X::A
    X̄::A
end

struct OptimisationResult{T,I,A}
    iter::I
    j_value::T
    X::A
end

"""
    optimise(model, objective, X0, options)
"""
function optimise(model, objective, X0, options)
    # unpack options
    (; line_search, callback, maxiter, j_tol, x_tol) = options

    # copy the initial solution, needed for the line search
    X = copy(X0)

    # gradient (adjoint)
    X̄ = similar(X)

    # search direction
    P = similar(X)

    # compute initial objective value
    J0 = J(X, objective, model)

    # compute initial gradient
    ∇J!(X̄, X, objective, model)

    # store previous gradient to compute β
    X̄p = similar(X̄)

    # initialise the search direction with the direction of the steepest descent
    @. P = -X̄

    # initial step size
    α = line_search.α_min

    # convergence flags
    x_converged = false
    j_converged = false

    # report initial results if needed
    if !isnothing(callback)
        callback(OptmisationState(0, α, J0, NaN, NaN, X, X̄))
    end

    # gradient descent loop
    iter = 1
    while iter <= maxiter
        # find a suitable step size
        α = find_step_line_search(model, objective, line_search, α, X, X0, X̄, P, J0)

        # update the solution
        @. X = X0 + α * P

        # save the previous gradient
        copy!(X̄p, X̄)

        # compute gradient
        ∇J!(X̄, X, objective, model)

        # Hager-Zhang rule
        Y    = X̄ .- X̄p
        dkyk = dot(P, Y)
        ηk   = -inv(sqrt(dot(P, P)) * min(0.01, sqrt(dot(X̄, X̄))))
        β    = max(dot(Y .- (2.0 * dot(Y, Y) / dkyk) .* P, X̄) / dkyk, ηk)

        # update search direction
        @. P = P * β - X̄

        # report intermediate results if needed
        J1 = J(X, objective, model)

        j_change = abs(J1 - J0) / J1
        x_change = mapreduce(abs, max, X̄)

        j_converged = j_change < j_tol
        x_converged = x_change < x_tol

        if j_converged || x_converged
            break
        end

        # save the previous solution and objective value
        copy!(X0, X)
        J0 = J1

        # report intermediate results
        if !isnothing(callback)
            callback(OptmisationState(iter, α, J0, j_change, x_change, X, X̄))
        end

        iter += 1
    end

    return OptimisationResult(iter, J0, X)
end

"""
    find_step_line_search(model, objective, line_search, α0, X, X0, X̄, P, J0)

Perform a two-way backtracking line search to find a suitable step size for the optimisation algorithm.
"""
function find_step_line_search(model, objective, line_search, α0, X, X0, X̄, P, J0)
    (; c, τ, α_min, α_max) = line_search

    # if the dot product of the gradient and the search direction is non-negative,
    # the search direction is not a descent direction and should be restarted
    m = dot(X̄, P)

    if m >= 0
        @warn "non-descent direction detected, restarting with the direction of steepest descent"
        @. P = -X̄
        m = dot(X̄, P)
    end

    # parameter for the Armijo-Goldstein condition
    t = -c * m
    α = α0

    # update the solution with the initial step size
    @. X = X0 + α * P

    # flag to indicate if the step size is accepted
    accepted = false

    # check the Armijo-Goldstein condition
    if J0 - J(X, objective, model) >= α * t
        # increase the step size repeatedly
        while α < α_max
            α_new = min(α / τ, α_max)

            @. X = X0 + α_new * P

            # check the Armijo-Goldstein condition
            if !(J0 - J(X, objective, model) >= α_new * t)
                break
            else
                α = α_new
            end
        end
        # always accept the step size since the Armijo-Goldstein condition is satisfied
        accepted = true
    else
        # decrease the step size repeatedly
        while α > α_min
            α = max(α * τ, α_min)
            @. X = X0 + α * P

            if J0 - J(X, objective, model) >= α * t
                accepted = true
                break
            end
        end
    end

    if !accepted
        @warn "line search failed to find a suitable step size"
    end

    return α
end
