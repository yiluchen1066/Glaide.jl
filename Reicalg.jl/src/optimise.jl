Base.@kwdef struct BacktrackingLineSearch
    τ::Float64 = 0.5
    c::Float64 = 0.5
    α_min::Float64
    α_max::Float64
end

struct OptimisationOptions{LS,CB}
    line_search::LS
    callback::CB
    maxiter::Int
    f_tol::Float64
    x_tol::Float64
end

struct OptmisationState{T,I,A}
    iter::I
    α::T
    J::T
    X::A
    X̄::A
end

function optimise(model, objective, X0, options)
    # unpack options
    (; line_search, callback, maxiter, f_tol, x_tol) = options

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

    # store the norm of the gradient
    rnorm0 = dot(X̄, X̄)

    # initial step size
    α = line_search.α_min

    # report initial results if needed
    isnothing(callback) || callback(0, α, J0, X, X̄)

    # gradient descent loop
    for iter in 1:maxiter
        # find a suitable step size
        α = find_step_line_search(model, objective, line_search, α, X, X0, X̄, P, J0)

        # update the solution
        @. X = X0 + α * P

        # save the current solution and gradient
        copy!(X0, X)
        copy!(X̄p, X̄)

        # compute gradient
        ∇J!(X̄, X, objective, model)

        # modified Polak-Ribière-Polyak (PRP+) rule
        rnorm  = dot(X̄, X̄)
        β      = max((rnorm - dot(X̄, X̄p)) / rnorm0, 0.0)
        rnorm0 = rnorm

        # update search direction
        @. P = P * β - X̄

        # report intermediate results if needed
        J0 = J(X, objective, model)
        isnothing(callback) || callback(iter, α, J0, X, X̄)
    end

    return X
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
        error("line search failed to find a suitable step size.")
    end

    return α
end
