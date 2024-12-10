using Enzyme, StaticArrays, CUDA

d_dx(S::StaticMatrix{3,<:Any}) = S[SVector(2, 3), :] - S[SVector(1, 2), :]

F(x) = sum(d_dx(x))

# Working CPU version
function foo_cpu(A, B)
    Bl   = @SMatrix [B[1] for _ in 1:3, _ in 1:3]
    r̄, r = Enzyme.autodiff(Enzyme.ReverseWithPrimal, Const(F), Active, Active(Bl))
    A[1] = r / sum(r̄[1])
    return
end

A  = zeros(1)
B  = zeros(1)
dA = zeros(1)
dB = zeros(1)

foo_cpu(A, B)
Enzyme.autodiff(Enzyme.Reverse, Const(foo_cpu), Const, Duplicated(A, dA), Duplicated(B, dB)) # error

# Broken GPU version
function foo_gpu(A, B)
    Bl   = @SMatrix [B[1] for _ in 1:3, _ in 1:3]
    r̄, r = Enzyme.autodiff_deferred(Enzyme.ReverseWithPrimal, Const(F), Active, Active(Bl))
    A[1] = r / sum(r̄[1])
    return
end

function dfoo_gpu(A, B)
    Enzyme.autodiff_deferred(Enzyme.Reverse, Const(foo_gpu), Const, A, B)
    return
end

A  = CUDA.zeros(Float64, 1)
B  = CUDA.zeros(Float64, 1)
dA = CUDA.zeros(Float64, 1)
dB = CUDA.zeros(Float64, 1)

@cuda foo_gpu(A, B)
@cuda dfoo_gpu(Duplicated(A, dA), Duplicated(B, dB)) # crash
