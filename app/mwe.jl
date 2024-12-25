using Enzyme, ForwardDiff, DiffResults, StaticArrays

d_dx(S::StaticMatrix{3,<:Any}) = S[SVector(2, 3), :] .- S[SVector(1, 2), :]

F(x) = sum(d_dx(x))

function foo(A, B)
    Bl   = @SMatrix [B[1] for _ in 1:3, _ in 1:3]
    dr   = ForwardDiff.gradient!(DiffResults.GradientResult(Bl), F, Bl)
    A[1] = DiffResults.value(dr) / sum(DiffResults.gradient(dr))
    # A[1] = F(Bl) / sum(ForwardDiff.gradient(F, Bl)) # this works
    return
end

A  = zeros(1)
B  = zeros(1)
dA = zeros(1)
dB = zeros(1)

foo(A, B)
Enzyme.autodiff(Enzyme.Reverse, Const(foo), Const, Duplicated(A, dA), Duplicated(B, dB)) # error
