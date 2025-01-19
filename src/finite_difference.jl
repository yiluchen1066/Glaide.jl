@inline δˣₐ(S::StaticMatrix{3,<:Any}) = S[SA[2, 3], :] .- S[SA[1, 2], :]
@inline δʸₐ(S::StaticMatrix{<:Any,3}) = S[:, SA[2, 3]] .- S[:, SA[1, 2]]

@inline δˣ(S::StaticMatrix{3,3}) = S[SA[2, 3], SA[2]] .- S[SA[1, 2], SA[2]]
@inline δʸ(S::StaticMatrix{3,3}) = S[SA[2], SA[2, 3]] .- S[SA[2], SA[1, 2]]

@inline δˣ(S::StaticMatrix{2,1}) = S[2] - S[1]
@inline δʸ(S::StaticMatrix{1,2}) = S[2] - S[1]

@inline av4(S::StaticMatrix{2,3}) = 0.25 .* (S[SA[1], SA[1, 2]] .+ S[SA[2], SA[1, 2]] .+
                                             S[SA[1], SA[2, 3]] .+ S[SA[2], SA[2, 3]])

@inline av4(S::StaticMatrix{3,2}) = 0.25 .* (S[SA[1, 2], SA[1]] .+ S[SA[1, 2], SA[2]] .+
                                             S[SA[2, 3], SA[1]] .+ S[SA[2, 3], SA[2]])

@inline av4(S::StaticMatrix{3,3}) = 0.25 .* (S[SA[1, 2], SA[1, 2]] .+ S[SA[1, 2], SA[2, 3]] .+
                                             S[SA[2, 3], SA[1, 2]] .+ S[SA[2, 3], SA[2, 3]])

@inline innˣ(S::StaticMatrix{3,<:Any}) = S[SA[2], :]
@inline innʸ(S::StaticMatrix{<:Any,3}) = S[:, SA[2]]

@inline avˣ(S::StaticMatrix{3,3}) = 0.5 .* (S[SA[1, 2], SA[2]] .+ S[SA[2, 3], SA[2]])
@inline avʸ(S::StaticMatrix{3,3}) = 0.5 .* (S[SA[2], SA[1, 2]] .+ S[SA[2], SA[2, 3]])

@inline avˣ(S::StaticMatrix{3,2}) = 0.5 .* (S[SA[1, 2], :] .+ S[SA[2, 3], :])
@inline avʸ(S::StaticMatrix{2,3}) = 0.5 .* (S[:, SA[1, 2]] .+ S[:, SA[2, 3]])

@inline avˣ(S::StaticMatrix{2,2}) = 0.5 .* (S[SA[1], :] .+ S[SA[2], :])
@inline avʸ(S::StaticMatrix{2,2}) = 0.5 .* (S[:, SA[1]] .+ S[:, SA[2]])

@inline avˣ(S::StaticMatrix{2,1}) = 0.5 * (S[1] + S[2])
@inline avʸ(S::StaticMatrix{1,2}) = 0.5 * (S[1] + S[2])

@inline lˣ(S::StaticMatrix{3,3}) = S[SA[1, 2], SA[2]]
@inline rˣ(S::StaticMatrix{3,3}) = S[SA[2, 3], SA[2]]
@inline lʸ(S::StaticMatrix{3,3}) = S[SA[2], SA[1, 2]]
@inline rʸ(S::StaticMatrix{3,3}) = S[SA[2], SA[2, 3]]

Base.@propagate_inbounds function st3x3(M, ix, iy)
    nx, ny = oftype.((ix, iy), size(M))
    # neighbor indices
    di = oneunit(ix)
    dj = oneunit(iy)
    iW = max(ix - di, di)
    iE = min(ix + di, nx)
    iS = max(iy - dj, dj)
    iN = min(iy + dj, ny)
    return SMatrix{3,3}(M[iW, iS], M[ix, iS], M[iE, iS],
                        M[iW, iy], M[ix, iy], M[iE, iy],
                        M[iW, iN], M[ix, iN], M[iE, iN])
end
