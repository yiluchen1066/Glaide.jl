Base.@propagate_inbounds δˣₐ(S::StaticMatrix{3,<:Any}) = S[SVector(2, 3), :] .- S[SVector(1, 2), :]
Base.@propagate_inbounds δʸₐ(S::StaticMatrix{<:Any,3}) = S[:, SVector(2, 3)] .- S[:, SVector(1, 2)]

Base.@propagate_inbounds δˣ(S::StaticMatrix{3,3}) = S[SVector(2, 3), SVector(2)] .- S[SVector(1, 2), SVector(2)]
Base.@propagate_inbounds δʸ(S::StaticMatrix{3,3}) = S[SVector(2), SVector(2, 3)] .- S[SVector(2), SVector(1, 2)]

Base.@propagate_inbounds δˣ(S::StaticMatrix{2,1}) = S[2] - S[1]
Base.@propagate_inbounds δʸ(S::StaticMatrix{1,2}) = S[2] - S[1]

Base.@propagate_inbounds av4(S::StaticMatrix{2,3}) = 0.25 .* (S[SVector(1), SVector(1, 2)] .+ S[SVector(2), SVector(1, 2)] .+
                                                              S[SVector(1), SVector(2, 3)] .+ S[SVector(2), SVector(2, 3)])

Base.@propagate_inbounds av4(S::StaticMatrix{3,2}) = 0.25 .* (S[SVector(1, 2), SVector(1)] .+ S[SVector(1, 2), SVector(2)] .+
                                                              S[SVector(2, 3), SVector(1)] .+ S[SVector(2, 3), SVector(2)])

Base.@propagate_inbounds innˣ(S::StaticMatrix{3,<:Any}) = S[SVector(2), :]
Base.@propagate_inbounds innʸ(S::StaticMatrix{<:Any,3}) = S[:, SVector(2)]

Base.@propagate_inbounds avˣ(S::StaticMatrix{3,3}) = 0.5 .* (S[SVector(1, 2), SVector(2)] .+ S[SVector(2, 3), SVector(2)])
Base.@propagate_inbounds avʸ(S::StaticMatrix{3,3}) = 0.5 .* (S[SVector(2), SVector(1, 2)] .+ S[SVector(2), SVector(2, 3)])

Base.@propagate_inbounds avˣ(S::StaticMatrix{2,1}) = 0.5 .* (S[1] + S[2])
Base.@propagate_inbounds avʸ(S::StaticMatrix{1,2}) = 0.5 .* (S[1] + S[2])

Base.@propagate_inbounds lˣ(S::StaticMatrix{3,3}) = S[SVector(1, 2), SVector(2)]
Base.@propagate_inbounds rˣ(S::StaticMatrix{3,3}) = S[SVector(2, 3), SVector(2)]
Base.@propagate_inbounds lʸ(S::StaticMatrix{3,3}) = S[SVector(2), SVector(1, 2)]
Base.@propagate_inbounds rʸ(S::StaticMatrix{3,3}) = S[SVector(2), SVector(2, 3)]

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
