lerp(a, b, t) = b * t + a * (oneunit(t) - t)

function download_raster(url, zip_entry, path)
    if ispath(path)
        @info "path '$path' already exists, skipping download..."
        return
    end

    # download data into an in-memory buffer
    data = take!(Downloads.download(url, IOBuffer()))

    # interpret binary blob as ZIP archive (which it is)
    archive = ZipReader(data)

    dir = dirname(path)
    if !isdir(dir)
        mkpath(dir)
    end

    # save the file from the zip archive to disk
    open(path, "w") do io
        write(io, zip_readentry(archive, zip_entry))
    end

    return
end

# copied from GlacioTools.jl, workaround to avoid segfaults in Rasters.jl
function coords_as_ranges(raster_like; sigdigits=0)
    x, y = dims(raster_like)

    if sigdigits == 0
        x = X(LinRange(x[1], x[end], length(x)))
        y = Y(LinRange(y[1], y[end], length(y)))
    else
        x = X(LinRange(round(x[1]; sigdigits), round(x[end]; sigdigits), length(x)))
        y = Y(LinRange(round(y[1]; sigdigits), round(y[end]; sigdigits), length(y)))
    end

    dx = x[2] - x[1]
    dy = y[2] - y[1]

    @assert abs(dx) == abs(dy) "abs(dx) and abs(dy) not equal ($dx, $dy)"
    return x, y
end

@views av1(a) = @. 0.5 * (a[1:end-1] + a[2:end])

# linearly interpolate velocity to the grid nodes
@views av4(A) = @. 0.25 * (A[1:end-1, 1:end-1] +
						   A[2:end  , 1:end-1] +
						   A[2:end  , 2:end  ] +
						   A[1:end-1, 2:end  ])

# least squares fit of the data
lsq_fit(x, y) = (x' * x) \ (x' * y)

function remove_components!(A; threshold=0.0, min_length=1)
	L    = label_components(A .> threshold)
	Ll   = component_lengths(L)
	L_rm = findall(Ll .<= min_length)
	Li   = component_indices(L)[L_rm]
	for I in Li
		A[I] .= 0.0
	end
	return
end

function smooth_lap!(A)
	@. A[2:end-1, 2:end-1] += 1 / 8 * (A[1:end-2, 2:end-1] +
									   A[3:end  , 2:end-1] +
									   A[2:end-1, 1:end-2] +
									   A[2:end-1, 3:end  ] -
									   4 * A[2:end-1, 2:end-1])
	@. A[1  , :] .= A[2    , :]
	@. A[end, :] .= A[end-1, :]
	@. A[:  , 1] .= A[:    , 2]
	@. A[:, end] .= A[:, end-1]
	return
end
