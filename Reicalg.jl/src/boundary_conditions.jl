#! format: off
function _bc_x!(H, B)
    @get_index_1d
    @inbounds if i <= size(H, 2)
        H[1  , i] = H[2    , i] + (B[2    , i] - B[1  , i])
        H[end, i] = H[end-1, i] + (B[end-1, i] - B[end, i])
    end
    return
end

function _bc_y!(H, B)
    @get_index_1d
    @inbounds if i <= size(H, 1)
        H[i, 1  ] = H[i, 2    ] + (B[i, 2    ] - B[i, 1  ])
        H[i, end] = H[i, end-1] + (B[i, end-1] - B[i, end])
    end
    return
end

# homogeneous Nemann

function _bc_x!(H)
    @get_index_1d
    @inbounds if i <= size(H, 2)
        H[1  , i] = H[2    , i]
        H[end, i] = H[end-1, i]
    end
    return
end

function _bc_y!(H)
    @get_index_1d
    @inbounds if i <= size(H, 1)
        H[i, 1  ] = H[i, 2    ]
        H[i, end] = H[i, end-1]
    end
    return
end
#! format: on

function bc!(H, B)
    # bc in x
    nthreads, nblocks = launch_config(size(H, 2))
    @cuda threads = nthreads blocks = nblocks _bc_x!(H, B)

    # bc in y
    nthreads, nblocks = launch_config(size(H, 1))
    @cuda threads = nthreads blocks = nblocks _bc_y!(H, B)

    return
end

function bc!(H)
    # bc in x
    nthreads, nblocks = launch_config(size(H, 2))
    @cuda threads = nthreads blocks = nblocks _bc_x!(H)

    # bc in y
    nthreads, nblocks = launch_config(size(H, 1))
    @cuda threads = nthreads blocks = nblocks _bc_y!(H)

    return
end
