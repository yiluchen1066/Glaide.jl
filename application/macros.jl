#! format: off
# get CUDA indices
macro get_indices()
    esc(:( begin
        ix = (blockIdx().x-1) * blockDim().x + threadIdx().x;
        iy = (blockIdx().y-1) * blockDim().y + threadIdx().y;
    end ))
end 
#CUDA specific 

# averaging
macro av_xy(A) esc(:( 0.25*($A[ix, iy    ] + $A[ix + 1, iy    ] +
                            $A[ix, iy + 1] + $A[ix + 1, iy + 1]) )) end
macro av_xa(A) esc(:( 0.5*($A[ix, iy] + $A[ix + 1, iy    ]) )) end
macro av_ya(A) esc(:( 0.5*($A[ix, iy] + $A[ix    , iy + 1]) )) end 
# derivatives
macro d_xa(A)  esc(:( $A[ix + 1, iy    ] - $A[ix    , iy    ] )) end 
macro d_ya(A)  esc(:( $A[ix    , iy + 1] - $A[ix    , iy    ] )) end
macro d_xi(A)  esc(:( $A[ix + 1, iy + 1] - $A[ix    , iy + 1] )) end 
macro d_yi(A)  esc(:( $A[ix + 1, iy + 1] - $A[ix + 1, iy    ] )) end
#! format: on

@views avx(A) = 0.5 .* (A[1:end-1, :] + A[2:end, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1] + A[:, 2:end])