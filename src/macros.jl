#! format: off
# get CUDA indices
macro get_indices()
    esc(:( begin
        ix = (blockIdx().x-Int32(1)) * blockDim().x + threadIdx().x;
        iy = (blockIdx().y-Int32(1)) * blockDim().y + threadIdx().y;
    end ))
end

macro get_index_1d() esc(:( i = (blockIdx().x-Int32(1)) * blockDim().x + threadIdx().x; )) end
#! format: on
