function SIA_fields(nx, ny)
    return (B       = CUDA.zeros(Float64, nx, ny),
            H       = CUDA.zeros(Float64, nx, ny),
            H_old   = CUDA.zeros(Float64, nx, ny),
            V       = CUDA.zeros(Float64, nx - 1, ny - 1),
            D       = CUDA.zeros(Float64, nx - 1, ny - 1),
            As      = CUDA.zeros(Float64, nx - 1, ny - 1),
            mb_mask = CUDA.zeros(Float64, nx - 2, ny - 2),
            r_H     = CUDA.zeros(Float64, nx - 2, ny - 2),
            d_H     = CUDA.zeros(Float64, nx - 2, ny - 2),
            dH_dτ   = CUDA.zeros(Float64, nx - 2, ny - 2),
            ELA     = CUDA.zeros(Float64, nx - 2, ny - 2))
end

function SIA_adjoint_fields(nx, ny)
    return (∂J_∂H = CUDA.zeros(Float64, nx, ny),
            H̄     = CUDA.zeros(Float64, nx, ny),
            D̄     = CUDA.zeros(Float64, nx - 1, ny - 1),
            V̄     = CUDA.zeros(Float64, nx - 1, ny - 1),
            ψ     = CUDA.zeros(Float64, nx - 2, ny - 2),
            r̄_H   = CUDA.zeros(Float64, nx - 2, ny - 2))
end
