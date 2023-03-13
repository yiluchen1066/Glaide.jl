using Enzyme

nx,ny = 500, 500 
macro get_thread_idx(A) esc(:( begin ix =(blockIdx().x-1) * blockDim().x + threadIdx().x; iy = (blockIdx().y-1) * blockDim().y+threadIdx().y;end )) end 

#fill in the Matrix H 

function compute_D!(D, H, nx, ny)
    @get_thread_idx(H)
    if ix <= nx && iy <= ny 
        D[ix,iy] = H[ix,iy]^2 
    end 
    return 
end 

function compute_q!(qHx, qHy, D, H, nx, ny) 
    @get_thread_idx(H) 
    if ix <= nx && iy <= ny 
        qHx[ix,iy] = D[ix,iy]^2 + H[ix,iy] 
        qHy[ix,iy] = D[ix,iy]^2 - H[ix,iy] 
    end 
    return 
end 

function residual!(RH, qHx, qHy, H, nx, ny)
    @get_thread_idx(RH)
    if ix <= nx && iy <= ny 
        RH[ix, iy] = qHx[ix,iy] + qHy[ix,iy] + sin(H[ix,iy])
    end 
    return 
end 

# compute dR_dH 
#residual!(RH, qHx, qHy, H, nx, ny)
function grad_residual_H_1!() 
    Enzyme.autodiff_deferred(residual!, Duplicated(tmp1, tmp2), Duplicated(qHx, dR_qHx), Duplicated(qHy, dR_qHy), Duplicated(H, dR_H), Const(nx),  Const(ny)) 
    return 
end 

function grad_residual_H_2!()
    Enzyme.autodiff_deferred(compute_q!, Duplicated(qHx, dR_qHx), Duplicated(qHy, dR_qHy), )
    return 
end



