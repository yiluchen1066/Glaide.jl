using Enzyme

nx,ny = 500, 500 

H = zeros(Float64, (nx,ny)) 

#fill in the Matrix H 

function compute_D!()
    @get_thread_idx()
    if ix <= nx && iy <= ny 
        D[ix,iy] = H[ix,iy]^2 
    end 
    return 
end 

function compute_q!() 
    @get_thread_idx() 
    if ix <= nx && iy <= ny 
        qHx[ix,iy] = D[ix,iy]^2 + H[ix,iy] 
        qHy[ix,iy] = D[ix,iy]^2 - H[ix,iy] 
    end 
    return 
end 

function residual!()
    @get_thread_idx()
    if ix <= nx && iy <= ny 
        RH[ix, iy] = qHx[ix,iy] * qHy[ix,iy] + sin(H[ix,iy])
    end 
    return 
end 

# compute dR_dH 

