function rhone_application()
    # load the data 

    # parameters and rescaling 

    # pack into name tuples 

    # run 3 inversions 
    # H steadystate inversion 
    inversion_steadystate(geometry, observed, initial, physics, numerics, optim_params; do_vis=false)

    # flux steadystate inversion

    inversion_steadystate(geometry, observed, initial, physics, numerics, optim_params; do_vis=false)
    # snapshot flux inversion 
    snapshot()

    #visualization for As_H_steady As_flux_steady As_snapshot


    # visualization 


    return 
end 



#load the data: H, B, xc, yc, qmag 
B_rhone, S_rhone, xc, yc = load_data("Rhone_data_padding/Archive/Rhone_BedElev_cr.tif", "Rhone_data_padding/Archive/Rhone_SurfElev_cr.tif")

# rescaling 

# power law exponent
npow = 3

# dimensionally independent physics 
lsc   = 1.0 # length scale [m]
tsc   = 1.0 # time scale   [s]

# non-dimensional numbers 
Aρgn     = ...# 1/tsc/lsc^n
s_f      = 0.026315789473684213 # sliding to ice flow ratio: s_f   = asρgn0/aρgn0/lx^2
b_max_nd = 4.706167536706325e-12 # m_max/lx*tsc m_max = 2.0 m/a lx = 1e3 tsc = 
β1tsc    = 2.353083768353162e-10 # ratio between characteristic time scales of ice flow and accumulation/ablation βtsc = β0*tsc 
β2tsc    = 3.5296256525297436e-10
γ_nd     = 1e2

# geometry 
lx_l, ly_l           = 25.0, 20.0  # horizontal length to characteristic length ratio
w1_l, w2_l           = 100.0, 10.0 # width to charactertistic length ratio 
B0_l                 = 0.35  # maximum bed rock elevation to characteristic length ratio
z_ela_l_1, z_ela_l_2 = 0.215, 0.09 # ela to domain length ratio z_ela_l = 
#numerics 
H_cut_l = 1.0e-6

# dimensionally dependent parameters 
lx, ly           = lx_l * lsc, ly_l * lsc  # 250000, 200000
w1, w2           = w1_l * lsc^2, w2_l * lsc^2 # 1e10, 1e9
z_ELA_0, z_ELA_1 = z_ela_l_1 * lsc, z_ela_l_2 * lsc # 2150, 900
B0               = B0_l * lsc # 3500
asρgn0_syn       = s_f_syn * aρgn0 * lsc^2 #5.7e-20*(ρg)^n = 5.7e-20*(910*9.81)^3 = 4.055141889402214e-8
asρgn0           = s_f * aρgn0 * lsc^2 #5.0e-18*(ρg)^n = 3.54627498316e-6
b_max            = b_max_nd * lsc / tsc  #2.0 m/a = 6.341958396752917e-8 m/s
β0, β1           = β1tsc / tsc, β2tsc / tsc  # 3.1709791983764586e-10, 4.756468797564688e-10
H_cut            = H_cut_l * lsc # 1.0e-2
γ0               = γ_nd * lsc^(2 - 2npow) * tsc^(-2) #1.0e-2










