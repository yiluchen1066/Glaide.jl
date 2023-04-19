using JLD2

(B, H_obs, as_syn, as_ini_vis, xc, yc, nx, ny,gd_niter) = load("synthetic_data/synthetic_static.jld2", "B", "H_obs", "as_syn", "as_ini_vis", "xc", "yc","nx", "ny", "gd_niter")

anim = @animate for gd_iter = 1:gd_niter 
    (H, as) = load("synthetic_data/synthetic_$gd_iter.jld2", "H", "as")
    p1=heatmap(xc, yc, Array(H'); xlabel ="X", ylabel="Y", title ="Ice thickness", xlims=extrema(xc), ylims=extrema(yc),levels=20, color =:turbo, aspect_ratio = 1,cbar=true)
    vline!(xc[nx ÷2])
    p2=plot(Array(H[nx÷2,:]),yc;xlabel="H",ylabel="Y", label="Current H (cross section)", legend=:bottom)
    plot!(Array(H_obs[nx÷2,:]),yc; xlabel="H", ylabel="Y", title="Ice thickness", label="Synthetic H (cross section)",  legend=:bottom)
             
    p3=heatmap(xc[1:end-1], yc[1:end-1], Array(log10.(as)'); xlabel="X", ylabel="Y", xlims=extrema(xc), ylims=extrema(yc),label="as", title="Sliding coefficient as", aspect_ratio=1)
    p4=plot(Array(log10.(as[nx÷2,:])),yc[1:end-1]; xlabel="as", ylabel="Y", title="Sliding coefficient as",color=:blue, lw = 3, label="Current as (cross section)", legend=true)
    plot!(Array(log10.(as_ini_vis[nx÷2,:])),yc[1:end-1]; xlabel="as", ylabel="Y", color=:green, lw=3, label="Initial as for inversion", legend=true)
    plot!(Array(log10.(as_syn[nx÷2,:])),yc[1:end-1];xlabel="as", ylabel="Y", color=:red, lw= 3, label="Synthetic as", legend=true)
    #p5 = plot(iter_evo, J_evo; shape = :circle, xlabel="Iterations", ylabel="J_old/J_ini", title="misfit", yaxis=:log10)
    display(plot(p1,p2,p3,p4; layout=(2,2)))
    #display(plot(p5;  size=(490,490)))
             
end 

gif(anim, "adjoint_bench_2D.gif"; fps=5)
