# make your scripts automatically re-activate your project
cd(@__DIR__)
using Pkg; Pkg.activate("../..")

# load packages
using my_ml_for_pt
using DelimitedFiles
using LaTeXStrings
using Plots
using Random
ENV["GKSwstype"]="nul"

# set linear system size and corresponding data folder
L=60
# L=60
data_folder = "../../data/ising/L="*string(L)*"/"

# set path to save folder
save_folder = "./results/L="*string(L)*"/"

# critical parameter value for the Ising model
p_crit = Float32(2/log(1+sqrt(2)))

# define parameter ranges
p_min=0.05f0
p_max = 10.0f0
dp=0.05f0
p_range = collect(p_min:dp:p_max)
p_range_LBC = collect(p_min-dp/2:dp:p_max+dp/2)

# import Monte Carlo data

# unique energies at p=p_1
energy = readdlm(data_folder*"E_T="*string(0)*".txt",eltype(dp))

# number of times each energy is drawn at T=T_1
num = readdlm(data_folder*"num_T="*string(0)*".txt",eltype(dp))

# iterate over entire parameter range
energies=[energy]
numbers=[num]
for index in 1:length(p_range)-1
    energyy = readdlm(data_folder*"E_T="*string(index)*".txt",eltype(dp))
    numm = readdlm(data_folder*"num_T="*string(index)*".txt",eltype(dp))
    push!(energies,energyy)
    push!(numbers,numm)
end
unique_energies = unique(vcat(energies...))
n_samples = sum(numbers[1])

# construct probability distribution from Monte Carlo data
# distr(x,p) gives probability to draw the sample x at parameter value p
# samples correspond to enumeration of unique energies with integers
distr,samples = MLP.distr_approx(energies,unique_energies,numbers,p_range)

# supervised learning using analytical expression
# returns optimal predictions, indicator, and loss
pred_opt_SL, indicator_opt_SL, loss_opt_SL = MLP.get_indicators_SL_analytical(samples, distr, p_range, dp, p_min, p_max)

# prediction-based method using analytical expression
# returns optimal predictions, indicator, and loss
pred_opt_PBM, indicator_opt_PBM, loss_opt_PBM = MLP.get_indicators_PBM_analytical(samples, distr, p_range, dp)

# learning by confusion using analytical expression
# returns optimal indicator and loss
indicator_opt_LBC, loss_opt_LBC = MLP.get_indicators_LBC_analytical(samples, distr, p_range, p_range_LBC)

# save output in save_folder
open(save_folder*"p_range_SL.txt", "w") do io
    writedlm(io, p_range)
end

open(save_folder*"p_range_PBM.txt", "w") do io
    writedlm(io, p_range)
end

open(save_folder*"p_range_LBC.txt", "w") do io
    writedlm(io, p_range_LBC)
end

open(save_folder*"pred_opt_SL.txt", "w") do io
    writedlm(io, pred_opt_SL)
end

open(save_folder*"indicator_opt_SL.txt", "w") do io
    writedlm(io, indicator_opt_SL)
end

open(save_folder*"loss_opt_SL.txt", "w") do io
    writedlm(io, [loss_opt_SL])
end

open(save_folder*"pred_opt_PBM.txt", "w") do io
    writedlm(io, pred_opt_PBM)
end

open(save_folder*"indicator_opt_PBM.txt", "w") do io
    writedlm(io, indicator_opt_PBM)
end

open(save_folder*"loss_opt_PBM.txt", "w") do io
    writedlm(io, [loss_opt_PBM])
end

open(save_folder*"indicator_opt_LBC.txt", "w") do io
    writedlm(io, indicator_opt_LBC)
end

open(save_folder*"loss_opt_LBC.txt", "w") do io
    writedlm(io, loss_opt_LBC)
end

# plot and save results in save_folder
plt = plot(p_range,pred_opt_SL,dpi=300,c="black",label=false)
vline!([p_crit],label=L"p_{c}",c="red")
xlabel!(L"$p$")
ylabel!(L"$\hat{y}^{\mathrm{opt}}_{\mathrm{SL}}$")
savefig(save_folder*"pred_opt_SL.png")

plt = plot(p_range[2:end-1],indicator_opt_SL,dpi=300,c="black",label=false)
vline!([p_crit],label=L"p_{c}",c="red")
xlabel!(L"$p$")
ylabel!(L"$I^{\mathrm{opt}}_{\mathrm{SL}}$")
savefig(save_folder*"indicator_opt_SL.png")

plt = plot(p_range,pred_opt_PBM,dpi=300,c="black",label=false)
vline!([p_crit],label=L"p_{c}",c="red")
xlabel!(L"$p$")
ylabel!(L"$\hat{y}^{\mathrm{opt}}_{\mathrm{PBM}}$")
savefig(save_folder*"pred_opt_PBM.png")

plt = plot(p_range[2:end-1],indicator_opt_PBM,dpi=300,c="black",label=false)
vline!([p_crit],label=L"p_{c}",c="red")
xlabel!(L"$p$")
ylabel!(L"$I^{\mathrm{opt}}_{\mathrm{PBM}}$")
savefig(save_folder*"indicator_opt_PBM.png")

plt = plot(p_range_LBC,indicator_opt_LBC,dpi=300,c="black",label=false)
vline!([p_crit],label=L"p_{c}",c="red")
xlabel!(L"$p$")
ylabel!(L"$I^{\mathrm{opt}}_{\mathrm{LBC}}$")
savefig(save_folder*"indicator_opt_LBC.png")

plt = plot(p_range_LBC,loss_opt_LBC,dpi=300,c="black",label=false)
vline!([p_crit],label=L"p_{c}",c="red")
xlabel!(L"$p$")
ylabel!(L"$\mathcal{L}^{\mathrm{opt}}_{\mathrm{LBC}}$")
savefig(save_folder*"loss_opt_LBC.png")
