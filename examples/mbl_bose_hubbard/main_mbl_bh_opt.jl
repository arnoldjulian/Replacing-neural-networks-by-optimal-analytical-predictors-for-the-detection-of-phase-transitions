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
L=8
# L=8
data_folder = "../../data/mbl_bose_hubbard/L="*string(L)*"/"

# set path to save folder
save_folder = "./results/L="*string(L)*"/"

# critical parameter values for many-body localization transition from literature
p_crit1 = 4.0
p_crit2 = 7.0

# define parameter ranges
p_min = 0.1
p_max = 20.0
n_points = 200
p_range=collect(LinRange(p_min, p_max, n_points))
dp = p_range[2]-p_range[1]
p_range_LBC = collect(p_min-dp/2:dp:p_max+dp/2)

# import data

# probabilities associated with each basis state (Fock basis) at p=p_1
prob = vcat(readdlm(data_folder*"probs_W="*string(1)*".txt")...)

# iterate over entire parameter range
probss = [prob]
for index in 2:201-1
	probb = vcat(readdlm(data_folder*"probs_W="*string(index)*".txt")...)
	push!(probss,probb)
end
samples = collect(1:length(probss[1]))
n_samples = length(samples)

# construct probability distribution from data
# distr(x,p) gives probability to draw the sample x at parameter value p
# samples correspond to enumeration of unique basis states with integers
function distribution(sample,p,p_range,probss)
	p_indx = Int(round((p-p_range[1])/(p_range[2]-p_range[1])))+1
	return probss[p_indx][Int(sample)]
end
distr = (x,p)->distribution(x,p,p_range,probss)

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
vline!([p_crit1,p_crit2],label=L"p_{c}",c="red")
xlabel!(L"$p$")
ylabel!(L"$\hat{y}^{\mathrm{opt}}_{\mathrm{SL}}$")
savefig(save_folder*"pred_opt_SL.png")

plt = plot(p_range[2:end-1],indicator_opt_SL,dpi=300,c="black",label=false)
vline!([p_crit1,p_crit2],label=L"p_{c}",c="red")
xlabel!(L"$p$")
ylabel!(L"$I^{\mathrm{opt}}_{\mathrm{SL}}$")
savefig(save_folder*"indicator_opt_SL.png")

plt = plot(p_range,pred_opt_PBM,dpi=300,c="black",label=false)
vline!([p_crit1,p_crit2],label=L"p_{c}",c="red")
xlabel!(L"$p$")
ylabel!(L"$\hat{y}^{\mathrm{opt}}_{\mathrm{PBM}}$")
savefig(save_folder*"pred_opt_PBM.png")

plt = plot(p_range[2:end-1],indicator_opt_PBM,dpi=300,c="black",label=false)
vline!([p_crit1,p_crit2],label=L"p_{c}",c="red")
xlabel!(L"$p$")
ylabel!(L"$I^{\mathrm{opt}}_{\mathrm{PBM}}$")
savefig(save_folder*"indicator_opt_PBM.png")

plt = plot(p_range_LBC,indicator_opt_LBC,dpi=300,c="black",label=false)
vline!([p_crit1,p_crit2],label=L"p_{c}",c="red")
xlabel!(L"$p$")
ylabel!(L"$I^{\mathrm{opt}}_{\mathrm{LBC}}$")
savefig(save_folder*"indicator_opt_LBC.png")

plt = plot(p_range_LBC,loss_opt_LBC,dpi=300,c="black",label=false)
vline!([p_crit1,p_crit2],label=L"p_{c}",c="red")
xlabel!(L"$p$")
ylabel!(L"$\mathcal{L}^{\mathrm{opt}}_{\mathrm{LBC}}$")
savefig(save_folder*"loss_opt_LBC.png")