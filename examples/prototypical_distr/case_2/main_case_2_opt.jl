# make your scripts automatically re-activate your project
cd(@__DIR__)
using Pkg; Pkg.activate("../../..")

# load packages
using ml_for_pt
using DelimitedFiles
using LaTeXStrings
using Plots
using Random
ENV["GKSwstype"]="nul"

# set path to save folder
save_folder = "./results/"

# set critical value of tuning parameter
p_crit = 1.0f0

# define parameter ranges
p_min = 0.1f0
p_max = 3.0f0
dp = 0.05f0
p_range = collect(p_min:dp:p_max)
p_min_indx = 1
p_max_indx = length(p_range)
p_range_LBC = collect(p_min-dp/2:dp:p_max+dp/2)

# construct probability distribution
distr,samples = MLP.jump_distr(p_crit=p_crit)

# construct data matrix that contains probabilties for all unique samples at each parameter value
data = zeros(eltype(p_range[1]),(length(samples),length(p_range)))
for sample_indx in 1:length(samples)
  sample = samples[sample_indx]
  for p_indx in 1:length(p_range)
    p = p_range[p_indx]
    data[sample_indx,p_indx] = distr(sample,p)
  end
end

# supervised learning using analytical expression
# returns optimal predictions, indicator, and loss
pred_opt_SL, indicator_opt_SL, loss_opt_SL = MLP.get_indicators_SL_analytical(data, p_range, dp, p_min_indx, p_max_indx)

# prediction-based method using analytical expression
# returns optimal predictions, indicator, and loss
pred_opt_PBM, indicator_opt_PBM, loss_opt_PBM = MLP.get_indicators_PBM_analytical(data, p_range, dp)

# learning by confusion using analytical expression
# returns optimal indicator and loss
indicator_opt_LBC, loss_opt_LBC = MLP.get_indicators_LBC_analytical(data, p_range)

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
plt = plot(p_range, pred_opt_SL, dpi=300, c="black", label=false)
vline!([p_crit], label=L"p_{c}", c="red")
xlabel!(L"$p$")
ylabel!(L"$\hat{y}^{\mathrm{opt}}_{\mathrm{SL}}$")
savefig(save_folder*"pred_opt_SL.png")

plt = plot(p_range[2:end-1], indicator_opt_SL, dpi=300, c="black", label=false)
vline!([p_crit], label=L"p_{c}", c="red")
xlabel!(L"$p$")
ylabel!(L"$I^{\mathrm{opt}}_{\mathrm{SL}}$")
savefig(save_folder*"indicator_opt_SL.png")

plt = plot(p_range, pred_opt_PBM, dpi=300, c="black", label=false)
vline!([p_crit], label=L"p_{c}", c="red")
xlabel!(L"$p$")
ylabel!(L"$\hat{y}^{\mathrm{opt}}_{\mathrm{PBM}}$")
savefig(save_folder*"pred_opt_PBM.png")

plt = plot(p_range[2:end-1], indicator_opt_PBM, dpi=300, c="black", label=false)
vline!([p_crit], label=L"p_{c}", c="red")
xlabel!(L"$p$")
ylabel!(L"$I^{\mathrm{opt}}_{\mathrm{PBM}}$")
savefig(save_folder*"indicator_opt_PBM.png")

plt = plot(p_range_LBC, indicator_opt_LBC, dpi=300, c="black", label=false)
vline!([p_crit], label=L"p_{c}", c="red")
xlabel!(L"$p$")
ylabel!(L"$I^{\mathrm{opt}}_{\mathrm{LBC}}$")
savefig(save_folder*"indicator_opt_LBC.png")

plt = plot(p_range_LBC, loss_opt_LBC, dpi=300, c="black", label=false)
vline!([p_crit], label=L"p_{c}", c="red")
xlabel!(L"$p$")
ylabel!(L"$\mathcal{L}^{\mathrm{opt}}_{\mathrm{LBC}}$")
savefig(save_folder*"loss_opt_LBC.png")
