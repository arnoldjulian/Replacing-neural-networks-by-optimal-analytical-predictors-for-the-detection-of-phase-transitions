# make your scripts automatically re-activate your project
cd(@__DIR__)
using Pkg; Pkg.activate("../../..")

# load packages
using ml_for_pt
using DelimitedFiles
using LaTeXStrings
using Plots
using Random
using Flux
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
p_range_LBC = collect(p_min-dp/2:dp:p_max+dp/2)

# construct probability distribution
distr,samples = MLP.jump_distr(p_crit=p_crit)

# optionally run methods using the analytical expressions to compare with neural network results

# supervised learning using analytical expression
# returns optimal predictions, indicator, and loss
pred_opt_SL, indicator_opt_SL, loss_opt_SL = MLP.get_indicators_SL_analytical(samples, distr, p_range, dp, p_min, p_max)

# prediction-based method using analytical expression
# returns optimal predictions, indicator, and loss
pred_opt_PBM, indicator_opt_PBM, loss_opt_PBM = MLP.get_indicators_PBM_analytical(samples, distr, p_range, dp)

# learning by confusion using analytical expression
# returns optimal indicator and loss
indicator_opt_LBC, loss_opt_LBC = MLP.get_indicators_LBC_analytical(samples, distr, p_range, p_range_LBC)


# construct dataset for training neural networks
dataset = MLP.get_dataset_proto(p_range, distr, samples)
dataset_train_SL = hcat(dataset[:, 1:length(samples)], dataset[:, end-length(samples)+1:end])

# standardize inputs
inputs_one_hot = reshape(samples, 1, length(samples))
mean_train, std_train = MLP.get_dataset_stats(dataset,inputs_one_hot, length(p_range))
inputs_one_hot_stand = (inputs_one_hot.-mean_train)./std_train
mean_train_SL, std_train_SL = MLP.get_dataset_stats(dataset_train_SL, inputs_one_hot, length(p_range))
inputs_one_hot_stand_SL = (inputs_one_hot.-mean_train_SL)./std_train_SL


# supervised learning using neural networks

# initialize neural network
n_nodes_SL = [size(inputs_one_hot)[1], 64, 2]
NN_SL = Chain(
  Dense(n_nodes_SL[1], n_nodes_SL[2], relu),
  Dense(n_nodes_SL[2], n_nodes_SL[end], identity))
pnn_SL, re_SL = Flux.destructure(NN_SL)

# set hyperparameters
lr_SL = 0.001f0
epochs_SL = 1000
saveat_SL = 100
opt_SL = ADAM(lr_SL)
verbose = true

# train neural network
# returns predictions, indicators, and loss saved at epochs specified by saveat_SL variable
pred_logger_SL, losses_SL, NN_logger_SL = MLP.get_indicators_SL_numerical(pnn_SL,  re_SL, dataset_train_SL, dataset, epochs_SL, p_range, dp, p_min, p_max, opt_SL, inputs_one_hot_stand, verbose=verbose, saveat=saveat_SL)

# extract results at last save point
pred_NN_SL = pred_logger_SL[end][1]
indicator_NN_SL = pred_logger_SL[end][2]
loss_NN_SL = pred_logger_SL[end][3]

# save results of supervised learning in save_folder
open(save_folder*"hyperparams_SL.txt", "w") do io
  writedlm(io, [saveat_SL, lr_SL, epochs_SL])
end

open(save_folder*"NN_logger_SL.txt", "w") do io
  writedlm(io, NN_logger_SL)
end

for i in 1:length(pred_logger_SL)
  open(save_folder*"pred_logger_SL_"*string(i)*".txt", "w") do io
    writedlm(io, pred_logger_SL[i][1])
  end

  open(save_folder*"indicator_logger_SL_"*string(i)*".txt", "w") do io
    writedlm(io, pred_logger_SL[i][2])
  end

  open(save_folder*"loss_logger_SL_"*string(i)*".txt", "w") do io
    writedlm(io, pred_logger_SL[i][3])
  end

end

open(save_folder*"pred_NN_SL.txt", "w") do io
  writedlm(io, pred_NN_SL)
end

open(save_folder*"indicator_NN_SL.txt", "w") do io
  writedlm(io, indicator_NN_SL)
end

open(save_folder*"loss_NN_SL.txt", "w") do io
  writedlm(io, loss_NN_SL)
end

open(save_folder*"NN_structure_SL.txt", "w") do io
  writedlm(io, n_nodes_SL)
end

open(save_folder*"losses_SL.txt", "w") do io
  writedlm(io, losses_SL)
end

# plot and save results of supervised learning in save_folder
plt = plot(p_range, pred_NN_SL, dpi=300, c="black", label="NN", ylims=(0.0, 1.0))
plot!(p_range, pred_opt_SL, dpi=300, c="blue", label="analytical", ylims=(0.0, 1.0))
vline!([p_crit], label=L"p_{c}", c="red")
xlabel!(L"$p$")
ylabel!(L"$\hat{y}_{\mathrm{SL}}$")
savefig(save_folder*"pred_NN_SL.png")

plt = plot(p_range[2:end-1], indicator_NN_SL, dpi=300, c="black", label="NN")
plot!(p_range[2:end-1], indicator_opt_SL, dpi=300, c="blue", label="analytical")
vline!([p_crit], label=L"p_{c}", c="red")
xlabel!(L"$p$")
ylabel!(L"$I_{\mathrm{SL}}$")
savefig(save_folder*"indicator_NN_SL.png")

plt = plot(collect(1:length(losses_SL)), losses_SL, dpi=300, c="black", label="NN")
hline!([loss_opt_SL], label="analytical", c="blue")
xlabel!("Epochs")
ylabel!(L"$\mathcal{L}_{\mathrm{SL}}$")
savefig(save_folder*"loss_NN_SL.png")


# prediction-based method using neural networks

# initialize neural network
n_nodes_PBM = [size(inputs_one_hot)[1], 64, 1]
NN_PBM = Chain(
  Dense(n_nodes_PBM[1], n_nodes_PBM[2], relu),
  Dense(n_nodes_PBM[2], n_nodes_PBM[end], identity))
pnn_PBM, re_PBM = Flux.destructure(NN_PBM)

# set hyperparameters
lr_PBM = 0.001f0
epochs_PBM = 1000
saveat_PBM = 100
opt_PBM = ADAM(lr_PBM)
verbose = true

# train neural network
# returns predictions, indicators, and loss saved at epochs specified by saveat_PBM variable
pred_logger_PBM, losses_PBM, NN_logger_PBM = MLP.get_indicators_PBM_numerical(pnn_PBM, re_PBM, dataset, epochs_PBM, p_range, dp, opt_PBM, inputs_one_hot_stand, verbose=verbose, saveat=saveat_PBM)

# extract results at last save point
pred_NN_PBM = pred_logger_PBM[end][1]
indicator_NN_PBM = pred_logger_PBM[end][2]
loss_NN_PBM = pred_logger_PBM[end][3]

# save results of prediction-based method in save_folder
open(save_folder*"hyperparams_PBM.txt", "w") do io
  writedlm(io, [saveat_PBM, lr_PBM, epochs_PBM])
end

open(save_folder*"NN_logger_PBM.txt", "w") do io
  writedlm(io, NN_logger_PBM)
end

for i in 1:length(pred_logger_PBM)
  open(save_folder*"pred_logger_PBM_"*string(i)*".txt", "w") do io
    writedlm(io, pred_logger_PBM[i][1])
  end

  open(save_folder*"indicator_logger_PBM_"*string(i)*".txt", "w") do io
    writedlm(io, pred_logger_PBM[i][2])
  end

  open(save_folder*"loss_logger_PBM_"*string(i)*".txt", "w") do io
    writedlm(io, pred_logger_PBM[i][3])
  end

end

open(save_folder*"pred_NN_PBM.txt", "w") do io
  writedlm(io, pred_NN_PBM)
end

open(save_folder*"indicator_NN_PBM.txt", "w") do io
  writedlm(io, indicator_NN_PBM)
end

open(save_folder*"loss_NN_PBM.txt", "w") do io
  writedlm(io, loss_NN_PBM)
end

open(save_folder*"NN_structure_PBM.txt", "w") do io
  writedlm(io, n_nodes_PBM)
end

open(save_folder*"losses_PBM.txt", "w") do io
  writedlm(io, losses_PBM)
end

# plot and save results of prediction-based method in save_folder
plt = plot(p_range, pred_NN_PBM, dpi=300, c="black", label="NN", ylims=(p_range[1], p_range[end]))
plot!(p_range, pred_opt_PBM, dpi=300, c="blue", label="analytical", ylims=(p_range[1],p_range[end]))
vline!([p_crit], label=L"p_{c}", c="red")
xlabel!(L"$p$")
ylabel!(L"$\hat{y}_{\mathrm{PBM}}$")
savefig(save_folder*"pred_NN_PBM.png")

plt = plot(p_range[2:end-1], indicator_NN_PBM, dpi=300, c="black", label="NN")
plot!(p_range[2:end-1], indicator_opt_PBM, dpi=300, c="blue", label="analytical")
vline!([p_crit], label=L"p_{c}", c="red")
xlabel!(L"$p$")
ylabel!(L"$I^{\mathrm{NN}}_{\mathrm{PBM}}$")
savefig(save_folder*"indicator_NN_PBM.png")

plt = plot(collect(1:length(losses_PBM)), losses_PBM, dpi=300, c="black", label="NN")
hline!([loss_opt_PBM], label="analytical", c="blue")
xlabel!("Epochs")
ylabel!(L"$\mathcal{L}_{\mathrm{PBM}}$")
savefig(save_folder*"loss_NN_PBM.png")


# learning by confusion using neural networks

# initialize neural network
n_nodes_LBC = [size(inputs_one_hot)[1], 64, 2]
NN_LBC = Chain(
  Dense(n_nodes_LBC[1], n_nodes_LBC[2], relu),
  Dense(n_nodes_LBC[2], n_nodes_LBC[end], identity))
pnn_LBC, re_LBC = Flux.destructure(NN_LBC)

# set hyperparameters
lr_LBC = 0.001f0
epochs_LBC = 1000
saveat_LBC = epochs_LBC
opt_LBC = ADAM(lr_LBC)
verbose = true

indicator_NN_LBC = []
loss_NN_LBC = []
for indx in collect(1:length(p_range_LBC))
  pred_logger_LBC, losses_LBC, NN_logger_LBC = MLP.get_indicators_LBC_numerical_fixed_p(deepcopy(pnn_LBC), re_LBC, dataset, epochs_LBC, p_range, dp, opt_LBC, p_range_LBC, indx, inputs_one_hot_stand, saveat=saveat_LBC, verbose=verbose)

  indicator_NN_LBC_p = pred_logger_LBC[end][1][1]
  push!(indicator_NN_LBC, indicator_NN_LBC_p)
  loss_NN_LBC_p = pred_logger_LBC[end][2][1]
  push!(loss_NN_LBC, loss_NN_LBC_p)
end

# save results of learning by confusion in save_folder
open(save_folder*"hyperparams_LBC.txt", "w") do io
  writedlm(io, [saveat_LBC, lr_LBC, epochs_LBC])
end

open(save_folder*"indicator_NN_LBC.txt", "w") do io
  writedlm(io, indicator_NN_LBC)
end

open(save_folder*"loss_NN_LBC.txt", "w") do io
  writedlm(io, loss_NN_LBC)
end

open(save_folder*"NN_structure_LBC.txt", "w") do io
  writedlm(io, n_nodes_LBC)
end

# plot and save results of learning by confusion in save_folder
plt = plot(p_range_LBC, indicator_NN_LBC, dpi=300, c="black", label="NN")
plot!(p_range_LBC, indicator_opt_LBC, dpi=300, c="blue", label="analytical")
vline!([p_crit], label=L"p_{c}", c="red")
xlabel!(L"$p$")
ylabel!(L"$I_{\mathrm{LBC}}$")
savefig(save_folder*"indicator_NN_LBC.png")

plt = plot(p_range_LBC, loss_NN_LBC, dpi=300, c="black", label="NN")
plot!(p_range_LBC, loss_opt_LBC, dpi=300, c="blue", label="analytical")
vline!([p_crit], label=L"p_{c}",c="red")
xlabel!(L"$p$")
ylabel!(L"$\mathcal{L}_{\mathrm{LBC}}$")
savefig(save_folder*"loss_NN_LBC.png")
