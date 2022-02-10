cd(@__DIR__)
using Pkg; Pkg.activate("../..")

using my_ml_for_pt
# using BenchmarkTools
# using Profile
using DelimitedFiles
using LaTeXStrings
ENV["GKSwstype"]="nul"

using Plots
using SavitzkyGolay
using Flux
using Random

################################################################################

L=parse(Int,ARGS[1])
run_id=0
p_crit = Float32(2/log(1+sqrt(2)))
p_min=0.05f0
p_max = 10.0f0
dp=0.05f0
p_range = collect(p_min:dp:p_max)
p_range_LBC = collect(p_min-dp/2:dp:p_max+dp/2)

folder_data = "./Data/"
folder_methods = "./Methods/L="*string(L)*"/"
folder_figures = "./Figures/L="*string(L)*"/"

analytical=true
plotting=true
saving=true

################################################################################

energy = readdlm(folder_data*"L="*string(L)*"/run="*string(run_id)*"/E_T="*string(0)*".txt",eltype(dp))
num = readdlm(folder_data*"L="*string(L)*"/run="*string(run_id)*"/num_T="*string(0)*".txt",eltype(dp))
energies=[energy]
numbers=[num]
for index in 1:length(p_range)-1
	energyy = readdlm(folder_data*"L="*string(L)*"/run="*string(run_id)*"/E_T="*string(index)*".txt",eltype(dp))
	numm = readdlm(folder_data*"L="*string(L)*"/run="*string(run_id)*"/num_T="*string(index)*".txt",eltype(dp))
	push!(energies,energyy)
	push!(numbers,numm)
end
unique_energies = unique(vcat(energies...))
n_samples = sum(numbers[1])

distr,samples = MLP.distr_approx(energies,unique_energies,numbers,p_range)
distr_const,samples_const = MLP.constant_distr()
dataset = MLP.get_dataset(p_range,distr,samples)
dataset_train_SL = MLP.get_modified_dataset_train_SL(distr,samples,p_range,n_samples)
# dataset_train_SL = MLP.get_unmodified_dataset_train_SL(distr,samples,p_range,n_samples)

inputs_one_hot = reshape(unique_energies,1,length(unique_energies))
mean_train,std_train = MLP.get_dataset_stats(dataset,inputs_one_hot,length(p_range))
inputs_one_hot_stand = (inputs_one_hot.-mean_train)./std_train
mean_train_SL,std_train_SL = MLP.get_dataset_stats(dataset_train_SL,inputs_one_hot,length(p_range))
inputs_one_hot_stand_SL = (inputs_one_hot.-mean_train_SL)./std_train_SL

################################################################################
n_nodes_LBC=[size(inputs_one_hot)[1],64,64,64,2]

NN_LBC = Chain(
	Dense(n_nodes_LBC[1], n_nodes_LBC[2], relu),
	Dense(n_nodes_LBC[2], n_nodes_LBC[3], relu),
	Dense(n_nodes_LBC[3], n_nodes_LBC[4], relu),
	Dense(n_nodes_LBC[4], n_nodes_LBC[end], identity))
pnn_LBC, re_LBC = Flux.destructure(NN_LBC)


indx = parse(Int,ARGS[2])

######################
lr1_LBC = 0.001f0
epochs1_LBC = 1000
#epochs1_LBC = 10
saveat1_LBC=50
#saveat1_LBC=10
opt1_LBC = ADAM(lr1_LBC)
stochastic_LBC = false
batchsize_stochastic_LBC = Int(round(size(dataset)[2]/10))
verbose = true

pred_logger_LBC1, losses_LBC1, NN_logger_LBC1 = MLP.get_indicators_LBC_numerical_fixed_p(deepcopy(pnn_LBC),re_LBC,dataset,epochs1_LBC,p_range,dp,opt1_LBC,p_range_LBC,indx,inputs_one_hot_stand,saveat=saveat1_LBC,verbose=verbose,stochastic=stochastic_LBC,batchsize_stochastic=batchsize_stochastic_LBC)

indicator_NN_LBC = pred_logger_LBC1[end][1]
loss_NN_LBC = pred_logger_LBC1[end][2]

pred_logger_LBC = pred_logger_LBC1
losses_LBC = losses_LBC1
NN_logger_LBC = NN_logger_LBC1

######################

if saving
	open(folder_methods*"LBC/hyperparams_LBC_p="*string(indx)*".txt", "w") do io
		writedlm(io, [saveat1_LBC, batchsize_stochastic_LBC, size(dataset)[2], lr1_LBC,epochs1_LBC])
	end

	open(folder_methods*"LBC/NN_logger_LBC"*string(indx)*".txt", "w") do io
		writedlm(io, NN_logger_LBC)
	end

	for i in 1:length(pred_logger_LBC)
		open(folder_methods*"LBC/indicator_logger_LBC_p="*string(indx)*"_"*string(i)*".txt", "w") do io
			writedlm(io, pred_logger_LBC[i][1])
		end

		open(folder_methods*"LBC/loss_logger_LBC_p="*string(indx)*"_"*string(i)*".txt", "w") do io
			writedlm(io, pred_logger_LBC[i][2])
		end

	end

	open(folder_methods*"LBC/indicator_1_NN_LBC_p="*string(indx)*".txt", "w") do io
		writedlm(io, indicator_NN_LBC)
	end

	open(folder_methods*"LBC/loss_NN_LBC_p="*string(indx)*".txt", "w") do io
		writedlm(io, loss_NN_LBC)
	end

	open(folder_methods*"LBC/losses_LBC_p="*string(indx)*".txt", "w") do io
		writedlm(io, losses_LBC)
	end

	open(folder_methods*"LBC/p_range_LBC.txt", "w") do io
		writedlm(io, p_range_LBC)
	end
end
