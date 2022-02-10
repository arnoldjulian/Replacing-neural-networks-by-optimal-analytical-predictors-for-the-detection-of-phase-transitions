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
run_id=1
p_crit1 = 4.0
p_crit2 = 8.0
p_min = 0.1
p_max = 20.0
n_points = 200
p_range = collect(LinRange(p_min,p_max,n_points))
dp=p_range[2]-p_range[1]
p_range_LBC = collect(p_min-dp/2:dp:p_max+dp/2)

folder_data = "./Data/L="*string(L)*"/"
folder_methods = "./Methods/L="*string(L)*"/"
folder_figures = "./Figures/L="*string(L)*"/"

analytical=true
plotting=true
saving=true

################################################################################

prob = vcat(readdlm(folder_data*"probs_W="*string(1)*".txt")...)
probss = [prob]

for index in 2:201-1
	probb =vcat(readdlm(folder_data*"probs_W="*string(index)*".txt")...)
	push!(probss,probb)
end
samples = collect(1:length(probss[1]))
n_samples = length(samples)

function distribution(sample,p,probss)
	p_indx = Int(round((p-p_range[1])/(p_range[2]-p_range[1])))+1
	return probss[p_indx][Int(sample)]
end

distr = (x,p)->distribution(x,p,probss)
distr_const,samples_const = MLP.constant_distr()
dataset = MLP.get_dataset(p_range,distr,samples)
dataset_train_SL = MLP.get_training_data_SL(dataset,p_range,p_max,p_min)

inputs_one_hot = transpose(readdlm(folder_data*"strings.txt",eltype(dp)))
mean_train,std_train = MLP.get_dataset_stats(dataset,inputs_one_hot,length(p_range))
inputs_one_hot_stand = (inputs_one_hot.-mean_train)./std_train
mean_train_SL,std_train_SL = MLP.get_dataset_stats(dataset_train_SL,inputs_one_hot,length(p_range))
inputs_one_hot_stand_SL = (inputs_one_hot.-mean_train_SL)./std_train_SL

################################################################################
n_nodes_LBC=[size(inputs_one_hot)[1],128,128,64,64,64,2]
NN_LBC = Chain(
	Dense(n_nodes_LBC[1], n_nodes_LBC[2], relu),
	Dense(n_nodes_LBC[2], n_nodes_LBC[3], relu),
	Dense(n_nodes_LBC[3], n_nodes_LBC[4], relu),
	Dense(n_nodes_LBC[4], n_nodes_LBC[5], relu),
	Dense(n_nodes_LBC[5], n_nodes_LBC[6], relu),
	Dense(n_nodes_LBC[6], n_nodes_LBC[end], identity))
pnn_LBC, re_LBC = Flux.destructure(NN_LBC)


indx = parse(Int,ARGS[2])

######################
lr1_LBC = 0.001f0
epochs1_LBC = 500
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
