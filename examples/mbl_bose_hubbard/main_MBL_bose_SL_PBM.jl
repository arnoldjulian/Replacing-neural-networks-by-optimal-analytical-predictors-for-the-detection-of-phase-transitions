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
L=10
method= 1

L=parse(Int,ARGS[1])
method = parse(Int,ARGS[2])
run_id=1
p_crit1 = 4.0
p_crit2 = 8.0
p_min = 0.1
p_max = 20.0
n_points = 200
p_range = collect(LinRange(p_min,p_max,n_points))
dp=p_range[2]-p_range[1]
p_range_LBC = collect(p_min-dp/2:dp:p_max+dp/2)

# folder_data = "/media/julian/Samsung_T5/ml_for_pt/final_data/Kitaev/data/"
folder_data = "./Data/L="*string(L)*"/"
folder_methods = "./Methods/L="*string(L)*"/"
folder_figures = "./Figures/L="*string(L)*"/"

analytical=false
plotting=false
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
if method == 0
	#analytically
	if analytical
		pred_opt_SL, indicator_opt_SL, loss_opt_SL = MLP.get_indicators_SL_analytical(samples, distr, p_range, dp, p_min, p_max)
	end

	#numerically
	n_nodes_SL=[size(inputs_one_hot)[1],128,128,64,64,64,2]

	NN_SL = Chain(
		Dense(n_nodes_SL[1], n_nodes_SL[2], relu),
		Dense(n_nodes_SL[2], n_nodes_SL[3], relu),
		Dense(n_nodes_SL[3], n_nodes_SL[4], relu),
		Dense(n_nodes_SL[4], n_nodes_SL[5], relu),
		Dense(n_nodes_SL[5], n_nodes_SL[6], relu),
		Dense(n_nodes_SL[6], n_nodes_SL[end], identity))
	pnn_SL, re_SL = Flux.destructure(NN_SL)

	######################

	lr1_SL = 0.001f0
	epochs1_SL = 10000
	saveat1_SL=50
	opt1_SL = ADAM(lr1_SL)
	stochastic_SL = false
	batchsize_stochastic_SL = Int(round(size(dataset_train_SL)[2]/10))
	# batchsize_stochastic_SL = 512
	verbose = true

	pred_logger_SL1, losses_SL1, NN_logger_SL1 = MLP.get_indicators_SL_numerical(pnn_SL, re_SL,dataset_train_SL,dataset,epochs1_SL,p_range,dp,p_min,p_max,opt1_SL,inputs_one_hot_stand,verbose=verbose,saveat=saveat1_SL,stochastic=stochastic_SL,batchsize_stochastic=batchsize_stochastic_SL)

	pred_NN_SL = pred_logger_SL1[end][1]
	indicator_NN_SL = pred_logger_SL1[end][2]
	loss_NN_SL = pred_logger_SL1[end][3]

	pred_logger_SL = pred_logger_SL1
	losses_SL = losses_SL1
	NN_logger_SL = NN_logger_SL1

	#####################
	if saving
		open(folder_methods*"SL/hyperparams_SL.txt", "w") do io
			writedlm(io, [saveat1_SL, batchsize_stochastic_SL, size(dataset_train_SL)[2], lr1_SL, epochs1_SL])
		end

		open(folder_methods*"SL/NN_logger_SL.txt", "w") do io
			writedlm(io, NN_logger_SL)
		end

		for i in 1:length(pred_logger_SL)
			open(folder_methods*"SL/pred_logger_SL_"*string(i)*".txt", "w") do io
				writedlm(io, pred_logger_SL[i][1])
			end

			open(folder_methods*"SL/indicator_logger_SL_"*string(i)*".txt", "w") do io
				writedlm(io, pred_logger_SL[i][2])
			end

			open(folder_methods*"SL/loss_logger_SL_"*string(i)*".txt", "w") do io
				writedlm(io, pred_logger_SL[i][3])
			end

		end

		open(folder_methods*"SL/indicator_1_NN_SL.txt", "w") do io
			writedlm(io, pred_NN_SL)
		end

		open(folder_methods*"SL/indicator_2_NN_SL.txt", "w") do io
			writedlm(io, indicator_NN_SL)
		end

		open(folder_methods*"SL/loss_NN_SL.txt", "w") do io
			writedlm(io, loss_NN_SL)
		end

		open(folder_methods*"SL/NN_structure_SL.txt", "w") do io
			writedlm(io, n_nodes_SL)
		end

		open(folder_methods*"SL/losses_SL.txt", "w") do io
			writedlm(io, losses_SL)
		end
	end

else
	################################################################################
	#analytically
	if analytical
		pred_opt_PBM, indicator_opt_PBM, loss_opt_PBM = MLP.get_indicators_PBM_analytical(samples, distr, p_range, dp)
	end

	#numerically
	n_nodes_PBM=[size(inputs_one_hot)[1],128,128,64,64,64,1]

	NN_PBM = Chain(
		Dense(n_nodes_PBM[1], n_nodes_PBM[2], relu),
		Dense(n_nodes_PBM[2], n_nodes_PBM[3], relu),
		Dense(n_nodes_PBM[3], n_nodes_PBM[4], relu),
		Dense(n_nodes_PBM[4], n_nodes_PBM[5], relu),
		Dense(n_nodes_PBM[5], n_nodes_PBM[6], relu),
		Dense(n_nodes_PBM[6], n_nodes_PBM[end], identity))
	pnn_PBM, re_PBM = Flux.destructure(NN_PBM)

	######################

	lr1_PBM = 0.001f0
	epochs1_PBM = 1000
	#epochs1_PBM = 100
	saveat1_PBM=50
	opt1_PBM = ADAM(lr1_PBM)
	stochastic_PBM = false
	batchsize_stochastic_PBM = Int(round(size(dataset)[2]/10))
	verbose = true

	pred_logger_PBM1, losses_PBM1, NN_logger_PBM1 =MLP.get_indicators_PBM_numerical(pnn_PBM,re_PBM,dataset,epochs1_PBM,p_range,dp,opt1_PBM,inputs_one_hot_stand,verbose=verbose,saveat=saveat1_PBM,stochastic=stochastic_PBM,batchsize_stochastic=batchsize_stochastic_PBM)

	pred_NN_PBM = pred_logger_PBM1[end][1]
	indicator_NN_PBM = pred_logger_PBM1[end][2]
	loss_NN_PBM = pred_logger_PBM1[end][3]

	pred_logger_PBM = pred_logger_PBM1
	losses_PBM = losses_PBM1
	NN_logger_PBM = NN_logger_PBM1

	#####################
	if saving
		open(folder_methods*"PBM/hyperparams_PBM.txt", "w") do io
			writedlm(io, [saveat1_PBM, batchsize_stochastic_PBM, size(dataset)[2], lr1_PBM, epochs1_PBM])
		end

		open(folder_methods*"PBM/NN_logger_PBM.txt", "w") do io
			writedlm(io, NN_logger_PBM)
		end

		for i in 1:length(pred_logger_PBM)
			open(folder_methods*"PBM/pred_logger_PBM_"*string(i)*".txt", "w") do io
				writedlm(io, pred_logger_PBM[i][1])
			end

			open(folder_methods*"PBM/indicator_logger_PBM_"*string(i)*".txt", "w") do io
				writedlm(io, pred_logger_PBM[i][2])
			end

			open(folder_methods*"PBM/loss_logger_PBM_"*string(i)*".txt", "w") do io
				writedlm(io, pred_logger_PBM[i][3])
			end

		end

		open(folder_methods*"PBM/indicator_1_NN_PBM.txt", "w") do io
			writedlm(io, pred_NN_PBM)
		end

		open(folder_methods*"PBM/indicator_2_NN_PBM.txt", "w") do io
			writedlm(io, indicator_NN_PBM)
		end

		open(folder_methods*"PBM/loss_NN_PBM.txt", "w") do io
			writedlm(io, loss_NN_PBM)
		end

		open(folder_methods*"PBM/NN_structure_PBM.txt", "w") do io
			writedlm(io, n_nodes_PBM)
		end

		open(folder_methods*"PBM/losses_PBM.txt", "w") do io
			writedlm(io, losses_PBM)
		end
	end
end
