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

L=60
run_id=0 #from exact enumeration (run=0 for MC data)
data_folder = "/media/julian/Samsung_T51/ml_for_pt/final_data/Ising/data/"

p_crit = Float32(2/log(1+sqrt(2)))
p_min=0.05f0
p_max = 10.0f0
dp=0.05f0
p_range = collect(p_min:dp:p_max)
p_range_LBC = collect(p_min-dp/2:dp:p_max+dp/2)

energy = readdlm(data_folder*"L="*string(L)*"/run="*string(run_id)*"/E_T="*string(0)*".txt",eltype(dp))
num = readdlm(data_folder*"L="*string(L)*"/run="*string(run_id)*"/num_T="*string(0)*".txt",eltype(dp))
energies=[energy]
numbers=[num]
for index in 1:length(p_range)-1
	energyy = readdlm(data_folder*"L="*string(L)*"/run="*string(run_id)*"/E_T="*string(index)*".txt",eltype(dp))
	numm = readdlm(data_folder*"L="*string(L)*"/run="*string(run_id)*"/num_T="*string(index)*".txt",eltype(dp))
	push!(energies,energyy)
	push!(numbers,numm)
end
unique_energies = unique(vcat(energies...))
n_samples = sum(numbers[1])

folder_data = "./Methods/L="*string(L)*"/run="*string(run_id)*"/"
folder_figures = "./Figures/L="*string(L)*"/run="*string(run_id)*"/"

################################################################################

distr_const,samples_const = MLP.constant_distr()
distr,samples = MLP.distr_approx(energies,unique_energies,numbers,p_range)
dataset = MLP.get_dataset(p_range,distr,samples)
# dataset_train_SL = MLP.get_training_data_SL(dataset,p_range,p_max,p_min)
dataset_train_SL = MLP.get_modified_dataset_train_SL(distr,samples,p_range,n_samples)
dataset_test_SL = dataset

# create standardized versions
mean_train,std_train = MLP.get_dataset_stats(dataset,length(p_range))
dataset_stand = deepcopy(dataset)
dataset_stand[1,:] = (dataset_stand[1,:].-mean_train)./std_train

mean_train_SL,std_train_SL = MLP.get_dataset_stats(dataset_train_SL,2)
dataset_train_SL_stand = deepcopy(dataset_train_SL)
dataset_train_SL_stand[1,:] = (dataset_train_SL_stand[1,:].-mean_train_SL)./std_train_SL

dataset_test_SL_stand = deepcopy(dataset_test_SL)
dataset_test_SL_stand[1,:] = (dataset_test_SL_stand[1,:].-mean_train_SL)./std_train_SL

################################################################################

n_nodes_SL=[64,64]

NN_SL = Chain(
	Dense(1, n_nodes_SL[1], relu),
	Dense(n_nodes_SL[1], n_nodes_SL[2], relu),
	Dense(n_nodes_SL[2], 2, identity))
pnn_SL, re_SL = Flux.destructure(NN_SL)

lr_SL = 0.001f0
epochs_SL = 1
opt_SL = ADAM(lr_SL)
batchsize_SL = Int(round(size(dataset_train_SL)[2]))
saveat_SL=epochs_SL
stochastic_SL = true
n_batches_train_stochastic_SL = 10
standardized_SL = true

# run SL analytically
pred_logger_SL, losses, NN_logger_SL = MLP.get_indicators_SL_numerical(pnn_SL, re_SL,dataset_train_SL_stand,dataset_test_SL_stand,epochs_SL,p_range,dp,p_min, p_max,opt_SL,verbose=true,trained=false,saveat=saveat_SL,stochastic=stochastic_SL,n_batches_train_stochastic=n_batches_train_stochastic_SL)
pred_NN_SL = pred_logger_SL[end][1]
indicator_NN_SL = pred_logger_SL[end][2]
loss_NN_SL = pred_logger_SL[end][3]


pred_opt_SL, indicator_opt_SL, loss_opt_SL = MLP.get_indicators_SL_analytical(samples, distr, p_range, dp, p_min, p_max)
sg_window_size_SL = 23
sg_order_SL = 4
indicator_opt_smooth_SL = savitzky_golay(pred_opt_SL, sg_window_size_SL,sg_order_SL,deriv=1,rate=1/dp).y*-1


plt = plot(p_range[2:end-1],indicator_NN_SL)
plot!(p_range[2:end-1],indicator_opt_SL)
vline!([p_crit])
# savefig("sl_stand_2.png")

plt =plot(1:length(losses),losses)
hline!([loss_opt_SL])


open(folder_data*"hyperparams_SL_peak.txt", "w") do io
	writedlm(io, [saveat_SL, lr_SL, epochs_SL, standardized_SL, stochastic_SL, n_batches_train_stochastic_SL])
end

open(folder_data*"NN_logger_SL_peak.txt", "w") do io
	writedlm(io, NN_logger_SL)
end

for i in 1:length(pred_logger_SL)
	open(folder_data*"pred_logger_SL_"*string(i)*"_peak.txt", "w") do io
		writedlm(io, pred_logger_SL[i][1])
	end

	open(folder_data*"indicator_logger_SL_"*string(i)*"_peak.txt", "w") do io
		writedlm(io, pred_logger_SL[i][2])
	end

	open(folder_data*"loss_logger_SL_"*string(i)*"_peak.txt", "w") do io
		writedlm(io, pred_logger_SL[i][3])
	end

end

open(folder_data*"indicator_1_NN_SL_peak.txt", "w") do io
	writedlm(io, pred_NN_SL)
end


open(folder_data*"indicator_2_NN_SL_peak.txt", "w") do io
	writedlm(io, indicator_NN_SL)
end

open(folder_data*"loss_NN_SL_peak.txt", "w") do io
	writedlm(io, loss_NN_SL)
end

open(folder_data*"NN_structure_SL_peak.txt", "w") do io
	writedlm(io, n_nodes_SL)
end

################################################################################

n_nodes_PBM = [64,64,64,64,64]

NN_PBM = Chain(
	Dense(1, n_nodes_PBM[1], relu),
	Dense(n_nodes_PBM[1], n_nodes_PBM[2], relu),
	Dense(n_nodes_PBM[2], n_nodes_PBM[3], relu),
	Dense(n_nodes_PBM[3], n_nodes_PBM[4], relu),
	Dense(n_nodes_PBM[4], n_nodes_PBM[5], relu),
	Dense(n_nodes_PBM[5], 2, identity))
pnn_PBM, re_PBM = Flux.destructure(NN_PBM)


lr_PBM = 0.001f0
epochs_PBM = 100
opt_PBM = ADAM(lr_PBM)
saveat_PBM=epochs_PBM
stochastic_PBM = false
n_batches_train_stochastic_PBM = 5
standardized_PBM = false

pred_logger_PBM, losses, NN_logger_PBM = MLP.get_indicators_PBM_numerical(pnn_PBM,re_PBM,dataset,epochs_PBM,p_range,dp,opt_PBM,verbose=true,trained=false,saveat=saveat_PBM,stochastic=stochastic_PBM,n_batches_train_stochastic=n_batches_train_stochastic_PBM)
pred_NN_PBM = pred_logger_PBM[end][1]
indicator_NN_PBM = pred_logger_PBM[end][2]
loss_NN_PBM = pred_logger_PBM[end][3]

pred_opt_PBM, indicator_opt_PBM, loss_opt_PBM = MLP.get_indicators_PBM_analytical(samples, distr, p_range, dp)
sg_window_size_PBM = 23
sg_order_PBM = 4
indicator_opt_smooth_PBM = savitzky_golay(pred_opt_PBM, sg_window_size_PBM,sg_order_PBM, deriv=1,rate=1/dp).y.-1

plt = plot(p_range,pred_NN_PBM,label="NN")
plot!(p_range,pred_opt_PBM,label="analytical")

plt = plot(p_range[2:end-1],indicator_NN_PBM,label="NN")
plot!(p_range[2:end-1],indicator_opt_PBM,label="analytical")
vline!([p_crit])
# savefig("pbm_non_standardized_2.png")

plt =plot(1:length(loss_NN_PBM),loss_NN_PBM)
hline!([loss_opt_PBM])


open(folder_data*"hyperparams_PBM_peak.txt", "w") do io
	writedlm(io, [saveat_PBM, lr_PBM, epochs_PBM, standardized_PBM, stochastic_PBM, n_batches_train_stochastic_PBM])
end

open(folder_data*"NN_logger_PBM_peak.txt", "w") do io
	writedlm(io, NN_logger_PBM)
end

for i in 1:length(pred_logger_PBM)
	open(folder_data*"pred_logger_PBM_"*string(i)*"_peak.txt", "w") do io
		writedlm(io, pred_logger_PBM[i][1])
	end

	open(folder_data*"indicator_logger_PBM_"*string(i)*"_peak.txt", "w") do io
		writedlm(io, pred_logger_PBM[i][2])
	end

	open(folder_data*"loss_logger_PBM_"*string(i)*"_peak.txt", "w") do io
		writedlm(io, pred_logger_PBM[i][3])
	end

end

open(folder_data*"indicator_1_NN_PBM_peak.txt", "w") do io
	writedlm(io, pred_NN_PBM)
end

open(folder_data*"indicator_2_NN_PBM_peak.txt", "w") do io
	writedlm(io, indicator_NN_PBM)
end

open(folder_data*"loss_NN_PBM_peak.txt", "w") do io
	writedlm(io, loss_NN_PBM)
end

################################################################################

n_nodes_LBC = [64,64,64,64,64]

NN_LBC = Chain(
	Dense(1, n_nodes_LBC[1], relu),
	Dense(n_nodes_LBC[1], n_nodes_LBC[2], relu),
	Dense(n_nodes_LBC[2], n_nodes_LBC[3], relu),
	Dense(n_nodes_LBC[3], n_nodes_LBC[4], relu),
	Dense(n_nodes_LBC[4], n_nodes_LBC[5], relu),
	Dense(n_nodes_LBC[5], 2, identity))
pnn_LBC, re_LBC = Flux.destructure(NN_LBC)

lr_LBC = 0.001f0
epochs_LBC = 500
opt_LBC = ADAM(lr_LBC)
saveat_LBC=epochs_LBC
stochastic_LBC = true
n_batches_train_stochastic_LBC = 1
standardized_LBC = true
batchsize_stochastic_LBC=Int(round(length(dataset[1,:])/50))

lr_LBC = 0.0001f0
epochs_LBC = 100
opt_LBC = ADAM(lr_LBC)
saveat_LBC=epochs_LBC
stochastic_LBC = true
n_batches_train_stochastic_LBC = 1
standardized_LBC = true
batchsize_stochastic_LBC=Int(round(length(dataset[1,:])/20))

lr_LBC = 0.0001f0
epochs_LBC = 10
opt_LBC = ADAM(lr_LBC)
saveat_LBC=epochs_LBC
stochastic_LBC = false
n_batches_train_stochastic_LBC = 1
standardized_LBC = true
batchsize_stochastic_LBC=Int(round(length(dataset[1,:])/20))


# indicator_NN_LBC, loss_NN_LBC,NN_logger_LBC = MLP.get_indicators_LBC_numerical(pnn1, re1, dataset,epochs,p_range,dp,opt,p_range_LBC,verbose=true,trained=false,stochastic=false,n_batches_train_stochastic=10)

##
# indicator_NN_LBC, loss_NN_LBC,NN_logger_LBC = MLP.get_indicators_LBC_numerical_parallel(pnn_LBC, re_LBC, dataset,epochs_LBC,p_range,dp,opt_LBC,p_range_LBC,verbose=true,trained=false,stochastic=stochastic_LBC,n_batches_train_stochastic=n_batches_train_stochastic_LBC,batchsize_stochastic=batchsize_stochastic_LBC)
##

# accuracy,end_loss,NN_logger = MLP.get_indicators_LBC_numerical_fixed_p(deepcopy(pnn1),re1,dataset,epochs,p_range,dp,opt,p_range_LBC,1,verbose=true,trained=false,stochastic=false,n_batches_train_stochastic=10)

# accuracy,end_loss,NN_logger = MLP.get_indicators_LBC_numerical_fixed_p(pnn_LBC,re_LBC,dataset_stand,epochs_LBC,p_range,dp,opt_LBC,p_range_LBC,10,verbose=true,trained=false,stochastic=false,n_batches_train_stochastic=n_batches_train_stochastic_LBC,batchsize_stochastic=batchsize_stochastic_LBC)

accuracy,end_loss,NN_logger = MLP.get_indicators_LBC_numerical_fixed_p(pnn_LBC,re_LBC,dataset_stand,epochs_LBC,p_range,dp,opt_LBC,p_range_LBC,10,verbose=true,trained=false,stochastic=stochastic_LBC,n_batches_train_stochastic=n_batches_train_stochastic_LBC,batchsize_stochastic=batchsize_stochastic_LBC)

accuracies = []
end_losses = []
p_vals = []
for indx in collect(10:20:length(p_range_LBC))

	lr_LBC = 0.001f0
	epochs_LBC = 500
	opt_LBC = ADAM(lr_LBC)
	saveat_LBC=epochs_LBC
	stochastic_LBC = true
	n_batches_train_stochastic_LBC = 1
	standardized_LBC = true
	batchsize_stochastic_LBC=Int(round(length(dataset[1,:])/50))

	accuracy,end_loss,NN_logger = MLP.get_indicators_LBC_numerical_fixed_p(deepcopy(pnn_LBC),re_LBC,dataset_stand,epochs_LBC,p_range,dp,opt_LBC,p_range_LBC,indx,verbose=true,trained=false,stochastic=stochastic_LBC,n_batches_train_stochastic=n_batches_train_stochastic_LBC,batchsize_stochastic=batchsize_stochastic_LBC)

	lr_LBC = 0.0001f0
	epochs_LBC = 100
	opt_LBC = ADAM(lr_LBC)
	saveat_LBC=epochs_LBC
	stochastic_LBC = true
	n_batches_train_stochastic_LBC = 1
	standardized_LBC = true
	batchsize_stochastic_LBC=Int(round(length(dataset[1,:])/20))

	accuracy,end_loss,NN_logger = MLP.get_indicators_LBC_numerical_fixed_p(NN_logger[end],re_LBC,dataset_stand,epochs_LBC,p_range,dp,opt_LBC,p_range_LBC,indx,verbose=true,trained=false,stochastic=stochastic_LBC,n_batches_train_stochastic=n_batches_train_stochastic_LBC,batchsize_stochastic=batchsize_stochastic_LBC)

	# lr_LBC = 0.0001f0
	# epochs_LBC = 10
	# opt_LBC = ADAM(lr_LBC)
	# saveat_LBC=epochs_LBC
	# stochastic_LBC = false
	# n_batches_train_stochastic_LBC = 1
	# standardized_LBC = true
	# batchsize_stochastic_LBC=Int(round(length(dataset[1,:])/20))
	#
	# accuracy,end_loss,NN_logger = MLP.get_indicators_LBC_numerical_fixed_p(NN_logger[end],re_LBC,dataset_stand,epochs_LBC,p_range,dp,opt_LBC,p_range_LBC,indx,verbose=true,trained=false,stochastic=stochastic_LBC,n_batches_train_stochastic=n_batches_train_stochastic_LBC,batchsize_stochastic=batchsize_stochastic_LBC)

	push!(accuracies,accuracy)
	push!(end_losses,end_loss)
	push!(p_vals,p_range_LBC[indx])
end

plt = scatter(p_vals,accuracies,label="NN")
plot!(p_range_LBC,indicator_opt_LBC,label="analytical")
vline!([p_crit])

plt = scatter(p_vals,end_losses,label="NN")
plot!(p_range_LBC,loss_opt_LBC,label="analytical")
vline!([p_crit])

# indicator_NN_LBC, loss_NN_LBC,NN_logger_LBC = MLP.get_indicators_LBC_numerical_parallel(pnn1, re1, dataset,epochs,p_range,dp,opt,p_range_LBC,verbose=false,trained=false,stochastic=false,n_batches_train_stochastic=10)

indicator_opt_LBC, loss_opt_LBC = MLP.get_indicators_LBC_analytical(samples, distr, p_range, p_range_LBC)
sg_window_size_LBC = 23
sg_order_LBC = 4
indicator_opt_smooth_LBC = savitzky_golay(indicator_opt_LBC, sg_window_size_LBC,sg_order_LBC, deriv=0,rate=1/dp).y
background, _ = MLP.get_indicators_LBC_analytical(samples_const, distr_const, p_range, p_range_LBC)
indicator_opt_subtr_LBC = indicator_opt_smooth_LBC.-background

plt = plot(p_range_LBC,indicator_NN_LBC,label="NN")
plot!(p_range_LBC,indicator_opt_LBC,label="analytical")
vline!([p_crit])

plt = plot(p_range_LBC,loss_NN_LBC,label="NN")
plot!(p_range_LBC,loss_opt_LBC,label="analytical")


open(folder_data*"hyperparams_LBC.txt", "w") do io
	writedlm(io, [saveat, lr, epochs, batchsize])
end

open(folder_data*"NN_logger_LBC.txt", "w") do io
	writedlm(io, NN_logger_LBC)
end

open(folder_data*"indicator_1_NN_LBC.txt", "w") do io
	writedlm(io, indicator_NN_LBC)
end

open(folder_data*"loss_NN_LBC.txt", "w") do io
	writedlm(io, loss_NN_LBC)
end

################################################################################


plt = plot(p_range,pred_NN_SL,dpi=300,c="black",label="numerical")
plot!(p_range,pred_opt_SL,dpi=300,c="blue",label="analytical")
vline!([p_crit],label=L"p_{c}",c="red")
xlabel!(L"p")
ylabel!("Output layer")
savefig(folder_figures*"indicator_1_SL_main.png")

plt = plot(p_range[2:end-1],indicator_NN_SL,dpi=300,c="black",label="numerical")
plot!(p_range[2:end-1],indicator_opt_SL,dpi=300,c="blue",label="analytical")
plot!(p_range,indicator_opt_smooth_SL,dpi=300,c="green",label="analytical smooth")
vline!([p_crit],label=L"p_{c}",c="red")
xlabel!(L"p")
ylabel!(L"I")
savefig(folder_figures*"indicator_2_SL_main.png")

plt = plot(collect(1:epochs),loss_NN_SL,dpi=300,c="black",label="numerical")
hline!([loss_opt_SL],label=L"p_{c}",c="red")
xlabel!(L"Epochs")
ylabel!(L"Loss")
savefig(folder_figures*"loss_SL.png")

plt = plot(p_range,pred_NN_PBM,dpi=300,c="black",label="numerical")
plot!(p_range,pred_opt_PBM,dpi=300,c="blue",label="analytical")
vline!([p_crit],label=L"p_{c}",c="red")
xlabel!(L"p")
ylabel!(L"p_{pred}")
savefig(folder_figures*"indicator_1_PBM_main.png")

plt = plot(p_range[2:end-1],indicator_NN_PBM,dpi=300,c="black",label="numerical")
plot!(p_range[2:end-1],indicator_opt_PBM,dpi=300,c="blue",label="analytical")
plot!(p_range,indicator_opt_smooth_PBM,dpi=300,c="green",label="analytical smooth")
vline!([p_crit],label=L"p_{c}",c="red")
xlabel!(L"p")
ylabel!(L"I")
savefig(folder_figures*"indicator_2_PBM_main.png")

plt = plot(collect(1:epochs),loss_NN_PBM,dpi=300,c="black",label="numerical")
hline!([loss_opt_PBM],label=L"p_{c}",c="red")
xlabel!(L"Epochs")
ylabel!(L"Loss")
savefig(folder_figures*"loss_PBM.png")

plt = plot(p_range_LBC,indicator_NN_LBC,dpi=300,c="black",label="numerical")
plot!(p_range_LBC,indicator_opt_LBC,dpi=300,c="blue",label="analytical")
plot!(twinx(),p_range_LBC,indicator_opt_subtr_LBC,dpi=300,c="orange",label=false)
plot!(p_range_LBC,indicator_opt_smooth_LBC,dpi=300,c="green",label="analytical smooth")
vline!([p_crit],label=L"p_{c}",c="red")
xlabel!(L"p")
ylabel!("Accuracy")
savefig(folder_figures*"indicator_1_LBC_main.png")

plt = plot(p_range_LBC,loss_NN_LBC,dpi=300,c="black",label="numerical")
plot!(p_range_LBC,loss_opt_LBC,dpi=300,c="red",label="analytical")
xlabel!(L"p")
ylabel!("Loss")
savefig(folder_figures*"loss_LBC.png")
