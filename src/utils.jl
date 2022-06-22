# compute L2 norm
function sqnorm(x)
  return sum(abs2, x)
end

# compute L2 regularization term for a given NN with lambda as regularization strength
function L2_penalty(lambda, pnn)
  return lambda*sum(map(x->x^2,pnn))
end

# compute binary crossentropy loss
function crossentropy(p, l)
  # add epsilon perturbation for floating-point stability
  return -(l*log(p+eps(eltype(p))) + (1-l)*log(1-p + eps(eltype(p))))
end

# construct the training data set for applying SL with NNs, where we modify the underlying probability distributions to combat deviations due to finite sample statistics (applicable for data generated from Boltzmann distributions)
function get_training_data_SL_modified(data, p_max_indx, p_min_indx, n_samples)
  data_train_SL = hcat(data[:,1:p_min_indx], data[:,p_max_indx:size(data)[2]])
  unseen_indices = findall(x->x==zero(eltype(data[1,1])), sum(data_train_SL,dims=2)[:,1])

  data_train_SL[unseen_indices,end] = ones(eltype(data[1,1]),length(unseen_indices))/n_samples

  return data_train_SL
end

# construct the training data set for applying SL with NNs, where we do not modify the underlying probability distributions (applicable for data generated from Boltzmann distributions)
function get_training_data_SL(data, p_max_indx, p_min_indx)
  return hcat(data[:,1:p_min_indx], data[:,p_max_indx:size(data)[2]])
end

# compute mean and standard deviation of input features given a data set (used for standardization)
function get_dataset_stats(data, inputs)
  mu = map(i->sum(data.*inputs[i,:])/size(data)[2],1:size(inputs)[1])
  musq = map(i->sum(data.*(inputs[i,:].^2))/size(data)[2],1:size(inputs)[1])

  return mu, sqrt.(musq.-mu.^2)
end
