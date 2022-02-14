# compute L2 norm
function sqnorm(x)
  return sum(abs2, x)
end

# compute binary crossentropy loss
function crossentropy(p, l)

  # add epsilon perturbation for floating-point stability
  return -1*(l*log(p+eps(float(eltype(p))))+(1-l)*log(1-p+eps(float(eltype(p)))))
end

# get a batch of indicies for NN training
function get_batches(indices, batchsize, batch)
  if batch*batchsize > length(indices)
    if (batch-1)*batchsize+1 > length(indices)
      error("batchsize too large")
    else
      return indices[1+batchsize*(batch-1):end]
    end
  else
    return indices[1+batchsize*(batch-1):batch*batchsize]
  end
end

# construct data set for NN training based on prototypical probability distributions (cases 1-3)
function get_dataset_proto(p_range, distribution, samples)
  dataset = zeros(eltype(p_range[1]), (3, length(p_range)*length(samples)))
  count = 1
  for i in 1:length(p_range)
    for j in 1:length(samples)
      dataset[:, count] = [j, distribution(samples[j], p_range[i]), i]
      count += 1
    end
  end
  return dataset
end

# construct data set for NN training based on a given probability distributions
function get_dataset(p_range, distribution, samples)
  dataset_ul, n_unique = get_dataset_unlabelled(p_range, distribution, samples)

  dataset = zeros(eltype(p_range[1]), (3, n_unique))
  count = 1
  len_p_train = 0
  for i in 1:length(dataset_ul)
    data = dataset_ul[i]
    p = p_range[i]

    for j in 1:length(data)
      freq = data[j][2]
      if freq != zero(eltype(freq))
        dataset[:, count] = [data[j][1], freq, i*one(eltype(freq))]
        count += 1
      end
    end
  end
  return dataset
end

# construct data set for NN training based on a given probability distributions, where labels are not yet included
function get_dataset_unlabelled(p_range, distribution, samples)
  data, counter = prepare_dataset(p_range[1], distribution, samples)
  datasets = [data]
  for indxp in 2:length(p_range)
    data,count = prepare_dataset(p_range[indxp], distribution, samples)
    push!(datasets, data)
    counter += count
  end
  return datasets, Int(counter)
end

# prepare list of samples and count number of unique samples
function prepare_dataset(p, distribution, samples)
  counter = zero(eltype(length(samples)))
  dist = distribution(samples[1][1],p)
  if dist != zero(eltype(dist))
    counter += one(eltype(counter))
  end

  data = [[samples[1][1], dist]]
  for indx in 2:length(samples)
    dist = distribution(samples[indx][1], p)
    if dist != zero(eltype(dist))
      counter += one(eltype(counter))
    end
    push!(data, [samples[indx][1], dist])
  end
  return data, counter
end

# construct the training data set for applying SL with NNs, where we modify the underlying probability distributions to combat deviations due to finite sample statistics
function get_modified_dataset_train_SL(distr, samples, p_range, n_samples; gs_index=1)
  new_dataset_train_SL = zeros(eltype(p_range[1]), (3, gs_index+length(samples)))

  for gs_indx in 1:gs_index
    new_dataset_train_SL[:, gs_indx]=[samples[gs_indx], distr(samples[gs_indx], p_range[1]), 1]
  end
  num_samples = n_samples

  for indx in collect(1:gs_index)
    sample = samples[indx]
    p = length(p_range)
    new_dataset_train_SL[:, indx+gs_index]=[sample, distr(sample,p_range[end]), p]
  end

  for indx in collect(gs_index+1:length(samples))
    sample = samples[indx]
    prob = Int(round(distr(sample,p_range[end])*n_samples))

    # add an instance of each sample which has not appeared in the training set to the set of samples associated with the largest sampled temperature value
    if prob == zero(eltype(p_range[1]))
      prob = one(eltype(p_range[1]))
      num_samples += one(eltype(n_samples))
    end
    p = length(p_range)

    new_dataset_train_SL[:, indx+gs_index]=[sample, prob, p]
  end

  new_dataset_train_SL[2, gs_index+1:end] = new_dataset_train_SL[2, gs_index+1:end]./num_samples

  return new_dataset_train_SL
end

# construct the training data set for applying SL with NNs, where we do not modify the underlying probability distributions
function get_unmodified_dataset_train_SL(distr, samples, p_range, n_samples)
  new_dataset_train_SL = zeros(eltype(p_range[1]), (3, 1+length(samples)))

  new_dataset_train_SL[:, 1] = [samples[1], distr(samples[1], p_range[1]), 1]

  for indx in collect(1:length(samples))
    sample = samples[indx]
    prob = distr(sample, p_range[end])
    p = length(p_range)
    new_dataset_train_SL[:, indx+1] = [sample, prob, p]
  end

  new_dataset_train_SL[2, 2:end] = new_dataset_train_SL[2, 2:end]

  return new_dataset_train_SL
end

# compute mean and standard deviation of input features given a data set (used for standardization)
function get_dataset_stats(dataset, inputs, n_points)
  mu = zeros(eltype(dataset[2, 1]), size(inputs)[1])
  musq = zeros(eltype(dataset[2, 1]), size(inputs)[1])
  for indx in collect(1:size(dataset)[2])
    mu .+= dataset[2, indx]*inputs[Int(dataset[1, indx])]
    musq .+= dataset[2, indx]*inputs[Int(dataset[1, indx])]^2
  end
  mu = mu./n_points
  musq = musq./n_points
  return mu, sqrt.(musq.-mu.^2)
end
