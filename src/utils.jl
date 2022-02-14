# compute L2 norm
function sqnorm(x)
  return sum(abs2, x)
end

# compute binary crossentropy loss
function crossentropy(p, l)

  # add epsilon perturbation for floating-point stability
  return -1*(l*log(p+eps(float(eltype(p))))+(1-l)*log(1-p+eps(float(eltype(p)))))
end

function prepare_dataset_tilted(p, distribution, n_samples, samples)
  data = []
  for x in samples
    push!(data, (x, n_samples*distribution(x, p)))
  end
  return data
end

function get_datasets_tilted(p_range, distribution, n_samples, samples)
  datasets = []
  for p in p_range
    push!(datasets, prepare_dataset_tilted(p, distribution, n_samples, samples))
  end
  return datasets
end

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

function get_stats(datasets)
  mu = zero(datasets[1][1][2])
  musq =zero(datasets[1][1][2])
  for dataset in datasets
    for (sample, prob) in dataset
      mu += sample*prob
      musq += (sample^2)*prob
    end
  end
  mu = mu/length(datasets)
  musq = musq/length(datasets)
  return mu, sqrt(musq-mu^2)
end

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

function get_training_data_SL(dataset_SL, p_range, p_max, p_min)
  train = [dataset_SL[:,1]]
  for i in 2:size(dataset_SL)[2]
    ele = dataset_SL[:, i]
    if p_range[Int(ele[3])] <= p_min || p_range[Int(ele[3])] >= p_max
      push!(train,ele)
    end
  end
  return hcat(train...)
end

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

function five_point_stencil(y, dp)
  derivative = zeros(length(y)-4)
  i = 1
  for index in 3:length(y)-2
    derivative[i] = (y[index-2]-8*y[index-1]+8*y[index+1]-y[index+2])/(12*dp)
    i += 1
  end
  return derivative
end

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

function get_dataset_train_SL_opt(dataset_train_SL, distr, p_range, p_min, p_max, dp)
  p_range_1 = Tuple(collect(p_range[1]:dp:p_min))
  p_range_2 = Tuple(collect(p_max:dp:p_range[end]))
  SLc = MLP.SL_cache([zero(eltype(dp))], [zero(eltype(dp))], [zero(eltype(dp))], [zero(eltype(dp))], [zero(eltype(dp))], [zero(eltype(dp))], [zero(eltype(dp)), zero(eltype(dp)), zero(eltype(dp))])

  relevant_samples = unique(map(x->Int(x), dataset_train_SL[1,:]))
  labels = map(x -> MLP.get_pred_opt_x_SL!(x, distr, p_range_1, p_range_2, SLc), relevant_samples)

  return transpose(hcat(relevant_samples, labels))
end


function get_dataset_train_PBM_opt(dataset, distr, p_range)
  relevant_samples = map(x -> Int(x), unique(dataset[1, :]))
  PBMc = MLP.PBM_cache([zero(eltype(p_range[1]))], [zero(eltype(p_range[1]))],[zero(eltype(p_range[1]))], [zero(eltype(p_range[1]))], [zero(eltype(p_range[1])), zero(eltype(p_range[1]))])

  labels = map(x -> MLP.get_pred_opt_x_PBM!(x, distr, p_range, PBMc), relevant_samples)

  return transpose(hcat(relevant_samples, labels))
end

function get_dataset_train_LBC_opt(dataset, distr, p_range, p_range_LBC)
  relevant_samples = map(x -> Int(x), unique(dataset[1, :]))
  LBCc = MLP.LBC_cache([zero(eltype(p_range[1]))], [zero(eltype(p_range[1]))], [zero(eltype(p_range[1]))], [zero(eltype(p_range[1]))], [zero(eltype(p_range[1]))], [zero(eltype(p_range[1])), zero(eltype(p_range[1]))])

  labels = map(x -> MLP.get_pred_opt_x_LBC!(x, distr, p_range, p_range_LBC[1], LBCc), relevant_samples)

  dataset_train = [transpose(hcat(relevant_samples, labels))]
  for i in collect(2:length(p_range_LBC))
    labels = map(x -> MLP.get_pred_opt_x_LBC!(x, distr, p_range, p_range_LBC[i], LBCc), relevant_samples)

    push!(dataset_train, transpose(hcat(relevant_samples, labels)))
  end

  return dataset_train
end

function get_dataset_LBC_sorted(dataset, distr, p_range, p_range_LBC; standardized=true)
  mu, std = get_dataset_stats(dataset, length(p_range))

  dataset_train_LBC_opt = MLP.get_dataset_train_LBC_opt(dataset, distr, p_range, p_range_LBC)

  dataset_train_LBC_opt_1 = dataset_train_LBC_opt[1]
  indices = reverse(sortperm(dataset_train_LBC_opt_1[2, :]))
  new_dataset = deepcopy(dataset)
  for i in 1:length(indices)
    current = indices[i]
    indxes = findall(x -> x==current,dataset[1, :])
    for j in indxes
      new_dataset[1, j] = convert(eltype(dataset[1, 1]), i)
    end
  end

  new_dataset[1, :] = (new_dataset[1, :].-mu)./std
  new_datasets = [new_dataset]
  for k in 1:length(dataset_train_LBC_opt)
    dataset_train_LBC_opt_k = dataset_train_LBC_opt[k]
    indices = reverse(sortperm(dataset_train_LBC_opt_k[2, :]))
    new_datasett = deepcopy(dataset)
    for i in 1:length(indices)
      current = indices[i]
      indxes = findall(x -> x==current,dataset[1, :])
      for j in indxes
        new_datasett[1, j] = convert(eltype(dataset[1, 1]), i)
      end
    end

    new_datasett[1, :] = (new_datasett[1, :].-mu)./std

    push!(new_datasets, new_datasett)
  end

  return new_datasets
end

function get_dataset_PBM_sorted(dataset, distr, p_range; standardized=true)
  mu, std = get_dataset_stats(dataset, length(p_range))
  dataset_train_PBM_opt = MLP.get_dataset_train_PBM_opt(dataset, distr, p_range)

  indices = sortperm(dataset_train_PBM_opt[2, :])
  new_dataset = deepcopy(dataset)
  for i in 1:length(indices)
    current = indices[i]
    indxes = findall(x -> x==current, dataset[1, :])
    for j in indxes
      new_dataset[1, j] = convert(eltype(dataset[1, 1]), i)
    end
  end

  new_dataset[1, :] = (new_dataset[1, :].-mu)./std

  return new_dataset
end

function get_dataset_SL_sorted(dataset, dataset_train_SL, distr, p_range, p_min, p_max, dp; standardized=true)
  mu, std = get_dataset_stats(dataset_train_SL, length(p_range))
  dataset_SL_opt = MLP.get_dataset_train_SL_opt(dataset, distr, p_range, p_min, p_max, dp)

  indices = reverse(sortperm(dataset_SL_opt[2, :]))
  new_dataset = deepcopy(dataset)
  new_dataset_train = deepcopy(dataset_train_SL)
  for i in 1:length(indices)
    current = indices[i]
    indxes = findall(x -> x==current,dataset[1, :])
    for j in indxes
      new_dataset[1, j] = convert(eltype(dataset[1, 1]), i)
    end

    indxes = findall(x -> x==current,dataset_train_SL[1, :])
    for j in indxes
      new_dataset_train[1, j] = convert(eltype(dataset_train_SL[1, 1]), i)
    end

  end

  new_dataset[1, :] = (new_dataset[1, :].-mu)./std
  new_dataset_train[1, :] = (new_dataset_train[1, :].-mu)./std

  return new_dataset, new_dataset_train
end
