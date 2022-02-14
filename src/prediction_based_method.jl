# computation using analytical expression

# define cache
struct PBM_cache{cnumtype, cdentype, cprobtype, cpredxtype, crestype}
  cnum::cnumtype
  cden::cdentype
  cprob::cprobtype
  cpredx::cpredxtype
  cres::crestype
end

# get optimal predictions for specific sample
function get_pred_opt_x_PBM!(x, distribution, p_range,PBMc)
  @unpack cnum, cden, cprob, cpredx, cres = PBMc

  cnum[1] = zero(eltype(cnum[1]))
  cden[1] = zero(eltype(cden[1]))

  # compute numerator and denominator
  for p in p_range
    cprob[1] = distribution(x, p)
    cnum[1] += cprob[1]*p
    cden[1] += cprob[1]
  end

  if cden[1] == zero(eltype(cden[1]))
    return zero(eltype(cden[1]))
  else
    return cnum[1]/cden[1]
  end
end

# get mean optimal predictions for fixed value of tuning parameter
function get_pred_opt_p_PBM!(samples, distribution, p_range, p_tar,PBMc)
  @unpack cnum, cden, cprob, cpredx, cres = PBMc

  cres[1] = zero(eltype(cres[1]))

  for x in samples
    cpredx[1] = get_pred_opt_x_PBM!(x, distribution, p_range,PBMc)
    cprob[1] = distribution(x, p_tar)

    # contribution to optimal prediction
    cres[1] += cprob[1]*cpredx[1]

    # contribution to optimal loss
    cres[2] += cprob[1]*(p_tar-cpredx[1])^2
  end
  return nothing
end

# compute optimal predictions and indicators, as well as optimal loss of PBM
function get_indicators_PBM_analytical(samples, distribution, p_range, dp)

  caches=[PBM_cache([zero(eltype(dp))], [zero(eltype(dp))], [zero(eltype(dp))],[zero(eltype(dp))], [zero(eltype(dp)), zero(eltype(dp))]) for i in 1:Threads.nthreads()]
  pred_PBM_opt = zeros(eltype(dp), length(p_range))

  # start parallel computation for sampled values of tuning parameter
  Threads.@threads for indxp in collect(1:length(p_range))
    get_pred_opt_p_PBM!(samples, distribution, p_range, p_range[indxp], caches[Threads.threadid()])
    pred_PBM_opt[indxp] = caches[Threads.threadid()].cres[1]
  end

  # sum up all individual loss contributions
  cres2 = zero(eltype(dp))
  for i in 1:Threads.nthreads()
    cres2 += caches[i].cres[2]
  end

  # compute indicator using symmetric difference quotient
  return pred_PBM_opt, map(x -> x-1,(circshift(pred_PBM_opt, -1).-circshift(pred_PBM_opt, 1))./(2*dp))[2:end-1], cres2/length(p_range)
end


# computation using neural networks

# loss function where each prediction is weighted by probability associated with input
function main_loss_PBM_weighted(NN, pnn, data, p_range, inputs)
  indices = convert.(Int, data[1, :])
  input = inputs[:, indices]

  pred = NN(pnn)(input)
  labels = reshape([p_range[Int(i)] for i in data[3, :]], 1, length(data[3, :]))
  freq = reshape(data[2, :], 1, length(data[2, :]))

  return sum(freq.*((pred.-labels).^2))
end

# loss function without weighting
function main_loss_PBM_stochastic(NN, pnn, dataset, p_range, inputs)
  indices = convert.(Int, dataset[1, :])
  input = inputs[:, indices]

  pred = NN(pnn)(input)
  labels = reshape([p_range[Int(i)] for i in dataset[3, :]], 1, length(dataset[3, :]))

  return sum((pred.-labels).^2)/length(dataset[3, :])
end

# train NN in PBM based on weighted loss function
function train_PBM_weighted(NN, pnn, dataset, epochs, p_range, dp, opt, batchsize, n_batches, inputs; verbose=false, saveat=epochs)
  NN_logger = [zeros(eltype(p_range[1]), length(pnn))]
  pred_logger = [(zeros(eltype(p_range[1]), length(p_range)), zeros(eltype(p_range[1]), length(p_range)-2), zeros(eltype(p_range[1]), 1))]

  pnn_best = zeros(eltype(pnn[1]), length(pnn))
  best_loss = zero(eltype(pnn[1]))

  losses = zeros(eltype(p_range[1]), epochs)
  indices = collect(1:size(dataset)[2])
  for epoch in 1:epochs
    Random.shuffle!(indices)

    # compute gradient batch-wise
    grad = zeros(eltype(p_range[1]), length(pnn))
    for batch in 1:n_batches
      randint = get_batches(indices, batchsize, batch)
      data = reshape(dataset[:, randint], 3, length(randint))
      val, back = Flux.Zygote.pullback(p -> main_loss_PBM_weighted(NN, p, data, p_range, inputs), pnn)
      grad .+= back(one(val))[1]
      losses[epoch] += val
    end

    # update NN parameters based on overall gradient
    Flux.Optimise.update!(opt, pnn, grad./length(p_range))
    losses[epoch] =losses[epoch]/length(p_range)

    # keep track of best performing NN
    if epoch == 1
      best_loss = losses[epoch]
    elseif losses[epoch] < best_loss
      println("better")
      println(best_loss)
      println(losses[epoch])
      pnn_best = deepcopy(pnn)
      best_loss = losses[epoch]
    end

    # save at regular intervals
    if epoch % saveat == 0
      push!(NN_logger, pnn_best)
      push!(pred_logger, predict_PBM(dataset, p_range, pnn_best, dp, NN, batchsize, n_batches, inputs, calc_loss=true))
    end

    verbose && println("epoch: $epoch / $(epochs)")
    verbose && println("loss: "*string(losses[epoch]))
  end
  return  losses, NN_logger, pred_logger
end

# train NN in PBM based on ``stochastic'' loss function
function train_PBM_stochastic(NN,pnn, dataset, epochs, p_range, dp, opt, batchsize, batchsize_stochastic, n_batches, n_batches_train_stochastic, inputs; verbose=false, saveat=epochs)
  NN_logger = [zeros(eltype(p_range[1]), length(pnn))]
  pred_logger = [(zeros(eltype(p_range[1]), length(p_range)), zeros(eltype(p_range[1]), length(p_range)-2), zeros(eltype(p_range[1]), 1))]

  pnn_best = zeros(eltype(pnn[1]), length(pnn))
  best_loss = zero(eltype(pnn[1]))

  losses = zeros(eltype(p_range[1]), epochs)
  for epoch in 1:epochs

    # compute loss and gradient batchwise
    for batch in 1:n_batches_train_stochastic
      randint = sample(1:length(dataset[1, :]), Weights(dataset[2, :]), batchsize_stochastic)
      data = reshape(dataset[:, randint], 3, length(randint))
      val, back = Flux.Zygote.pullback(p -> main_loss_PBM_stochastic(NN, p, data, p_range, inputs), pnn)
      grad = back(one(val))[1]

      # update NN parameters based on batch gradient
      Flux.Optimise.update!(opt, pnn, grad)
      losses[epoch] += val
    end
    losses[epoch] =losses[epoch]/n_batches_train_stochastic

    # keep track of best performing NN
    if epoch == 1
      best_loss = losses[epoch]
    elseif losses[epoch] < best_loss
      println("better")
      println(best_loss)
      println(losses[epoch])
      pnn_best = deepcopy(pnn)
      best_loss = losses[epoch]
    end

    # save at regular intervals
    if epoch % saveat == 0
      push!(NN_logger, pnn_best)
      push!(pred_logger, predict_PBM(dataset, p_range, pnn_best, dp, NN, batchsize, n_batches, inputs, calc_loss=true))
    end

    verbose && println("epoch: $epoch / $(epochs)")
    verbose && println("loss: "*string(losses[epoch]))
  end
  return  losses, NN_logger, pred_logger
end

# compute predictions, indicators, and loss value based on current NN
function predict_PBM(dataset, p_range, pnn, dp, NN, batchsize, n_batches, inputs; calc_loss=false, loss=zero(eltype(p_range[1])))

  # compute predictions and loss batch-wise
  predictions = zeros(eltype(p_range[1]), length(p_range))
  indices = collect(1:size(dataset)[2])
  for batch in 1:n_batches
    rand_int = get_batches(indices, batchsize ,batch)
    data = reshape(dataset[:, rand_int], 3, length(rand_int))

    indicess = convert.(Int,data[1, :])
    input = inputs[:, indicess]

    pred = NN(pnn)(input)
    for indxp in 1:length(rand_int)
      predictions[Int(data[3, indxp])] += data[2, indxp]*pred[1, indxp]
    end

    if calc_loss
      loss += main_loss_PBM_weighted(NN, pnn, data, p_range, inputs)
    end
  end

  if calc_loss
    loss = loss/length(p_range)
  end

  # compute indicator using symmetric difference quotient
  return predictions,map(x -> x-1,(circshift(predictions, -1).-circshift(predictions, 1))./(2*dp))[2:end-1], [loss]
end

# apply PBM using neural networks
function get_indicators_PBM_numerical(pnn, NN, dataset, epochs, p_range, dp, opt, inputs; verbose=false, trained=false, saveat=epochs, batchsize=length(dataset[1,:]), stochastic=false, n_batches_train_stochastic=10, batchsize_stochastic=length(dataset[1,:]), dataset_train=dataset)

  n_batches = ceil(eltype(batchsize), size(dataset)[2]/batchsize)

  if !trained
    if stochastic
      losses, NN_logger, pred_logger = train_PBM_stochastic(NN, pnn, dataset, epochs, p_range, dp, opt, batchsize, batchsize_stochastic, n_batches, n_batches_train_stochastic, inputs, verbose=verbose, saveat=saveat)
    else
      losses, NN_logger, pred_logger = train_PBM_weighted(NN, pnn, dataset, epochs, p_range, dp, opt, batchsize, n_batches, inputs, verbose=verbose, saveat=saveat)
    end

    pred_logger = pred_logger[2:end]
    NN_logger = NN_logger[2:end]

  else
    predictions, indicator, loss = predict_PBM(dataset, p_range, pnn, dp, NN, batchsize, n_batches, inputs, calc_loss=true)
    losses = zeros(eltype(dp), epochs)
    NN_logger = [pnn]
    pred_logger = [(predictions, indicator, loss)]
  end

  return pred_logger, losses, NN_logger
end
