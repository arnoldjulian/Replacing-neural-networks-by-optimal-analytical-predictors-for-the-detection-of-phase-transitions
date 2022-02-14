# computation using analytical expression

# define cache
struct LBC_cache{cP1type, cP2type, cprobtype, cpredxtype, clabeltype, crestype}
  cP1::cP1type
  cP2::cP2type
  cprob::cprobtype
  cpredx::cpredxtype
  clabel::clabeltype
  cres::crestype
end

# get optimal predictions for specific sample
function get_pred_opt_x_LBC!(x, distribution, p_range, p_tar, LBCc)
  @unpack cP1, cP2, cprob, cpredx, clabel, cres = LBCc

  cP1[1] = zero(eltype(cP1[1]))
  cP2[1] = zero(eltype(cP2[1]))
  for p in p_range
    cprob[1] = distribution(x, p)
    if p <= p_tar
      cP1[1] += cprob[1]
    else
      cP2[1] += cprob[1]
    end
  end

  if cP1[1] + cP2[1] == zero(eltype(cP1[1]))
    return zero(eltype(cP1[1]))
  else
    # adjust depending on the choice of data labelling
    # alternative: cP2[1]/(cP1[1]+cP2[1])
    return cP1[1]/(cP1[1]+cP2[1])
  end
end

function get_e_opt_p_LBC!(samples, distribution, p_range, p_tar, LBCc)
  @unpack cP1, cP2, cprob, cpredx, clabel, cres = LBCc

  cres[1] = zero(eltype(cres[1]))
  cres[2] = zero(eltype(cres[2]))
  for p in p_range
    if p <= p_tar
      clabel[1] = one(eltype(clabel[1]))
    else
      clabel[1] = zero(eltype(clabel[1]))
    end

    for x in samples
      cpredx[1] = get_pred_opt_x_LBC!(x, distribution, p_range, p_tar, LBCc)
      cprob[1] = distribution(x, p)
      cres[2] += cprob[1]*crossentropy(cpredx[1], clabel[1])
      cres[1] += cprob[1]*min(cpredx[1], one(eltype(cpredx[1]))-cpredx[1])

    end
  end
  return nothing
end

# compute optimal indicator and loss of LBC across entire range of tuning parameter
function get_indicators_LBC_analytical(samples, distribution, p_range, p_range_LBC)

  caches=[LBC_cache([zero(eltype(p_range[1]))], [zero(eltype(p_range[1]))], [zero(eltype(p_range[1]))], [zero(eltype(p_range[1]))], [zero(eltype(p_range[1]))], [zero(eltype(p_range[1])),  zero(eltype(p_range[1]))]) for i in 1:length(p_range_LBC)]

  @sync for indxp in collect(1:length(p_range_LBC))
    Threads.@spawn get_e_opt_p_LBC!(samples, distribution, p_range, p_range_LBC[indxp], caches[indxp])
  end

  acc_LBC_opt = zeros(eltype(p_range_LBC[1]), length(p_range_LBC))
  loss_LBC_opt = zeros(eltype(p_range_LBC[1]), length(p_range_LBC))
  for indxp in collect(1:length(p_range_LBC))
    acc_LBC_opt[indxp] = 1-caches[indxp].cres[1]/length(p_range)
    loss_LBC_opt[indxp] = caches[indxp].cres[2]/length(p_range)
  end


  return acc_LBC_opt, loss_LBC_opt
end

# compute optimal indicator and loss of LBC at fixed value of tuning parameter
function get_indicators_LBC_analytical_fixed_p(samples, distribution, p_range, p_range_LBC, indxp)

  cache=LBC_cache([zero(eltype(p_range[1]))], [zero(eltype(p_range[1]))], [zero(eltype(p_range[1]))], [zero(eltype(p_range[1]))], [zero(eltype(p_range[1]))], [zero(eltype(p_range[1])), zero(eltype(p_range[1]))])

  get_e_opt_p_LBC!(samples, distribution, p_range, p_range_LBC[indxp], cache)
  acc_LBC_opt = 1-cache.cres[1]/length(p_range)
  loss_LBC_opt = cache.cres[2]/length(p_range)

  return acc_LBC_opt, loss_LBC_opt
end


# computation using neural networks


function main_loss_LBC_weighted(NN, pnn, dataset, p_tar, p_range, inputs)
  indices = convert.(Int, dataset[1,:])
  input = inputs[:, indices]

  pred = NN(pnn)(input)
  pred = Flux.softmax(pred)
  loss = zero(eltype(p_tar))
  for i in 1:size(pred)[2]
    if p_range[Int(dataset[3, i])] <= p_tar
      loss += dataset[2, i]*MLP.crossentropy(pred[1, i], one(eltype(pred[1, i])))
    else
      loss += dataset[2, i]*MLP.crossentropy(pred[1, i], zero(eltype(pred[1, i])))
    end
  end
  return loss
end

function main_loss_LBC_stochastic(NN, pnn, dataset, p_tar, p_range, inputs)
  indices = convert.(Int, dataset[1, :])
  input = inputs[:, indices]

  pred = NN(pnn)(input)
  pred = Flux.softmax(pred)
  loss = zero(eltype(p_tar))
  for i in 1:size(pred)[2]
    if p_range[Int(dataset[3, i])] <= p_tar
      loss += MLP.crossentropy(pred[1, i], one(eltype(pred[1, i])))
    else
      loss += MLP.crossentropy(pred[1, i], zero(eltype(pred[1, i])))
    end
  end
  return loss/length(dataset[1, :])
end

function train_LBC_weighted(NN, pnn, dataset, epochs, p_range, opt, p_tar, batchsize, n_batches, inputs; verbose=false, saveat=epochs)
  NN_logger = [zeros(eltype(p_range[1]), length(pnn))]
  pred_logger = [(zeros(eltype(p_range[1]), 1), zeros(eltype(p_range[1]), 1))]

  pnn_best = zeros(eltype(pnn[1]), length(pnn))
  best_loss = zero(eltype(pnn[1]))

  losses = zeros(eltype(p_range[1]), epochs)
  indices = collect(1:size(dataset)[2])
  for epoch in 1:epochs
    Random.shuffle!(indices)
    grad = zeros(eltype(p_range[1]), length(pnn))
    for batch in 1:n_batches
      randint = get_batches(indices, batchsize, batch)
      data = reshape(dataset[:, randint], 3, length(randint))
      val, back = Flux.Zygote.pullback(p -> main_loss_LBC_weighted(NN, p,data, p_tar, p_range, inputs), pnn)
      grad .+= back(one(val))[1]
      losses[epoch] += val
    end
    Flux.Optimise.update!(opt, pnn, grad./length(p_range))
    losses[epoch] = losses[epoch]/length(p_range)

    if epoch == 1
      best_loss = losses[epoch]
    elseif losses[epoch] < best_loss
      println("better")
      println(best_loss)
      println(losses[epoch])
      pnn_best = deepcopy(pnn)
      best_loss = losses[epoch]
    end

    if epoch % saveat == 0
      push!(NN_logger, pnn_best)
      push!(pred_logger, predict_LBC(dataset, p_range, pnn_best, NN, batchsize, n_batches, p_tar, inputs, calc_loss=true))
    end

    verbose && println("epoch: $epoch")
    verbose && println("loss: "*string(losses[epoch]))
  end
  return losses, NN_logger, pred_logger
end

function train_LBC_stochastic(NN, pnn, dataset, epochs, p_range, opt, p_tar, batchsize, n_batches_stochastic, n_batches, inputs; verbose=false, saveat=epochs)
  NN_logger = [zeros(eltype(p_range[1]), length(pnn))]
  pred_logger = [(zeros(eltype(p_range[1]), 1), zeros(eltype(p_range[1]), 1))]

  pnn_best = deepcopy(pnn)
  best_loss = zero(eltype(pnn[1]))

  losses = zeros(eltype(p_range[1]), epochs)
  for epoch in 1:epochs
    for batch in 1:n_batches_stochastic
      randint = sample(1:length(dataset[1, :]), Weights(dataset[2, :]), batchsize)
      data = reshape(dataset[:, randint], 3, length(randint))
      val, back = Flux.Zygote.pullback(p -> main_loss_LBC_stochastic(NN, p, data, p_tar, p_range, inputs), pnn)
      grad = back(one(val))[1]
      Flux.Optimise.update!(opt, pnn, grad)
      losses[epoch] += val
    end
    losses[epoch] = losses[epoch]/n_batches_stochastic

    if epoch == 1
      best_loss = losses[epoch]
    elseif losses[epoch] < best_loss
      println("better")
      println(best_loss)
      println(losses[epoch])
      pnn_best = deepcopy(pnn)
      best_loss = losses[epoch]
    end

    if epoch % saveat == 0
      push!(NN_logger, pnn_best)
      push!(pred_logger, predict_LBC(dataset, p_range, pnn_best, NN, batchsize, n_batches, p_tar, inputs,calc_loss=true))
    end

    verbose && println("epoch: $epoch")
    verbose && println("loss: "*string(losses[epoch]))
  end
  return losses, NN_logger, pred_logger
end

function predict_LBC(dataset, p_range, pnn, NN, batchsize, n_batches, p_tar, inputs; calc_loss=false, loss=zero(eltype(p_range[1])))
  error = zero(eltype(p_range[1]))

  indices = collect(1:size(dataset)[2])
  for batch in 1:n_batches
    randint = get_batches(indices, batchsize, batch)
    data = reshape(dataset[:, randint], 3, length(randint))

    indicess = convert.(Int, data[1, :])
    input = inputs[:, indicess]

    pred = NN(pnn)(input)
    pred = Flux.softmax(pred)
    for indxp in 1:length(randint)
      if p_range[Int(data[3, indxp])]<=p_tar
        label = one(eltype(pred[1, indxp]))
      else
        label = zero(eltype(pred[1, indxp]))
      end

      if pred[1, indxp] <= 0.5*one(eltype(pred[1, indxp]))
        binary_pred = zero(eltype(pred[1, indxp]))
      else
        binary_pred = one(eltype(pred[1, indxp]))
      end

      error += data[2, indxp]*abs(label-binary_pred)
    end


    if calc_loss
      loss += main_loss_LBC_weighted(NN, pnn, data, p_tar, p_range, inputs)
    end

  end

  if calc_loss
    loss = loss/length(p_range)
  end

  return [1-error/length(p_range)], [loss]
end

function get_indicators_LBC_numerical_fixed_p(pnn, NN, dataset, epochs, p_range, dp, opt, p_range_LBC, indx_ptar, inputs; verbose=false, trained=false, saveat=epochs, batchsize=length(dataset[1, :]), stochastic=false, n_batches_train_stochastic=10, batchsize_stochastic=length(dataset[1, :]), train_opt=false)

  n_batches = ceil(eltype(batchsize), size(dataset)[2]/batchsize)
  p_tar = p_range_LBC[indx_ptar]
  # verbose && println("p: $p_tar")

  if !trained
    if stochastic
      losses, NN_logger, pred_logger = train_LBC_stochastic(NN, pnn, dataset, epochs, p_range, deepcopy(opt), p_tar, batchsize_stochastic, n_batches_train_stochastic, n_batches, inputs, verbose=verbose, saveat=saveat)

      # train_LBC_stochastic(NN,pnn,dataset,epochs,p_range,opt,p_tar,batchsize,n_batches_stochastic,n_batches,inputs;verbose=false,saveat=epochs)

    else
      losses, NN_logger, pred_logger = train_LBC_weighted(NN, pnn, dataset, epochs, p_range, deepcopy(opt), p_tar, batchsize, n_batches, inputs, verbose=verbose, saveat=saveat)
    end

    pred_logger = pred_logger[2:end]
    NN_logger = NN_logger[2:end]

  else
    accuracy, loss = predict_LBC(dataset, p_range, pnn, NN, batchsize, n_batches, p_range_LBC[indx_ptar], inputs, calc_loss=true)

    losses = zeros(eltype(p_range[1]), epochs)
    NN_logger = [pnn]
    pred_logger = [(accuracy, loss)]

  end

  return pred_logger, losses, NN_logger
end
