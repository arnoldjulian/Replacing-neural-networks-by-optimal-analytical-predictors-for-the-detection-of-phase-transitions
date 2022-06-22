# computation using analytical expression

# get optimal indicator and loss for a specific bipartition point
function get_indicators_LBC_analytical_fixed_p(data, p_range, p_tar_indx)
  p1 = sum((@view data[:,1:p_tar_indx-1]),dims=2)[:,1]
  p2 = sum((@view data[:,p_tar_indx:end]),dims=2)[:,1]
  pred_opt = p1./(p1.+p2)

  error = sum(data.*(min.(pred_opt,ones(eltype(p_range[1]),length(pred_opt)).-pred_opt)))
  loss = sum((crossentropy.(pred_opt, 1)')*(@view data[:, 1:p_tar_indx-1])) + sum((crossentropy.(pred_opt, 0)')*(@view data[:, p_tar_indx:length(p_range)]))

  return 1-error/length(p_range), loss/length(p_range)
end

# compute optimal indicator and loss of LBC across entire range of tuning parameter
function get_indicators_LBC_analytical(data, p_range)
  acc_LBC_opt = zeros(eltype(p_range[1]),length(p_range)+1)
  loss_LBC_opt = zeros(eltype(p_range[1]),length(p_range)+1)

  # start parallel computation for sampled values of tuning parameter
  Threads.@threads for p_tar_indx in 1:length(p_range)+1
    acc_LBC_opt[p_tar_indx], loss_LBC_opt[p_tar_indx] = get_indicators_LBC_analytical_fixed_p(data, p_range, p_tar_indx)
  end

  return acc_LBC_opt, loss_LBC_opt
end


# computation using neural networks

# loss function where each prediction is weighted by probability associated with input
function main_loss_LBC(NN, pnn, data, p_tar_indx, p_range, inputs)
  pred = @view Flux.softmax(NN(pnn)(inputs))[1,:]

  return (sum((crossentropy.(pred, 1)')*(@view data[:, 1:p_tar_indx-1])) + sum((crossentropy.(pred, 0)')*(@view data[:, p_tar_indx:length(p_range)])))/length(p_range)
end

# train NN in LBC
function train_LBC(NN, pnn, data, epochs, p_range, opt, p_tar_indx, inputs; verbose=false, saveat=epochs, savebest=true, lambda=zero(eltype(p_range[1])))
  NN_logger = [zeros(eltype(p_range[1]), length(pnn))]
  pred_logger = [(zeros(eltype(p_range[1]), 1), zeros(eltype(p_range[1]), 1))]

  pnn_best = zeros(eltype(pnn[1]), length(pnn))
  best_loss = zero(eltype(pnn[1]))

  losses = zeros(eltype(p_range[1]), epochs)
  for epoch in 1:epochs

    # compute loss and gradient
    val, back = Flux.Zygote.pullback(p -> main_loss_LBC(NN, p, data, p_tar_indx, p_range, inputs), pnn)
    val_reg, back_reg = Flux.Zygote.pullback(p -> L2_penalty(lambda, p), pnn)
    grad = back(one(val))[1].+back_reg(one(val_reg))[1]
    losses[epoch] = val+val_reg

    # update NN parameters based on overall gradient
    Flux.Optimise.update!(opt, pnn, grad)

    # keep track of best performing NN
    if savebest
      if epoch == 1
        best_loss = losses[epoch]
      elseif losses[epoch] < best_loss
        verbose && println("better")
        verbose && println(best_loss)
        verbose && println(losses[epoch])
        pnn_best = deepcopy(pnn)
        best_loss = losses[epoch]
      end
    else
      best_loss = losses[epoch]
      pnn_best = deepcopy(pnn)
    end

    # save at regular intervals
    if epoch % saveat == 0
      push!(NN_logger, pnn_best)
      push!(pred_logger, predict_LBC(data, p_range, pnn_best, NN, p_tar_indx, inputs, calc_loss=false, loss=losses[epoch], lambda=lambda))
    end

    verbose && println("epoch: $epoch")
    verbose && println("loss: "*string(losses[epoch]))
  end
  return losses, NN_logger, pred_logger
end

# compute indicator and loss value based on current NN
function predict_LBC(data, p_range, pnn, NN, p_tar_indx, inputs; calc_loss=false, loss=zero(eltype(p_range[1])), lambda=zero(eltype(p_range[1])))
  pred = @view Flux.softmax(NN(pnn)(inputs))[1,:]

  binary_pred = [if x <= one(eltype(x))/2 zero(eltype(x)) else one(eltype(x)) end for x in pred]
  labels = [if i <= p_tar_indx-1 one(eltype(p_range[1])) else zero(eltype(p_range[1])) end for i in 1:length(p_range)]

  error = zero(eltype(p_range[1]))
  for i in 1:length(p_range)
    error += sum((@view data[:,i]).*(abs.(binary_pred.-labels[i])))
  end

  if calc_loss
    loss = sum((crossentropy.(pred, 1)')*(@view data[:, 1:p_tar_indx-1])) + sum((crossentropy.(pred, 0)')*(@view data[:, p_tar_indx:length(p_range)]))/length(p_range) + L2_penalty(lambda, pnn)
  end

  return [1-error/length(p_range)], [loss]
end

# perform LBC for fixed bipartition point using neural networks
function get_indicators_LBC_numerical_fixed_p(pnn, NN, data, epochs, p_range, dp, opt, p_range_LBC, p_tar_indx, inputs; verbose=false, trained=false, saveat=epochs, savebest=true, lambda=zero(eltype(p_range[1])))

  if !trained
    losses, NN_logger, pred_logger = train_LBC(NN, pnn, data, epochs, p_range, opt, p_tar_indx, inputs, verbose=verbose, saveat=saveat, savebest=savebest, lambda=lambda)

    pred_logger = pred_logger[2:end]
    NN_logger = NN_logger[2:end]

  else
    accuracy, loss = predict_LBC(data, p_range, pnn, NN, p_tar_indx, inputs, calc_loss=true, lambda=lambda)

    losses = zeros(eltype(p_range[1]), epochs)
    NN_logger = [pnn]
    pred_logger = [(accuracy, loss)]

  end

  return pred_logger, losses, NN_logger
end
