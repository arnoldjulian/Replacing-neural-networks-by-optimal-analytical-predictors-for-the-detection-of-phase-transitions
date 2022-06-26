# computation using analytical expression

# compute optimal predictions and indicators, as well as optimal loss of PBM
function get_indicators_PBM_analytical(full_data, p_range, dp)

  # start parallel computation for sampled values of tuning parameter
  pred_opt = zeros(eltype(p_range[1]),size(full_data)[1])
  Threads.@threads for i in 1:size(full_data)[1]
    pred_opt[i] = sum((@view full_data[i,:]).*p_range)/(sum((@view full_data[i,:]))+eps(eltype(dp)))
  end

  mean_pred_opt = zeros(eltype(p_range[1]),length(p_range))
  loss_opt = zeros(eltype(p_range[1]),length(p_range))
  Threads.@threads for i in 1:length(p_range)
    mean_pred_opt[i] = sum((@view full_data[:,i]).*pred_opt)
    loss_opt[i] = sum((@view full_data[:,i]).*((pred_opt.-p_range[i]).^2))
  end

  # compute indicator using symmetric difference quotient
  return mean_pred_opt, map(i->((mean_pred_opt[i+1]-mean_pred_opt[i-1])/(2*dp)).-1,2:length(p_range)-1), mean(loss_opt)
end

# compute optimal predictions and indicators, as well as optimal loss of PBM without multiple threads (non-default option; useful if only a single CPU is available)
function get_indicators_PBM_analytical_not_threaded(full_data, p_range, dp)
  pred_opt = sum(full_data.*p_range',dims=2)./(sum(full_data,dims=2).+eps(eltype(dp)))
  mean_pred_opt = sum(full_data.*pred_opt,dims=1)[1,:]
  loss_opt = mean(p_range.^2) + mean(sum(full_data.*(pred_opt.^2),dims=1)) - 2*mean(mean_pred_opt.*p_range)

  # compute indicator using symmetric difference quotient
  return mean_pred_opt, map(i->((mean_pred_opt[i+1]-mean_pred_opt[i-1])/(2*dp)).-1,2:length(p_range)-1), loss_opt
end

# computation using neural networks

# loss function where each prediction is weighted by probability associated with input
function main_loss_PBM(NN,pnn,data,p_range,inputs)
  pred = @view NN(pnn)(inputs)[1,:]

  return mean(p_range.^2) + mean(sum(data.*(pred.^2),dims=1)) - 2*mean((@view sum(data.*pred,dims=1)[1,:]).*p_range)
end

# train NN in PBM
function train_PBM(NN, pnn, data, epochs, p_range, dp, opt, inputs; verbose=false, saveat=epochs, savebest=true, lambda=zero(eltype(p_range[1])))
  NN_logger = [zeros(eltype(p_range[1]), length(pnn))]
  pred_logger = [(zeros(eltype(p_range[1]), length(p_range)), zeros(eltype(p_range[1]), length(p_range)-2), zeros(eltype(p_range[1]), 1))]

  pnn_best = zeros(eltype(pnn[1]), length(pnn))
  best_loss = zero(eltype(pnn[1]))

  losses = zeros(eltype(p_range[1]), epochs)
  for epoch in 1:epochs

    # compute loss and gradient
    val, back = Flux.Zygote.pullback(p -> main_loss_PBM(NN, p, data, p_range, inputs), pnn)
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
      push!(pred_logger, predict_PBM(data, p_range, pnn_best, dp, NN, inputs, calc_loss=false, loss=losses[epoch], lambda=lambda))
    end

    verbose && println("epoch: $epoch / $(epochs)")
    verbose && println("loss: "*string(losses[epoch]))
  end
  return  losses, NN_logger, pred_logger
end

# compute predictions, indicators, and loss value based on current NN
function predict_PBM(data, p_range, pnn, dp, NN, inputs; calc_loss=false, loss=zero(eltype(p_range[1])), lambda=zero(eltype(p_range[1])))

  # compute predictions, indicator, and loss
  pred = @view NN(pnn)(inputs)[1,:]
  mean_pred = sum(data.*pred,dims=1)[1,:]
  indicator = map(i->((mean_pred[i+1]-mean_pred[i-1])/(2*dp)).-1,2:length(p_range)-1)
  if calc_loss
    loss = mean(p_range.^2) + mean(sum(data.*(pred.^2),dims=1)) - 2*mean(mean_pred.*p_range) + L2_penalty(lambda, pnn)
  end

  return mean_pred, indicator, [loss]
end

# perform PBM using neural networks
function get_indicators_PBM_numerical(pnn, NN, data, epochs, p_range, dp, opt, inputs; verbose=false, trained=false, saveat=epochs, savebest=true, lambda=zero(eltype(p_range[1])))

  if !trained
    losses, NN_logger, pred_logger = train_PBM(NN, pnn, data, epochs, p_range, dp, opt, inputs, verbose=verbose, saveat=saveat, savebest=savebest, lambda=lambda)

    pred_logger = pred_logger[2:end]
    NN_logger = NN_logger[2:end]

  else
    predictions, indicator, loss = predict_PBM(data, p_range, pnn, dp, NN, inputs, calc_loss=true, lambda=lambda)
    losses = zeros(eltype(dp), epochs)
    NN_logger = [pnn]
    pred_logger = [(predictions, indicator, loss)]
  end

  return pred_logger, losses, NN_logger
end
