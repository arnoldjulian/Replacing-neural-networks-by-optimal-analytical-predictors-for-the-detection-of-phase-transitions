# computation using analytical expression

# compute optimal predictions and indicators, as well as optimal loss of SL
function get_indicators_SL_analytical(data, p_range, dp, p_min_indx, p_max_indx)
  p1 = @view sum(data[:,1:p_min_indx],dims=2)[:,1]
  p2 = @view sum(data[:,p_max_indx:end],dims=2)[:,1]

  pred = [if p1[i]+p2[i] != zero(eltype(p_range[1])) p1[i]/(p1[i]+p2[i]) else zero(eltype(p_range[1])) end for i in 1:size(data)[1]]


  loss = sum((crossentropy.(pred,1)')*(@view data[:,1:p_min_indx])) + sum((crossentropy.(pred,0)')*(@view data[:,p_max_indx:length(p_range)]))


  pred_SL_opt = sum(data.*pred,dims=1)[1,:]

  # compute indicator using symmetric difference quotient
  ind_SL_opt = zeros(eltype(p_range[1]),length(p_range)-2)
  for i in 2:length(p_range)-1
    ind_SL_opt[i-1] = -1*(pred_SL_opt[i+1]-pred_SL_opt[i-1])/(2*dp)
  end

  return pred_SL_opt, ind_SL_opt, loss/(p_min_indx+length(p_range)-p_max_indx)
end

# compute optimal predictions and indicators, as well as optimal loss of SL with multiple threads (non-default option; useful if training region is large)
function get_indicators_SL_analytical_threaded(data, p_range, dp, p_min_indx, p_max_indx)
  p1 = @view sum(data[:,1:p_min_indx],dims=2)[:,1]
  p2 = @view sum(data[:,p_max_indx:end],dims=2)[:,1]

  pred = [if p1[i]+p2[i] != zero(eltype(p_range[1])) p1[i]/(p1[i]+p2[i]) else zero(eltype(p_range[1])) end for i in 1:size(data)[1]]


  # start parallel computation for sampled values of tuning parameter
  loss = zero(eltype(p_range[1]))
  Threads.@threads for i in 1:p_min_indx
    loss += sum((crossentropy.(pred,1)')*data[:,i])
  end

  Threads.@threads for i in p_max_indx:length(p_range)
    loss += sum((crossentropy.(pred,0)')*data[:,i])
  end

  pred_SL_opt = zeros(eltype(p_range[1]),length(p_range))
  Threads.@threads for i in 1:length(p_range)
    pred_SL_opt[i] = sum(data[:,i].*pred)
  end

  # compute indicator using symmetric difference quotient
  ind_SL_opt = zeros(eltype(p_range[1]),length(p_range)-2)
  for i in 2:length(p_range)-1
    ind_SL_opt[i-1] = -1*(pred_SL_opt[i+1]-pred_SL_opt[i-1])/(2*dp)
  end

  return pred_SL_opt, ind_SL_opt, loss/(p_min_indx+length(p_range)-p_max_indx)
end

# computation using neural networks

# loss function where each prediction is weighted by probability associated with input
function main_loss_SL(NN,pnn,data_train,p_range,p_min_indx,p_max_indx,inputs, len_p_train)
  pred = @view Flux.softmax(NN(pnn)(inputs))[1,:]
  return (sum((@view data_train[:,1:p_min_indx]).*crossentropy.(pred,ones(eltype(pred[1]),length(pred)))) + sum((@view data_train[:,p_min_indx+1:p_min_indx+1+(length(p_range)-p_max_indx)]).*crossentropy.(pred,zeros(eltype(pred[1]),length(pred)))))/len_p_train
end

# train NN in SL
function train_SL(NN, pnn, data_train, data_test, epochs, p_range, dp, p_min_indx, p_max_indx, opt, len_p_train, inputs; verbose=false, saveat=epochs, savebest=true, lambda=zero(eltype(p_range[1])))
  NN_logger = [zeros(eltype(p_range[1]), length(pnn))]
  pred_logger = [(zeros(eltype(p_range[1]), length(p_range)), zeros(eltype(p_range[1]), length(p_range)-2), zeros(eltype(p_range[1]), 1))]

  pnn_best = zeros(eltype(pnn[1]), length(pnn))
  best_loss = zero(eltype(pnn[1]))

  losses = zeros(eltype(p_range[1]), epochs)
  indices = collect(1:size(data_train)[2])
  for epoch in 1:epochs

    # compute loss and gradient
    val, back = Flux.Zygote.pullback(p -> main_loss_SL(NN, p, data_train, p_range, p_min_indx, p_max_indx, inputs, len_p_train), pnn)

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
      push!(pred_logger, predict_SL(data_train, data_test, p_range, p_min_indx, p_max_indx, dp, NN, pnn_best, len_p_train, inputs, calc_loss=false, loss=losses[epoch], lambda=lambda))
    end

    verbose && println("epoch: $epoch / $(epochs)")
    verbose && println("loss: "*string(losses[epoch]))
  end

  return losses, NN_logger, pred_logger
end

# compute predictions, indicators, and loss value based on current NN
function predict_SL(data_train, data_test, p_range, p_min_indx, p_max_indx, dp, NN, pnn, len_p_train, inputs; calc_loss=false, loss=zero(eltype(p_range[1])), lambda=zero(eltype(p_range[1])))

  # compute predictions
  pred = @view Flux.softmax(NN(pnn)(inputs))[1,:]
  pred_SL = sum(data_test.*pred,dims=1)[1,:]

  # compute indicator using symmetric difference quotient
  ind_SL = zeros(eltype(p_range[1]),length(p_range)-2)
  for i in 2:length(p_range)-1
      ind_SL[i-1] = -1*(pred_SL[i+1]-pred_SL[i-1])/(2*dp)
  end

  # compute loss
  if calc_loss
    loss = main_loss_SL(NN, pnn, data_train, p_range, p_min_indx, p_max_indx, inputs, len_p_train) + L2_penalty(lambda, pnn)
  end

  return pred_SL, ind_SL, [loss]
end

# perform SL using neural networks
function get_indicators_SL_numerical(pnn, NN, data_train, data_test, epochs, p_range, dp, p_min_indx, p_max_indx, opt, inputs; verbose=false, trained=false, saveat=epochs, savebest=true, lambda=zero(eltype(p_range[1])))

  len_p_train = p_min_indx + (length(p_range)-p_max_indx)

  if !trained
    losses, NN_logger, pred_logger = train_SL(NN, pnn, data_train, data_test, epochs, p_range, dp, p_min_indx, p_max_indx, opt, len_p_train, inputs, verbose=verbose, saveat=saveat, savebest=savebest, lambda=lambda)

    pred_logger = pred_logger[2:end]
    NN_logger = NN_logger[2:end]

  else
    predictions, indicator, loss = predict_SL(data_train, data_test, p_range, p_min_indx, p_max_indx, dp, NN, pnn, len_p_train, inputs, calc_loss=true, lambda=lambda)

    losses = zeros(eltype(dp), epochs)
    NN_logger = [pnn]
    pred_logger = [(predictions, indicator, loss)]
  end

  return pred_logger, losses, NN_logger
end
