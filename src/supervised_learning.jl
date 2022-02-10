struct SL_cache{cP1type,cP2type,cprobtype,cpredxtype,cweighttype,clabeltype,crestype}
    cP1::cP1type
    cP2::cP2type
    cprob::cprobtype
    cpredx::cpredxtype
    cweight::cweighttype
    clabel::clabeltype
    cres::crestype
end

# analytical part
function get_pred_opt_x_SL!(x, distribution, p_range_1, p_range_2,SLc)
    @unpack cP1,cP2,cprob,cpredx,cweight,clabel,cres = SLc

    cP1[1] = zero(eltype(cP1))
    cP2[1] = zero(eltype(cP2))

    for p in p_range_1
        cP1[1] +=distribution(x,p)
    end

    for p in p_range_2
        cP2[1] +=distribution(x,p)
    end

    if cP1[1]+cP2[1] == zero(eltype(cP2[1]))
        return zero(eltype(cP2[1]))
    else
        return cP1[1]/(cP1[1]+cP2[1])
        # return cP2[1]/(cP1[1]+cP2[1])
    end
end

function l_add!(p_min, p_max, p_tar, cweight,clabel)
    if p_tar <=p_min
        cweight[1] = one(eltype(cweight[1]))
        clabel[1]  = one(eltype(clabel[1]))
    elseif p_tar >=p_max
        cweight[1] = one(eltype(cweight[1]))
        clabel[1]  = zero(eltype(clabel[1]))
    else
        cweight[1] = zero(eltype(cweight[1]))
        clabel[1]  = zero(eltype(clabel[1]))
    end
    return nothing
end

function get_pred_opt_p_SL!(samples, distribution, p_range_1, p_range_2, p_tar,SLc)
    @unpack cP1,cP2,cprob,cpredx,cweight,clabel,cres = SLc

    cres[1] = zero(eltype(cres[1]))

    l_add!(p_range_1[end], p_range_2[1], p_tar,cweight,clabel)
    cres[3] = cres[3]+cweight[1]
    for x in samples
        cprob[1] = distribution(x,p_tar)
        cpredx[1] = get_pred_opt_x_SL!(x, distribution, p_range_1, p_range_2, SLc)
        if cpredx[1] == zero(eltype(cres[1]))
            # println(cprob[1])
        end
        cres[1] += cprob[1]*cpredx[1]

        # loss += prob*(pred_x-label)^2
        # loss += prob*Flux.crossentropy([pred_x,1-pred_x], [label,1-label])
        cres[2]+=cweight[1]*cprob[1]*crossentropy(cpredx[1],clabel[1])
    end
    return nothing
end

function get_indicators_SL_analytical(samples, distribution, p_range, dp, p_min, p_max)
    p_range_1 = Tuple(collect(p_range[1]:dp:p_min))
    if p_range[end] == Inf
        p_range_2 = Tuple(p_range[end])
    else
        p_range_2 = Tuple(collect(p_max:dp:p_range[end]))
    end

    pred_SL_opt = zeros(eltype(dp),length(p_range))
    loss = zero(eltype(dp))

    # cP1 = [zero(eltype(dp))]
    # cP2 = [zero(eltype(dp))]
    # cprob = [zero(eltype(dp))]
    # cpredx = [zero(eltype(dp))]
    # cweight = [zero(eltype(dp))]
    # clabel = [zero(eltype(dp))]
    # cres = [zero(eltype(dp)), zero(eltype(dp)), zero(eltype(dp))]

    caches=[SL_cache([zero(eltype(dp))],[zero(eltype(dp))],[zero(eltype(dp))],[zero(eltype(dp))],[zero(eltype(dp))],[zero(eltype(dp))],[zero(eltype(dp)), zero(eltype(dp)), zero(eltype(dp))]) for i in 1:Threads.nthreads()]

    Threads.@threads for indxp in collect(1:length(p_range))
        get_pred_opt_p_SL!(samples, distribution, p_range_1, p_range_2, p_range[indxp], caches[Threads.threadid()])
        pred_SL_opt[indxp] = caches[Threads.threadid()].cres[1]
    end

    cres2 = zero(eltype(dp))
    cres3 = zero(eltype(dp))
    for i in 1:Threads.nthreads()
        cres2 += caches[i].cres[2]
        cres3 +=caches[i].cres[3]
    end


    # caches=[SL_cache([zero(eltype(dp))],[zero(eltype(dp))],[zero(eltype(dp))],[zero(eltype(dp))],[zero(eltype(dp))],[zero(eltype(dp))],[zero(eltype(dp)), zero(eltype(dp)), zero(eltype(dp))]) for i in 1:length(p_range)]
    #
    # @sync for indxp in collect(1:length(p_range))
    #     Threads.@spawn get_pred_opt_p_SL!(samples, distribution, p_range_1, p_range_2, p_range[indxp], caches[indxp])
    # end
    #
    # cres2 = zero(eltype(dp))
    # cres3 = zero(eltype(dp))
    # for indxp in collect(1:length(p_range))
    #     pred_SL_opt[indxp] = caches[indxp].cres[1]
    #     cres2 += caches[indxp].cres[2]
    #     cres3 += caches[indxp].cres[3]
    # end

    return pred_SL_opt, (-1*(circshift(pred_SL_opt, -1).-circshift(pred_SL_opt, 1))./(2*dp))[2:end-1], cres2/cres3
end

#old function for separate calculation of the loss
function get_loss_opt_SL(samples, distribution, p_range, p_min, p_max, loss_type)
    p_min_indx = findall(x->x==p_min, p_range)[1]
    p_max_indx = findall(x->x==p_max, p_range)[1]
    p_min_range = collect(1:p_min_indx)
    p_max_range = collect(p_max_indx:length(p_range))
    ranges = [p_min_range,p_max_range]

    loss = zero(eltype(p_max))
    for range_indx in collect(1:2)
        range = ranges[range_indx]
        for i in range
            p=p_range[i]
            loss_p = zero(eltype(p_max))

            for x in samples
                pred = get_pred_opt_x_SL(x, distribution, p_range, p_min, p_max)
                if range_indx == 1
                    label = one(eltype(pred))
                else
                    label = zero(eltype(pred))
                end

                if loss_type == "MSE"
                    loss_x = (pred-label)^2
                elseif loss_type == "CE"
                    loss_x = Flux.crossentropy([pred,1-pred], [label,1-label])

                else
                    error("Your loss function is currently not supported.")
                end

                loss_p +=distribution(x,p)*loss_x
            end
            loss+=loss_p
        end
    end

    return loss/(length(p_min_range)+length(p_max_range))
end

# numerical part

# implement training points apart from direct boundaries
# need to be able to save trained NNs and re-evaluate without retraining (also save parameters during training)
# implement batchwise training
function main_loss_SL_weighted(NN,pnn,data,p_range,p_min,p_max,inputs)
    indices = convert.(Int,data[1,:])
    input = inputs[:,indices]

    pred = NN(pnn)(input)
    pred=Flux.softmax(pred)
    loss = zero(eltype(p_range[1]))
    for i in 1:size(pred)[2]
        if p_range[Int(data[3,i])] <=p_min
            loss+= data[2,i]*MLP.crossentropy(pred[1,i],one(eltype(loss)))
        elseif p_range[Int(data[3,i])] >=p_max
            loss+= data[2,i]*MLP.crossentropy(pred[1,i],zero(eltype(loss)))
        else
            error("wrong asignment of training data")
        end
    end
    # return loss + 0.0001*sum(MLP.sqnorm, pnn)
    return loss
end

function main_loss_SL_stochastic(NN,pnn,data,p_range,p_min,p_max,inputs)
    indices = convert.(Int,data[1,:])
    input = inputs[:,indices]

    pred = NN(pnn)(input)
    pred=Flux.softmax(pred)
    loss = zero(eltype(p_range[1]))
    for i in 1:size(pred)[2]
        if p_range[Int(data[3,i])] <=p_min
            loss+= MLP.crossentropy(pred[1,i],one(eltype(loss)))
        elseif p_range[Int(data[3,i])] >=p_max
            loss+= MLP.crossentropy(pred[1,i],zero(eltype(loss)))
        else
            error("wrong asignment of training data")
        end
    end
    # return loss + 0.0001*sum(MLP.sqnorm, pnn)
    return loss/length(data[3,:])
end

function main_loss_SL_opt(NN,pnn,data,p_range,p_min,p_max,inputs)
    indices = convert.(Int,data[1,:])
    input = inputs[:,indices]

    pred = NN(pnn)(input)
    pred=Flux.softmax(pred)
    loss = zero(eltype(p_range[1]))
    for i in 1:size(pred)[2]
        loss+= MLP.crossentropy(pred[1,i],data[2,i])
    end
    # return loss + 0.0001*sum(MLP.sqnorm, pnn)
    return loss/length(data[1,:])
end

function train_SL_weighted(NN,pnn,data_train,data_test,epochs,p_range,dp,p_min,p_max,opt,batchsize,n_batches_train,n_batches_test,len_p_train,inputs;verbose=false,saveat=epochs)
    NN_logger = [zeros(eltype(p_range[1]),length(pnn))]
    pred_logger = [(zeros(eltype(p_range[1]),length(p_range)),zeros(eltype(p_range[1]),length(p_range)-2),zeros(eltype(p_range[1]),1))]

    pnn_best = zeros(eltype(pnn[1]),length(pnn))
    best_loss = zero(eltype(pnn[1]))

    losses = zeros(eltype(p_range[1]),epochs)
    indices = collect(1:size(data_train)[2])
    for epoch in 1:epochs
        Random.shuffle!(indices)
        grad = zeros(eltype(p_range[1]),length(pnn))
        for batch in 1:n_batches_train
            randint = get_batches(indices,batchsize,batch)
            data = reshape(data_train[:,randint],3,length(randint))
            val, back = Flux.Zygote.pullback(p -> main_loss_SL_weighted(NN,p,data,p_range,p_min,p_max,inputs), pnn)
            grad .+= back(one(val))[1]
            losses[epoch] += val
        end
        Flux.Optimise.update!(opt, pnn, grad./len_p_train)
        losses[epoch] =losses[epoch]/len_p_train

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
            push!(NN_logger,pnn_best)
            push!(pred_logger,predict_SL(data_train,data_test,p_range,p_min,p_max,dp,NN,pnn_best,batchsize,n_batches_train,n_batches_test,len_p_train,inputs,calc_loss=true))
        end

        verbose && println("epoch: $epoch / $(epochs)")
        verbose && println("loss: "*string(losses[epoch]))
    end
    return losses, NN_logger, pred_logger
end

function train_SL_stochastic(NN,pnn,data_train,data_test,epochs,p_range,dp,p_min,p_max,opt,batchsize,batchsize_stochastic,n_batches_train,n_batches_train_stochastic,n_batches_test,len_p_train,inputs;verbose=false,saveat=epochs)
    NN_logger = [zeros(eltype(p_range[1]),length(pnn))]
    pred_logger = [(zeros(eltype(p_range[1]),length(p_range)),zeros(eltype(p_range[1]),length(p_range)-2),zeros(eltype(p_range[1]),1))]

    pnn_best = zeros(eltype(pnn[1]),length(pnn))
    best_loss = zero(eltype(pnn[1]))

    losses = zeros(eltype(p_range[1]),epochs)
    for epoch in 1:epochs
        for batch in 1:n_batches_train_stochastic
            randint = sample(1:length(data_train[1,:]),Weights(data_train[2,:]),batchsize_stochastic)
            data = reshape(data_train[:,randint],3,length(randint))
            val, back = Flux.Zygote.pullback(p -> main_loss_SL_stochastic(NN,p,data,p_range,p_min,p_max,inputs), pnn)
            grad = back(one(val))[1]
            Flux.Optimise.update!(opt, pnn, grad)
            losses[epoch] += val
        end
        losses[epoch] =losses[epoch]/n_batches_train_stochastic

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
            push!(NN_logger,pnn_best)
            push!(pred_logger,predict_SL(data_train,data_test,p_range,p_min,p_max,dp,NN,pnn_best,batchsize,n_batches_train,n_batches_test,len_p_train,inputs,calc_loss=true))
        end

        verbose && println("epoch: $epoch / $(epochs)")
        verbose && println("loss: "*string(losses[epoch]))
    end
    return losses, NN_logger, pred_logger
end

function train_SL_opt(NN,pnn,data_train,data_test,epochs,p_range,dp,p_min,p_max,opt,batchsize,batchsize_stochastic,n_batches_train,n_batches_train_stochastic,n_batches_test,len_p_train,inputs;verbose=false,saveat=epochs)
    NN_logger = [zeros(eltype(p_range[1]),length(pnn))]
    pred_logger = [(zeros(eltype(p_range[1]),length(p_range)),zeros(eltype(p_range[1]),length(p_range)-2),zeros(eltype(p_range[1]),1))]

    pnn_best = zeros(eltype(pnn[1]),length(pnn))
    best_loss = zero(eltype(pnn[1]))

    losses = zeros(eltype(p_range[1]),epochs)
    for epoch in 1:epochs
        val, back = Flux.Zygote.pullback(p -> main_loss_SL_opt(NN,p,data_train,p_range,p_min,p_max,inputs), pnn)
        grad = back(one(val))[1]
        Flux.Optimise.update!(opt, pnn, grad)

        losses[epoch] = val

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
            push!(NN_logger,pnn_best)
            push!(pred_logger,predict_SL(data_train,data_test,p_range,p_min,p_max,dp,NN,pnn_best,batchsize,n_batches_train,n_batches_test,len_p_train,inputs,calc_loss=true))
        end

        verbose && println("epoch: $epoch / $(epochs)")
        verbose && println("loss: "*string(losses[epoch]))
    end
    return losses, NN_logger, pred_logger
end

function predict_SL(data_train,data_test,p_range,p_min,p_max,dp,NN,pnn,batchsize,n_batches_train,n_batches_test,len_p_train,inputs;calc_loss=false,loss=zero(eltype(p_range[1])))

    predictions = zeros(eltype(p_range[1]),length(p_range))
    indices = collect(1:size(data_test)[2])
    for batch in 1:n_batches_test
        rand_int = get_batches(indices,batchsize,batch)
        data = reshape(data_test[:,rand_int],3,length(rand_int))

        indicess = convert.(Int,data[1,:])
        input = inputs[:,indicess]

        pred = NN(pnn)(input)
        pred=Flux.softmax(pred)
        for indxp in 1:length(rand_int)
            predictions[Int(data[3,indxp])] += data[2,indxp]*pred[1,indxp]
        end
    end

    if calc_loss
        indices = collect(1:size(data_train)[2])
        for batch in 1:n_batches_train
            rand_int = get_batches(indices,batchsize,batch)
            data = reshape(data_train[:,rand_int],3,length(rand_int))
            loss += main_loss_SL_weighted(NN,pnn,data,p_range,p_min,p_max,inputs)
        end
        loss = loss/len_p_train
    end

    return predictions, -1*((circshift(predictions, -1).-circshift(predictions, 1))./(2*dp))[2:end-1], [loss]
end

function get_indicators_SL_numerical(pnn, NN,data_train,data_test,epochs,p_range,dp,p_min,p_max,opt,inputs;verbose=false,trained=false,saveat=epochs,batchsize=length(data_train[1,:]),stochastic=false,n_batches_train_stochastic=1,batchsize_stochastic=length(data_train[1,:]),train_opt=false)

    n_batches_train = ceil(eltype(batchsize),size(data_train)[2]/batchsize)
    n_batches_test = ceil(eltype(batchsize),size(data_test)[2]/batchsize)
    len_p_train = length(collect(p_range[1]:dp:p_min))+length(collect(p_max:dp:p_range[end]))

    if !trained
        if stochastic
            losses, NN_logger, pred_logger = train_SL_stochastic(NN,pnn,data_train,data_test,epochs,p_range,dp,p_min,p_max,opt,batchsize,batchsize_stochastic,n_batches_train,n_batches_train_stochastic,n_batches_test,len_p_train,inputs,verbose=verbose,saveat=saveat)
        elseif train_opt
            losses, NN_logger, pred_logger = train_SL_opt(NN,pnn,data_train,data_test,epochs,p_range,dp,p_min,p_max,opt,batchsize,batchsize_stochastic,n_batches_train,n_batches_train_stochastic,n_batches_test,len_p_train,inputs,verbose=verbose,saveat=saveat)
        else
            losses, NN_logger, pred_logger = train_SL_weighted(NN,pnn,data_train,data_test,epochs,p_range,dp,p_min,p_max,opt,batchsize,n_batches_train,n_batches_test,len_p_train,inputs,verbose=verbose,saveat=saveat)
        end

        pred_logger = pred_logger[2:end]
        NN_logger = NN_logger[2:end]

        # predictions, indicator = predict_SL(data_test,p_range,trained_pnn,dp,NN,batchsize,n_batches_test)
    else
        predictions, indicator, loss = predict_SL(data_train,data_test,p_range,p_min,p_max,dp,NN,pnn,batchsize,n_batches_train,n_batches_test,len_p_train,inputs,calc_loss=true)

        losses = zeros(eltype(dp),epochs)
        NN_logger = [pnn]
        pred_logger = [(predictions,indicator,loss)]
    end

    return pred_logger, losses, NN_logger
end
