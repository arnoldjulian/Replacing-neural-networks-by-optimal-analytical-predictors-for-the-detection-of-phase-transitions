function jump_distr(;p_crit=1.0f0)
    function stepfunc(x,p,p_crit)
        if p <= p_crit
            if x == zero(eltype(x))
                return one(eltype(p))
            else
                return zero(eltype(p))
            end
        else
            if x == zero(eltype(x))
                return zero(eltype(p))
            else
                return one(eltype(p))
            end
        end
    end
    samples=[0,1]
    return (x,p)->stepfunc(x,p,p_crit), samples
end

function constant_distr()
    function uniform(x,p)
        return 0.5f0
    end
    samples=[0,1]
    return uniform, samples
end

function continuous_distr()
    function paramagnet_distr(x,p)
        if x == 0
            return 1.0-0.5*(tanh(p)+1)
        else
            return 0.5*(tanh(p)+1)
        end
    end
    samples=[0,1]
    return paramagnet_distr, samples
end

function tilt_distr(p_range;p_crit=2.0f0)
    function tilt(p,p_crit)
        a_left = -1.0f0
        b_left = 5.0f0
        a_right = - 1.0f0
        b_right = 2.0f0
        if p <= p_crit
            x=a_left*p+b_left
        else
            x=a_right*p+b_right
        end

        return x
    end
    samples = map(p->tilt(p,p_crit),p_range)

    function tilt_distr(x,p,p_range)
        target = samples[findall(x->x==p, p_range)[1]]
        if target == x
            return one(eltype(p))
        else
            return zero(eltype(p))
        end
    end
    return (x,p)->tilt_distr(x,p,p_range), samples
end

function boltzmann_factor_energy(energy, T)
    if T == Inf
        return one(eltype(T))
    else
        return exp(-energy/T)
    end
end

function get_probabilities(p_range,energies)
    probs = zeros(eltype(p_range[1]),length(p_range),size(energies)[1])
    for indx_p in 1:length(p_range)
        p = p_range[indx_p]
        partition_func = zero(eltype(p))
        for i in 1:size(energies)[1]
            energy = energies[i,1]
            num = energies[i,2]
            probs[indx_p,i] = num*boltzmann_factor_energy(energy, p)
            partition_func += probs[indx_p,i]
        end
        if partition_func == Inf
            probs[indx_p,:] = zeros(eltype(p_range[1]),length(probs[indx_p,:]))
        else
            probs[indx_p,:] = probs[indx_p,:]./partition_func
        end
    end
    return probs
end

function thermal_distr(x,p,probs,p_range)
    if p == Inf
        p_indx = length(p_range)
    else
        p_indx = Int(round((p-p_range[1])/(p_range[2]-p_range[1])))+1
    end

    partition_func = probs[p_indx,Int(x)]
    return partition_func
end

function ising_exact_distr(energies,p_range)
    probs = get_probabilities(p_range,energies)
    return (x,p)->thermal_distr(x,p,probs,p_range)
end

function distr_approx_x_p(full_data,p_range,x,p)
    p_indx = Int(round((p-p_range[1])/(p_range[2]-p_range[1])))+1
    return full_data[p_indx,x]
end

function distr_approx(energies,unique_energies,numbers,p_range)
    full_data =zeros(eltype(p_range[1]),(length(p_range),length(unique_energies)))
    for i in 1:length(p_range)
        numtot = zero(eltype(p_range[1]))
        for j in 1:length(unique_energies)
            number_list = numbers[i]
            energy_list = energies[i]
            sample = unique_energies[j]
            indices = findall(x->x==sample, energy_list)
            if length(indices) == 0
                full_data[i,j] = zero(eltype(sample))
            else
                full_data[i,j] = number_list[indices[1]]
                numtot+=number_list[indices[1]]
            end
        end
        full_data[i,:] = full_data[i,:]/numtot
    end

    return (x,p)->distr_approx_x_p(full_data,p_range,x,p), collect(1:length(unique_energies))
end

function distr_approx_numerical(energies,unique_energies,numbers,p_range)
    full_data =zeros(eltype(p_range[1]),(length(p_range),length(unique_energies)))
    for i in 1:length(p_range)
        numtot = zero(eltype(p_range[1]))
        for j in 1:length(unique_energies)
            number_list = numbers[i]
            energy_list = energies[i]
            sample = unique_energies[j]
            indices = findall(x->x==sample, energy_list)
            if length(indices) == 0
                full_data[i,j] = zero(eltype(sample))
            else
                full_data[i,j] = number_list[indices[1]]
                numtot+=number_list[indices[1]]
            end
        end
        full_data[i,:] = full_data[i,:]/numtot
    end

    return (x,p)->distr_approx_x_p(full_data,p_range,x,p), collect(1:length(unique_energies))
end
