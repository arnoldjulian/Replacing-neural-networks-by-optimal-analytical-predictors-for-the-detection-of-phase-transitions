# construct constant uniform distribution (case 1)
function constant_distr()

  # define constant function
  function uniform(x, p)
    return 0.5f0
  end

  # binary samples
  samples = [0, 1]
  return uniform, samples
end

# construct distribution with jump at p_crit (case 2)
function jump_distr(;p_crit=1.0f0)

  # define step function
  function stepfunc(x, p, p_crit)
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

  # binary samples
  samples = [0, 1]
  return (x, p) -> stepfunc(x, p, p_crit), samples
end

# construct ``tilted'' distribution with jump at p_crit (case 3)
function tilt_distr(p_range; p_crit = 2.0f0)

  # tilt function
  function tilt(p, p_crit)
    a_left = -1.0f0
    b_left = 5.0f0
    a_right = -1.0f0
    b_right = 2.0f0
    if p <= p_crit
      x = a_left*p+b_left
    else
      x = a_right*p+b_right
    end

    return x
  end

  # sample from tilt function
  samples = map(p -> tilt(p, p_crit), p_range)
  function tilt_distr(x ,p, p_range)
    target = samples[findall(x -> x==p, p_range)[1]]
    if target == x
      return one(eltype(p))
    else
      return zero(eltype(p))
    end
  end
  return (x, p) -> tilt_distr(x, p, p_range), samples
end

# probability for specific sample at fixed value of tuning parameter
function distr_approx_x_p(full_data, p_range, x, p)
  p_indx = Int(round((p-p_range[1])/(p_range[2]-p_range[1])))+1
  return full_data[p_indx, x]
end

# construct probability distribution based on Monte Carlo samples
function distr_approx(energies, unique_energies, numbers, p_range)
  full_data =zeros(eltype(p_range[1]),(length(p_range),length(unique_energies)))
  for i in 1:length(p_range)
    numtot = zero(eltype(p_range[1]))
    for j in 1:length(unique_energies)
      number_list = numbers[i]
      energy_list = energies[i]
      sample = unique_energies[j]
      indices = findall(x -> x==sample, energy_list)
      if length(indices) == 0
        full_data[i, j] = zero(eltype(sample))
      else
        full_data[i, j] = number_list[indices[1]]
        numtot += number_list[indices[1]]
      end
    end
    full_data[i, :] = full_data[i, :]/numtot
  end

  return transpose(full_data), (x, p) -> distr_approx_x_p(full_data, p_range, x, p), collect(1:length(unique_energies))
end
