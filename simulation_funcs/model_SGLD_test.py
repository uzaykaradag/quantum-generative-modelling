#############################################
# Stochastic Gradient Langevin Dynamics (SGLD)
#############################################
# This code block implements the Stochastic Gradient Langevin Dynamics (SGLD) algorithm.
import matplotlib.pyplot as plt
import qujax
import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp, random, vmap, grad, value_and_grad, jit


from .helper_funcs  import *


@jit
def param_to_mmd_and_grad_(param, param_to_st, data_probs, bandwidth_sq, data):
    cost_val = param_to_mmd(param, param_to_st, data_probs, bandwidth_sq, data)
    cost_grad = grad(param_to_mmd)(param, param_to_st, data_probs, bandwidth_sq, data)
    return cost_val, cost_grad

## Functions
def run_simulation_SGLD(stochastic_flag,init_param,n_params,h, Nsteps, beta, batch_size, train_keys, param_to_st, data_probs,bandwidth_sq, data):
    from jax import numpy as jnp, random, value_and_grad
    import time

    bar_length = 30
    
    params = jnp.zeros((Nsteps, n_params))
    params = params.at[0].set(init_param)
    
    cost_vals = jnp.zeros(Nsteps - 1)
    
    start_time = time.time() # Record the start time

    for step in range(1, Nsteps):
        if stochastic_flag:
            cost_val, cost_grad = value_and_grad(param_to_mmd_stochastic)(params[step - 1], param_to_st, data, bandwidth_sq, batch_size)
        else:
            #cost_val, cost_grad = value_and_grad(param_to_mmd)(params[step - 1], param_to_st, data_probs, bandwidth_sq, data)
            cost_val, cost_grad = param_to_mmd_and_grad_(params[step - 1], param_to_st, data_probs, bandwidth_sq, data)

      
        cost_vals = cost_vals.at[step - 1].set(cost_val)
        
        # stepsize = get_stepsize(step)
        
        new_param = (
            params[step - 1]
            - h * cost_grad
            + jnp.sqrt(2 * h / beta)
            * random.normal(train_keys[step - 1], shape=(n_params,))
        )
        params = params.at[step].set(new_param)

        progress = (step+1) / Nsteps
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        elapsed_time = time.time() - start_time # Compute the elapsed time
        remaining_time = (elapsed_time / step) * (Nsteps - step) # Estimate the remaining time
        print(f'\rProgress: |{bar}| {progress:.2%} | Iteration: {step}/{Nsteps} | Cost: {cost_val:.6f} | Elapsed Time: {elapsed_time:.2f}s | Estimated Remaining Time: {remaining_time:.2f}s', end='')
    
    print() # Move to the next line after the progress bar
    
    total_run_time = time.time() - start_time # Compute the total run time
    
    return params, cost_vals, total_run_time

def run_model_SGLD(data, hyperparameters, stochastic_flag=True, print_flag=False): 
    from jax import numpy as jnp, random, vmap

    h,beta, n_steps, n_qubits, circuit_depth, batch_size = hyperparameters.values()
    number_of_data_points = len( data) 
    init_rad = 0.001 / jnp.pi
    # get_stepsize = lambda step: (step + 10) ** (-1 / 3)
    random_key = random.PRNGKey(0)
    init_key, train_key = random.split(random_key)
    
    ## These take awhile for large data sets
    # computes pairwise
    #dist_mat = vmap(lambda a: vmap(lambda b: (a - b) ** 2)(data))(data)
    
    # get bandwidth
    bandwidth_sq = 5.7478714 #jnp.median(dist_mat) / 2
    
    results = {}
    
    
    gates, qubit_inds, param_inds, n_params = get_circuit(n_qubits, circuit_depth)
    param_to_st = qujax.get_params_to_statetensor_func(gates, qubit_inds, param_inds)
    data_probs = jnp.ones(len(data)) / len(data)
    global param_to_mmd_and_grad
    param_to_mmd_and_grad = jit(value_and_grad(param_to_mmd))
    # param_to_mmd_and_grad = lambda func: jit(value_and_grad(param_to_mmd))
    
    init_param = random.uniform(
        init_key, shape=(n_params,), minval=-init_rad, maxval=init_rad
    )
    
    train_keys = random.split(train_key, n_steps - 1)
    params, cost_vals, total_run_time = run_simulation_SGLD(stochastic_flag,init_param, n_params,
                                                       h,n_steps, beta, batch_size, train_keys,
                                                       param_to_st, data_probs,bandwidth_sq, data)
    
    if print_flag:
        plt.plot(cost_vals)
        plt.title(f'Cost for beta={beta}')
        plt.xlabel("Iteration")
        plt.ylabel("MMD")
        plt.ylim(0, 0.4)
        plt.show()
    
    final_params = params[-1]
    final_st = param_to_st(final_params)
    
    return (jnp.square(jnp.abs(final_st.flatten())), cost_vals, total_run_time, params, param_to_st)
 


