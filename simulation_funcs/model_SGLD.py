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

## Functions
def run_simulation_SGLD(noise_flag,init_param,n_params, Nsteps, beta, train_keys, param_to_st, data_probs,bandwidth_sq, data):
    from jax import numpy as jnp, random, value_and_grad
    import time

    bar_length = 30
    
    params = jnp.zeros((Nsteps, n_params))
    params = params.at[0].set(init_param)
    
    cost_vals = jnp.zeros(Nsteps - 1)
    
    start_time = time.time() # Record the start time
    
    for step in range(1, Nsteps):
        cost_val, cost_grad = value_and_grad(param_to_mmd)(params[step - 1], param_to_st, data_probs, bandwidth_sq, data)
          # Add noise to the cost_grad
        noise = 0
        if noise_flag:
            noise = random.normal(train_keys[step - 1], shape=cost_grad.shape, dtype=jnp.float32)
        cost_grad = cost_grad + noise
      
        cost_vals = cost_vals.at[step - 1].set(cost_val)
        
        stepsize = get_stepsize(step)
        
        new_param = (
            params[step - 1]
            - stepsize * cost_grad
            + jnp.sqrt(2 * stepsize / beta)
            * random.normal(train_keys[step - 1], shape=(n_params,))
        )
        params = params.at[step].set(new_param)

        progress = step / Nsteps
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        elapsed_time = time.time() - start_time # Compute the elapsed time
        remaining_time = (elapsed_time / step) * (Nsteps - step) # Estimate the remaining time
        print(f'\rProgress: |{bar}| {progress:.2%} | Iteration: {step}/{Nsteps} | Cost: {cost_val:.6f} | Elapsed Time: {elapsed_time:.2f}s | Estimated Remaining Time: {remaining_time:.2f}s', end='')
    
    print() # Move to the next line after the progress bar
    
    total_run_time = time.time() - start_time # Compute the total run time
    
    return params, cost_vals, total_run_time

def run_model_SGLD(data,noise_flag=True, beta=100,n_steps=500, n_qubits=8 , circuit_depth=3, print_flag=False): 
    from jax import numpy as jnp, random, vmap

    number_of_data_points = len( data) 
    init_rad = 0.001 / jnp.pi
    # get_stepsize = lambda step: (step + 10) ** (-1 / 3)
    random_key = random.PRNGKey(0)
    init_key, train_key = random.split(random_key)
    
    ## These take awhile for large data sets
    # computes pairwise
    dist_mat = vmap(lambda a: vmap(lambda b: (a - b) ** 2)(data))(data)
    
    # get bandwidth
    bandwidth_sq = jnp.median(dist_mat) / 2
    
    results = {}
    
    
    gates, qubit_inds, param_inds, n_params = get_circuit(n_qubits, circuit_depth)
    param_to_st = qujax.get_params_to_statetensor_func(gates, qubit_inds, param_inds)
    data_probs = jnp.ones(len(data)) / len(data)
    # param_to_mmd_and_grad = jit(value_and_grad(param_to_mmd))
    
    init_param = random.uniform(
        init_key, shape=(n_params,), minval=-init_rad, maxval=init_rad
    )
    
    train_keys = random.split(train_key, n_steps - 1)
    params, cost_vals, total_run_time = run_simulation_SGLD(noise_flag,init_param, n_params,
                                                       n_steps, beta, train_keys,
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
    
    return (jnp.square(jnp.abs(final_st.flatten())), cost_vals, total_run_time, params)
 


