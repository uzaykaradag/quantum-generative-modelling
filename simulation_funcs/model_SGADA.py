#############################################
# Stochastic Gradient Adaptive Langevin-Thermostat (SGADA)
#############################################

import matplotlib.pyplot as plt
import qujax
import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp, random, vmap, grad, value_and_grad, jit

from .helper_funcs  import *


def G(p, params_for_sgada):
    from jax import numpy as jnp
    import numpy as np
    M, M_inv, mu_inv, N_d, kBT, sigma, sigma_a, _ = params_for_sgada
    return mu_inv * (p.T @ M_inv @ p - N_d * kBT)

def A_step(q,p,xi,h, params_for_sgada):
    from jax import numpy as jnp
    import numpy as np
    M, M_inv, mu_inv, N_d, kBT, sigma, sigma_a, _ = params_for_sgada
    q = q + h * M_inv @ p
    return q,p,xi

def B_step(q,p,xi,h,force, params_for_sgada):
    p = p + h * force(q) 
    return q,p,xi

def O_step(q,p,xi,h, params_for_sgada):
    from jax import numpy as jnp
    import numpy as np
    M, M_inv, mu_inv, N_d, kBT, sigma, sigma_a, _ = params_for_sgada
    
    term1 = np.exp(-xi*h) * p
    term2 = sigma_a * np.sqrt((1 - np.exp(-2 * xi * h)) / (2 * xi)) * np.random.randn(p.shape[0])
    p = term1 + term2
    return q,p,xi

def D_step(q,p,xi,h, params_for_sgada):
    from jax import numpy as jnp
    import numpy as np
    xi = xi + h * G(p, params_for_sgada)
    return q,p,xi

def ld_BADODAB(q,p,xi,h,force, params_for_sgada):

    q, p, xi = np.copy(q), np.copy(p), np.copy(xi)
    q, p, xi = B_step(q,p,xi,h/2,force,params_for_sgada)
    q, p, xi = A_step(q,p,xi,h/2, params_for_sgada)
    q, p, xi = D_step(q,p,xi,h/2, params_for_sgada)
    q, p, xi = O_step(q,p,xi,h,   params_for_sgada)
    q, p, xi = D_step(q,p,xi,h/2, params_for_sgada)
    q, p, xi = A_step(q,p,xi,h/2, params_for_sgada)
    q, p, xi = B_step(q,p,xi,h/2,force, params_for_sgada)

    return q, p, xi


def force_for_SGADA(q, param_to_st, data_probs, bandwidth_sq, data):
    cost_val, cost_grad = value_and_grad(param_to_mmd)(q, param_to_st, data_probs, bandwidth_sq, data)
    return cost_val, cost_grad

def run_model_SGADA(data, hyperparameters, stochastic_flag = True, print_flag=False):
    from jax import numpy as jnp, random, vmap
    import matplotlib.pyplot as plt
    import qujax
    
    init_rad = 0.001 / jnp.pi
    random_key = random.PRNGKey(0)
    init_key, train_key = random.split(random_key)
    # get bandwidth
    dist_mat = vmap(lambda a: vmap(lambda b: (a - b) ** 2)(data))(data)
    bandwidth_sq = 5.7478714 # for GMM #jnp.median(dist_mat) / 2 

    ## params
    h, M, mu_inv, N_d, beta, sigma, sigma_a, n_steps, n_qubits, circuit_depth, batch_size = hyperparameters.values()
    gates, qubit_inds, param_inds, n_params = get_circuit(n_qubits, circuit_depth)
    M_inv = jnp.linalg.inv(M) # Omit this eventually and use a solve later as we should never invert a matrix
    kBT     = 1 / beta 
    # combine them 
    params_for_sgada = (M, M_inv, mu_inv, N_d, kBT, sigma, sigma_a, batch_size)
    
    param_to_st = qujax.get_params_to_statetensor_func(gates, qubit_inds, param_inds)
    data_probs = jnp.ones(len(data)) / len(data)
    
    init_param = random.uniform(
    init_key, shape=(n_params,), minval=-init_rad, maxval=init_rad
    )
  
    params = jnp.zeros((n_steps, n_params))
    params = params.at[0].set(init_param)
    
    cost_vals = jnp.zeros(n_steps - 1)

    train_keys = random.split(train_key, n_steps - 1)
    # run the sim
    params, p_traj, xi_traj, cost_vals, total_run_time = run_simulation_SGADA(
        stochastic_flag=stochastic_flag, 
        q0=params[0],
        p0=np.random.randn(n_params),
        xi0=0.0,
        h=0.1, 
        step_function= ld_BADODAB, 
        force= lambda q: -force_for_SGADA(q, param_to_st, data_probs, bandwidth_sq, data)[1], 
        Nsteps=n_steps, 
        params_for_sgada = params_for_sgada,
        train_keys=train_keys,
        param_to_st=param_to_st,
        data_probs=data_probs,
        bandwidth_sq= bandwidth_sq, 
        data=data)

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



def run_simulation_SGADA(stochastic_flag,q0, p0, xi0,h,step_function,force,Nsteps,
                         params_for_sgada,train_keys, param_to_st,data_probs,bandwidth_sq,data):
    import time
    
    bar_length = 30
    
    q_traj = [q0]
    p_traj = [p0]
    xi_traj = [xi0]
    cost_vals = []

    q = q0
    p = p0
    xi = xi0
    
    start_time = time.time()  # Record the sstart time
    _, _, _, _, _, _, _, batch_size= params_for_sgada
    for n in range(Nsteps):

        if stochastic_flag:
            cost_val, cost_grad = value_and_grad(param_to_mmd_stochastic)(q_traj[n - 1], param_to_st, data, bandwidth_sq, batch_size)
        else:
            cost_val, cost_grad = value_and_grad(param_to_mmd)(q_traj[n - 1], param_to_st, data_probs, bandwidth_sq, data)
        
        q,p,xi = step_function(q, p, xi, h, force, params_for_sgada)
        cost_vals.append(cost_val)

        q_traj += [q]
        p_traj += [p]
        xi_traj += [xi]
        
        progress = (n+1) / Nsteps
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        elapsed_time = time.time() - start_time  # Compute the elapsed time
        remaining_time = (elapsed_time / (n + 1)) * (Nsteps - n - 1)  # Estimate the remaining time

        
        print(f'\rProgress: |{bar}| {progress:.2%} | Iteration: {n+1}/{Nsteps} | Cost: {cost_val:.6f} | Elapsed Time: {elapsed_time:.2f}s | Estimated Remaining Time: {remaining_time:.2f}s', end='')    
    print()  # Move to the next line after the progress bar
    
    total_run_time = time.time() - start_time  # Compute the total run time
    
    return q_traj, p_traj, xi_traj, cost_vals, total_run_time


