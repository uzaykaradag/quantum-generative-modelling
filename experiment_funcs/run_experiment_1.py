import os
import pickle
from simulation_funcs.model_SGADA import *
from simulation_funcs.model_SGLD import *
from jax.experimental import multihost_utils

def run_experiment(params):
    n_qubits, circuit_depth, data, sgada_hyperparams, sgld_hyperparams, output_folder, stochastic_flag = params

    print(f"Running experiment for n_qubits={n_qubits}, circuit_depth={circuit_depth}")

    gates, qubit_inds, param_inds, n_params = get_circuit(n_qubits, circuit_depth)

    # Update hyperparameters with the current n_qubits and circuit_depth
    sgada_hyperparams['n_qubits'] = n_qubits
    sgada_hyperparams['circuit_depth'] = circuit_depth
    sgada_hyperparams['M'] = jnp.eye(n_params)

    sgld_hyperparams['n_qubits'] = n_qubits
    sgld_hyperparams['circuit_depth'] = circuit_depth

    # Run SGADA model
    sgada_output, sgada_cost_vals, sgada_run_time, sgada_params, sgada_param_to_st = run_model_SGADA(
        data, sgada_hyperparams, stochastic_flag=stochastic_flag, print_flag=False
    )
    model_SGADA = {'output':sgada_output, 'cost_vals':sgada_cost_vals, 'run_time':sgada_run_time, 'params':sgada_params, 'param_to_st':sgada_param_to_st}

    # Run SGLD model
    sgld_output, sgld_cost_vals, sgld_run_time, sgld_params, sgld_param_to_st = run_model_SGLD(
        data, sgld_hyperparams, stochastic_flag=stochastic_flag, print_flag=False
    )
    
    model_SGLD  = {'output': sgld_output, 'cost_vals':sgld_cost_vals, 'run_time':sgld_run_time, 'params':sgld_params, 'param_to_st':sgld_param_to_st}

    # Save results to files
    filename = f"results_nqubits_{n_qubits}_depth_{circuit_depth}.pkl"
    filepath = os.path.join(output_folder, filename)

    results = {
        'n_qubits': n_qubits,
        'circuit_depth': circuit_depth,
        'sgld': model_SGLD, 
        'sqada': model_SGADA
    }

    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to {filepath}")

def run_experiment_1(data, qubit_range, depth_range, sgada_hyperparams, sgld_hyperparams, output_folder, stochastic_flag=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    params_list = [(n_qubits, circuit_depth, data, sgada_hyperparams, sgld_hyperparams, output_folder, stochastic_flag)
                   for n_qubits in qubit_range for circuit_depth in depth_range]

    multihost_utils.spawn(run_experiment, args_list=params_list)
