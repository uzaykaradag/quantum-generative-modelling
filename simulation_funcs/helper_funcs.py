 ##################################
 ##### helper_funcs ###############
 ##################################
 # Set of functions to assist the main effort... 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp, random, vmap, grad, value_and_grad, jit

## Plot function 
def plot_histogram_and_line(data, final_st):
    fig, ax = plt.subplots()
    
    # Plot histogram for data
    ax.hist(data, bins=50, density=True, alpha=0.5, label="Data")
    
    # Plot line for Final Parameter
    ax.plot(final_st, label="Final Parameter", color='orange')
    
    # Plot line for Averaged over parameters
   # ax.plot(data_avg, label="Averaged over parameters", linestyle='--', color='orange')
    
    # Adjust plot settings
    ax.set_xlim(data.min(), data.max())
    ax.set_ylabel("Probability")
    ax.set_xlabel("Data $\mu$m")
    ax.legend()
    
    return fig

## Simulation functions
def expected_kernel(kernel, data1, weights1, data2, weights2):
    from jax import numpy as jnp, random, vmap, grad, value_and_grad, jit

    def row_eval(data1_single):
        return (vmap(kernel, in_axes=(None, 0))(data1_single, data2) * weights2).sum()

    return (vmap(row_eval)(data1) * weights1).sum()


def mmd(kernel, data1, weights1, data2, weights2):

    return (
        expected_kernel(kernel, data1, weights1, data1, weights1)
        - 2 * expected_kernel(kernel, data1, weights1, data2, weights2)
        + expected_kernel(kernel, data2, weights2, data2, weights2)
    )

def get_stepsize( step):
    return (step + 10) **(-1/3)

def get_circuit(n_qubits, depth):
    n_params = 2 * n_qubits * (depth + 1)

    gates = ["H"] * n_qubits + ["Rx"] * n_qubits + ["Ry"] * n_qubits
    qubit_inds = [[i] for i in range(n_qubits)] * 3
    param_inds = [[]] * n_qubits + [[i] for i in range(n_qubits * 2)]

    k = 2 * n_qubits

    for _ in range(depth):
        for i in range(0, n_qubits - 1):
            gates.append("CZ")
            qubit_inds.append([i, i + 1])
            param_inds.append([])
        for i in range(n_qubits):
            gates.append("Rx")
            qubit_inds.append([i])
            param_inds.append([k])
            k += 1
        for i in range(n_qubits):
            gates.append("Ry")
            qubit_inds.append([i])
            param_inds.append([k])
            k += 1
    return gates, qubit_inds, param_inds, n_params


def gaussian_kernel(s1, s2, bandwidth_sq):

    return jnp.exp(-jnp.square(s1 - s2) / bandwidth_sq)

def param_to_mmd(param, param_to_st, data_probs, bandwidth_sq, data):
    from jax import numpy as jnp
    st = param_to_st(param)
    probs = jnp.square(jnp.abs(st.flatten()))
    return mmd(lambda s1, s2: gaussian_kernel(s1, s2, bandwidth_sq), jnp.arange(st.size), probs, data, data_probs)

def compute_av_probs(params, burn_in):
    from jax import numpy as jnp, vmap
    probs = vmap(lambda p: jnp.square(jnp.abs(param_to_st(p).flatten())))(params[burn_in:])
    return probs.mean(axis=0)
