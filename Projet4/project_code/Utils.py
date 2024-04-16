##########################################################################

# Titre: Utils.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 26/03/2024

##########################################################################
'''
# Description:

This file contains all general methods that needs to be called to help the quantum chemistry.

'''

###########################################################################

# IMPORTS

###########################################################################

from qiskit import transpile, QuantumCircuit, assemble
from qiskit.providers.backend import Backend
import numpy as np
from numpy.typing import NDArray
from typing import List, Union
import matplotlib.pyplot as plt

###########################################################################

# Methods

###########################################################################

def bitstring_to_bits(bit_string: str) -> NDArray[np.bool_]:
    """
    Convert bitstring to bool array.
    Args:
    bit_string (str): The bitstring to convert.
    Returns:
    NDArray[np.bool_]: The converted bool array.
    """
    return np.array([x == '1' for x in bit_string], dtype=bool)[::-1]

def execute_job(circuit: List[QuantumCircuit] , backend: Backend, execute_opts: dict) -> dict:
    """
    execute jobs on the provided backend.
    Args:
    circuit (List[QuantumCircuit]): List of job to execute
    backend (Backend): Provided backend
    execute_opts (dict): dictionnary of execution options
    Returns:
    dict: The results of the measurements
    """
    if len(circuit) != 1:
        transpiled_qc = transpile(list(circuit), backend)
        queue_job = assemble(transpiled_qc)
        job = backend.run(queue_job, **execute_opts)
    else:
        transpiled_qc = transpile(circuit[0], backend)
        job = backend.run(transpiled_qc, **execute_opts)
    return job.result()

def plot_results(distances: NDArray[np.float32], energy: NDArray[np.float32], name: str):
    """
    Create a plot of the two arguments.
    Args:
    distances (NDArray[np.float32]): x coordinates
    energy (NDArray[np.float32]): y coordinates
    name (str): name of the plot

    """
    plt.scatter(distances, energy)
    plt.xlabel('distance')
    plt.ylabel('energy')
    plt.title(name)
    plt.grid(True)
    plt.show()

def validate_results(estimated_values, exact_values):
    """
    Create a plot of the two arguments.
    Args:
    estimated_values (NDArray[np.float32]): estimated values
    exact_values (NDArray[np.float32]): exact values
    Returns:
    float: the mean squared error
    """
    mean_squared_error = np.mean((estimated_values - exact_values) ** 2)
    
    return mean_squared_error


def save_quantum_evolution_plot(time_values, exact_expected_values, trotter_expected_values, observables):

    num_observables = len(observables)
    fig, axs = plt.subplots(num_observables, sharex=True, sharey=True)
    fig.suptitle("expected values for every observables over time")

    for i in range(num_observables):
        axs[i].set_title(observables[i].paulis)
        axs[i].plot(time_values, exact_expected_values[i], color = "red", label="Exact expected values")
        axs[i].plot(time_values, trotter_expected_values[i], color = "blue", label="Trotter expected values")
        axs[i].legend(loc="best")
        
    plt.savefig("trotter_plot")
