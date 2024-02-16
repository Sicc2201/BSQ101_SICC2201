
##########################################################################

# Titre: State_tomography.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 15/02/2024

##########################################################################
'''
# Description: 

This file contains all methods that implement the algorithm.

# Methods:

- state_tomography(
    state_circuit: QuantumCircuit,
    backend: Backend,
    execute_opts : dict = dict()) -> NDArray[np.complex_]:

-  calculate_density_matrix(pauli_list: PauliList, expectation_values: List):

- calculate_stateVector(density_matrix)
'''

###########################################################################

# IMPORTS

###########################################################################
from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit.quantum_info import PauliList
from typing import List
import numpy as np
from numpy.typing import NDArray

#custom library
import Utils
import Pauli_operations as po

###########################################################################

# METHODS

###########################################################################

def state_tomography(
    state_circuit: QuantumCircuit,
    backend: Backend,
    execute_opts : dict = dict()) -> NDArray[np.complex_]:

    pauli_list = Utils.create_all_pauli(state_circuit.num_qubits)
    expectation_values = po.estimate_expectation_values(pauli_list,state_circuit,backend,execute_opts)
    scaled_expectation_values =  [value /(2**state_circuit.num_qubits) for value in expectation_values]

    density_matrix = calculate_density_matrix(pauli_list, scaled_expectation_values)
    state_vector = calculate_state_vector(density_matrix)

    return state_vector

def calculate_density_matrix(pauli_list: PauliList, expectation_values: NDArray[np.float_]):
    density_matrix = np.sum(np.multiply(value, pauli.to_matrix()) for value, pauli in zip(expectation_values, pauli_list)) # optimization possible??

    return density_matrix

def calculate_state_vector(density_matrix):

    eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
    max_value = np.argmax(eigenvalues)
    state_vector = eigenvectors[:, max_value]
    return state_vector

def validate_state_vector():
    return