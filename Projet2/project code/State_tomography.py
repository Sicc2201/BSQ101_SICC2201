
##########################################################################

# Titre: utils.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 02/02/2024

##########################################################################
'''
# Description: 

This file contains all methods that implement the algorithm.

# Methods:

- diag_pauli_expectation_value(pauli: Pauli, counts: dict) -> float:

- diagonalize_pauli_with_circuit(pauli : Pauli) -> Tuple[Pauli, QuantumCircuit]:

- estimate_expectation_values(
    pauli_list: PauliList,
    state_circuit: QuantumCircuit,
    backend: Backend,
    execute_opts : dict = dict()) -> NDArray[np.float_]:

- expectation_value_from_measurement(state_circuit, pauli, backend, execute_opts):

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
from qiskit import QuantumCircuit, transpile
from qiskit.providers.backend import Backend
from qiskit.quantum_info import Pauli, PauliList
from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import sqrtm


#custom library
import Utils

###########################################################################

# METHODS

###########################################################################


def diag_pauli_expectation_value(pauli: Pauli, counts: dict) -> float:

    assert(np.all(~pauli.x)) # verify if Pauli diagonal

    # utils.save_histogram_png(counts, "pauli_" + str(index))
    total_counts = 0
    expectation_value = 0
    for bit_str, count in counts.items():
        eigenvalue = 1 - 2 * (np.dot(Utils.bitstring_to_bits(bit_str), pauli.z) % 2)
        expectation_value += eigenvalue * count
        total_counts += count

    return expectation_value / total_counts

def diagonalize_pauli_with_circuit(pauli : Pauli) -> Tuple[Pauli, QuantumCircuit]:
    num_qubits = pauli.num_qubits

    z_bits = pauli.z
    x_bits = pauli.x
    index = num_qubits

    qc = QuantumCircuit(num_qubits)

    for index,  (z, x) in enumerate(zip(z_bits, x_bits)):
        index -= 1 # a check/ avec l<ajout du enumerate
        if x == 0:
            print(index, ": Z or I gate, apply nothing")
        elif z == 0:
            print(index, ": X gate, apply H")
            qc.h(index)
        else:
            print(index, ": Y gate, apply HS")
            qc.h(index)
            qc.sdg(index)

    qc.compose(pauli, qc.qubits)
    diag_z_bits = np.logical_or(z_bits, x_bits)
    print("z_bits: ", z_bits)
    print("x_bits: ", x_bits)
    print("diag z_bits: ", diag_z_bits)

    pauli = Pauli((diag_z_bits, np.zeros(num_qubits, dtype=bool)))

    assert(np.all(~pauli.x))

    return (pauli, qc)

def estimate_expectation_values(
    pauli_list: PauliList,
    state_circuit: QuantumCircuit,
    backend: Backend,
    execute_opts : dict = dict()) -> NDArray[np.float_]:

    diag_pauli_list = []
    expectation_values = np.empty(4 ** state_circuit.num_qubits)
    index = 0
    for pauli in pauli_list:
        expectation_values[index] = expectation_value_from_measurement(state_circuit, pauli, backend, execute_opts)
        index += 1

    return expectation_values

def expectation_value_from_measurement(state_circuit, pauli, backend, execute_opts):
        diag_pauli, pauli_qc = diagonalize_pauli_with_circuit(pauli)
        pauli_circuit = state_circuit.compose(pauli_qc, state_circuit.qubits)
        pauli_circuit.measure_all()
        counts = Utils.execute_job(pauli_circuit, backend, execute_opts)
        print("job done")
        return diag_pauli_expectation_value(diag_pauli, counts)

def state_tomography(
    state_circuit: QuantumCircuit,
    backend: Backend,
    execute_opts : dict = dict()) -> NDArray[np.complex_]:

    pauli_list = Utils.create_all_pauli(state_circuit.num_qubits)
    expectation_values = estimate_expectation_values(pauli_list,state_circuit,backend,execute_opts)
    scaled_expectation_values =  [value /(2**state_circuit.num_qubits) for value in expectation_values]

    density_matrix = calculate_density_matrix(pauli_list, scaled_expectation_values)
    print(density_matrix)

    state_vector = calculate_stateVector(density_matrix)

    return state_vector

def calculate_density_matrix(pauli_list: PauliList, expectation_values: List):
    density_matrix = sum(np.multiply(value, pauli.to_matrix()) for value, pauli in zip(expectation_values, pauli_list))

    return density_matrix

def calculate_stateVector(density_matrix):

    # probabilities, statevectors = np.linalg.eigh(density_matrix)

    # je ne suis pas sûr comment trouver le state vector avec la matrice de densité - J'ai trouvé cette façon sur internet
    # un indice pour savoir comment faire en commentaire serait apprécié
    sqrt_density_matrix = sqrtm(density_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(sqrt_density_matrix)
    max_index = np.argmax(eigenvalues)
    state_vector = eigenvectors[:, max_index]
    state_vector /= np.linalg.norm(state_vector)
    return state_vector