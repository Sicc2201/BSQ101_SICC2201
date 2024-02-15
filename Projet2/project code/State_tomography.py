
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

#custom library
import Utils

###########################################################################

# METHODS

###########################################################################


def diag_pauli_expectation_value(pauli: Pauli, counts: dict) -> float:

    assert(np.all(~pauli.x))

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

    qc = QuantumCircuit(num_qubits)
    for index, (z, x) in enumerate(zip(z_bits, x_bits)):
        if x == 0:
            print(num_qubits - index - 1, ": Z or I gate, apply nothing")
        elif z == 0:
            print(num_qubits - index - 1, ": X gate, apply H")
            qc.h(num_qubits - index - 1)
        else:
            print(num_qubits - index - 1, ": Y gate, apply HS")
            qc.h(num_qubits - index - 1)
            qc.sdg(num_qubits - index - 1)

    pauli = diagonalize_pauli(z_bits, x_bits)

    assert(np.all(~pauli.x))

    return (pauli, qc)

def diagonalize_pauli(z_bits, x_bits):
    diag_z_bits = np.logical_or(z_bits, x_bits)
    return Pauli((diag_z_bits, np.zeros(len(z_bits))), dtype=bool)

def measure_pauli_circuit(state_circuit, pauli_qc):
        pauli_circuit = state_circuit.compose(pauli_qc, state_circuit.qubits)
        pauli_circuit.measure_all()
        return pauli_circuit


def estimate_expectation_values(
    pauli_list: PauliList,
    state_circuit: QuantumCircuit,
    backend: Backend,
    execute_opts : dict = dict()) -> NDArray[np.float_]:

    jobs = np.empty(len(pauli_list))
    diag_pauli_list = np.empty(len(pauli_list))
    for index, pauli in enumerate(pauli_list):
        diag_pauli, pauli_qc = diagonalize_pauli_with_circuit(pauli)
        pauli_measurement = measure_pauli_circuit(state_circuit, pauli_qc)
        jobs[index] = pauli_measurement
        diag_pauli_list[index] = diag_pauli

    counts = Utils.execute_job(jobs, backend, execute_opts)
    
    expectation_values = [diag_pauli_expectation_value(diag_pauli, counts) for diag_pauli in diag_pauli_list]
    return expectation_values

def state_tomography(
    state_circuit: QuantumCircuit,
    backend: Backend,
    execute_opts : dict = dict()) -> NDArray[np.complex_]:

    pauli_list = Utils.create_all_pauli(state_circuit.num_qubits)
    expectation_values = estimate_expectation_values(pauli_list,state_circuit,backend,execute_opts)
    scaled_expectation_values =  [value /(2**state_circuit.num_qubits) for value in expectation_values]

    density_matrix = calculate_density_matrix(pauli_list, scaled_expectation_values)
    print(density_matrix)

    state_vector = calculate_state_vector(density_matrix)

    return state_vector

def calculate_density_matrix(pauli_list: PauliList, expectation_values: List):
    density_matrix = np.sum(np.multiply(value, pauli.to_matrix()) for value, pauli in zip(expectation_values, pauli_list)) # optimization possible??

    return density_matrix

def calculate_state_vector(density_matrix):

    eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
    max_value = np.argmax(eigenvalues)
    state_vector = eigenvectors[:, max_value]
    state_vector /= np.linalg.norm(state_vector)
    return state_vector