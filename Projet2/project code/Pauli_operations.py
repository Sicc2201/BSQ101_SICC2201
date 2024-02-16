
##########################################################################

# Titre: Pauli_operation.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 15/02/2024

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
from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit.quantum_info import Pauli, PauliList
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

#custom library
import Utils

def diag_pauli_expectation_value(pauli: Pauli, counts: dict) -> float:

    assert(np.all(~pauli.x))

    # utils.save_histogram_png(counts, "pauli_" + str(index))
    total_counts = 0
    expectation_value = 0
    for bit_str, count in counts.items():
        eigenvalue = 1 - 2 * (np.sum(np.multiply(Utils.bitstring_to_bits(bit_str), pauli.z)) % 2)
        expectation_value += eigenvalue * count
        total_counts += count

    return expectation_value / total_counts

def diagonalize_pauli_with_circuit(pauli : Pauli) -> Tuple[Pauli, QuantumCircuit]:
    num_qubits = pauli.num_qubits

    z_bits = pauli.z
    x_bits = pauli.x

    qc = QuantumCircuit(num_qubits)
    for index, (z, x) in enumerate(zip(z_bits, x_bits)):
        
        if z ==1 and x == 1:
            qc.sdg(num_qubits - index - 1)
            qc.h(num_qubits - index - 1)
        if x == 1:
            qc.h(num_qubits - index - 1)

    pauli = diagonalize_pauli(z_bits, x_bits)

    assert(np.all(~pauli.x))

    return (pauli, qc)

def diagonalize_pauli(z_bits, x_bits):
    diag_z_bits = np.logical_or(z_bits, x_bits)
    return Pauli((diag_z_bits, np.zeros(len(z_bits), dtype=bool)))

def measure_pauli_circuit(state_circuit, pauli_qc):
        pauli_circuit = state_circuit.compose(pauli_qc, state_circuit.qubits)
        pauli_circuit.measure_all()
        return pauli_circuit

def estimate_expectation_values(
    pauli_list: PauliList,
    state_circuit: QuantumCircuit,
    backend: Backend,
    execute_opts : dict = dict()) -> NDArray[np.float_]:

    jobs = np.empty(len(pauli_list), dtype=object)
    diag_pauli_list = np.empty(len(pauli_list), dtype=object)
    for index, pauli in enumerate(pauli_list):
        diag_pauli, pauli_qc = diagonalize_pauli_with_circuit(pauli)
        pauli_measurement = measure_pauli_circuit(state_circuit, pauli_qc)
        jobs[index] = pauli_measurement
        diag_pauli_list[index] = diag_pauli

    results = Utils.execute_job(jobs, backend, execute_opts)
    expectation_values = [diag_pauli_expectation_value(diag_pauli, counts) for diag_pauli, counts in zip(diag_pauli_list, results.get_counts())]
    return expectation_values