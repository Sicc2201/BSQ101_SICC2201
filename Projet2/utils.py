
##########################################################################

# Titre: utils.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 02/02/2024

##########################################################################
'''
# Description: 


'''

###########################################################################

# IMPORTS

###########################################################################
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import XGate, ZGate,HGate, SdgGate, SGate
from qiskit.providers.backend import Backend
from qiskit.quantum_info import Pauli, PauliList
from typing import Tuple, List
from itertools import product
import numpy as np
from numpy.typing import NDArray


#custom library
import QuantumUtils as utils

###########################################################################

# METHODS

###########################################################################

def calculate_trace(pauli):
    print("calculate_trace")
    pauli_matrix = pauli.to_matrix()
    trace = np.trace(pauli_matrix)
    return trace


# useful to calculate eigen values
def bitstring_to_bits(bit_string: str) -> NDArray[np.bool_]:
    print("bitstring_to_bits")
    return np.array([False if  i == "0" else True for i in bit_string])[::-1]


def diag_pauli_expectation_value(pauli: Pauli, counts: dict, index) -> float:
    print("diag_pauli_expectation_value")
    assert(np.all(~pauli.x)) # verify if Pauli diagonal

    # calculate sum of its eigen values
    # utils.save_histogram_png(counts, "pauli_" + str(index))
    print(pauli, " : ", counts)

    # bitstring_to_bits(counts.keys)

    trace = calculate_trace(pauli)

    expectation_value = 0

    return expectation_value


# diagonalize circuit
def diagonalize_pauli_with_circuit(pauli : Pauli) -> Tuple[Pauli, QuantumCircuit]:
    print("diagonalize_pauli_with_circuit")
    num_qubits = pauli.num_qubits

    z_bits = pauli.z
    x_bits = pauli.x
    index = num_qubits

    qc = QuantumCircuit(num_qubits)
    for z, x in zip(z_bits, x_bits):
        index -= 1
        if x == 0:
            print(index, ": Z or I gate, apply nothing")
        elif z == 0:
            print(index, ": X gate, apply HZH")
            pauli.dot(HGate, qargs=[index], inplace=True)
            pauli.dot(XGate, qargs=[index], inplace=True)
            pauli.dot(HGate, qargs=[index], inplace=True)
        else:
            print(index, ": Y gate, apply SHZHS")
            pauli.dot(SGate, qargs=[index], inplace=True)
            pauli.dot(HGate, qargs=[index], inplace=True)
            pauli.dot(ZGate, qargs=[index], inplace=True)
            pauli.dot(HGate, qargs=[index], inplace=True)
            pauli.dot(SdgGate, qargs=[index], inplace=True)

    # for z, x in zip(z_bits, x_bits):
    #     index -= 1
    #     if x == 0:
    #         print(index, ": Z or I gate, apply nothing")
    #     elif z == 0:
    #         print(index, ": X gate, apply HZH")
    #         qc.h(index)
    #         qc.x(index)
    #         qc.h(index)
    #     else:
    #         print(index, ": Y gate, apply SHZHS")
    #         qc.s(index)
    #         qc.h(index)
    #         qc.z(index)
    #         qc.h(index)
    #         qc.sdg(index)
    
    qc.append(pauli, qc.qubits)


    assert(np.all(~pauli.x)) # verify the diagonalization

    return (pauli, qc)


# calculate "valeurs moyennes" from the circuit to find state vector
def estimate_expectation_values(
    pauli_list: PauliList,
    state_circuit: QuantumCircuit,
    backend: Backend,
    execute_opts : dict = dict()) -> NDArray[np.float_]:

    print("estimate_expectation_values")
    expectation_values = []
    index = 0
    for pauli in pauli_list:
        diag_pauli, qc = diagonalize_pauli_with_circuit(pauli)
        pauli_circuit = state_circuit.compose(qc)
        pauli_circuit.measure_all()
        transpiled_qc = transpile(pauli_circuit, backend)
        job = backend.run(transpiled_qc, options = execute_opts)
        counts = job.result().get_counts()
        expected_value = diag_pauli_expectation_value(diag_pauli, counts, index)
        expectation_values.append(expected_value)
        index += 1

    return expectation_values


def state_tomography(
    state_circuit: QuantumCircuit,
    backend: Backend,
    execute_opts : dict = dict()) -> NDArray[np.complex_]:

    print("state_tomography")

    pauli_list = create_all_pauli(state_circuit.num_qubits)
    expected_values = estimate_expectation_values(pauli_list,state_circuit,backend,execute_opts)

    density_matrix = calculate_density_matrix()

    statevector = 0

    return statevector

def create_all_pauli(num_qubits: int) -> PauliList:
    print("create_all_pauli")

    pauli_bits_combinations = product([0,1], repeat=num_qubits)
    pauli_zx_permutations = list(product(pauli_bits_combinations, repeat=2))
    pauli_list = [Pauli(pauli) for pauli in pauli_zx_permutations]
    return PauliList(pauli_list)

def create_random_quantum_circuit(num_qubits):
    print("create_random_quantum_circuit")
    qc = QuantumCircuit(num_qubits)
    qc.h(qc.qubits)

    return qc

def calculate_density_matrix():
    return