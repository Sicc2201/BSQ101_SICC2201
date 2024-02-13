
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
from scipy.linalg import sqrtm


#custom library
import QuantumUtils as utils

###########################################################################

# METHODS

###########################################################################


# useful to calculate eigen values
def bitstring_to_bits(bit_string: str) -> NDArray[np.bool_]:
    return np.array([x == '1' for x in bit_string], dtype=bool)


def diag_pauli_expectation_value(pauli: Pauli, counts: dict) -> float:
    print("diag_pauli_expectation_value")
    assert(np.all(~pauli.x)) # verify if Pauli diagonal

    # utils.save_histogram_png(counts, "pauli_" + str(index))
    total_counts = 0
    expectation_value = 0
    for bit_str, count in counts.items():
        eigenvalue = 1 - 2 * (np.dot(bitstring_to_bits(bit_str), pauli.z) % 2)
        print(eigenvalue)
        expectation_value += eigenvalue * count
        total_counts += count

    return expectation_value / total_counts


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
    print("diag x_bits: ", diag_z_bits)

    pauli = Pauli((diag_z_bits, np.zeros(num_qubits, dtype=bool)))

    assert(np.all(~pauli.x)) # verify the diagonalization

    return (pauli, qc)


# calculate "valeurs moyennes" from the circuit to find state vector
def estimate_expectation_values(
    pauli_list: PauliList,
    state_circuit: QuantumCircuit,
    backend: Backend,
    execute_opts : dict = dict()) -> NDArray[np.float_]:

    diag_pauli_list = []
    print("estimate_expectation_values")
    expectation_values = np.empty(4 ** state_circuit.num_qubits)
    index = 0
    for pauli in pauli_list:
        expectation_values[index], diag_pauli = expectation_value_from_measurement(state_circuit, pauli, backend, execute_opts)
        index += 1
        diag_pauli_list.append(diag_pauli) 

    return expectation_values, PauliList(diag_pauli_list)

def expectation_value_from_measurement(state_circuit, pauli, backend, execute_opts):
        diag_pauli, pauli_qc = diagonalize_pauli_with_circuit(pauli)
        pauli_circuit = state_circuit.compose(pauli_qc, state_circuit.qubits)
        pauli_circuit.measure_all()
        counts = utils.execute_job(pauli_circuit, backend, execute_opts)
        print("job done")
        return diag_pauli_expectation_value(diag_pauli, counts), diag_pauli

def state_tomography(
    state_circuit: QuantumCircuit,
    backend: Backend,
    execute_opts : dict = dict()) -> NDArray[np.complex_]:

    print("state_tomography")

    pauli_list = create_all_pauli(state_circuit.num_qubits)
    expectation_values, diag_pauli_list = estimate_expectation_values(pauli_list,state_circuit,backend,execute_opts)
    expectation_values = (1/(2**state_circuit.num_qubits)) * expectation_values

    density_matrix = calculate_density_matrix(diag_pauli_list, expectation_values)
    print(density_matrix)

    state_vector = calculate_stateVector(density_matrix)

    return state_vector

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

def calculate_density_matrix(pauli_list: PauliList, expectation_values: List):
    density_matrix = sum(np.multiply(value, pauli.to_matrix()) for value, pauli in zip(expectation_values, pauli_list))

    return density_matrix

def calculate_stateVector(density_matrix):

    # je ne suis pas sûr comment trouver le state vector avec la matrice de densité - J'ai trouvé cette façon sur internet
    sqrt_density_matrix = sqrtm(density_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(sqrt_density_matrix)
    max_index = np.argmax(eigenvalues)
    state_vector = eigenvectors[:, max_index]
    state_vector /= np.linalg.norm(state_vector)
    return state_vector