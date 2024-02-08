
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
from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit.quantum_info import Pauli, PauliList
from typing import Tuple, List
from itertools import product, permutations
import numpy as np
from numpy.typing import NDArray

###########################################################################

# METHODS

###########################################################################

def bitstring_to_bits(bit_string: str) -> NDArray[np.bool_]:

    return np.array([False if  i == "0" else True for i in bit_string])[::-1]


def diag_pauli_expectation_value(pauli: Pauli, counts: dict) -> float:

    assert(np.all(~pauli.x)) # verify if Pauli 
    expectation_value = 0

    return expectation_value


def diagonalize_pauli_with_circuit(pauli : Pauli) -> Tuple[Pauli, QuantumCircuit]:

    diagonal_pauli = Pauli(...)
    circuit = QuantumCircuit(...)
    assert(np.all(~diagonal_pauli.x)) # verify the diagonalization

    return diagonal_pauli, circuit


def estimate_expectation_values(
    paulis: PauliList,
    state_circuit: QuantumCircuit,
    backend: Backend,
    execute_opts : dict = dict()) -> NDArray[np.float_]:

    expectation_values = 0

    return expectation_values


def state_tomography(
    state_circuit: QuantumCircuit,
    backend: Backend,
    execute_opts : dict = dict()) -> NDArray[np.complex_]:

    statevector = 0

    return statevector


def create_all_pauli(num_qubits: int) -> PauliList:

    # pauli_lists = []

    # pauli_string = ["I", "X", "Y", "Z"]
    # print(permutations(pauli_strings))

    pauli_strings = "IXYZ"
    pauli_combinations = product(pauli_strings, repeat=num_qubits)
    
    pauli_chains = []
    for combination in pauli_combinations:
        pauli_string = ''.join(combination)
        pauli = Pauli(pauli_string)
        pauli_chains.append(pauli)
    
    return PauliList(pauli_chains)