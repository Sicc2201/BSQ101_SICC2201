
##########################################################################

# Titre: Quantum_chemistry.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 26/03/2024

##########################################################################
'''
# Description: 

Ce fichier contient toutes fonctions qui gèrent le processus d'évolution d'un système quantique.

'''

###########################################################################

# IMPORTS

###########################################################################
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers.backend import Backend
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import List, Callable, Union
from scipy.optimize import OptimizeResult, minimize

#custom library
import Utils
import Pauli_operations as po

###########################################################################

# METHODS

###########################################################################

def exact_evolution(
initial_state: QuantumCircuit,
hamiltonian: SparsePauliOp,
time_values: NDArray[np.float_],
observables: List[SparsePauliOp],
):
    """
    Simulate the exact evolution of a quantum system in state initial_state under a given
    hamiltonian for different time_values. The result is a series of expected values
    for given observables.
    Args:
    time_values (NDArray[np.float_]): An array of time values.
    initial_state (QuantumCircuit): The circuit preparing the initial quantum state.
    hamiltonian (SparsePauliOp): The Hamiltonien of the system
    observables (List[SparsePauliOp]): The observable to be measure at each the the
    time_values.

    Returns:
    NDArray[np.float_]: The expected values of the observable. Should be of shape
    (len(time_values), len(observables)).
    """

    observables_expected_values = np.empty(len(time_values), len(observables))

    w, v = diagonalize_hamiltonian(hamiltonian)

    evolve_matrix = np.einsum("sk, ik, jk -> sij", w, v, v.conj())




    return observables_expected_values


def diagonalize_hamiltonian(hamiltonian: SparsePauliOp, time_values: NDArray[np.float_]):
    a, v = np.linalg.eigh(hamiltonian.to_matrix())
    energy_matrix = np.diag(a)

    w = time_values[:, None] * energy_matrix[:, None]

    print("diag energy: ", energy_matrix)
    print("w: ", w)

    return w, v



def trotter_evolution(
initial_state: QuantumCircuit,
hamiltonian: SparsePauliOp,
time_values: NDArray[np.float_],
observables: List[SparsePauliOp],
num_trotter_steps: NDArray[np.int_],
):
    """
    Simulate, using Trotterisation, the evolution of a quantum system in state initial_state
    under a given hamiltonian for different time_values. The result is a series of
    expected values for given observables.

    Args:
    time_values (NDArray[np.float_]): An array of time values.
    initial_state (QuantumCircuit): The circuit preparing the initial quantum state.
    hamiltonian (SparsePauliOp): The Hamiltonien of the system
    observables (List[SparsePauliOp]): The observable to be measure at each the the
    time_values.
    num_trotter_steps: (NDArray[np.int_]): The number of steps of the Trotterization for
    each time_values.

    Returns:
    NDArray[np.float_]: The expected values of the observable. Should be of shape
    (len(time_values), len(observables)).
    """
    observables_expected_values = np.empty(len(time_values), len(observables))

    trotter_circuit = trotter_circuit(hamiltonian, time_values, num_trotter_steps)



    return observables_expected_values

def trotter_circuit(
hamiltonian: SparsePauliOp,
total_duration: Union[float, Parameter],
num_trotter_steps: int,
) -> QuantumCircuit:
    """
    Construct the QuantumCircuit using the first order Trotter formula.

    Args:
    hamiltonian (SparsePauliOp): The Hamiltonian which governs the evolution.
    total_duration (Union[float, Parameter]): The duration of the complete evolution.
    num_trotter_steps (int): The number of trotter steps.

    Returns:
    QuantumCircuit: The circuit of the Trotter evolution operator
    """

    qc = QuantumCircuit(1)

    return qc

def random_pauli_op(dimension):
    """
    Generate a random SparsePauliOp of dimension `dimension`.
    """
    pauli_labels = ['I', 'X', 'Y', 'Z']
    pauli_matrix = np.random.choice(pauli_labels, size=dimension)
    return SparsePauliOp.from_label(''.join(pauli_matrix))

def create_hamiltonian(num_qubits, density=0.5):

    
    # """
    # Generate a random Hamiltonian with `num_terms` terms of dimension `dimension`.
    # """
    # hamiltonian = SparsePauliOp.zeros(num_qubits)  # Start with a zero operator

    # for _ in range(num_terms):
    #     coefficient = np.random.uniform(-1, 1)  # Random coefficient between -1 and 1
    #     pauli_op = random_pauli_op(num_qubits)
    #     term = coefficient * pauli_op
    #     hamiltonian += term

    # return hamiltonian

    # Generate random Pauli strings
    pauli_strings = []
    for _ in range(int(num_qubits * density)):
        pauli = "".join(np.random.choice(["I", "X", "Y", "Z"]) for _ in range(num_qubits))
        pauli_strings.append(Pauli(pauli))

    # Generate random coefficients
    coefficients = np.random.rand(len(pauli_strings))

    # Create SparsePauliOp
    sparse_pauli_op = SparsePauliOp(pauli_strings, coefficients)
    return sparse_pauli_op

def create_initial_state(num_qubits: int):
    qc = QuantumCircuit(num_qubits)
    return qc

def create_observables(num_qubits: int):
    ones = np.ones(num_qubits)
    zeros = np.zeros(num_qubits)
    pauli_x = Pauli(zeros, ones)
    pauli_y = Pauli(ones, ones)
    pauli_z = Pauli(ones, zeros)
    return PauliList([pauli_x, pauli_y, pauli_z])

