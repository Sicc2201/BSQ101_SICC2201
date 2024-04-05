
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
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp, Statevector
import numpy as np
from numpy.typing import NDArray
from typing import List, Union

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

    observables_expected_values = np.empty((len(time_values), len(observables)))

    w, v = diagonalize_hamiltonian(hamiltonian, time_values)

    b = Statevector(initial_state)

    a0 = v.conj()*b

    a1 = w[0] * a0

    b1 = v * a1

    a2 = w[1] * a1

    #evolve_matrix = np.einsum("sk, ik, jk -> sij", w, v, v.conj())

    return observables_expected_values

def diagonalize_hamiltonian(hamiltonian: SparsePauliOp, time_values: NDArray[np.float_]):
    e, v = np.linalg.eigh(hamiltonian.to_matrix())

    w = time_values[:, None] * e[:, None]

    evolution_operator = np.exp(w)

    print("diag energy: ", e)
    # print("w: ", w)


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
    observables_expected_values = np.empty((len(time_values), len(observables)))
    
    for step in num_trotter_steps:
        trotter_qc = trotter_circuit(hamiltonian, time, num_trotter_steps)

        for time in time_values:
            initial_state.compose(trotter_circuit(hamiltonian, time, num_trotter_steps))
            # mesurer pour tous les obsrvables

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
    magnetic_field = hamiltonian.coeffs
    num_qubits = hamiltonian.num_qubits
    Rz_param = Parameter("wt")
    qc = QuantumCircuit(num_qubits)

    # only for magnetic field in Z for now
    if num_qubits > 1:
        cnot_controls = [(i, i + 1) for i in range(num_qubits - 1)]
        for control, target in cnot_controls:
            qc.cx(control, target)
        qc.rz(Rz_param, num_qubits -  1)
        for control, target in reversed(cnot_controls):
            qc.cx(control, target)
    else:
        qc.rz(Rz_param, num_qubits -  1)

    qc.bind_parameters(total_duration*magnetic_field[0]/num_trotter_steps)
    return qc

def random_pauli_op(dimension):
    """
    Generate a random SparsePauliOp of dimension `dimension`.
    """
    pauli_labels = ['I', 'X', 'Y', 'Z']
    pauli_matrix = np.random.choice(pauli_labels, size=dimension)
    return SparsePauliOp.from_label(''.join(pauli_matrix))

def create_random_hamiltonian(num_qubits, dimension = 2): 
    """
    Generate a random Hamiltonian with `num_terms` terms of dimension `dimension`.
    """

    # Generate random Pauli strings
    pauli_strings = []
    for _ in range(int(num_qubits * dimension)):
        pauli = "".join(np.random.choice(["I", "X", "Y", "Z"]) for _ in range(num_qubits))
        pauli_strings.append(Pauli(pauli))

    # Generate random coefficients
    coefficients = np.random.rand(len(pauli_strings))

    # Create SparsePauliOp
    sparse_pauli_op = SparsePauliOp(pauli_strings, coefficients)
    return sparse_pauli_op

def create_single_spin_hamiltonian(theta: float):
    return SparsePauliOp(["Z", "Y"], [np.cos(theta)*(-0.5), np.sin(theta)*(-0.2)])

def create_two_spin_hamiltonian(theta: float):
    return SparsePauliOp(["Z", "Y"], [np.cos(theta)*(-0.5), np.sin(theta)*(-0.2)])

def create_random_initial_state(num_qubits: int):
    qc = QuantumCircuit(num_qubits)
    return qc

def create_single_spin_initial_state():
    qc = QuantumCircuit(1)
    qc.h(0)
    return qc

def create_two_spin_initial_state():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.x(1)

    return qc

def create_observables(num_qubits: int):
    ones = np.ones(num_qubits)
    zeros = np.zeros(num_qubits)
    pauli_x = Pauli((zeros, ones))
    pauli_y = Pauli((ones, ones))
    pauli_z = Pauli((ones, zeros))
    return PauliList([pauli_x, pauli_y, pauli_z])

