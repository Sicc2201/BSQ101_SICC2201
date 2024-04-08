
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
from qiskit.primitives import Estimator
from qiskit.circuit import Parameter
from qiskit.providers.backend import Backend
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp, Statevector
import numpy as np
from numpy.typing import NDArray
from typing import List, Union
import matplotlib.pyplot as plt

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
    b = np.empty(len(time_values))    
    estimator = Estimator()

    exponent_matrix, v = diagonalize_hamiltonian(hamiltonian, time_values)
    b0 = Statevector(initial_state)

    # for 2 qubits for now
    b1 = v * exponent_matrix[0] * v.conj() * b0
    b2 = v * exponent_matrix[1] * v.conj() * b1

    # je suis un peu confus opur cette partie, maintenant que j'ai mes state vector pour mes états à chaque temps, qu'est-ce que je veux en extraire?

    return observables_expected_values

def diagonalize_hamiltonian(hamiltonian: SparsePauliOp, time_values: NDArray[np.float_]):
    e, v = np.linalg.eigh(hamiltonian.to_matrix())

    w = time_values[:, None] * e[:, None]

    exponent_matrix = np.exp(-1j * w)

    return exponent_matrix, v

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

    estimator = Estimator()
    observables_expected_values = np.array([]) 
    
    num_qubits = hamiltonian.num_qubits
    
    # assuming optimization if circuits created only one time since it is the same for every circuit
    control_operator_qc = create_control_not_steps(num_qubits)

    for time in time_values:
        jobs = []       
        for step in range(num_trotter_steps):
            initial_state &= create_trotter_qc(hamiltonian, time, num_trotter_steps, control_operator_qc)

        # pour créer les observables_expected_values, je dois faire une liste de circuits de la taille de la quantité d'observable, 
        # mais est-ce que tous les circuis sont pareil et estimator se charge d'effectuer les bonne operations sur mes ciurcuits pour calculer les observables? comme j'ai fait?
        for observable in range(len(observables)):
            jobs.append(initial_state)

        results = estimator.run(jobs, observables)
        observables_expected_values = np.concatenate((observables_expected_values, results.result().values))
        print("obsevables shape: ", observables_expected_values.shape)


    observables_expected_values = np.reshape(observables_expected_values, (len(time_values), len(observables)))
    print(observables_expected_values.shape)

    return observables_expected_values

def create_trotter_qc(
hamiltonian: SparsePauliOp,
total_duration: Union[float, Parameter],
num_trotter_steps: int,
control_operator_qc
) -> QuantumCircuit:

    trotter_qc = QuantumCircuit(hamiltonian.num_qubits)
    for magnetic_field, pauli_list in zip(hamiltonian.coeffs, hamiltonian.paulis):
        diag_pauli_qc = create_diag_pauli_circuit(pauli_list)
        trotter_qc &= trotter_circuit(hamiltonian, total_duration,  num_trotter_steps, control_operator_qc, magnetic_field, diag_pauli_qc)
        
    return trotter_qc

def trotter_circuit(
hamiltonian: SparsePauliOp,
total_duration: Union[float, Parameter],
num_trotter_steps: int,
control_operator_qc,
magnetic_field,
diag_operator_circuit
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

    trotter_qc = diag_operator_circuit.inverse()

    trotter_qc &= control_operator_qc.inverse()
    trotter_qc.rz(total_duration*(-2)*magnetic_field.real/num_trotter_steps, hamiltonian.num_qubits -  1)
    trotter_qc &= control_operator_qc

    trotter_qc &= diag_operator_circuit

    return trotter_qc

def create_control_not_steps(num_qubits: int):

    cnot_controls = [(i, i + 1) for i in range(num_qubits - 1)]
    qc = QuantumCircuit(num_qubits)
    if num_qubits > 1:
        for control, target in cnot_controls:
            qc.cx(control, target)

    return qc

def create_diag_pauli_circuit(pauli_list: PauliList):
    diag_qc = QuantumCircuit(pauli_list.num_qubits)
    for pauli in pauli_list:
        _, pauli_qc = po.diagonalize_pauli_with_circuit(pauli)
        diag_qc &= pauli_qc
    return diag_qc

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
    return SparsePauliOp(["IZ", "IZ"], [1.05, 0.95])

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

    observables = []
    zeros = np.zeros((num_qubits, num_qubits))
    identity = np.eye(num_qubits)

    pauli_z = np.concatenate((zeros, identity), axis=0)
    pauli_z = np.concatenate((pauli_z, identity), axis=0)
    pauli_x = np.concatenate((identity, identity), axis=0)
    pauli_x = np.concatenate((pauli_x, zeros), axis=0)

    for z, x in zip(pauli_z, pauli_x):
        observables.append(SparsePauliOp(PauliList.from_symplectic(z, x)))

    return observables

