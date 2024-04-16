
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

    exponent_matrix, v = diagonalize_hamiltonian(hamiltonian, time_values)
    initial_statevector = Statevector(initial_state)

    evolution_operator = np.einsum("ki, sk, kj -> sij", v, exponent_matrix, v.conj())

    evolved_evolution_operator = np.einsum("sij, j-> si", evolution_operator, initial_statevector)

    observables_matrix = [np.stack(observable.to_matrix()) for observable in observables]

    observables_expected_values = np.einsum("si, pij, sj -> sp", evolved_evolution_operator, observables_matrix, evolved_evolution_operator.conj())

    return observables_expected_values

def diagonalize_hamiltonian(hamiltonian: SparsePauliOp, time_values: NDArray[np.float_]):
    e, v = np.linalg.eigh(hamiltonian.to_matrix())
    w = np.einsum("s, i -> si", time_values, e)
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
    evolved_state = initial_state.copy()
    jobs = []   
    for time, step in zip(time_values, num_trotter_steps):
        evolved_state = evolved_state.compose(trotter_circuit_per_time(hamiltonian, time, step)) 
        for _ in range(len(observables)):
            jobs.append(evolved_state)

    print("Computing trotter estimation")
    results = estimator.run(jobs, [observable.paulis for observable in observables] * len(time_values))
    observables_expected_values = results.result().values

    observables_expected_values = np.reshape(observables_expected_values, (len(time_values), len(observables)))

    return observables_expected_values

def trotter_circuit_per_time(
hamiltonian: SparsePauliOp,
total_duration: Union[float, Parameter],
num_trotter_steps: int
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
   
    trotter_qc = QuantumCircuit(hamiltonian.num_qubits)
    for _ in range(num_trotter_steps):
        trotter_qc = trotter_qc.compose(trotter_step(hamiltonian, total_duration/num_trotter_steps))

    return trotter_qc

def trotter_step(
hamiltonian: SparsePauliOp,
trotter_time_step: float
) -> QuantumCircuit:
    """
    Construct the QuantumCircuit of a single trotter step.

    Args:
    hamiltonian (SparsePauliOp): The Hamiltonian which governs the evolution.
    trotter_time_step: float: time value per trotter step.

    Returns:
    QuantumCircuit: The circuit of the Trotter evolution operator
    """   
    
    trotter_step_qc = QuantumCircuit(hamiltonian.num_qubits)
    for coeff, pauli_list in zip(hamiltonian.coeffs, hamiltonian.paulis):
        trotter_step_qc = trotter_step_qc.compose(hamiltonian_pauli_circuit((-2)*coeff*trotter_time_step, pauli_list))

    return trotter_step_qc

def hamiltonian_pauli_circuit(rz_theta: float, pauli_list: PauliList) -> QuantumCircuit:
    """
    Construct the QuantumCircuit for a single Pauli chain of the Hamiltonian.

    Args:
    rz_theta (float): phase parameter of the R_z gate .
    pauli_list (PauliList): The pauli chain on which we build the circuit

    Returns:
    QuantumCircuit: QuantumCircuit for a single Pauli chain of the Hamiltonian
    """
    diag_pauli_list, diag_pauli_qc = create_diag_pauli_circuit(pauli_list)  
    control_step_qc, last_qubit_index = create_cx_steps(diag_pauli_list)

    hamiltonian_pauli_qc = diag_pauli_qc.copy()
    hamiltonian_pauli_qc = hamiltonian_pauli_qc.compose(control_step_qc)
    if last_qubit_index != None:
        hamiltonian_pauli_qc.rz(rz_theta.real, last_qubit_index)
    hamiltonian_pauli_qc = hamiltonian_pauli_qc.compose(control_step_qc.inverse())
    hamiltonian_pauli_qc = hamiltonian_pauli_qc.compose(diag_pauli_qc.inverse())

    return hamiltonian_pauli_qc

def create_cx_steps(pauli_list: PauliList) -> QuantumCircuit:
    """
    Construct the QuantumCircuit for the cnot steps.

    Args:
    pauli_list (PauliList): The pauli chain on which we build the circuit

    Returns:
    QuantumCircuit: QuantumCircuit that contains the cnot steps.
    """
    qc = QuantumCircuit(pauli_list.num_qubits)
    z_bits = pauli_list.z[0]
    index_array = np.where(z_bits[z_bits == 1])[0]

    last_qubit_index = index_array[-1]
    
    if(len(index_array) > 1):
        controls = np.delete(index_array, -1, 0)
        targets = np.delete(index_array, 0, 0)
        qc.cx(controls, targets)

    return qc, last_qubit_index

def create_diag_pauli_circuit(pauli_list: PauliList) -> QuantumCircuit:
    diag_qc = QuantumCircuit(pauli_list.num_qubits)
    diag_paulis_z = []
    diag_paulis_x = []
    for pauli in pauli_list:
        diag_pauli, pauli_qc = po.diagonalize_pauli_with_circuit(pauli)
        diag_qc.compose(pauli_qc)
        diag_paulis_z.append(diag_pauli.z)
        diag_paulis_x.append(diag_pauli.x)

    diag_pauli_z = np.array(diag_paulis_z).reshape(1,pauli_list.num_qubits)
    diag_pauli_x = np.array(diag_paulis_x).reshape(1,pauli_list.num_qubits)
    return PauliList.from_symplectic(diag_pauli_z, diag_pauli_x), diag_qc

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
    return SparsePauliOp(["IZ", "IZ", "XX"], [1.05, 0.95, 0.20])

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

def create_observables(num_qubits: int) -> List[SparsePauliOp]:

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

