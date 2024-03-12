
##########################################################################

# Titre: Quantum_chemistry.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 15/02/2024

##########################################################################
'''
# Description: 

Ce fichier contient toutes fonctions qui gèrent le processus.
# Methods:

- state_tomography(
    state_circuit: QuantumCircuit,
    backend: Backend,
    execute_opts : dict = dict()) -> NDArray[np.complex_]: Gère la structure de la tomographie.

-  calculate_density_matrix(pauli_list: PauliList, expectation_values: List): Calcule la matrice de densité en multipliant la valeur moyenne des chaînes de
Pauli avec la chaîne de Pauli associée

- calculate_stateVector(density_matrix) : Calcule le vecteur d'etat en trouvant le vecteur propre associé à la valeur propre la
plus grande de la matrice de densité du système.
'''

###########################################################################

# IMPORTS

###########################################################################
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers.backend import Backend
from qiskit.quantum_info import PauliList, SparsePauliOp
import numpy as np
from numpy.typing import NDArray
from typing import List, Callable
from scipy.optimize import OptimizeResult, minimize

#custom library
import Utils
import Pauli_operations as po

###########################################################################

# METHODS

###########################################################################

def quantum_chemistry(
one_body: NDArray[np.complex_],
two_body: NDArray[np.complex_],
state_circuit: QuantumCircuit,
backend: Backend,
execute_opts : dict = dict()) -> NDArray[np.complex_]:



    return 0

def create_initial_quantum_circuit(num_qubits : int):
    qc = QuantumCircuit(num_qubits, num_qubits)
    ry_param = Parameter("theta")
    qc.ry(ry_param, 1)
    qc.x(1)
    qc.cx(1, 0)
    qc.x(1)
    qc.cx(0, 2)
    qc.cx(1, 3)

    print(qc)

    return qc


def annihilation_operators_with_jordan_wigner(num_states: int) -> List[SparsePauliOp]:
    """
    Builds the annihilation operators as sum of two Pauli Strings for given number offermionic
    states using the Jordan Wigner mapping.
    Args:
    num_states (int): Number of fermionic states.
    Returns:
    List[SparsePauliOp]: The annihilation operators
    """
    # on sait le nombre d'annihilator
    annihilation_operators = np.empty(num_states, dtype=SparsePauliOp)
    z1_bits = np.zeros(num_states, dtype=bool)
    z2_bits = np.zeros(num_states, dtype=bool)
    print(z2_bits)

    for index in range(num_states):
        x1_bits = np.zeros(num_states, dtype=bool)
        x1_bits[index] = True
        z2_bits[index] = True

        if index != 0:
            z1_bits[index - 1] = True
            z2_bits[index - 1] = True
        
        paulis = PauliList.from_symplectic([z1_bits, z2_bits], [x1_bits, x1_bits])
        annihilation_operators[index] = 0.5 * SparsePauliOp(paulis, [1, 1j])
        print(paulis)

    return annihilation_operators

def build_qubit_hamiltonian(
one_body: NDArray[np.complex_],
two_body: NDArray[np.complex_],
annihilation_operators: List[SparsePauliOp],
creation_operators: List[SparsePauliOp],
) -> SparsePauliOp:
    """
    Build a qubit Hamiltonian from the one body and two body fermionic Hamiltonians.
    Args:
    one_body (NDArray[np.complex_]): The matrix for the one body Hamiltonian
    two_body (NDArray[np.complex_]): The array for the two body Hamiltonian
    annihilation_operators (List[SparsePauliOp]): List of sums of two Pauli strings
    creation_operators (List[SparsePauliOp]): List of sums of two Pauli strings (adjoint of
    annihilation_operators)
    Returns:
    SparsePauliOp: The total Hamiltonian as a sum of Pauli strings
    """
    qubit_hamiltonian = 0
    one_body_sum = 0
    two_body_sum = 0

    for i in range(len(annihilation_operators)):
        for j in range(len(annihilation_operators)):
            one_body_sum += one_body[i][j]*creation_operators[i].compose(annihilation_operators[j])
            for k in range(len(annihilation_operators)):
                for l in range(len(annihilation_operators)): 
                    a_ij = creation_operators[i].compose(creation_operators[j])
                    a_kl = annihilation_operators[k].compose(annihilation_operators[l])
                    two_body_sum += two_body[i][j][k][l]*a_ij.compose(a_kl)

    qubit_hamiltonian = one_body_sum + 0.5*two_body_sum
    
    return qubit_hamiltonian

def minimize_expectation_value(
observable: SparsePauliOp,
ansatz: QuantumCircuit,
starting_params: list,
backend: Backend,
minimizer: Callable,
execute_opts: dict = {},
) -> OptimizeResult:
    """
    Uses the minimizer to search for the minimal expection value of the observable for the
    state that the ansatz produces given some parameters.
    Args:
    observable (SparsePauliOp): The observable which the expectation value will be
    minimized.
    ansatz (QuantumCircuit): A paramtrized quantum circuit used to produce quantum state.
    starting_params (list): The initial parameter of the circuit used to start the
    minimization.
    backend (Backend): A Qiskit backend on which the cirucit will be executed.
    minimizer (Callable): A callable function, based on scipy.optimize.minimize which only
    takes a function and starting params as inputs.
    execute_opts (dict, optional): Options to be passed to the Qsikit execute function.
    Returns:
    OptimizeResult: The result of the optimization
    """

    result = minimizer(cost_function, starting_params)

    return result

def minimizer(cost_function:Callable, starting_params)-> OptimizeResult:
    return minimize(cost_function, starting_params, method='COBYLA')

def cost_function(params):
    return


def exact_minimal_eigenvalue(observable: SparsePauliOp) -> float:
    """
    Computes the minimal eigenvalue of an observable.

    Args:
    observable (SparsePauliOp): The observable to diagonalize.

    Returns:
    float: The minimal eigenvalue of the observable.
    """

    return #minimal_eigenvalue