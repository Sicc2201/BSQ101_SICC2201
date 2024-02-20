
##########################################################################

# Titre: State_tomography.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 15/02/2024

##########################################################################
'''
# Description: 

Ce fichier contient toutes fonctions qui gèrent la tomographie.
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
from qiskit.providers.backend import Backend
from qiskit.quantum_info import PauliList, Statevector
import numpy as np
from numpy.typing import NDArray

#custom library
import Utils
import Pauli_operations as po

###########################################################################

# METHODS

###########################################################################

def state_tomography(
    state_circuit: QuantumCircuit,
    backend: Backend,
    execute_opts : dict = dict()) -> NDArray[np.complex_]:

    pauli_list = Utils.create_all_pauli(state_circuit.num_qubits)
    expectation_values = po.estimate_expectation_values(pauli_list,state_circuit,backend,execute_opts)
    scaled_expectation_values =  [value /(2**state_circuit.num_qubits) for value in expectation_values]

    density_matrix = calculate_density_matrix(pauli_list, scaled_expectation_values)
    state_vector = calculate_state_vector(density_matrix)

    return state_vector

def calculate_density_matrix(pauli_list: PauliList, expectation_values: NDArray[np.float_]) -> NDArray[np.complex_]:
    density_matrix = np.sum(np.multiply(value, pauli.to_matrix()) for value, pauli in zip(expectation_values, pauli_list))

    return density_matrix

def calculate_state_vector(density_matrix : NDArray[np.complex_]) -> NDArray[np.complex_]:

    eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
    max_value = np.argmax(eigenvalues)
    state_vector = eigenvectors[:, max_value]
    return state_vector

def validate_state_vector(qc : QuantumCircuit, estimated_state_vector : NDArray[np.complex_]):

    expected_state_vector = np.array(Statevector(qc))
    print("vecteur d'état espéré: ", expected_state_vector)
    print("vecteur d'état estimé: ", estimated_state_vector)
    inner_product = np.dot(np.conj(estimated_state_vector), expected_state_vector)
    norm_product = np.linalg.norm(estimated_state_vector) * np.linalg.norm(expected_state_vector)
    return np.abs(inner_product / norm_product)**2