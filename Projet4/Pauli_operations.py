
##########################################################################

# Titre: Pauli_operation.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 15/02/2024

##########################################################################
'''
# Description: 

 Ce fichier contient toutes fonctions qui gèrent les opérations à utiliser sur les Pauli et chaînes de Pauli utilisés dans la tomographie.

# Methods:

- diag_pauli_expectation_value(pauli: Pauli, counts: dict) -> float: Calcule la valeur moyenne d'une chaîne de Pauli diagonale.

- diagonalize_pauli_with_circuit(pauli : Pauli) -> Tuple[Pauli, QuantumCircuit]: Applique les transformations nécessaires au circuit mystère pour mesurer dans la bonne base de calcul selon la chaîne de Pauli en entrée et diagonalise la pauli en entrée.

- diagonalize_pauli(z_bits : NDArray[np.bool_], x_bits: NDArray[np.bool_]) -> Pauli : Diagonalise une chaîne de Pauli selon sa représentation zx.

- measure_pauli_circuit(state_circuit : QuantumCircuit, pauli_qc : QuantumCircuit) -> QuantumCircuit : Ajoute une chaîne de Pauli diagonale au circuit mystère et le mesure.

- estimate_expectation_values(
    pauli_list: PauliList,
    state_circuit: QuantumCircuit,
    backend: Backend,
    execute_opts : dict = dict()) -> NDArray[np.float_]: Calcule les valeurs moyennes des chaînes de Pauli.


'''

###########################################################################

# IMPORTS

###########################################################################
from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

#custom library
import Utils

def diag_pauli_expectation_value(pauli: Pauli, counts: dict) -> float:

    assert(np.all(~pauli.x))
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
        if x == 1:
            if z == 1:
                qc.sdg(index)
            qc.h(index)

    pauli = diagonalize_pauli(z_bits, x_bits)

    assert(np.all(~pauli.x))

    return (pauli, qc)

def diagonalize_pauli(z_bits : NDArray[np.bool_], x_bits: NDArray[np.bool_]) -> Pauli:
    diag_z_bits = np.logical_or(z_bits, x_bits)
    return Pauli((diag_z_bits, np.zeros(len(z_bits), dtype=bool)))

def measure_pauli_circuit(state_circuit : QuantumCircuit, pauli_qc : QuantumCircuit) -> QuantumCircuit:
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
