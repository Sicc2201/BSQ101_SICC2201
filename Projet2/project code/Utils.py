##########################################################################

# Titre: QuantumUtils.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 23/01/2024

##########################################################################
'''
# Description:

This file contains all general methods that needs to be called to create the circuit.


# Methods:

- bitstring_to_bits(bit_string: str) : convert a bit string array into a bool array.

- save_histogram_png(counts: dict, title: str) : saves an histogram of your results as a png file.

- execute_job(circuit: QuantumCircuit, backend: Backend, execute_opts: dict : run the quantum job and return the counts

-  create_all_pauli(num_qubits: int) -> PauliList: Create all possible 4^nb_qubits Pauli gates.

-  create_random_quantum_circuit(num_qubits): create a secret random circuit (just H for now)

'''

###########################################################################

# IMPORTS

###########################################################################

import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit import transpile, QuantumCircuit
from qiskit.providers.backend import Backend
import numpy as np
from numpy.typing import NDArray
from itertools import product
from qiskit.quantum_info import Pauli, PauliList

###########################################################################

# Methods

###########################################################################

def save_histogram_png(counts: dict, title: str):

    figure, plot = plt.subplots(figsize=(9, 8))

    plot_histogram(counts, ax=plot)

    plot.set_title(title + " result counts")
    plot.set_xlabel("Possible solutions")
    plot.set_ylabel("Counts")
    plt.savefig(title + ".png")  

def execute_job(circuit: QuantumCircuit, backend: Backend, execute_opts: dict) -> dict:
    print("run job")
    transpiled_qc = transpile(circuit, backend)
    job = backend.run(transpiled_qc, execute_opts)
    return job.result().get_counts()

def bitstring_to_bits(bit_string: str) -> NDArray[np.bool_]:
    return np.array([x == '1' for x in bit_string], dtype=bool)[::-1]

def create_all_pauli(num_qubits: int) -> PauliList:
    pauli_bits_combinations = product([0,1], repeat=num_qubits)
    pauli_zx_permutations = list(product(pauli_bits_combinations, repeat=2))
    pauli_list = [Pauli(pauli) for pauli in pauli_zx_permutations]
    return PauliList(pauli_list)

def create_random_quantum_circuit(num_qubits):
    qc = QuantumCircuit(num_qubits)
    qc.h(qc.qubits)

    return qc
