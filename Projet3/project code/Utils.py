##########################################################################

# Titre: Utils.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 15/02/2024

##########################################################################
'''
# Description:

This file contains all general methods that needs to be called to create the circuit.


# Methods:

- extract_data(file_name:str, file_path: str): 

- execute_job(circuit: QuantumCircuit, backend: Backend, execute_opts: dict : run the quantum job and return the counts

-  create_all_pauli(num_qubits: int) -> PauliList: Create all possible 4^nb_qubits Pauli gates.

-  create_random_quantum_circuit(num_qubits : int): create a secret random circuit

'''

###########################################################################

# IMPORTS

###########################################################################

from qiskit import transpile, QuantumCircuit, assemble
from qiskit.providers.backend import Backend
from qiskit.circuit.library import ZGate, HGate, XGate, YGate, SGate
import numpy as np
import os
from numpy.typing import NDArray
from itertools import product
from qiskit.quantum_info import Pauli, PauliList
from typing import List
import random

###########################################################################

# Methods

###########################################################################

def execute_job(circuit: List[QuantumCircuit] , backend: Backend, execute_opts: dict) -> dict:
    transpiled_qc = transpile(list(circuit), backend)
    queue_job = assemble(transpiled_qc)
    job = backend.run(queue_job, **execute_opts)
    return job.result()

def bitstring_to_bits(bit_string: str) -> NDArray[np.bool_]:
    return np.array([x == '1' for x in bit_string], dtype=bool)[::-1]

def create_all_pauli(num_qubits: int) -> PauliList:
    pauli_bits_combinations = product([0,1], repeat=num_qubits)
    pauli_zx_permutations = list(product(pauli_bits_combinations, repeat=2))
    pauli_list = [Pauli(pauli) for pauli in pauli_zx_permutations]
    return PauliList(pauli_list)


def extract_data(filename:str, datapath: str):
    filepath = os.path.join(datapath, filename)
    npzfile = np.load(filepath)
    return npzfile["distance"], npzfile["one_body"], npzfile["two_body"], npzfile["nuclear_repulsion_energy"]


