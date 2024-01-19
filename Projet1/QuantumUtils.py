##########################################################################

# Titre: QuantumUtils.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 19/01/2024

##########################################################################
'''
# Description:

This file contains all general methods that needs to be called to create the circuit that are not specificly Grover related.


# Methods:



'''

###########################################################################

# IMPORTS

###########################################################################
from qiskit import QuantumCircuit


# from sklearn.cluster import KMeans

###########################################################################

# Methods

###########################################################################

def initialize_s(qc: QuantumCircuit, first: int, last: int):
    """Apply a H-gate to 'qubits' in qc"""
    for q in range(last - first):
        qc.h(q + first)
    return qc


def mesure_qubits(qc, nqubits):
    for i in range(nqubits):
        qc.measure(i, i)


def quantum_results_to_boolean(results, atoms: list):

    boolean_solutions = {}
    threshold = calculate_threshold(results)

    for res in results:
        if res[1] > threshold:
            index_participant = [index for index, char in enumerate(res[0][::-1]) if char == "1"]

            if len(index_participant) == 1:
                boolean_solutions[atoms[index_participant[0]]] = True

            elif len(index_participant) > 1:
                culprits = []
                for index in index_participant:
                    culprits.append(atoms[index])
                boolean_solutions[tuple(culprits)] = True
            else:
                raise ValueError("Ther is no solution")

    return boolean_solutions


def calculate_threshold(results):
    # nb_cluster = 2
    # kmeans = KMeans(nb_cluster)
    # kmeans.fit([t[1] for t in results])
    # centroids = kmeans.cluster_centers_
    # sorted_centroids = np.sort(centroids, axis=0)
    # threshold = (sorted_centroids[0] + sorted_centroids[1]) / 2
    threshold = 500
    print("threshold: ", threshold)
    return threshold