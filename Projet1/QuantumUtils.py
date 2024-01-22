##########################################################################

# Titre: QuantumUtils.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 21/01/2024

##########################################################################
'''
# Description:

This file contains all general methods that needs to be called to create the circuit that are not specificly Grover related.


# Methods:

- initialize_s(qc: QuantumCircuit, first: int, last: int) : 

- mesure_qubits(qc, nqubits) : Mesures the qubits on a quantum circuit from a starting point to an end point.

- quantum_results_to_boolean(results: list, atoms: list) : Converts the results from the quantum operation to an easy to understand format.

- calculate_threshold(results: list) : Calculates the threshold to separate the True and False clusters in the data with a K-means.

'''

###########################################################################

# IMPORTS

###########################################################################
from qiskit import QuantumCircuit


# from sklearn.cluster import KMeans

###########################################################################

# Methods

###########################################################################

"""Commentaires
Est-ce que c'est plus clean que juste
    qc.h(range(last - first))
? Est-ce que ça vaut une fonction?
"""
def initialize_s(qc: QuantumCircuit, first: int, last: int):
    """Apply a H-gate to 'qubits' in qc"""
    for q in range(last - first):
        qc.h(q + first)
    return qc

"""Commentaires
Même chose ici. Cela pourrait se faire en une ligne dans le fichier de Grover
"""
def mesure_qubits(qc: QuantumCircuit, nqubits: int):
    for i in range(nqubits):
        qc.measure(i, i)

"""Commentaires
Ce n'est pas clair ici que results est une list de tuple. Ça rend le code difficile à comprend au début.
Cette fontion devrait retourner une list de dict. Chaque dict est une solution.
Les keys sont les variables (atoms) et les value True/False.
"""
def quantum_results_to_boolean(results: list, atoms: list):

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
                raise ValueError("There is no solution")

    return boolean_solutions

"""Commentaires
Bonne idée qui reste a être implémentée
"""
def calculate_threshold(results: list):
    # nb_cluster = 2
    # kmeans = KMeans(nb_cluster)
    # kmeans.fit([t[1] for t in results])
    # centroids = kmeans.cluster_centers_
    # sorted_centroids = np.sort(centroids, axis=0)
    # threshold = (sorted_centroids[0] + sorted_centroids[1]) / 2
    threshold = 500
    print("threshold: ", threshold)
    return threshold

""" La convention des noms de fichier n'est pas claire. """
