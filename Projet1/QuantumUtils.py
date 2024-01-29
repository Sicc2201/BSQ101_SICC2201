##########################################################################

# Titre: QuantumUtils.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 23/01/2024

##########################################################################
'''
# Description:

This file contains all general methods that needs to be called to create the circuit that are not specificly Grover related.


# Methods:

- quantum_results_to_boolean(results: list, atoms: list) : Converts the results from the quantum operation to an easy to understand format.

- calculate_threshold(results: list) : Calculates the threshold to separate the True and False clusters in the data with a K-means.

- save_histogram_png(counts: dict, title: str) : saves an histogram of your results as a png file.

'''

###########################################################################

# IMPORTS

###########################################################################

# from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from sympy import Symbol
import numpy as np

###########################################################################

# Methods

###########################################################################

def quantum_results_to_boolean(results: dict, atoms: list[Symbol]) -> list[dict]:

    boolean_solutions = []
    print(results)
    threshold = calculate_threshold(list(results.values()))

    for res in results.items():
        solution = False
        dict_solution = {}
        index_participant = [index for index, char in enumerate(res[0][::-1]) if char == "1"]
        for index in index_participant:
            if res[1] > threshold:
                dict_solution[atoms[index]] = True
                solution = True
            else:
                dict_solution[atoms[index]] = False
        if solution == True:
            boolean_solutions.append(dict_solution)

    return boolean_solutions


def calculate_threshold(results: list) -> float:
    # data = np.array(results).reshape(-1, 1)
    # kmeans = KMeans(n_clusters = 2)
    # kmeans.fit(data)
    # centroids = kmeans.cluster_centers_
    # sorted_centroids = np.sort(centroids, axis=0)
    # threshold = (sorted_centroids[0] + sorted_centroids[1]) / 2
    threshold = 500
    print("threshold: ", threshold)
    return threshold

def save_histogram_png(counts: dict, title: str):

    figure, plot = plt.subplots()

    plot_histogram(counts, ax=plot)

    plot.set_title(title + " result counts")
    plot.set_xlabel("Qubits")
    plot.set_ylabel("Counts")
    plt.savefig(title + ".png")  

