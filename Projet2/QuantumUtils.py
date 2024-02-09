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

- validate_grover_solutions(results: list[dict], cnf: And) : validate the result from the sumulation with sympy.

'''

###########################################################################

# IMPORTS

###########################################################################

import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit import transpile
import numpy as np

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

def execute_job(circuit, backend, execute_opts):
    transpiled_qc = transpile(circuit, backend)
    job = backend.run(transpiled_qc, options = execute_opts)
    return job.result().get_counts()