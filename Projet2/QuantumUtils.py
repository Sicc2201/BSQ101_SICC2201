##########################################################################

# Titre: QuantumUtils.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 23/01/2024

##########################################################################
'''
# Description:

This file contains all general methods that needs to be called to create the circuit that are not specificly Grover related.


# Methods:

- save_histogram_png(counts: dict, title: str) : saves an histogram of your results as a png file.

- execute_job(circuit: QuantumCircuit, backend: Backend, execute_opts: dict : run the quantum job and return the counts

'''

###########################################################################

# IMPORTS

###########################################################################

import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit import transpile, QuantumCircuit
from qiskit.providers.backend import Backend
import numpy as np

from qiskit.tools.monitor import job_monitor

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

def execute_job(circuit: QuantumCircuit, backend: Backend, execute_opts: dict):
    print("run job")
    transpiled_qc = transpile(circuit, backend)
    job = backend.run(transpiled_qc, execute_opts)
    return job.result().get_counts()