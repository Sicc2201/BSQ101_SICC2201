##########################################################################

# Titre: main.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 12/01/2024

##########################################################################
'''
# Description: 

This file is the main file, the base project is build here.

'''

###########################################################################

# IMPORTS

###########################################################################

# Custom libraries
import IBMQ_credentials
import GroverUtils as grover
import booleanProblems as bool
from qiskit.visualization import plot_histogram, array_to_latex, plot_gate_map
import matplotlib.pyplot as plt


###########################################################################

# MAIN

###########################################################################


def main():

    ibmq_token = "put your token here"
    backend_name = "ibmq_qasm_simulator"

    ####  uncomment if this is the first time you connect to your account  ##########

    # IBMQ_credentials.ibmq_connexion(ibmq_token)

    #################################################################################

    provider, backend = IBMQ_credentials.ibmq_provider(backend_name)
    
    cnf_cake = bool.create_cake_problem()
    cnf_pincus = bool.create_pincus_problem()

    qc, result = grover.solve_sat_with_grover(cnf_cake, grover.cnf_to_oracle(cnf_cake), backend)
    print(result)
    
    plt.show()

    return 0

if __name__ == "__main__":
    main()
