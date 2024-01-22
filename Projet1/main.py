##########################################################################

# Titre: main.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 21/01/2024

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

"""Commentaires
Il faudrait valider que la solution est la bonne. Si le format d'une solution est bien un dict on peut simplement faire
    cnf_cake.subs(solution)
Peut-être solutionner les deux problèmes un à la suite de l'autre?
C'est le bon endroit pour afficher des histogrammes et autre.
"""
def main():

    ibmq_token = "put your token here"
    backend_name = "ibmq_qasm_simulator"

    ####  uncomment if this is the first time you connect to your IBMQ account  ##########

    # IBMQ_credentials.ibmq_connexion(ibmq_token)

    #################################################################################

    provider, backend = IBMQ_credentials.ibmq_provider(backend_name)
    
    # get cnf from problems
    cnf_cake = bool.create_cake_problem()
    cnf_pincus = bool.create_pincus_problem()

    # calculate the results of the SAT
    cake_result = grover.solve_sat_with_grover(cnf_cake, grover.cnf_to_oracle(cnf_cake), backend)
    # pincus_result = grover.solve_sat_with_grover(cnf_pincus, grover.cnf_to_oracle(cnf_pincus), backend)
    plt.show()

    return 0

if __name__ == "__main__":
    main()
