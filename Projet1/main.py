##########################################################################

# Titre: main.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 23/01/2024

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
import BooleanProblems as bool
import QuantumUtils as utils

from sympy import *

###########################################################################

# MAIN

###########################################################################


def main():

    ibmq_token = "put your token here"
    backend_name = "ibmq_qasm_simulator"

    ####  uncomment if this is the first time you connect to your IBMQ account  ##########

    # IBMQ_credentials.ibmq_connexion(ibmq_token)

    #################################################################################

    provider, backend = IBMQ_credentials.ibmq_provider(backend_name)
    print("\n*****************************   Creating CNF   ***************************************\n")  
    # get cnf from problems
    cnf_cake = bool.create_cake_problem()
    cnf_pincus = bool.create_pincus_problem()

    # calculate the results of the SAT
    print("\n*****************************   Resolving Cake Problem   ***************************************\n")
    cake_result = grover.solve_sat_with_grover(cnf_cake, grover.cnf_to_oracle(cnf_cake), backend, True, "cake_problem")
    print("We have ", len(cake_result), " solution(s) -> ", cake_result, "\n")
    print("************************************************************************************************\n")

    print("*************************    Resolving Planet Pincus Problem   ***************************************\n")
    pincus_result = grover.solve_sat_with_grover(cnf_pincus, grover.cnf_to_oracle(cnf_pincus), backend, True, "pincus_problem")
    print("We have ", len(pincus_result), " solution(s) -> ", pincus_result, "\n")
    print("*********************************************************************************************************")

    utils.validate_grover_solutions(cake_result, cnf_cake)
    utils.validate_grover_solutions(pincus_result, cnf_pincus)

    return 0


if __name__ == "__main__":
    main()
