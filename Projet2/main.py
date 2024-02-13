##########################################################################

# Titre: main.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 02/02/2024

##########################################################################
'''
# Description: 

This file is the main file, the base project is build here.

'''

###########################################################################

# IMPORTS

###########################################################################

# python libraries


# Custom libraries
import IBMQ_credentials
import utils

###########################################################################

# MAIN

###########################################################################


def main():

    ibmq_token = "put your token here"
    backend_name = "ibmq_qasm_simulator"

    ####  uncomment if this is the first time you connect to your IBMQ account  ##########

    # IBMQ_credentials.ibmq_connexion(ibmq_token)

    #################################################################################
    # backend, execute_opts = 0
    provider, backend = IBMQ_credentials.ibmq_provider(backend_name)

  

    num_qubits = 2

    execute_opts = {'shots': 256}

    state_circuit = utils.create_random_quantum_circuit(num_qubits)

    state_vector = utils.state_tomography(state_circuit,backend,execute_opts)
    print(state_vector)


    return 0


if __name__ == "__main__":
    main()
