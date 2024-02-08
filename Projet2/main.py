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

    # provider, backend = IBMQ_credentials.ibmq_provider(backend_name)

    
    chains = utils.create_all_pauli(3)
    # Access the PauliList's pauli_list attribute
    print(chains[60].x)

    return 0


if __name__ == "__main__":
    main()
