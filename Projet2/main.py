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

    
    print(utils.bitstring_to_bits("0011001"))


    return 0


if __name__ == "__main__":
    main()
