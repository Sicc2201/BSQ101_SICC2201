##########################################################################

# Titre: main.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 15/02/2024

##########################################################################
'''
# Description: 

C'est ce fichier qui sera lancé par l'utilisateur pour lancer l'algorithme.

'''

###########################################################################

# IMPORTS

###########################################################################

# Custom libraries
import IBMQ_credentials
import Quantum_chemistry as chem
import Utils

###########################################################################

# MAIN

###########################################################################


def main():

    ibmq_token = "put your token here"
    backend_name = "ibmq_qasm_simulator"
    filename = "h2_mo_intergrals_d_0750.npz"
    datapath = "h2_mo_integrals/"

    ####  uncomment if this is the first time you connect to your IBMQ account  ##########

    # IBMQ_credentials.ibmq_connexion(ibmq_token)

    #################################################################################

    #_, backend = IBMQ_credentials.ibmq_provider(backend_name)

    backend = IBMQ_credentials.get_local_simulator()

    #dist, oneb, twob, energy = Utils.extract_data(filename, datapath)

    num_qubits = 4

    chem.annihilation_operators_with_jordan_wigner(num_qubits)


    return 0


if __name__ == "__main__":
    main()
