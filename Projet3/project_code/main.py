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
import numpy as np
from scipy.optimize import minimize


# Custom libraries
import IBMQ_credentials
import Quantum_chemistry as qchem
import Pauli_operations as po
import Utils

###########################################################################

# MAIN

###########################################################################


def main():

    ibmq_token = "put your token here"
    backend_name = "ibmq_qasm_simulator"
    filename = "h2_mo_integrals_d_0300.npz"
    datapath = "BSQ101_projects/Projet3/project_code/h2_mo_integrals"

    ####  uncomment if this is the first time you connect to your IBMQ account  ##########

    # IBMQ_credentials.ibmq_connexion(ibmq_token)

    #################################################################################

    #_, backend = IBMQ_credentials.ibmq_provider(backend_name)

    backend = IBMQ_credentials.get_local_simulator()
    execute_opts = {'shots': 1024}

    dist, oneb, twob, energy = Utils.extract_data(filename, datapath)


    # print('distance: ', dist, '\none_body: ', oneb, '\ntwo_body: ', twob, '\nenergy: ', energy)

    h2_orbitals = 4

    state_circuit = qchem.create_initial_quantum_circuit(h2_orbitals)


    annihilators = qchem.annihilation_operators_with_jordan_wigner(h2_orbitals)
    creators = [op.adjoint() for op in annihilators]

    hamiltonian = qchem.build_qubit_hamiltonian(oneb, twob, annihilators, creators)
    minimized_energy = qchem.minimize_expectation_value(hamiltonian, state_circuit, [0], backend, minimize, execute_opts)


    return 0


if __name__ == "__main__":
    main()
