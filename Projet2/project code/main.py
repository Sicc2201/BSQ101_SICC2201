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
import State_tomography
import Utils

###########################################################################

# MAIN

###########################################################################


def main():

    ibmq_token = "put your token here"
    backend_name = "ibmq_qasm_simulator"

    ####  uncomment if this is the first time you connect to your IBMQ account  ##########

    # IBMQ_credentials.ibmq_connexion(ibmq_token)

    #################################################################################

    #_, backend = IBMQ_credentials.ibmq_provider(backend_name)

    backend = IBMQ_credentials.get_local_simulator()

    num_qubits = 2
    num_gates_to_apply = num_qubits * 2 # number of random gates you want to apply in the random circuit (could be random too)

    execute_opts = {'shots': 1024}

    state_circuit = Utils.create_random_quantum_circuit(num_qubits, num_gates_to_apply)

    state_vector = State_tomography.state_tomography(state_circuit,backend,execute_opts)

    # fidelity range 0-1 where the closer it is to 1 the closer they are. Closer they are to 0, the more they are orthogonal.
    quantum_state_validation = State_tomography.validate_state_vector(state_circuit, state_vector)
    print("rapport de fidelite (0 = orthogonal, 1 = colineaire): ", quantum_state_validation)

    return 0


if __name__ == "__main__":
    main()
