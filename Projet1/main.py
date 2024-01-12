##########################################################################

# Titre: main.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 09/01/2024

##########################################################################
'''
# Description: 
'''

###########################################################################

# IMPORTS

###########################################################################

# Custom libraries
import IBMQ_credentials
import GroverUtils as utils

# import packages
from sympy import *
###########################################################################

# MAIN

###########################################################################


def main():

    ibmq_token = ""  #put your IBMQ account Token here
    backend = "ibmq_qasm_simulator"

    ####  uncomment if this is the first time you connect to your account  ##########
    # IBMQ_credentials.ibmq_connexion(ibmq_token)
    #################################################################################

    IBMQ_credentials.ibmq_provider(backend)
    
    Xa, Xb, Xc, Xd, Xe = symbols('Xa, Xb, Xc, Xd, Xe')
    Pa = (~Xe & ~Xb) | (Xe & Xb)
    Pb = (~Xc & ~Xe) | (Xc & Xe)
    Pc = (~Xe & ~Xa) | (Xe & Xa)
    Pd = (~Xc & ~Xb) | (Xc & Xb)
    Pe = (~Xd & ~Xa) | (Xd & Xa)

    Pg = Pa & Pb & Pc & Pd & Pe
    Pg_cnf = to_cnf(Pg)


    return Pa

if __name__ == "__main__":
    main()
