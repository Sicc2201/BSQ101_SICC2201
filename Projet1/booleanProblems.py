##########################################################################

# Titre: BooleanProblems.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 23/01/2024

##########################################################################
'''
# Description: 

This file contains all the boolean problems the projects needs to execute.
This is where you put the problems propositions.

# methods:

- create_cake_problem() -> cnf : Creates the cnf of the cake problem in the project1.

- create_pincus_problem() -> cnf : Creates the cnf of the Pincus planet problem in the project1.

'''

###########################################################################

# IMPORTS

###########################################################################

from sympy import symbols, to_cnf, And

def create_cake_problem() -> And:
    Alan, Ben, Chris, Dave, Emma = symbols('Alan, Ben, Chris, Dave, Emma')
    Pa = (~Emma & ~Ben) | (Emma & Ben)
    Pb = (~Chris & Emma) | (Chris & ~Emma)
    Pc = (~Emma & ~Alan) | (Emma & Alan)
    Pd = (~Chris & Ben) | (Chris & ~Ben)
    Pe = (~Dave & ~Alan) | (Dave & Alan)

    Pg = Pa & Pb & Pc & Pd & Pe
    cnf = to_cnf(Pg, True)

    return cnf

def create_pincus_problem() -> And:
        
    Peur, Joie, Malade, Bruyant = symbols('Peur, Joie, Malade, Bruyant')

    p1 = Malade | Peur | Bruyant

    p2 = ~Peur | Bruyant | Joie

    p3 = Malade | ~Bruyant | Joie

    p4 = ~Joie | Bruyant | ~Malade

    p5 = ~Peur | Malade | ~Joie

    p6 = Joie | ~Malade | Peur

    p7 = ~Peur | ~Bruyant | ~Malade

    cnf = p1 & p2 & p3 & p4 & p5 & p6 & p7
    return cnf