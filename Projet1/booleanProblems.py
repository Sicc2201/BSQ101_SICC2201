##########################################################################

# Titre: BooleanProblems.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 21/01/2024

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

"""Commentaires
Je crois que tu importes Not et que tu ne t'en sers pas.
"""
from sympy import symbols, to_cnf, Not


def create_cake_problem():
    Alan, Ben, Chris, Dave, Emma = symbols('Alan, Ben, Chris, Dave, Emma')
    Pa = (~Emma & ~Ben) | (Emma & Ben)
    Pb = (~Chris & Emma) | (Chris & ~Emma)
    Pc = (~Emma & ~Alan) | (Emma & Alan)
    Pd = (~Chris & Ben) | (Chris & ~Ben)
    Pe = (~Dave & ~Alan) | (Dave & Alan)

    Pg = Pa & Pb & Pc & Pd & Pe
    cnf = to_cnf(Pg, True)

    return cnf

"""Commentaires
Une forme normale conjonctive est une conjonction de disjonction. 
La fonction `create_pincus_problem` retourne une disjonction de conjonctions.
Il y a un problème dans la construction.
Évite de définir chacune de proposition 2 fois. Je comprend que c'est utile pour tester au début, mais assure-toi de faire le ménage ensuite.
"""
def create_pincus_problem():
        
    x1, x2, x3, x4 = symbols('x1, x2, x3, x4')

    p1 = ~x3 & ~x1 >> x4
    p1 = x3 & x1 & x4

    p2 = x1 & ~x4 >> x2
    p2 = ~x1 & x4 & x2

    p3 = ~x3 & x4 >> x2
    p3 = x3 & ~x4 & x2

    p4 = x2 & ~x4 >> ~x3
    p4 = ~x2 & x4 & ~x3

    p5 = x1 & ~x3 >> ~x2
    p5 = ~x1 & x3 & ~x2

    p6 = ~x2 & x3 >> x1
    p6 = x2 & ~x3 & x1

    p7 = x1 & x4 >> ~x3
    p7 = ~x1 & ~x4 & ~x3

    pg = p1 | p2 | p3 | p4 | p5 | p6 | p7
    cnf = pg # to_cnf(pg, True)
    return cnf
