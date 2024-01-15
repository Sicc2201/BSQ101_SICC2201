##########################################################################

# Titre: BooleanProblems.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 09/01/2024

##########################################################################
'''
# Description: 

This file contains all the boolean problems the projects needs to execute.
This is where you put the problems propositions.

# methods:

create_cake_problem() -> cnf : create the cnf of the cake problem in the project1.
create_pincus_problem() -> cnf : create the cnf of the Pincus planet problem in the project1.

'''

###########################################################################

# IMPORTS

###########################################################################

from sympy import symbols, to_cnf, Not


def create_cake_problem():
    Xa, Xb, Xc, Xd, Xe = symbols('Xa, Xb, Xc, Xd, Xe')
    Pa = (~Xe & ~Xb) | (Xe & Xb)
    Pb = (~Xc & ~Xe) | (Xc & Xe)
    Pc = (~Xe & ~Xa) | (Xe & Xa)
    Pd = (~Xc & ~Xb) | (Xc & Xb)
    Pe = (~Xd & ~Xa) | (Xd & Xa)

    Pg = Pa & Pb & Pc & Pd & Pe
    cnf = to_cnf(Pg)

    # filter out all clauses that contains a litteral and its negation ex: (Xa | ~Xa)
    # filtered_cnf = [clause for clause in cnf.args if not any(isinstance(arg, Not) and arg.args[0] == arg2 for arg in clause.args for arg2 in clause.args)]
    return cnf

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

    pg = p1 & p2 & p3 & p4 & p5 & p6 & p7
    cnf = to_cnf(pg)
    return cnf