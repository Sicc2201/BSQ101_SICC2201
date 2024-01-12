##########################################################################

# Titre: BooleanProblems.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 09/01/2024

##########################################################################
'''
# Description: 

This file contains all the boolean problems the projetcts needs to execute.
This is where you put the problems propositions.

'''

###########################################################################

# IMPORTS

###########################################################################

from simpy import symbols


def create_cake_problem():
        Xa, Xb, Xc, Xd, Xe = symbols('Xa, Xb, Xc, Xd, Xe')
        Pa = (~Xe & ~Xb) | (Xe & Xb)
        Pb = (~Xc & ~Xe) | (Xc & Xe)
        Pc = (~Xe & ~Xa) | (Xe & Xa)
        Pd = (~Xc & ~Xb) | (Xc & Xb)
        Pe = (~Xd & ~Xa) | (Xd & Xa)

        Pg = Pa & Pb & Pc & Pd & Pe
        return Pg

def create_pincus_problem():
        
        x1, x2, x3, x4 = symbols('x1, x2, x3, x4')
        p1 = ~x3 & ~x1 >> x4
        p2 = x1 & ~x4 >> x2
        p3 = ~x3 & x4 >> x2
        p4 = x2 & ~x4 >> ~x3
        p5 = x1 & ~x3 >> ~x2
        p6 = ~x2 & x3 >> x1
        p7 = x1 & x4 >> ~x3

        pg = p1 & p2 & p3 & p4 & p5 & p6 & p7
        return pg