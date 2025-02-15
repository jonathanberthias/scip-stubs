import pyscipopt.scip
from pyscipopt.scip import Constraint as Constraint, Model as Model, Variable as Variable, quicksum as quicksum

def add_piecewise_linear_cons(model: pyscipopt.scip.Model, X: pyscipopt.scip.Variable, Y: pyscipopt.scip.Variable, a: list, b: list) -> pyscipopt.scip.Constraint:
    """add constraint of the form y = f(x), where f is a piecewise linear function

    :param model: pyscipopt model to add the constraint to
    :param X: x variable
    :param Y: y variable
    :param a: array with x-coordinates of the points in the piecewise linear relation
    :param b: array with y-coordinate of the points in the piecewise linear relation

    Disclaimer: For the moment, can only model 2d piecewise linear functions
    Adapted from https://github.com/scipopt/PySCIPOpt/blob/master/examples/finished/piecewise.py
    """
