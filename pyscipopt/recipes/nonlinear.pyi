import pyscipopt.scip
from pyscipopt.scip import Model as Model

def set_nonlinear_objective(model: pyscipopt.scip.Model, expr, sense: str = ...):
    """
    Takes a nonlinear expression and performs an epigraph reformulation.
    """
