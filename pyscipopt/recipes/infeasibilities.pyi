import pyscipopt.scip
from pyscipopt.scip import Model as Model
from pyscipopt.scip import quicksum as quicksum

def get_infeasible_constraints(orig_model: pyscipopt.scip.Model, verbose: bool = ...):
    """
    Given a model, adds slack variables to all the constraints and minimizes a binary variable that indicates if they're positive.
    Positive slack variables correspond to infeasible constraints.
    """
