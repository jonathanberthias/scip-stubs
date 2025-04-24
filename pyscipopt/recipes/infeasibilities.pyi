from pyscipopt import Model, Variable

def get_infeasible_constraints(
    orig_model: Model, verbose: bool = False
) -> tuple[int, dict[str, Variable]]:
    """
    Given a model, adds slack variables to all the constraints and minimizes a binary variable that indicates if they're positive.
    Positive slack variables correspond to infeasible constraints.
    """
