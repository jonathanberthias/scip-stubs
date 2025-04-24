from typing import Any
from typing import Literal as L

from pyscipopt import Model
from pyscipopt.scip import Expr, GenExpr

def set_nonlinear_objective(
    model: Model,
    expr: Expr | GenExpr[Any],
    sense: L["minimize", "maximize"] = "minimize",
) -> None:
    """
    Takes a nonlinear expression and performs an epigraph reformulation.
    """
