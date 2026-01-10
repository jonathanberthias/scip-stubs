from collections.abc import Sequence

from pyscipopt import Constraint, Model, Variable

def add_piecewise_linear_cons(
    model: Model, X: Variable, Y: Variable, a: Sequence[float], b: Sequence[float]
) -> Constraint: ...
