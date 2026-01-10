from pyscipopt import Model, Variable

def get_infeasible_constraints(
    orig_model: Model, verbose: bool = False
) -> tuple[int, dict[str, Variable]]: ...
