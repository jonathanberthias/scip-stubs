from __future__ import annotations

import numpy as np

from pyscipopt import Constraint, Model, Variable, quicksum
from pyscipopt.scip import Column, Solution

# Variables in SCIP

scip = Model()

# Variable Types

x = scip.addVar(vtype="C", name="x")
assert x.vtype() == "CONTINUOUS"

# Dictionary of Variables

var_dict: dict[int, dict[int, Variable]] = {}
n = 5
m = 5
for i in range(n):
    var_dict[i] = {}
    for j in range(m):
        var_dict[i][j] = scip.addVar(vtype="B", name=f"x_{i}_{j}")

example_cons: dict[int, Constraint] = {}
for i in range(n):
    example_cons[i] = scip.addCons(
        quicksum(var_dict[i][j] for j in range(m)) == 1, name=f"cons_{i}"
    )

# List of Variables

n, m = 5, 5
var_list: list[list[Variable]] = [[None for i in range(m)] for i in range(n)]  # type: ignore[misc]  # pyright: ignore[reportAssignmentType]
for i in range(n):
    for j in range(m):
        var_list[i][j] = scip.addVar(vtype="B", name=f"x_{i}_{j}")

example_cons1: list[Constraint] = []
for i in range(n):
    example_cons1.append(
        scip.addCons(quicksum(var_list[i][j] for j in range(m)) == 1, name=f"cons_{i}")
    )

# Numpy array of Variables

n, m = 5, 5
var_array = np.zeros((n, m), dtype=object)  # dtype object allows arbitrary storage
for i in range(n):
    for j in range(m):
        var_array[i][j] = scip.addVar(vtype="B", name=f"x_{i}_{j}")

example_cons2 = np.zeros((n,), dtype=object)
for i in range(n):
    example_cons2[i] = scip.addCons(
        quicksum(var_dict[i][j] for j in range(m)) == 1, name=f"cons_{i}"
    )

a = np.random.uniform(size=(n, m))
c = a @ var_array

# Get Variables

scip_vars: list[Variable] = scip.getVars()

# Variable Information

scip.setObjective(2 * x)
assert x.getObj() == 2.0

var_val: float = scip.getVal(x)

if scip.getNSols() >= 1:
    scip_sol: Solution | None = scip.getBestSol()
    assert scip_sol is not None
    var_val1: float = scip_sol[x]

# What is a Column?

is_in_lp: bool = x.isInLP()
if is_in_lp:
    print("Variable is in LP!")
    print(f"Variable value in LP is {x.getLPSol()}")
else:
    print("Variable is not in LP!")

col: Column = x.getCol()

obj_coeff: float = col.getObjCoeff()
lp_val: float = col.getPrimsol()
lb: float = col.getLb()
ub: float = col.getUb()
x1: Variable = col.getVar()

# What is a Transformed Variable?

scip_vars1: list[Variable] = scip.getVars(transformed=True)

is_original: bool = scip_vars1[0].isOriginal()
