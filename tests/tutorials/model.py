from __future__ import annotations

from pathlib import Path

from pyscipopt import SCIP_PARAMEMPHASIS, SCIP_PARAMSETTING, Constraint, Model, Variable

# Create a Model, Variables, and Constraints

scip: Model = Model()

x: Variable = scip.addVar(vtype="C", lb=0, ub=None, name="x")
y: Variable = scip.addVar(vtype="C", lb=0, ub=None, name="y")
z: Variable = scip.addVar(vtype="C", lb=0, ub=None, name="z")
cons_1: Constraint = scip.addCons(x + y <= 5, name="cons_1")
cons_2: Constraint = scip.addCons(y + z >= 3, name="cons_2")
cons_3: Constraint = scip.addCons(x + y == 5, name="cons_3")
scip.setObjective(2 * x + 3 * y - 5 * z, sense="minimize")
scip.optimize()

# Query the Model for Solution Information


solve_time: float = scip.getSolvingTime()
num_nodes: int = scip.getNTotalNodes()
obj_val: float = scip.getObjVal()
for scip_var in [x, y, z]:
    print(f"Variable {scip_var.name} has value {scip.getVal(scip_var)}")

# Set / Get a Parameter

scip.setParam("limits/time", 20)

time_limit: bool | float | str = scip.getParam("limits/time")

param_dict = {"limits/time": 20}
scip.setParams(param_dict)

param_dict: dict[str, bool | float | str] = scip.getParams()

path_to_file = Path("params.set")
scip.readParams(path_to_file)

# Set Plugin-wide Parameters (Aggressiveness)

scip = Model()
scip.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)

# Set Solver Emphasis

scip = Model()
scip.setEmphasis(SCIP_PARAMEMPHASIS.FEASIBILITY)

# Copy a SCIP Model

scip_alternate_model = Model(sourceModel=scip)

scip.freeTransform()

scip.freeProb()
