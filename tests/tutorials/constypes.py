from __future__ import annotations

from pyscipopt import SCIP_PARAMSETTING, Constraint, Model, Variable, quicksum
from pyscipopt.scip import Column, Row, Solution

scip = Model()

# What is a Constraint?

x: Variable = scip.addVar(vtype="B", name="x")
y: Variable = scip.addVar(vtype="B", name="y")
z: Variable = scip.addVar(vtype="B", name="z")
# Linear constraint
linear_cons: Constraint = scip.addCons(x + y + z == 1, name="lin_cons")
# Non-linear constraint
nonlinear_cons: Constraint = scip.addCons(x * y + z == 1, name="nonlinear_cons")


# Quicksum

x1: list[Variable] = [scip.addVar(vtype="B", name=f"x_{i}") for i in range(1000)]

scip.addCons(quicksum(x1[i] for i in range(1000)) == 1, name="sum_cons")


# Constraint Information

linear_conshdlr_name: str = linear_cons.getConshdlrName()
assert linear_cons.isLinear()

if scip.getNSols() >= 1:
    scip_sol: Solution | None = scip.getBestSol()
    activity: float = scip.getActivity(linear_cons, scip_sol)
    slack: float = scip.getSlack(linear_cons, scip_sol)
scip.chgCoefLinear(linear_cons, x, 7)


scip.setPresolve(SCIP_PARAMSETTING.OFF)
scip.setHeuristics(SCIP_PARAMSETTING.OFF)
scip.disablePropagation()

dual_sol: float = scip.getDualsolLinear(linear_cons)

# Constraint Types


sos_cons: Constraint = scip.addConsSOS1([x, y, z], name="example_sos")
indicator_cons: Constraint = scip.addConsIndicator(
    x + y <= 1, binvar=z, name="example_indicator"
)

# What is a Row?

row: Row = scip.getRowLinear(linear_cons)


lhs: float = row.getLhs()
rhs: float = row.getRhs()
constant: float = row.getConstant()
cols: list[Column] = row.getCols()
vals: list[float] = row.getVals()
origin_cons_name: str = row.getConsOriginConshdlrtype()
