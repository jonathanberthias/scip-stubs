from typing_extensions import assert_type

from pyscipopt import Expr, Model, Variable

scip = Model()
assert_type(scip, Model)

x = scip.addVar(vtype="C", lb=0, ub=None, name="x")
y = scip.addVar(vtype="C", lb=0, ub=None, name="y")
z = scip.addVar(vtype="C", lb=0, ub=None, name="z")
assert_type(x, Variable)
assert_type(y, Variable)
assert_type(z, Variable)

assert_type(x + y, Expr)
