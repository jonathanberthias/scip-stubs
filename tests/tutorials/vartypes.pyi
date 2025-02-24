from typing_extensions import assert_type

from pyscipopt.scip import Column, Variable

# What is a Column?

x: Variable
assert_type(x.isInLP(), bool)
assert_type(x.getLPSol(), float)

col = x.getCol()
assert_type(col, Column)

assert_type(col.getObjCoeff(), float)
assert_type(col.getPrimsol(), float)
assert_type(col.getLb(), float)
assert_type(col.getUb(), float)
assert_type(col.getVar(), Variable)
