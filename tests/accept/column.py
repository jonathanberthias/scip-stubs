from typing_extensions import assert_type
from pyscipopt.scip import Column, Variable


def what_is_a_column(x: Variable):
    col = x.getCol()
    assert_type(col, Column)

    assert_type(col.getObjCoeff(), float)
    assert_type(col.getPrimsol(), float)
    assert_type(col.getLb(), float)
    assert_type(col.getUb(), float)
    assert_type(col.getVar(), Variable)
