from decimal import Decimal

from pyscipopt import LP
from pyscipopt.scip import Expr, Model, Term, Variable, buildGenExprObj

scip: Model = Model()
x: Variable = scip.addVar()
y: Variable = scip.addVar()
e = Expr({Term(x): 1})
g = buildGenExprObj(e)
d = Decimal("1.0")
t = Term()

lp = LP()
