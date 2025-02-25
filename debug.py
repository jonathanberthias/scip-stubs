import pyscipopt as pyscipopt
from pyscipopt.scip import Expr as Expr
from pyscipopt.scip import GenExpr as GenExpr
from pyscipopt.scip import Model, Term, Variable, buildGenExprObj
from pyscipopt.scip import ProdExpr as ProdExpr
from pyscipopt.scip import SumExpr as SumExpr

scip: Model = Model()
x: Variable = scip.addVar()
y: Variable = scip.addVar()
e = Expr({Term(x): 1})
g = buildGenExprObj(e)
