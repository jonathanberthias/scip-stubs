from decimal import Decimal
from typing_extensions import assert_type
from pyscipopt import Expr
from pyscipopt.scip import Constant, GenExpr, SumExpr, buildGenExprObj

e: Expr
g: GenExpr
d: Decimal

class ToFloat:
    def __float__(self) -> float: ...

assert_type(buildGenExprObj(1.0), Constant)
assert_type(buildGenExprObj(ToFloat()), Constant)
assert_type(buildGenExprObj(e), SumExpr)
assert_type(buildGenExprObj(g), GenExpr)
assert_type(buildGenExprObj(expr=e), SumExpr)

buildGenExprObj()  # pyright: ignore[reportCallIssue]
buildGenExprObj(1j)  # pyright: ignore[reportArgumentType, reportCallIssue]

# works at runtime
buildGenExprObj("1.0")  # pyright: ignore[reportArgumentType, reportCallIssue]
