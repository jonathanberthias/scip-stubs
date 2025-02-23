from decimal import Decimal

from typing_extensions import assert_type

from pyscipopt import Expr
from pyscipopt.scip import (
    Constant,
    GenExpr,
    PowExpr,
    ProdExpr,
    SumExpr,
    UnaryExpr,
    VarExpr,
    Variable,
    buildGenExprObj,
)

e: Expr
g: GenExpr
d: Decimal
x: Variable

class ToFloat:
    def __float__(self) -> float: ...

assert_type(buildGenExprObj(1.0), Constant)
assert_type(buildGenExprObj(expr=1.0), Constant)
assert_type(buildGenExprObj(d), Constant)
assert_type(buildGenExprObj(ToFloat()), Constant)

assert_type(buildGenExprObj(e), SumExpr)
assert_type(buildGenExprObj(Variable()), SumExpr)

assert_type(buildGenExprObj(g), GenExpr)
assert_type(buildGenExprObj(PowExpr()), GenExpr)
assert_type(buildGenExprObj(ProdExpr()), GenExpr)
assert_type(buildGenExprObj(UnaryExpr()), GenExpr)
assert_type(buildGenExprObj(VarExpr()), GenExpr)

buildGenExprObj()  # pyright: ignore[reportCallIssue]
buildGenExprObj(1j)  # pyright: ignore[reportArgumentType, reportCallIssue]

# works at runtime
buildGenExprObj("1.0")  # pyright: ignore[reportArgumentType, reportCallIssue]

assert_type(e + e, Expr)
assert_type(e + 1, Expr)
assert_type(1 + e, Expr)
assert_type(e + "1", Expr)
assert_type("1" + e, Expr)

assert_type(x + x, Expr)
assert_type(x + 1, Expr)
assert_type(1 + x, Expr)
assert_type(x + "1", Expr)
assert_type("1" + x, Expr)

assert_type(e + g, SumExpr)
assert_type(e + PowExpr(), SumExpr)

e + 1j  # pyright: ignore[reportOperatorIssue, reportUnusedExpression]
