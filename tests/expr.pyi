from collections.abc import Iterator
from decimal import Decimal
from typing import Literal

from typing_extensions import assert_type

from pyscipopt import Expr
from pyscipopt.scip import (
    Constant,
    ExprCons,
    GenExpr,
    PowExpr,
    ProdExpr,
    SumExpr,
    Term,
    UnaryExpr,
    VarExpr,
    Variable,
    buildGenExprObj,
)

e: Expr
g: GenExpr
d: Decimal
x: Variable
y: Variable

class ToFloat:
    def __float__(self) -> float: ...

# In many cases where floats are expected, a string repr of a float is also valid
# However, from a typing perspective, a general string is not valid
# so we forbid all strings (a call to float is enough to solve the type error)

# buildGenExprObj
assert_type(buildGenExprObj(1.0), Constant)
assert_type(buildGenExprObj(expr=1), Constant)
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

# Term.__init__
assert_type(Term(x), Term)
assert_type(Term(x, y), Term)
Term(e)  # pyright: ignore[reportArgumentType]
Term(1)  # pyright: ignore[reportArgumentType]
Term((x, y))  # pyright: ignore[reportArgumentType]

t: Term
# Term.__getitem__
assert_type(t[0], Variable)
t[x]  # pyright:ignore[reportArgumentType]

# Term.__hash__
assert_type(hash(t), int)

# Term.__add__
assert_type(t + t, Term)
t + x  # pyright: ignore[reportOperatorIssue]
t + e  # pyright: ignore[reportOperatorIssue]

# Expr.__init__
assert_type(Expr(), Expr)
assert_type(Expr(terms=None), Expr)
assert_type(Expr(terms={}), Expr)
assert_type(Expr({Term(x): 1}), Expr)

# Expr.__getitem__
assert_type(e[x], float)
assert_type(e[Term(x)], float)
assert_type(e[Term(x, y)], float)

e[0]  # pyright: ignore[reportArgumentType, reportCallIssue]
e[x,]  # pyright: ignore[reportArgumentType, reportCallIssue]
e[x, y]  # pyright: ignore[reportArgumentType, reportCallIssue]
e[e]  # pyright: ignore[reportArgumentType, reportCallIssue]

# Expr.__iter__
assert_type(iter(e), Iterator[Term])
# __next__ is defined but doesn't work, so let's make sure it gets flagged
next(e)  # pyright: ignore[reportArgumentType]

# Expr.__abs__
assert_type(abs(e), UnaryExpr)

# Expr.__(r)add__
assert_type(e + e, Expr)
assert_type(e + 1, Expr)
assert_type(e + d, Expr)
assert_type(1 + e, Expr)

assert_type(x + x, Expr)
assert_type(x + 1, Expr)
assert_type(1 + x, Expr)

assert_type(e + g, SumExpr)
assert_type(e + PowExpr(), SumExpr)

e + 1j  # pyright: ignore[reportOperatorIssue]
e + "1"  # pyright: ignore[reportOperatorIssue]
"1" + e  # pyright: ignore[reportOperatorIssue]

# Expr.__iadd__
e1: Expr
e2: Expr
e3: Expr
e4: Expr
e5: Expr

e1 += e
assert_type(e1, Expr)
e2 += 1
assert_type(e2, Expr)
# at runtime this will modify e3 to become a SumExpr
# using the `e3 += g` syntax is an error for pyright as it tries to
# assign the result back to e3
assert_type(e3.__iadd__(g), SumExpr)

e4 += "1"  # pyright: ignore[reportOperatorIssue]
e5 += 1j  # pyright: ignore[reportOperatorIssue]

# Expr.__(r)mul__
assert_type(e * 1, Expr)
assert_type(1 * e, Expr)
assert_type(e * d, Expr)
assert_type(d * e, Expr)
assert_type(e * e, Expr)

assert_type(e * g, ProdExpr)

e * "1"  # pyright: ignore[reportOperatorIssue]
"1" * e  # pyright: ignore[reportOperatorIssue]

# Expr.__(r)truediv__
assert_type(e / 2, Expr)
assert_type(2 / e, Expr)
assert_type(e / e, ProdExpr)
assert_type(e / g, ProdExpr)

e / "2"  # pyright: ignore[reportOperatorIssue]
"2" / e  # pyright: ignore[reportOperatorIssue]

e // 2  # pyright: ignore[reportOperatorIssue]
2 // e  # pyright: ignore[reportOperatorIssue]

# Expr.__pow__
assert_type(e**0, Expr | Literal[1] | PowExpr)  # actually returns Literal[1]
assert_type(e**2, Expr | Literal[1] | PowExpr)  # actually returns Expr
assert_type(e**1.5, Expr | Literal[1] | PowExpr)  # actually returns PowExpr
assert_type(e**d, Expr | Literal[1] | PowExpr)
assert_type(e**-1, Expr | Literal[1] | PowExpr)  # actually returns PowExpr

e**e  # pyright: ignore[reportOperatorIssue]
e ** "a"  # pyright: ignore[reportOperatorIssue]

# Expr.__neg__
assert_type(-e, Expr)
assert_type(-x, Expr)

# Expr.__sub__
assert_type(e - e, Expr)
assert_type(e - 1, Expr)
assert_type(e - g, SumExpr)

# Expr.__rsub__
assert_type(1 - e, Expr)
assert_type(d - e, Expr)

# Expr comparisons
assert_type(e <= e, ExprCons)
assert_type(e <= g, ExprCons)
assert_type(e <= 1, ExprCons)
assert_type(e <= d, ExprCons)

assert_type(e >= e, ExprCons)
assert_type(e >= g, ExprCons)
assert_type(e >= 1, ExprCons)
assert_type(e >= d, ExprCons)

assert_type(e == e, ExprCons)
assert_type(e == g, ExprCons)
assert_type(e == 1, ExprCons)
assert_type(e == d, ExprCons)

e < 1  # pyright: ignore[reportOperatorIssue]
e > 1  # pyright: ignore[reportOperatorIssue]
e != 1  # FIXME: this should be an error
e <= "1"  # pyright: ignore[reportOperatorIssue]
e >= "1"  # pyright: ignore[reportOperatorIssue]
e == "1"  # FIXME: this should be an error

# ExprCons.__init__
ExprCons(e)
ExprCons(g)
ExprCons(e, 0)
ExprCons(expr=g, lhs=1, rhs=None)
ExprCons(e, lhs=None, rhs=1)

ExprCons()  # pyright: ignore[reportCallIssue]
ExprCons(e, 1, 2, 3)  # pyright: ignore[reportCallIssue]
ExprCons(e, None, None)  # pyright: ignore[reportArgumentType, reportCallIssue]
ExprCons(e, e)  # pyright: ignore[reportArgumentType, reportCallIssue]
ExprCons(e, d)  # pyright: ignore[reportArgumentType, reportCallIssue]

# ExprCons comparisons
ec: ExprCons
# TODO: refine these by making ExprCons generic on the bounds, e.g.:
# class ExprCons[UB: (float, None), LB: (float, None)]
# to get:
# assert_type(ExprCons[float, None]() <= 1, ExprCons[float, float])

assert_type(ec <= 1, ExprCons)
assert_type(ec <= d, ExprCons)
assert_type(ec >= 1, ExprCons)
assert_type(ec >= d, ExprCons)

ec <= e  # pyright: ignore[reportOperatorIssue]
ec >= e  # pyright: ignore[reportOperatorIssue]
ec < 1  # pyright: ignore[reportOperatorIssue]
ec > 1  # pyright: ignore[reportOperatorIssue]
ec == 1  # FIXME: this should be an error

# Expr.__bool__
bool(ec)  # FIXME: this should be an error
1 <= e <= 2  # FIXME: this should be an error
