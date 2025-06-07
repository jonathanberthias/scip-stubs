from collections.abc import Iterator
from decimal import Decimal
from typing import Any, Literal

from typing_extensions import Literal as L
from typing_extensions import assert_type

from pyscipopt import Expr, scip
from pyscipopt.scip import (
    Constant,
    ExprCons,
    GenExpr,
    Operator,
    PowExpr,
    ProdExpr,
    SumExpr,
    Term,
    UnaryExpr,
    VarExpr,
    Variable,
    buildGenExprObj,
    expr_to_nodes,
    quickprod,
    quicksum,
)

e: Expr
g: GenExpr[L["+"]]
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

assert_type(buildGenExprObj(g), GenExpr[L["+"]])
assert_type(buildGenExprObj(PowExpr()), GenExpr[L["**"]])
assert_type(buildGenExprObj(ProdExpr()), GenExpr[L["prod"]])
assert_type(buildGenExprObj(UnaryExpr(Operator.fabs, g)), GenExpr[L["abs"]])
assert_type(buildGenExprObj(VarExpr(x)), GenExpr[L["var"]])

buildGenExprObj()  # type: ignore[call-overload] # pyright: ignore[reportCallIssue]
buildGenExprObj(1j)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType, reportCallIssue]

# works at runtime
buildGenExprObj("1.0")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType, reportCallIssue]

# Term.__init__
assert_type(Term(x), Term)
assert_type(Term(x, y), Term)
Term(e)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
Term(1)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
Term((x, y))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]

t: Term
# Term.__getitem__
assert_type(t[0], Variable)
t[x]  # type: ignore[index] # pyright:ignore[reportArgumentType]

# Term.__hash__
assert_type(hash(t), int)
assert_type({t}, set[Term])

# Term.__add__
assert_type(t + t, Term)
t + x  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
t + e  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

# Expr.__init__
assert_type(Expr(), Expr)
assert_type(Expr(terms=None), Expr)
assert_type(Expr(terms={}), Expr)
assert_type(Expr({Term(x): 1}), Expr)

# Expr.__getitem__
assert_type(e[x], float)
assert_type(e[Term(x)], float)
assert_type(e[Term(x, y)], float)

e[0]  # type: ignore[call-overload] # pyright: ignore[reportArgumentType, reportCallIssue]
e[x,]  # type: ignore[call-overload] # pyright: ignore[reportArgumentType, reportCallIssue]
e[x, y]  # type: ignore[call-overload] # pyright: ignore[reportArgumentType, reportCallIssue]
e[e]  # type: ignore[call-overload] # pyright: ignore[reportArgumentType, reportCallIssue]

# Expr.__iter__
assert_type(iter(e), Iterator[Term])
# __next__ is defined but doesn't work, so let's make sure it gets flagged
next(e)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType]

# Expr.__abs__
assert_type(abs(e), UnaryExpr[L["abs"]])

# Expr.__(r)add__
assert_type(e + e, Expr)
assert_type(e + 1, Expr)
# FIXME: this works at runtime
e + d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
assert_type(1 + e, Expr)

assert_type(x + x, Expr)
assert_type(x + 1, Expr)
assert_type(1 + x, Expr)

assert_type(e + g, SumExpr)
assert_type(e + PowExpr(), SumExpr)

e + 1j  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
e + "1"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
"1" + e  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

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

e4 += "1"  # type: ignore[call-overload] # pyright: ignore[reportOperatorIssue]
e5 += 1j  # type: ignore[call-overload] # pyright: ignore[reportOperatorIssue]

# Expr.__(r)mul__
assert_type(e * 1, Expr)
assert_type(1 * e, Expr)
# FIXME: this works at runtime
e * d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
assert_type(d * e, Expr)
assert_type(e * e, Expr)

assert_type(e * g, ProdExpr)

e * "1"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
"1" * e  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

# Expr.__(r)truediv__
assert_type(e / 2, Expr)
assert_type(2 / e, Expr)
assert_type(e / e, ProdExpr)
assert_type(e / g, ProdExpr)

e / "2"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
"2" / e  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

e // 2  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
2 // e  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

# Expr.__pow__
assert_type(e**0, Literal[1])  # special case
assert_type(e**2, Expr | PowExpr)  # actually returns Expr
assert_type(e**1.5, Expr | PowExpr)  # actually returns PowExpr
assert_type(e**d, Expr | PowExpr)
assert_type(e**-1, Expr | PowExpr)  # actually returns PowExpr

e**e  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
e ** "a"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

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

e < 1  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
e > 1  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
e != 1  # FIXME: this should be an error
e <= "1"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
e >= "1"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
e == "1"  # FIXME: this should be an error

# Expr is not hashable
# FIXME: mypy doesn't catch this?
{e}  # pyright: ignore[reportUnhashable]

# ExprCons.__init__
ExprCons(e)
ExprCons(g)
ExprCons(e, 0)
ExprCons(expr=g, lhs=1, rhs=None)
ExprCons(e, lhs=None, rhs=1)

ExprCons()  # type: ignore[call-overload] # pyright: ignore[reportCallIssue]
ExprCons(e, 1, 2, 3)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue]
ExprCons(e, None, None)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType, reportCallIssue]
ExprCons(e, e)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType, reportCallIssue]
ExprCons(e, d)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType, reportCallIssue]

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

ec <= e  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
ec >= e  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
ec < 1  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
ec > 1  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
ec == 1  # FIXME: this should be an error

# Expr.__bool__
bool(ec)  # FIXME: this should be an error
1 <= e <= 2  # FIXME: this should be an error

# quicksum
assert_type(quicksum(ex for ex in [e]), Expr)
assert_type(quicksum([g]), SumExpr)
assert_type(quicksum([e, g]), SumExpr)
assert_type(quicksum(termlist=[]), Expr)
assert_type(quicksum(termlist=range(3)), Expr)
assert_type(quicksum([d]), Expr)
assert_type(quicksum([1, d, e, g]), SumExpr)

# quickprod
assert_type(quickprod(ex for ex in [e]), Expr)
assert_type(quickprod([g]), ProdExpr)
assert_type(quickprod([e, g]), ProdExpr)
assert_type(quickprod(termlist=[]), Expr)
assert_type(quickprod(termlist=range(3)), Expr)
assert_type(quickprod([d]), Expr)
assert_type(quickprod([1, d, e, g]), ProdExpr)

# GenExpr.__abs__
assert_type(abs(g), UnaryExpr[L["abs"]])

# GenExpr.__add__
assert_type(g + 1, SumExpr)
assert_type(g + e, SumExpr)
assert_type(g + g, SumExpr)

# this fails at runtime, maybe a bug?
g + d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

# GenExpr.__mul__
assert_type(g * 2, ProdExpr)
assert_type(g * e, ProdExpr)
assert_type(g * g, ProdExpr)

# this fails at runtime, maybe a bug?
g * d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

# GenExpr.__pow__
assert_type(g**0, PowExpr)
assert_type(g**1.5, PowExpr)
assert_type(g**d, PowExpr)

g**e  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
g**g  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

# GenExpr.__truediv__
assert_type(g / 1, ProdExpr)
assert_type(g / e, ProdExpr)
assert_type(g / g, ProdExpr)

# this fails at runtime, maybe a bug?
g / d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

# GenExpr.__rtruediv__
assert_type(1 / g, ProdExpr)

# GenExpr.__neg__
assert_type(-g, ProdExpr)

# GenExpr.__sub__
assert_type(g - 1, SumExpr)
assert_type(g - e, SumExpr)
assert_type(g - g, SumExpr)

# this fails at runtime, maybe a bug?
g - d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

# GenExpr.__radd__
assert_type(1 + g, SumExpr)
assert_type(1.5 + g, SumExpr)

# this fails at runtime, maybe a bug?
d + g  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

# GenExpr.__rmul__
assert_type(1 * g, SumExpr)
assert_type(1.5 * g, SumExpr)

# this fails at runtime, maybe a bug?
d * g  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

# GenExpr.__rsub__
assert_type(1 - g, SumExpr)
assert_type(1.5 - g, SumExpr)

# this fails at runtime, maybe a bug?
d - g  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

# GenExpr comparisons
assert_type(g <= g, ExprCons)
assert_type(g <= e, ExprCons)
assert_type(g <= 1, ExprCons)
assert_type(g <= d, ExprCons)

assert_type(g >= g, ExprCons)
assert_type(g >= e, ExprCons)
assert_type(g >= 1, ExprCons)
assert_type(g >= d, ExprCons)

assert_type(g == g, ExprCons)
assert_type(g == e, ExprCons)
assert_type(g == 1, ExprCons)
assert_type(g == d, ExprCons)

g < 1  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
g > 1  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
g != 1  # FIXME: this should be an error
g <= "1"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
g >= "1"  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
g == "1"  # FIXME: this should be an error

# Other GenExpr methods
assert_type(g.degree(), float)
assert_type(g.getOp(), L["+"])

# SumExpr
assert_type(SumExpr(), SumExpr)
SumExpr(e)  # type: ignore[call-arg] # pyright: ignore[reportCallIssue]

assert_type(SumExpr().constant, float)
assert_type(SumExpr().coefs, list[float])
assert_type(SumExpr().getOp(), Literal["sum"])

# ProdExpr
assert_type(ProdExpr(), ProdExpr)
ProdExpr(e)  # type: ignore[call-arg] # pyright: ignore[reportCallIssue]
assert_type(ProdExpr().constant, float)
assert_type(ProdExpr().getOp(), Literal["prod"])

# VarExpr
assert_type(VarExpr(x), VarExpr)
assert_type(VarExpr(var=x), VarExpr)
VarExpr()  # type: ignore[call-arg] # pyright: ignore[reportCallIssue]
VarExpr(e)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]

assert_type(VarExpr(x).children, list[Variable])
assert_type(VarExpr(x).getOp(), Literal["var"])

# PowExpr
assert_type(PowExpr(), PowExpr)
PowExpr(1)  # type: ignore[call-arg] # pyright: ignore[reportCallIssue]

assert_type(PowExpr().expo, float)
assert_type(PowExpr().getOp(), Literal["**"])

# UnaryExpr
assert_type(UnaryExpr(Operator.exp, g), UnaryExpr[L["exp"]])
assert_type(UnaryExpr(Operator.log, g), UnaryExpr[L["log"]])
assert_type(UnaryExpr(Operator.sqrt, g), UnaryExpr[L["sqrt"]])
assert_type(UnaryExpr(Operator.sin, expr=g), UnaryExpr[L["sin"]])
assert_type(UnaryExpr(op=Operator.cos, expr=g), UnaryExpr[L["cos"]])
assert_type(UnaryExpr(op=Operator.fabs, expr=g), UnaryExpr[L["abs"]])

# Test all operations with string literals
assert_type(UnaryExpr(op="exp", expr=g), UnaryExpr[L["exp"]])
assert_type(UnaryExpr(op="log", expr=g), UnaryExpr[L["log"]])
assert_type(UnaryExpr(op="sqrt", expr=g), UnaryExpr[L["sqrt"]])
assert_type(UnaryExpr(op="sin", expr=g), UnaryExpr[L["sin"]])
assert_type(UnaryExpr(op="cos", expr=g), UnaryExpr[L["cos"]])
assert_type(UnaryExpr(op="abs", expr=g), UnaryExpr[L["abs"]])

UnaryExpr()  # type: ignore[call-arg] # pyright: ignore[reportCallIssue]
UnaryExpr(op="invalid", expr=g)  # type: ignore[type-var] # pyright: ignore[reportArgumentType]
UnaryExpr(Operator.prod, g)  # type: ignore[type-var] # pyright: ignore[reportArgumentType]
UnaryExpr(Operator.exp, 1)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
UnaryExpr(Operator.exp, e)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]

assert_type(UnaryExpr(Operator.exp, g).getOp(), L["exp"])
assert_type(UnaryExpr(Operator.log, g).getOp(), L["log"])
assert_type(UnaryExpr(Operator.sqrt, g).getOp(), L["sqrt"])
assert_type(UnaryExpr(Operator.sin, g).getOp(), L["sin"])
assert_type(UnaryExpr(Operator.cos, g).getOp(), L["cos"])
assert_type(UnaryExpr(Operator.fabs, g).getOp(), L["abs"])

assert_type(UnaryExpr(Operator.exp, g).children, list[GenExpr[Any]])

# Constant
assert_type(Constant(1), Constant)
assert_type(Constant(number=1.5), Constant)
Constant()  # type: ignore[call-arg] # pyright: ignore[reportCallIssue]
Constant(e)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
# allowed at runtime, but any operation with it will fail
Constant(d)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]

assert_type(Constant(1).number, float)
assert_type(Constant(1).getOp(), Literal["const"])

# GenExpr builders
assert_type(scip.exp(e), UnaryExpr[L["exp"]])
assert_type(scip.exp(expr=d), UnaryExpr[L["exp"]])
assert_type(scip.exp(g), UnaryExpr[L["exp"]])
assert_type(scip.log(e), UnaryExpr[L["log"]])
assert_type(scip.log(d), UnaryExpr[L["log"]])
assert_type(scip.log(expr=g), UnaryExpr[L["log"]])
assert_type(scip.sqrt(expr=e), UnaryExpr[L["sqrt"]])
assert_type(scip.sqrt(d), UnaryExpr[L["sqrt"]])
assert_type(scip.sqrt(g), UnaryExpr[L["sqrt"]])
assert_type(scip.sin(e), UnaryExpr[L["sin"]])
assert_type(scip.sin(d), UnaryExpr[L["sin"]])
assert_type(scip.sin(expr=g), UnaryExpr[L["sin"]])
assert_type(scip.cos(e), UnaryExpr[L["cos"]])
assert_type(scip.cos(expr=d), UnaryExpr[L["cos"]])
assert_type(scip.cos(g), UnaryExpr[L["cos"]])

# Misc
expr_to_nodes(g)
expr_to_nodes(e)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
