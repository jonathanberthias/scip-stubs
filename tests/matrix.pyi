from decimal import Decimal
from typing import Any

import numpy as np
from typing_extensions import assert_type

from pyscipopt.scip import Expr, MatrixExpr, MatrixExprCons, MatrixVariable, Variable

x: MatrixExpr

v: Variable
d: Decimal
arr: np.ndarray[tuple[Any, ...], np.dtype[np.float64]]

# MatrixExpr.sum
assert_type(x.sum(), Expr)

# Check axis argument
assert_type(x.sum(axis=None), Expr)
assert_type(x.sum(axis=1), MatrixExpr)
assert_type(x.sum(axis=(0,)), MatrixExpr)
assert_type(x.sum(axis=(0, 1)), MatrixExpr)

# Check keepdims argument
assert_type(x.sum(keepdims=False), Expr)
assert_type(x.sum(keepdims=True), MatrixExpr)
assert_type(x.sum(axis=None, keepdims=False), Expr)
assert_type(x.sum(axis=None, keepdims=True), MatrixExpr)
assert_type(x.sum(axis=0, keepdims=True), MatrixExpr)
assert_type(x.sum(axis=(0, 1), keepdims=False), MatrixExpr)

# Positional arguments don't work
x.sum(0)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]
x.sum(None, keepdims=True)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]
x.sum((0, 1), False)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]  # noqa: FBT003

# using other arguments should error
x.sum(dtype=int)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]  # ty: ignore[no-matching-overload]

# MatrixExpr.__le__/__ge__/__eq__

assert_type(x <= 2, MatrixExprCons)
assert_type(x == 2, MatrixExprCons)
assert_type(x >= 2, MatrixExprCons)

assert_type(x == v, MatrixExprCons)
assert_type(x <= v, MatrixExprCons)
assert_type(x >= v, MatrixExprCons)

assert_type(x >= arr, MatrixExprCons)
assert_type(x == arr, MatrixExprCons)
assert_type(x <= arr, MatrixExprCons)

# MatrixExpr.__add__
assert_type(x + x, MatrixExpr)
assert_type(x + v, MatrixExpr)
assert_type(x + d, MatrixExpr)
assert_type(x + arr, MatrixExpr)

# MatrixExpr.__sub__
assert_type(x - x, MatrixExpr)
assert_type(x - v, MatrixExpr)
assert_type(x - d, MatrixExpr)
assert_type(x - arr, MatrixExpr)

# MatrixExpr.__mul__
assert_type(x * x, MatrixExpr)
assert_type(x * v, MatrixExpr)
assert_type(x * d, MatrixExpr)
assert_type(x * arr, MatrixExpr)

# MatrixExpr.__truediv__
assert_type(x / x, MatrixExpr)
assert_type(x / v, MatrixExpr)
assert_type(x / d, MatrixExpr)
assert_type(x / arr, MatrixExpr)

# MatrixExpr.__pow__
assert_type(x**2, MatrixExpr)
x**v  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
assert_type(x**d, MatrixExpr)
assert_type(x**arr, MatrixExpr)

# MatrixExpr.__radd__
assert_type(2 + x, MatrixExpr)
assert_type(v + x, MatrixExpr)
assert_type(d + x, MatrixExpr)
# FIXME: this uses the numpy array addition, not our __radd__
assert_type(arr + x, MatrixExpr)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]

# MatrixExpr.__rsub__
assert_type(2 - x, MatrixExpr)
assert_type(v - x, MatrixExpr)
assert_type(d - x, MatrixExpr)
# FIXME: this uses the numpy array subtraction, not our __rsub__
assert_type(arr - x, MatrixExpr)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]

# MatrixExpr.__rmul__
assert_type(2 * x, MatrixExpr)
v * x  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
# FIXME: this works at runtime
d * x  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
# FIXME: this uses the numpy array multiplication, not our __rmul__
assert_type(arr * x, MatrixExpr)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]

# MatrixExpr.__rtruediv__
assert_type(2 / x, MatrixExpr)
v / x  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
# Note: this doesn't work at runtime, as opposed to mul
d / x  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
# FIXME: this uses the numpy array division, not our __rtruediv__
assert_type(arr / x, MatrixExpr)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]

# MatrixExpr.__iadd__
def iadd() -> None:
    y: MatrixExpr
    y += 3
    assert_type(y, MatrixExpr)
    y += d
    assert_type(y, MatrixExpr)
    y += v
    assert_type(y, MatrixExpr)
    y += arr
    assert_type(y, MatrixExpr)

    z: MatrixVariable
    assert_type(z, MatrixVariable)
    assert_type(z.__iadd__(1), MatrixExpr)

# FIXME: MatrixExprCons is broken, cannot chain comparisons
