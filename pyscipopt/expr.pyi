from _typeshed import Incomplete

class Expr:
    terms: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """terms is a dict of variables to coefficients.

        CONST is used as key for the constant term."""
    def degree(self, *args, **kwargs):
        """computes highest degree of terms"""
    def normalize(self, *args, **kwargs):
        """remove terms with coefficient of 0"""
    def __abs__(self):
        """abs(self)"""
    def __add__(self, other):
        """Return self+value."""
    def __eq__(self, other: object) -> bool:
        """Return self==value."""
    def __ge__(self, other: object) -> bool:
        """Return self>=value."""
    def __getitem__(self, index):
        """Return self[key]."""
    def __gt__(self, other: object) -> bool:
        """Return self>value."""
    def __iadd__(self, other):
        """Return self+=value."""
    def __iter__(self):
        """Implement iter(self)."""
    def __le__(self, other: object) -> bool:
        """Return self<=value."""
    def __lt__(self, other: object) -> bool:
        """Return self<value."""
    def __mul__(self, other):
        """Return self*value."""
    def __ne__(self, other: object) -> bool:
        """Return self!=value."""
    def __neg__(self):
        """-self"""
    def __next__(self): ...
    def __pow__(self, other):
        """Return pow(self, value, mod)."""
    def __radd__(self, other): ...
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __rmul__(self, other): ...
    def __rpow__(self, other):
        """Return pow(value, self, mod)."""
    def __rsub__(self, other): ...
    def __rtruediv__(self, other):
        """other / self"""
    def __setstate_cython__(self, *args, **kwargs): ...
    def __sub__(self, other):
        """Return self-value."""
    def __truediv__(self, other):
        """Return self/value."""

class ExprCons:
    expr: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def normalize(self, *args, **kwargs):
        """move constant terms in expression to bounds"""
    def __bool__(self) -> bool:
        """True if self else False"""
    def __eq__(self, other: object) -> bool:
        """Return self==value."""
    def __ge__(self, other: object) -> bool:
        """Return self>=value."""
    def __gt__(self, other: object) -> bool:
        """Return self>value."""
    def __le__(self, other: object) -> bool:
        """Return self<=value."""
    def __lt__(self, other: object) -> bool:
        """Return self<value."""
    def __ne__(self, other: object) -> bool:
        """Return self!=value."""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...

class GenExpr:
    children: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """ """
    def degree(self, *args, **kwargs):
        """Note: none of these expressions should be polynomial"""
    def getOp(self, *args, **kwargs):
        """returns operator of GenExpr"""
    def __abs__(self):
        """abs(self)"""
    def __add__(self, other):
        """Return self+value."""
    def __eq__(self, other: object) -> bool:
        """Return self==value."""
    def __ge__(self, other: object) -> bool:
        """Return self>=value."""
    def __gt__(self, other: object) -> bool:
        """Return self>value."""
    def __le__(self, other: object) -> bool:
        """Return self<=value."""
    def __lt__(self, other: object) -> bool:
        """Return self<value."""
    def __mul__(self, other):
        """Return self*value."""
    def __ne__(self, other: object) -> bool:
        """Return self!=value."""
    def __neg__(self):
        """-self"""
    def __pow__(self, other):
        """Return pow(self, value, mod)."""
    def __radd__(self, other): ...
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __rmul__(self, other): ...
    def __rpow__(self, other):
        """Return pow(value, self, mod)."""
    def __rsub__(self, other): ...
    def __rtruediv__(self, other):
        """other / self"""
    def __setstate_cython__(self, *args, **kwargs): ...
    def __sub__(self, other):
        """Return self-value."""
    def __truediv__(self, other):
        """Return self/value."""

class SumExpr(GenExpr):
    coefs: Incomplete
    constant: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...

class ProdExpr(GenExpr):
    constant: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...

class VarExpr(GenExpr):
    var: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...

class PowExpr(GenExpr):
    expo: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...

class UnaryExpr(GenExpr):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...

class Constant(GenExpr):
    number: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
