import dataclasses
import os
from collections.abc import Callable, Mapping
from enum import IntEnum
from io import TextIOWrapper
from typing import (
    Any,
    AnyStr,
    Generic,
    Iterable,
    Iterator,
    Sequence,
    SupportsFloat,
    TypedDict,
    TypeVar,
    overload,
    type_check_only,
)

from typing_extensions import (
    CapsuleType,
    NotRequired,
    Self,
    TypeAlias,
    deprecated,
    override,
)
from typing_extensions import Literal as L

_VTypes: TypeAlias = L[
    "C", "CONTINUOUS",
    "B", "BINARY",
    "I", "INTEGER",
    "M", "IMPLINT",
]  # fmt: skip
_VTypesLong: TypeAlias = L["CONTINUOUS", "BINARY", "INTEGER", "IMPLINT"]

##########
# expr.pxi
##########

# const = 'const'
# varidx = 'var'
# exp, log, sqrt, sin, cos = 'exp', 'log', 'sqrt', 'sin', 'cos'
# plus, minus, mul, div, power = '+', '-', '*', '/', '**'
# add = 'sum'
# prod = 'prod'
# fabs = 'abs'

class Op:
    const: L["const"]
    varidx: L["var"]
    exp: L["exp"]
    log: L["log"]
    sqrt: L["sqrt"]
    sin: L["sin"]
    cos: L["cos"]
    plus: L["+"]
    minus: L["-"]
    mul: L["*"]
    div: L["/"]
    power: L["**"]
    add: L["sum"]
    prod: L["prod"]
    fabs: L["abs"]

Operator: Op

_OpType: TypeAlias = L[
    "const",
    "var",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "+",
    "-",
    "*",
    "/",
    "**",
    "sum",
    "prod",
    "abs",
]
_OpT = TypeVar("_OpT", bound=_OpType)

_UnaryOp: TypeAlias = L["exp", "log", "sqrt", "sin", "cos", "+", "-", "abs"]

class Term:
    hashval: int
    ptrtuple: tuple[int, ...]
    vartuple: tuple[Variable, ...]
    def __init__(self, *vartuple: Variable) -> None: ...
    def __getitem__(self, index: int, /) -> Variable: ...
    @override
    def __hash__(self) -> int: ...
    def __len__(self, /) -> int: ...
    def __add__(self, other: Term, /) -> Term: ...

CONST: Term

@overload
def buildGenExprObj(expr: Expr) -> SumExpr:
    """helper function to generate an object of type GenExpr"""

@overload
def buildGenExprObj(expr: GenExpr[_OpT]) -> GenExpr[_OpT]:
    """helper function to generate an object of type GenExpr"""

@overload
def buildGenExprObj(expr: SupportsFloat) -> Constant:
    """helper function to generate an object of type GenExpr"""

# This case is valid at runtime if expr is the string repr of a real number
# (i.e., float(expr) does not raise), but expr is not converted to a float
# so the returned value is essentially unusable.
# @overload
# def buildGenExprObj(expr: str) -> Constant: ...

class Expr:
    terms: dict[Term, float]
    def __init__(self, /, terms: dict[Term, float] | None = None) -> None:
        """terms is a dict of variables to coefficients.

        CONST is used as key for the constant term."""
    @overload
    def __getitem__(self, index: Variable, /) -> float: ...
    @overload
    def __getitem__(self, index: Term, /) -> float: ...
    def __iter__(self, /) -> Iterator[Term]: ...
    def __abs__(self, /) -> UnaryExpr[L["abs"]]: ...
    @overload
    def __add__(self, other: Expr | SupportsFloat, /) -> Expr: ...
    @overload
    def __add__(self, other: GenExpr[Any], /) -> SumExpr: ...
    @overload
    def __iadd__(self, other: Expr | SupportsFloat, /) -> Self: ...
    @overload
    def __iadd__(self, other: GenExpr[Any], /) -> SumExpr: ...
    @overload
    def __mul__(self, other: Expr | SupportsFloat, /) -> Expr: ...
    @overload
    def __mul__(self, other: GenExpr[Any], /) -> ProdExpr: ...
    @overload
    def __truediv__(self, other: SupportsFloat, /) -> Expr: ...
    @overload
    def __truediv__(self, other: Expr | GenExpr[Any], /) -> ProdExpr: ...
    def __rtruediv__(self, other: SupportsFloat, /) -> Expr:
        """other / self"""
    def __pow__(
        self, other: SupportsFloat, mod: Any = None, /
    ) -> Expr | PowExpr | L[1]: ...
    def __neg__(self, /) -> Expr: ...
    @overload
    def __sub__(self, other: Expr | SupportsFloat, /) -> Expr: ...
    @overload
    def __sub__(self, other: GenExpr[Any], /) -> SumExpr: ...
    def __radd__(self, other: SupportsFloat, /) -> Expr: ...
    def __rmul__(self, other: SupportsFloat, /) -> Expr: ...
    def __rsub__(self, other: SupportsFloat, /) -> Expr: ...
    @override
    def __eq__(self, other: Expr | SupportsFloat | GenExpr[Any], /) -> ExprCons: ...  # type: ignore[override]
    def __ge__(self, other: Expr | SupportsFloat | GenExpr[Any], /) -> ExprCons: ...
    def __le__(self, other: Expr | SupportsFloat | GenExpr[Any], /) -> ExprCons: ...
    def normalize(self, /) -> None:
        """remove terms with coefficient of 0"""
    def degree(self, /) -> int:
        """computes highest degree of terms"""

class ExprCons:
    expr: Expr | GenExpr[Any]
    _lhs: float | None
    _rhs: float | None
    @overload
    def __init__(self, /, expr: Expr | GenExpr[Any], lhs: None = None) -> None: ...
    @overload
    def __init__(
        self, /, expr: Expr | GenExpr[Any], lhs: float, rhs: None = None
    ) -> None: ...
    @overload
    def __init__(
        self, /, expr: Expr | GenExpr[Any], lhs: float | None, rhs: float
    ) -> None: ...
    def normalize(self, /) -> None:
        """move constant terms in expression to bounds"""
    def __ge__(self, other: SupportsFloat, /) -> ExprCons: ...
    def __le__(self, other: SupportsFloat, /) -> ExprCons: ...

@overload
def quicksum(termlist: Iterable[Expr | SupportsFloat]) -> Expr:  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
    """add linear expressions and constants much faster than Python's sum
    by avoiding intermediate data structures and adding terms inplace
    """

@overload
def quicksum(termlist: Iterable[Expr | SupportsFloat | GenExpr[Any]]) -> SumExpr:
    """add linear expressions and constants much faster than Python's sum
    by avoiding intermediate data structures and adding terms inplace
    """

@overload
def quickprod(termlist: Iterable[Expr | SupportsFloat]) -> Expr:  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
    """multiply linear expressions and constants by avoiding intermediate
    data structures and multiplying terms inplace
    """

@overload
def quickprod(termlist: Iterable[Expr | SupportsFloat | GenExpr[Any]]) -> ProdExpr:
    """multiply linear expressions and constants by avoiding intermediate
    data structures and multiplying terms inplace
    """

class GenExpr(Generic[_OpT]):
    _op: _OpT
    children: list[GenExpr[Any]]
    def __init__(self, /) -> None: ...
    def __abs__(self, /) -> UnaryExpr[L["abs"]]: ...
    def __add__(self, other: Expr | float | GenExpr[Any], /) -> SumExpr: ...
    def __mul__(self, other: Expr | float | GenExpr[Any], /) -> ProdExpr: ...
    def __pow__(self, other: SupportsFloat, mod: Any = None, /) -> PowExpr: ...
    def __truediv__(self, other: Expr | float | GenExpr[Any], /) -> ProdExpr: ...
    def __rtruediv__(self, other: float, /) -> ProdExpr:
        """other / self"""
    def __neg__(self, /) -> ProdExpr: ...
    def __sub__(self, other: Expr | float | GenExpr[Any], /) -> SumExpr: ...
    def __radd__(self, other: float, /) -> SumExpr: ...
    def __rmul__(self, other: float, /) -> SumExpr: ...
    def __rsub__(self, other: float, /) -> SumExpr: ...
    @override
    def __eq__(self, other: Expr | SupportsFloat | GenExpr[Any], /) -> ExprCons: ...  # type: ignore[override]
    def __ge__(self, other: Expr | SupportsFloat | GenExpr[Any], /) -> ExprCons: ...
    def __le__(self, other: Expr | SupportsFloat | GenExpr[Any], /) -> ExprCons: ...
    def degree(self, /) -> float:
        """Note: none of these expressions should be polynomial"""
    def getOp(self, /) -> _OpT:
        """returns operator of GenExpr"""

class SumExpr(GenExpr[L["sum"]]):
    constant: float
    coefs: list[float]
    def __init__(self, /) -> None: ...

class ProdExpr(GenExpr[L["prod"]]):
    constant: float
    def __init__(self, /) -> None: ...

class VarExpr(GenExpr[L["var"]]):
    var: Variable
    children: list[Variable]  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleVariableOverride]
    def __init__(self, /, var: Variable) -> None: ...

class PowExpr(GenExpr[L["**"]]):
    expo: float
    def __init__(self, /) -> None: ...

_UnaryOpT = TypeVar("_UnaryOpT", bound=_UnaryOp)

class UnaryExpr(GenExpr[_UnaryOpT]):
    def __init__(self, op: _UnaryOpT, expr: GenExpr[Any]) -> None: ...

class Constant(GenExpr[L["const"]]):
    number: float
    def __init__(self, /, number: float) -> None: ...

def exp(expr: Expr | SupportsFloat | GenExpr[Any]) -> UnaryExpr[L["exp"]]:
    """returns expression with exp-function"""

def log(expr: Expr | SupportsFloat | GenExpr[Any]) -> UnaryExpr[L["log"]]:
    """returns expression with log-function"""

def sqrt(expr: Expr | SupportsFloat | GenExpr[Any]) -> UnaryExpr[L["sqrt"]]:
    """returns expression with sqrt-function"""

def sin(expr: Expr | SupportsFloat | GenExpr[Any]) -> UnaryExpr[L["sin"]]:
    """returns expression with sin-function"""

def cos(expr: Expr | SupportsFloat | GenExpr[Any]) -> UnaryExpr[L["cos"]]:
    """returns expression with cos-function"""

_Node: TypeAlias = tuple[str, list[Variable | float]]

def expr_to_nodes(expr: GenExpr[Any]) -> list[_Node]:
    """transforms tree to an array of nodes. each node is an operator and the position of the
    children of that operator (i.e. the other nodes) in the array"""

def value_to_array(val: float, nodes: list[_Node]) -> int:
    """adds a given value to an array"""

def expr_to_array(expr: GenExpr[Any], nodes: list[Node]) -> int:
    """adds expression to array"""

########
# lp.pxi
########

class LP:
    def __init__(
        self, name: str = "LP", sense: L["minimize", "maximize"] = "minimize"
    ) -> None:
        """
        Keyword arguments:
        name -- the name of the problem (default 'LP')
        sense -- objective sense (default minimize)
        """
    @property
    def name(self, /) -> str: ...
    def writeLP(self, /, filename: bytes) -> None:
        """Writes LP to a file.

        Keyword arguments:
        filename -- the name of the file to be used
        """
    def readLP(self, /, filename: bytes) -> None:
        """Reads LP from a file.

        Keyword arguments:
        filename -- the name of the file to be used
        """
    def infinity(self, /) -> float:
        """Returns infinity value of the LP."""
    def isInfinity(self, /, val: SupportsFloat) -> bool:
        """Checks if a given value is equal to the infinity value of the LP.

        Keyword arguments:
        val -- value that should be checked
        """
    def addCol(
        self,
        entries: Sequence[tuple[int, float]],
        obj: float = 0.0,
        lb: float = 0.0,
        ub: float | None = None,
    ) -> None:
        """Adds a single column to the LP.

        Keyword arguments:
        entries -- list of tuples, each tuple consists of a row index and a coefficient
        obj     -- objective coefficient (default 0.0)
        lb      -- lower bound (default 0.0)
        ub      -- upper bound (default infinity)
        """
    def addCols(
        self,
        entrieslist: Sequence[Sequence[tuple[int, float]]],
        objs: Sequence[float] | None = None,
        lbs: Sequence[float] | None = None,
        ubs: Sequence[float] | None = None,
    ) -> None:
        """Adds multiple columns to the LP.

        Keyword arguments:
        entrieslist -- list containing lists of tuples, each tuple contains a coefficient and a row index
        objs  -- objective coefficient (default 0.0)
        lbs   -- lower bounds (default 0.0)
        ubs   -- upper bounds (default infinity)
        """
    def delCols(self, firstcol: int, lastcol: int) -> None:
        """Deletes a range of columns from the LP.

        Keyword arguments:
        firstcol -- first column to delete
        lastcol  -- last column to delete
        """
    def addRow(
        self,
        entries: Sequence[tuple[int, float]],
        lhs: float = 0.0,
        rhs: float | None = None,
    ) -> None:
        """Adds a single row to the LP.

        Keyword arguments:
        entries -- list of tuples, each tuple contains a coefficient and a column index
        lhs     -- left-hand side of the row (default 0.0)
        rhs     -- right-hand side of the row (default infinity)
        """
    def addRows(
        self,
        entrieslist: Sequence[Sequence[tuple[int, float]]],
        lhss: Sequence[float] | None = None,
        rhss: Sequence[float] | None = None,
    ) -> None:
        """Adds multiple rows to the LP.

        Keyword arguments:
        entrieslist -- list containing lists of tuples, each tuple contains a coefficient and a column index
        lhss        -- left-hand side of the row (default 0.0)
        rhss        -- right-hand side of the row (default infinity)
        """
    def delRows(self, firstrow: int, lastrow: int) -> None:
        """Deletes a range of rows from the LP.

        Keyword arguments:
        firstrow -- first row to delete
        lastrow  -- last row to delete
        """
    def getBounds(
        self, firstcol: int = 0, lastcol: int | None = None
    ) -> tuple[list[float], list[float]] | None:
        """Returns all lower and upper bounds for a range of columns.

        Keyword arguments:
        firstcol -- first column (default 0)
        lastcol  -- last column (default ncols - 1)
        """
    def getSides(
        self, firstrow: int = 0, lastrow: float | None = None
    ) -> tuple[list[float], list[float]] | None:
        """Returns all left- and right-hand sides for a range of rows.

        Keyword arguments:
        firstrow -- first row (default 0)
        lastrow  -- last row (default nrows - 1)
        """
    def chgObj(self, col: int, obj: float) -> None:
        """Changes objective coefficient of a single column.

        Keyword arguments:
        col -- column to change
        obj -- new objective coefficient
        """
    def chgCoef(self, row: int, col: int, newval: float) -> None:
        """Changes a single coefficient in the LP.

        Keyword arguments:
        row -- row to change
        col -- column to change
        newval -- new coefficient
        """
    def chgBound(self, col: int, lb: float, ub: float) -> None:
        """Changes the lower and upper bound of a single column.

        Keyword arguments:
        col -- column to change
        lb  -- new lower bound
        ub  -- new upper bound
        """
    def chgSide(self, row: int, lhs: float, rhs: float) -> None:
        """Changes the left- and right-hand side of a single row.

        Keyword arguments:
        row -- row to change
        lhs -- new left-hand side
        rhs -- new right-hand side
        """
    def clear(self) -> None:
        """Clears the whole LP."""
    def nrows(self) -> int:
        """Returns the number of rows."""
    def ncols(self) -> int:
        """Returns the number of columns."""
    def solve(self, dual: bool = True) -> float:
        """Solves the current LP.

        Keyword arguments:
        dual -- use the dual or primal Simplex method (default: dual)
        """
    def getPrimal(self) -> list[float]:
        """Returns the primal solution of the last LP solve."""
    def isPrimalFeasible(self) -> bool:
        """Returns True iff LP is proven to be primal feasible."""
    def getDual(self) -> list[float]:
        """Returns the dual solution of the last LP solve."""
    def isDualFeasible(self) -> bool:
        """Returns True iff LP is proven to be dual feasible."""
    def getPrimalRay(self) -> list[float] | None:
        """Returns a primal ray if possible, None otherwise."""
    def getDualRay(self) -> list[float] | None:
        """Returns a dual ray if possible, None otherwise."""
    def getNIterations(self) -> int:
        """Returns the number of LP iterations of the last LP solve."""
    def getRedcost(self) -> list[float]:
        """Returns the reduced cost vector of the last LP solve."""
    def getBasisInds(self) -> list[int]:
        """Returns the indices of the basic columns and rows; index i >= 0 corresponds to column i, index i < 0 to row -i-1"""

#############
# benders.pxi
#############

@type_check_only
class BendersPresubsolveRes(TypedDict):
    infeasible: NotRequired[bool]  # default: False
    auxviol: NotRequired[bool]  # default: False
    skipsolve: NotRequired[bool]  # default: False
    result: L[
        PY_SCIP_RESULT.DIDNOTRUN,
        PY_SCIP_RESULT.FEASIBLE,
        PY_SCIP_RESULT.INFEASIBLE,
        PY_SCIP_RESULT.CONSADDED,
        PY_SCIP_RESULT.SEPARATED,
    ]

@type_check_only
class BendersSolvesubRes(TypedDict):
    objective: float
    result: L[
        PY_SCIP_RESULT.DIDNOTRUN,
        PY_SCIP_RESULT.FEASIBLE,
        PY_SCIP_RESULT.INFEASIBLE,
        PY_SCIP_RESULT.UNBOUNDED,
    ]

@type_check_only
class BendersPostsolveRes(TypedDict):
    merged: bool

@type_check_only
class BendersGetvarRes(TypedDict):
    mappedvar: Variable | None

class Benders:
    model: Model
    name: str
    def bendersfree(self) -> None:
        """calls destructor and frees memory of Benders decomposition"""
    def bendersinit(self) -> None:
        """initializes Benders deconposition"""
    def bendersexit(self) -> None:
        """calls exit method of Benders decomposition"""
    def bendersinitpre(self) -> None:
        """informs the Benders decomposition that the presolving process is being started"""
    def bendersexitpre(self) -> None:
        """informs the Benders decomposition that the presolving process has been completed"""
    def bendersinitsol(self) -> None:
        """informs Benders decomposition that the branch and bound process is being started"""
    def bendersexitsol(self) -> None:
        """informs Benders decomposition that the branch and bound process data is being freed"""
    def benderscreatesub(self, probnumber: int) -> None:
        """creates the subproblems and registers it with the Benders decomposition struct"""
    def benderspresubsolve(
        self,
        solution: Solution | None,
        enfotype: PY_SCIP_BENDERSENFOTYPE,
        checkint: bool,
    ) -> BendersPresubsolveRes:
        """sets the pre subproblem solve callback of Benders decomposition"""
    def benderssolvesubconvex(
        self, solution: Solution | None, probnumber: int, onlyconvex: bool
    ) -> BendersSolvesubRes:
        """sets convex solve callback of Benders decomposition"""
    def benderssolvesub(
        self, solution: Solution | None, probnumber: int
    ) -> BendersSolvesubRes:
        """sets solve callback of Benders decomposition"""
    def benderspostsolve(
        self,
        solution: Solution | None,
        enfotype: PY_SCIP_BENDERSENFOTYPE,
        mergecandidates: list[int],
        npriomergecands: int,
        checkint: bool,
        infeasible: bool,
    ) -> BendersPostsolveRes:
        """sets post-solve callback of Benders decomposition"""
    def bendersfreesub(self, probnumber: int) -> None:
        """frees the subproblems"""
    def bendersgetvar(self, variable: Variable, probnumber: int) -> BendersGetvarRes:
        """Returns the corresponding master or subproblem variable for the given variable. This provides a call back for the variable mapping between the master and subproblems."""

################
# benderscut.pxi
################

@type_check_only
class BenderscutExecRes(TypedDict):
    result: L[
        PY_SCIP_RESULT.DIDNOTRUN,
        PY_SCIP_RESULT.DIDNOTFIND,
        PY_SCIP_RESULT.CONSADDED,
        PY_SCIP_RESULT.FEASIBLE,
        PY_SCIP_RESULT.SEPARATED,
    ]

class Benderscut:
    model: Model
    benders: Benders
    name: str
    def benderscutfree(self) -> None: ...
    def benderscutinit(self) -> None: ...
    def benderscutexit(self) -> None: ...
    def benderscutinitsol(self) -> None: ...
    def benderscutexitsol(self) -> None: ...
    def benderscutexec(
        self,
        solution: Solution | None,
        probnumber: int,
        enfotype: PY_SCIP_BENDERSENFOTYPE,
    ) -> BenderscutExecRes: ...

################
# branchrule.pxi
################

_BranchRuleAllowedResultsCommon: TypeAlias = L[
    PY_SCIP_RESULT.CUTOFF,
    PY_SCIP_RESULT.CONSADDED,  # only allowed if allowaddcons is True
    PY_SCIP_RESULT.REDUCEDDOM,
    PY_SCIP_RESULT.BRANCHED,
    PY_SCIP_RESULT.DIDNOTFIND,
    PY_SCIP_RESULT.DIDNOTRUN,
]

@type_check_only
class BranchRuleExecTD(TypedDict):
    result: _BranchRuleAllowedResultsCommon | L[PY_SCIP_RESULT.SEPARATED]

@type_check_only
class BranchRuleExecPsTD(TypedDict):
    result: _BranchRuleAllowedResultsCommon

class Branchrule:
    model: Model
    def branchfree(self) -> None:
        """frees memory of branching rule"""
    def branchinit(self) -> None:
        """initializes branching rule"""
    def branchexit(self) -> None:
        """deinitializes branching rule"""
    def branchinitsol(self) -> None:
        """informs branching rule that the branch and bound process is being started"""
    def branchexitsol(self) -> None:
        """informs branching rule that the branch and bound process data is being freed"""
    def branchexeclp(self, allowaddcons: L[True]) -> BranchRuleExecTD:
        """executes branching rule for fractional LP solution"""
    def branchexecext(self, allowaddcons: L[True]) -> BranchRuleExecTD:
        """executes branching rule for external branching candidates"""
    def branchexecps(self, allowaddcons: L[True]) -> BranchRuleExecPsTD:
        """executes branching rule for not completely fixed pseudo solution"""

##############
# conshdlr.pxi
##############

@type_check_only
class ConshdlrConsTransRes(TypedDict):
    targetcons: NotRequired[Constraint]

@type_check_only
class ConshdlrConsInitLpRes(TypedDict):
    infeasible: bool

@type_check_only
class ConshdlrConsSepaRes(TypedDict):
    result: L[
        PY_SCIP_RESULT.CUTOFF,
        PY_SCIP_RESULT.CONSADDED,
        PY_SCIP_RESULT.REDUCEDDOM,
        PY_SCIP_RESULT.SEPARATED,
        PY_SCIP_RESULT.NEWROUND,
        PY_SCIP_RESULT.DIDNOTFIND,
        PY_SCIP_RESULT.DIDNOTRUN,
        PY_SCIP_RESULT.DELAYED,
    ]

@type_check_only
class ConshdlrEnfoRes(TypedDict):
    result: L[
        PY_SCIP_RESULT.CUTOFF,
        PY_SCIP_RESULT.CONSADDED,
        PY_SCIP_RESULT.REDUCEDDOM,
        PY_SCIP_RESULT.SEPARATED,
        PY_SCIP_RESULT.SOLVELP,
        PY_SCIP_RESULT.BRANCHED,
        PY_SCIP_RESULT.INFEASIBLE,
        PY_SCIP_RESULT.FEASIBLE,
    ]

@type_check_only
class ConshdlrEnfoPsRes(TypedDict):
    result: L[
        PY_SCIP_RESULT.CUTOFF,
        PY_SCIP_RESULT.CONSADDED,
        PY_SCIP_RESULT.REDUCEDDOM,
        PY_SCIP_RESULT.BRANCHED,
        PY_SCIP_RESULT.SOLVELP,
        PY_SCIP_RESULT.INFEASIBLE,
        PY_SCIP_RESULT.FEASIBLE,
        PY_SCIP_RESULT.DIDNOTRUN,
    ]

@type_check_only
class ConshdlrConsCheckRes(TypedDict):
    result: L[PY_SCIP_RESULT.FEASIBLE, PY_SCIP_RESULT.INFEASIBLE]

@type_check_only
class ConshdlrConsPropRes(TypedDict):
    result: L[
        PY_SCIP_RESULT.CUTOFF,
        PY_SCIP_RESULT.REDUCEDDOM,
        PY_SCIP_RESULT.DIDNOTFIND,
        PY_SCIP_RESULT.DIDNOTRUN,
        PY_SCIP_RESULT.DELAYED,
        PY_SCIP_RESULT.DELAYNODE,
    ]

@type_check_only
class ConshdlrConsPresolResultDict(TypedDict):
    nfixedvars: int
    naggrvars: int
    nchgvartypes: int
    nchgbds: int
    naddholes: int
    ndelconss: int
    naddconss: int
    nupgdconss: int
    nchgcoefs: int
    nchgsides: int
    result: L[
        PY_SCIP_RESULT.CUTOFF,
        PY_SCIP_RESULT.UNBOUNDED,
        PY_SCIP_RESULT.SUCCESS,
        PY_SCIP_RESULT.DIDNOTFIND,
        PY_SCIP_RESULT.DIDNOTRUN,
        PY_SCIP_RESULT.DELAYED,
    ]

@type_check_only
class ConshdlrConsGetnVarsRes(TypedDict):
    nvars: int
    success: bool

class Conshdlr:
    model: Model
    name: str
    def consfree(self) -> None:
        """calls destructor and frees memory of constraint handler"""
    def consinit(self, constraints: list[Constraint]) -> None:
        """calls initialization method of constraint handler"""
    def consexit(self, constraints: list[Constraint]) -> None:
        """calls exit method of constraint handler"""
    def consinitpre(self, constraints: list[Constraint]) -> None:
        """informs constraint handler that the presolving process is being started"""
    def consexitpre(self, constraints: list[Constraint]) -> None:
        """informs constraint handler that the presolving is finished"""
    def consinitsol(self, constraints: list[Constraint]) -> None:
        """informs constraint handler that the branch and bound process is being started"""
    def consexitsol(self, constraints: list[Constraint], restart: bool) -> None:
        """informs constraint handler that the branch and bound process data is being freed"""
    def consdelete(self, constraint: Constraint) -> None:
        """sets method of constraint handler to free specific constraint data"""
    def constrans(self, sourceconstraint: Constraint) -> ConshdlrConsTransRes:
        """sets method of constraint handler to transform constraint data into data belonging to the transformed problem"""
    def consinitlp(self, constraints: list[Constraint]) -> ConshdlrConsInitLpRes:
        """calls LP initialization method of constraint handler to separate all initial active constraints"""
    def conssepalp(self, constraints: list[Constraint], nusefulconss: int) -> None:
        """calls separator method of constraint handler to separate LP solution"""
    def conssepasol(
        self, constraints: list[Constraint], nusefulconss: int, solution: Solution
    ) -> ConshdlrConsSepaRes:
        """calls separator method of constraint handler to separate given primal solution"""
    def consenfolp(
        self, constraints: list[Constraint], nusefulconss: int, solinfeasible: bool
    ) -> ConshdlrEnfoRes:
        """calls enforcing method of constraint handler for LP solution for all constraints added"""
    def consenforelax(
        self,
        solution: Solution,
        constraints: list[Constraint],
        nusefulconss: int,
        solinfeasible: bool,
    ) -> ConshdlrEnfoRes:
        """calls enforcing method of constraint handler for a relaxation solution for all constraints added"""
    def consenfops(
        self,
        constraints: list[Constraint],
        nusefulconss: int,
        solinfeasible: bool,
        objinfeasible: bool,
    ) -> ConshdlrEnfoPsRes:
        """calls enforcing method of constraint handler for pseudo solution for all constraints added"""
    def conscheck(
        self,
        constraints: list[Constraint],
        solution: Solution,
        checkintegrality: bool,
        checklprows: bool,
        printreason: bool,
        completely: bool,
    ) -> ConshdlrConsCheckRes:
        """calls feasibility check method of constraint handler"""
    def consprop(
        self,
        constraints: list[Constraint],
        nusefulconss: int,
        nmarkedconss: int,
        proptiming: PY_SCIP_PROPTIMING,
    ) -> ConshdlrConsPropRes:
        """calls propagation method of constraint handler"""
    def conspresol(
        self,
        constraints: list[Constraint],
        nrounds: int,
        presoltiming: PY_SCIP_PRESOLTIMING,
        nnewfixedvars: int,
        nnewaggrvars: int,
        nnewchgvartypes: int,
        nnewchgbds: int,
        nnewholes: int,
        nnewdelconss: int,
        nnewaddconss: int,
        nnewupgdconss: int,
        nnewchgcoefs: int,
        nnewchgsides: int,
        result_dict: ConshdlrConsPresolResultDict,
    ) -> None:
        """calls presolving method of constraint handler"""
    def consresprop(self) -> None:
        """sets propagation conflict resolving method of constraint handler"""
    def conslock(
        self,
        constraint: Constraint | None,
        # 0 == LockType.MODEL
        # 1 == LockType.CONFLICT
        # The enum is not available in PySCIPOpt
        locktype: L[0, 1],
        nlockspos: int,
        nlocksneg: int,
    ) -> None:
        """variable rounding lock method of constraint handler"""
    def consactive(self, constraint: Constraint) -> None:
        """sets activation notification method of constraint handler"""
    def consdeactive(self, constraint: Constraint) -> None:
        """sets deactivation notification method of constraint handler"""
    def consenable(self, constraint: Constraint) -> None:
        """sets enabling notification method of constraint handler"""
    def consdisable(self, constraint: Constraint) -> None:
        """sets disabling notification method of constraint handler"""
    def consdelvars(self, constraints: list[Constraint]) -> None:
        """calls variable deletion method of constraint handler"""
    def consprint(self, constraint: Constraint) -> None:
        """sets constraint display method of constraint handler"""
    def conscopy(self) -> None:
        """sets copy method of both the constraint handler and each associated constraint"""
    def consparse(self) -> None:
        """sets constraint parsing method of constraint handler"""
    def consgetvars(self, constraint: Constraint) -> None:
        """sets constraint variable getter method of constraint handler"""
    def consgetnvars(self, constraint: Constraint) -> ConshdlrConsGetnVarsRes:
        """sets constraint variable number getter method of constraint handler"""
    def consgetdivebdchgs(self) -> None:
        """calls diving solution enforcement callback of constraint handler, if it exists"""
    def consgetpermsymgraph(self) -> None:
        """permutation symmetry detection graph getter callback, if it exists"""
    def consgetsignedpermsymgraph(self) -> None:
        """signed permutation symmetry detection graph getter callback, if it exists"""

############
# cutsel.pxi
############

@type_check_only
class CutSelSelectReturnTD(TypedDict, total=False):  # all entries are optional
    result: L[PY_SCIP_RESULT.SUCCESS, PY_SCIP_RESULT.DIDNOTFIND]
    # default: DIDNOTFIND
    cuts: Sequence[Row]
    # default: input cuts that may be reordered in-place
    nselectedcuts: int
    # default: 0

class Cutsel:
    model: Model
    def cutselfree(self) -> None:
        """frees memory of cut selector"""
    def cutselinit(self) -> None:
        """executed after the problem is transformed. use this call to initialize cut selector data."""
    def cutselexit(self) -> None:
        """executed before the transformed problem is freed"""
    def cutselinitsol(self) -> None:
        """executed when the presolving is finished and the branch-and-bound process is about to begin"""
    def cutselexitsol(self) -> None:
        """executed before the branch-and-bound process is freed"""
    def cutselselect(
        self, cuts: list[Row], forcedcuts: list[Row], root: bool, maxnselectedcuts: int
    ) -> CutSelSelectReturnTD:
        """first method called in each iteration in the main solving loop."""

###########
# event.pxi
###########

class Eventhdlr:
    model: Model
    name: str
    def eventcopy(self) -> None:
        """sets copy callback for all events of this event handler"""
    def eventfree(self) -> None:
        """calls destructor and frees memory of event handler"""
    def eventinit(self) -> None:
        """initializes event handler"""
    def eventexit(self) -> None:
        """calls exit method of event handler"""
    def eventinitsol(self) -> None:
        """informs event handler that the branch and bound process is being started"""
    def eventexitsol(self) -> None:
        """informs event handler that the branch and bound process data is being freed"""
    def eventdelete(self) -> None:
        """sets callback to free specific event data"""
    def eventexec(self, event: Event) -> None:
        """calls execution method of event handler"""

###############
# heuristic.pxi
###############

@type_check_only
class HeurExecResultTD(TypedDict):
    result: L[
        PY_SCIP_RESULT.FOUNDSOL,
        PY_SCIP_RESULT.DIDNOTFIND,
        PY_SCIP_RESULT.DIDNOTRUN,
        PY_SCIP_RESULT.DELAYED,
        PY_SCIP_RESULT.UNBOUNDED,
    ]

class Heur:
    model: Model
    name: str
    def heurfree(self) -> None:
        """calls destructor and frees memory of primal heuristic"""
    def heurinit(self) -> None:
        """initializes primal heuristic"""
    def heurexit(self) -> None:
        """calls exit method of primal heuristic"""
    def heurinitsol(self) -> None:
        """informs primal heuristic that the branch and bound process is being started"""
    def heurexitsol(self) -> None:
        """informs primal heuristic that the branch and bound process data is being freed"""
    def heurexec(
        self, heurtiming: PY_SCIP_HEURTIMING, nodeinfeasible: bool
    ) -> HeurExecResultTD:
        """should the heuristic the executed at the given depth, frequency, timing,..."""

############
# presol.pxi
############

@type_check_only
class PresolExecRes(TypedDict):
    result: L[
        PY_SCIP_RESULT.CUTOFF,
        PY_SCIP_RESULT.UNBOUNDED,
        PY_SCIP_RESULT.SUCCESS,
        PY_SCIP_RESULT.DIDNOTFIND,
        PY_SCIP_RESULT.DIDNOTRUN,
    ]
    nnewfixedvars: NotRequired[int]  # default: 0
    nnewaggrvars: NotRequired[int]  # default: 0
    nnewchgvartypes: NotRequired[int]  # default: 0
    nnewchgbds: NotRequired[int]  # default: 0
    nnewaddholes: NotRequired[int]  # default: 0
    nnewdelconss: NotRequired[int]  # default: 0
    nnewaddconss: NotRequired[int]  # default: 0
    nnewupgdconss: NotRequired[int]  # default: 0
    nnewchgcoefs: NotRequired[int]  # default: 0
    nnewchgsides: NotRequired[int]  # default: 0

class Presol:
    model: Model
    def presolfree(self) -> None:
        """frees memory of presolver"""
    def presolinit(self) -> None:
        """initializes presolver"""
    def presolexit(self) -> None:
        """deinitializes presolver"""
    def presolinitpre(self) -> None:
        """informs presolver that the presolving process is being started"""
    def presolexitpre(self) -> None:
        """informs presolver that the presolving process is finished"""
    def presolexec(
        self, nrounds: int, presoltiming: PY_SCIP_PRESOLTIMING
    ) -> PresolExecRes:
        """executes presolver"""

############
# pricer.pxi
############

@type_check_only
class PricerRedcostRes(TypedDict):
    result: L[PY_SCIP_RESULT.DIDNOTRUN, PY_SCIP_RESULT.SUCCESS]
    lowerbound: float
    stopearly: bool

@type_check_only
class PricerFarkasRes(TypedDict):
    result: L[PY_SCIP_RESULT.DIDNOTRUN, PY_SCIP_RESULT.SUCCESS]

class Pricer:
    model: Model
    def pricerfree(self) -> None:
        """calls destructor and frees memory of variable pricer"""
    def pricerinit(self) -> None:
        """initializes variable pricer"""
    def pricerexit(self) -> None:
        """calls exit method of variable pricer"""
    def pricerinitsol(self) -> None:
        """informs variable pricer that the branch and bound process is being started"""
    def pricerexitsol(self) -> None:
        """informs variable pricer that the branch and bound process data is being freed"""
    def pricerredcost(self) -> PricerRedcostRes:
        """calls reduced cost pricing method of variable pricer"""
    def pricerfarkas(self) -> PricerFarkasRes:
        """calls Farkas pricing method of variable pricer"""

################
# propagator.pxi
################

@type_check_only
class PropPresolResultDict(TypedDict):
    result: L[
        PY_SCIP_RESULT.CUTOFF,
        PY_SCIP_RESULT.UNBOUNDED,
        PY_SCIP_RESULT.SUCCESS,
        PY_SCIP_RESULT.DIDNOTFIND,
        PY_SCIP_RESULT.DIDNOTRUN,
    ]
    nfixedvars: int
    naggrvars: int
    nchgvartypes: int
    nchgbds: int
    naddholes: int
    ndelconss: int
    naddconss: int
    nupgdconss: int
    nchgcoefs: int
    nchgsides: int

@type_check_only
class PropExecRes(TypedDict):
    result: L[
        PY_SCIP_RESULT.CUTOFF,
        PY_SCIP_RESULT.REDUCEDDOM,
        PY_SCIP_RESULT.DIDNOTFIND,
        PY_SCIP_RESULT.DIDNOTRUN,
        PY_SCIP_RESULT.DELAYED,
        PY_SCIP_RESULT.DELAYNODE,
    ]

@type_check_only
class PropResPropRes(TypedDict):
    result: L[PY_SCIP_RESULT.SUCCESS, PY_SCIP_RESULT.DIDNOTFIND]

class Prop:
    model: Model
    def propfree(self) -> None:
        """calls destructor and frees memory of propagator"""
    def propinit(self) -> None:
        """initializes propagator"""
    def propexit(self) -> None:
        """calls exit method of propagator"""
    def propinitsol(self) -> None:
        """informs propagator that the prop and bound process is being started"""
    def propexitsol(self, restart: bool) -> None:
        """informs propagator that the prop and bound process data is being freed"""
    def propinitpre(self) -> None:
        """informs propagator that the presolving process is being started"""
    def propexitpre(self) -> None:
        """informs propagator that the presolving process is finished"""
    def proppresol(
        self,
        nrounds: int,
        presoltiming: PY_SCIP_PRESOLTIMING,
        nnewfixedvars: int,
        nnewaggrvars: int,
        nnewchgvartypes: int,
        nnewchgbds: int,
        nnewholes: int,
        nnewdelconss: int,
        nnewaddconss: int,
        nnewupgdconss: int,
        nnewchgcoefs: int,
        nnewchgsides: int,
        result_dict: PropPresolResultDict,
    ) -> None:
        """executes presolving method of propagator"""
    def propexec(self, proptiming: PY_SCIP_PROPTIMING) -> PropExecRes:
        """calls execution method of propagator"""
    def propresprop(
        self,
        confvar: Variable,
        inferinfo: int,
        # 0 == SCIP_BOUNDTYPE_LOWER
        # 1 == SCIP_BOUNDTYPE_UPPER
        bdtype: L[0, 1],
        relaxedbd: float,
    ) -> PropResPropRes:
        """resolves the given conflicting bound, that was reduced by the given propagator"""

##########
# sepa.pxi
##########

@type_check_only
class SepaExecResultTD(TypedDict):
    result: L[
        PY_SCIP_RESULT.CUTOFF,
        PY_SCIP_RESULT.CONSADDED,
        PY_SCIP_RESULT.REDUCEDDOM,
        PY_SCIP_RESULT.SEPARATED,
        PY_SCIP_RESULT.NEWROUND,
        PY_SCIP_RESULT.DIDNOTFIND,
        PY_SCIP_RESULT.DIDNOTRUN,
        PY_SCIP_RESULT.DELAYED,
    ]

class Sepa:
    model: Model
    name: str
    def sepafree(self) -> None:
        """calls destructor and frees memory of separator"""
    def sepainit(self) -> None:
        """initializes separator"""
    def sepaexit(self) -> None:
        """calls exit method of separator"""
    def sepainitsol(self) -> None:
        """informs separator that the branch and bound process is being started"""
    def sepaexitsol(self) -> None:
        """informs separator that the branch and bound process data is being freed"""
    def sepaexeclp(self) -> SepaExecResultTD:
        """calls LP separation method of separator"""
    def sepaexecsol(self, solution: Solution) -> SepaExecResultTD:
        """calls primal solution separation method of separator"""

############
# reader.pxi
############

@type_check_only
class ReaderRes(TypedDict):
    result: NotRequired[L[PY_SCIP_RESULT.DIDNOTRUN, PY_SCIP_RESULT.SUCCESS]]
    # default: DIDNOTRUN

class Reader:
    model: Model
    name: str
    def readerfree(self) -> None:
        """calls destructor and frees memory of reader"""
    def readerread(self, filename: str) -> ReaderRes:
        """calls read method of reader"""
    def readerwrite(
        self,
        file: TextIOWrapper,
        name: str,
        transformed: bool,
        # -1 == SCIP_OBJSENSE_MAXIMIZE
        # +1 == SCIP_OBJSENSE_MINIMIZE
        objsense: L[-1, 1],
        objscale: float,
        objoffset: float,
        binvars: list[Variable],
        intvars: list[Variable],
        implvars: list[Variable],
        contvars: list[Variable],
        fixedvars: list[Variable],
        startnvars: int,
        conss: list[Constraint],
        maxnconss: int,
        startnconss: int,
        genericnames: bool,
    ) -> ReaderRes:
        """calls write method of reader"""

###########
# relax.pxi
###########

@type_check_only
class RelaxExecRes(TypedDict):
    result: L[
        PY_SCIP_RESULT.CUTOFF,
        PY_SCIP_RESULT.CONSADDED,
        PY_SCIP_RESULT.REDUCEDDOM,
        PY_SCIP_RESULT.SEPARATED,
        PY_SCIP_RESULT.SUCCESS,
        PY_SCIP_RESULT.SUSPENDED,
        PY_SCIP_RESULT.DIDNOTRUN,
    ]
    lowerbound: float

class Relax:
    model: Model
    name: str
    def relaxfree(self) -> None:
        """calls destructor and frees memory of relaxation handler"""
    def relaxinit(self) -> None:
        """initializes relaxation handler"""
    def relaxexit(self) -> None:
        """calls exit method of relaxation handler"""
    def relaxinitsol(self) -> None:
        """informs relaxaton handler that the branch and bound process is being started"""
    def relaxexitsol(self) -> None:
        """informs relaxation handler that the branch and bound process data is being freed"""
    def relaxexec(self) -> RelaxExecRes:
        """callls execution method of relaxation handler"""

#############
# nodesel.pxi
#############

@type_check_only
class NodeselNodeselectTD(TypedDict):
    selnode: Node

class Nodesel:
    model: Model
    def nodefree(self) -> None:
        """frees memory of node selector"""
    def nodeinit(self) -> None:
        """executed after the problem is transformed. use this call to initialize node selector data."""
    def nodeexit(self) -> None:
        """executed before the transformed problem is freed"""
    def nodeinitsol(self) -> None:
        """executed when the presolving is finished and the branch-and-bound process is about to begin"""
    def nodeexitsol(self) -> None:
        """executed before the branch-and-bound process is freed"""
    def nodeselect(self) -> NodeselNodeselectTD:
        """first method called in each iteration in the main solving loop."""
    def nodecomp(self, node1: Node, node2: Node) -> int:
        """
        compare two leaves of the current branching tree

        It should return the following values:

          value < 0, if node 1 comes before (is better than) node 2
          value = 0, if both nodes are equally good
          value > 0, if node 1 comes after (is worse than) node 2.
        """

##########
# scip.pxi
##########

MAJOR: L[9]
MINOR: L[2]
PATCH: L[1]

def str_conversion(x: str) -> bytes: ...

_SCIP_BOUNDTYPE_TO_STRING: dict[int, str]

class PY_SCIP_RESULT(IntEnum):
    DIDNOTRUN = 1
    DELAYED = 2
    DIDNOTFIND = 3
    FEASIBLE = 4
    INFEASIBLE = 5
    UNBOUNDED = 6
    CUTOFF = 7
    SEPARATED = 8
    NEWROUND = 9
    REDUCEDDOM = 10
    CONSADDED = 11
    CONSCHANGED = 12
    BRANCHED = 13
    SOLVELP = 14
    FOUNDSOL = 15
    SUSPENDED = 16
    SUCCESS = 17
    DELAYNODE = 18

class PY_SCIP_PARAMSETTING(IntEnum):
    DEFAULT = 0
    AGGRESSIVE = 1
    FAST = 2
    OFF = 3

class PY_SCIP_PARAMEMPHASIS(IntEnum):
    DEFAULT = 0
    CPSOLVER = 1
    EASYCIP = 2
    FEASIBILITY = 3
    HARDLP = 4
    OPTIMALITY = 5
    COUNTER = 6
    PHASEFEAS = 7
    PHASEIMPROVE = 8
    PHASEPROOF = 9
    NUMERICS = 10
    BENCHMARK = 11

class PY_SCIP_STATUS(IntEnum):
    UNKNOWN = 0
    USERINTERRUPT = 1
    NODELIMIT = 2
    TOTALNODELIMIT = 3
    STALLNODELIMIT = 4
    TIMELIMIT = 5
    MEMLIMIT = 6
    GAPLIMIT = 7
    SOLLIMIT = 8
    BESTSOLLIMIT = 9
    RESTARTLIMIT = 10
    PRIMALLIMIT = 16
    DUALLIMIT = 17
    OPTIMAL = 11
    INFEASIBLE = 12
    UNBOUNDED = 13
    INFORUNBD = 14

StageNames: dict[int, str]

class PY_SCIP_STAGE(IntEnum):
    INIT = 0
    PROBLEM = 1
    TRANSFORMING = 2
    TRANSFORMED = 3
    INITPRESOLVE = 4
    PRESOLVING = 5
    EXITPRESOLVE = 6
    PRESOLVED = 7
    INITSOLVE = 8
    SOLVING = 9
    SOLVED = 10
    EXITSOLVE = 11
    FREETRANS = 12
    FREE = 13

class PY_SCIP_NODETYPE(IntEnum):
    FOCUSNODE = 0
    PROBINGNODE = 1
    SIBLING = 2
    CHILD = 3
    LEAF = 4
    DEADEND = 5
    JUNCTION = 6
    PSEUDOFORK = 7
    FORK = 8
    SUBROOT = 9
    REFOCUSNODE = 10

class PY_SCIP_PROPTIMING(IntEnum):
    BEFORELP = 0x1
    DURINGLPLOOP = 0x2
    AFTERLPLOOP = 0x4
    AFTERLPNODE = 0x8

class PY_SCIP_PRESOLTIMING(IntEnum):
    NONE = 0x2
    FAST = 0x4
    MEDIUM = 0x8
    EXHAUSTIVE = 0x10

class PY_SCIP_HEURTIMING(IntEnum):
    BEFORENODE = 0x1
    DURINGLPLOOP = 0x2
    AFTERLPLOOP = 0x4
    AFTERLPNODE = 0x8
    AFTERPSEUDONODE = 0x10
    AFTERLPPLUNGE = 0x20
    AFTERPSEUDOPLUNGE = 0x40
    DURINGPRICINGLOOP = 0x80
    BEFOREPRESOL = 0x100
    DURINGPRESOLLOOP = 0x200
    AFTERPROPLOOP = 0x400

EventNames: dict[int, str]

class PY_SCIP_EVENTTYPE(IntEnum):
    DISABLED = 0x0
    VARADDED = 0x1
    VARDELETED = 0x2
    VARFIXED = 0x4
    VARUNLOCKED = 0x8
    OBJCHANGED = 0x10
    GLBCHANGED = 0x20
    GUBCHANGED = 0x40
    LBTIGHTENED = 0x80
    LBRELAXED = 0x100
    UBTIGHTENED = 0x200
    UBRELAXED = 0x400
    GHOLEADDED = 0x800
    GHOLEREMOVED = 0x1000
    LHOLEADDED = 0x2000
    LHOLEREMOVED = 0x4000
    IMPLADDED = 0x8000
    PRESOLVEROUND = 0x20000
    NODEFOCUSED = 0x40000
    NODEFEASIBLE = 0x80000
    NODEINFEASIBLE = 0x100000
    NODEBRANCHED = 0x200000
    NODEDELETE = 0x400000
    FIRSTLPSOLVED = 0x800000
    LPSOLVED = 0x1000000
    LPEVENT = 0x1800000
    POORSOLFOUND = 0x2000000
    BESTSOLFOUND = 0x4000000
    ROWADDEDSEPA = 0x8000000
    ROWDELETEDSEPA = 0x10000000
    ROWADDEDLP = 0x20000000
    ROWDELETEDLP = 0x40000000
    ROWCOEFCHANGED = 0x80000000
    ROWCONSTCHANGED = 0x100000000  # noqa: PYI054
    ROWSIDECHANGED = 0x200000000  # noqa: PYI054
    SYNC = 0x400000000  # noqa: PYI054

    GBDCHANGED = 0x60
    LBCHANGED = 0x180
    UBCHANGED = 0x600
    BOUNDTIGHTENED = 0x280
    BOUNDRELAXED = 0x500
    BOUNDCHANGED = 0x780
    GHOLECHANGED = 0x1800
    LHOLECHANGED = 0x6000
    HOLECHANGED = 0x7800
    DOMCHANGED = 0x7F80
    VARCHANGED = 0x1FFFE
    VAREVENT = 0x1FFFF
    NODESOLVED = 0x380000
    NODEEVENT = 0x3C0000
    SOLFOUND = 0x6000000
    SOLEVENT = 0x6000000
    ROWCHANGED = 0x380000000  # noqa: PYI054
    ROWEVENT = 0x3F8000000  # noqa: PYI054

class PY_SCIP_LPSOLSTAT(IntEnum):
    NOTSOLVED = 0
    OPTIMAL = 1
    INFEASIBLE = 2
    UNBOUNDEDRAY = 3
    OBJLIMIT = 4
    ITERLIMIT = 5
    TIMELIMIT = 6
    ERROR = 7

class PY_SCIP_BRANCHDIR(IntEnum):
    DOWNWARDS = 0
    UPWARDS = 1
    FIXED = 2
    AUTO = 3

class PY_SCIP_BENDERSENFOTYPE(IntEnum):
    LP = 1
    RELAX = 2
    PSEUDO = 3
    CHECK = 4

class PY_SCIP_ROWORIGINTYPE(IntEnum):
    UNSPEC = 0
    CONSHDLR = 1
    CONS = 2
    SEPA = 3
    REOPT = 4

class PY_SCIP_SOLORIGIN(IntEnum):
    ORIGINAL = 0
    ZERO = 1
    LPSOL = 2
    NLPSOL = 3
    RELAXSOL = 4
    PSEUDOSOL = 5
    PARTIAL = 6
    UNKNOWN = 7

def PY_SCIP_CALL(rc: int) -> None: ...

class Event:
    data: object
    def getType(self) -> PY_SCIP_EVENTTYPE:
        """
        Gets type of event.

        Returns
        -------
        PY_SCIP_EVENTTYPE

        """
    def getName(self) -> str:
        """
        Gets name of event.

        Returns
        -------
        str

        """
    def getNewBound(self) -> float:
        """
        Gets new bound for a bound change event.

        Returns
        -------
        float

        """
    def getOldBound(self) -> float:
        """
        Gets old bound for a bound change event.

        Returns
        -------
        float

        """
    def getVar(self) -> Variable:
        """
        Gets variable for a variable event (var added, var deleted, var fixed,
        objective value or domain change, domain hole added or removed).

        Returns
        -------
        Variable

        """
    def getNode(self) -> Node:
        """
        Gets node for a node or LP event.

        Returns
        -------
        Node

        """
    def getRow(self) -> Row:
        """
        Gets row for a row event.

        Returns
        -------
        Row

        """
    @override
    def __hash__(self) -> int: ...

class Column:
    data: object
    def __init__(self) -> None: ...
    def getLPPos(self) -> int:
        """
        Gets position of column in current LP, or -1 if it is not in LP.

        Returns
        -------
        int

        """
    def getBasisStatus(self) -> L["lower", "basic", "upper", "zero"]:
        """
        Gets the basis status of a column in the LP solution

        Returns
        -------
        str
            Possible values are "lower", "basic", "upper", and "zero"

        Raises
        ------
        Exception
            If SCIP returns an unknown basis status

        Notes
        -----
        Returns basis status "zero" for columns not in the current SCIP LP.

        """
    def isIntegral(self) -> bool:
        """
        Returns whether the associated variable is of integral type (binary, integer, implicit integer).

        Returns
        -------
        bool

        """
    def getVar(self) -> Variable:
        """
        Gets variable this column represents.

        Returns
        -------
        Variable

        """
    def getPrimsol(self) -> float:
        """
        Gets the primal LP solution of a column.

        Returns
        -------
        float

        """
    def getLb(self) -> float:
        """
        Gets lower bound of column.

        Returns
        -------
        float

        """
    def getUb(self) -> float:
        """
        Gets upper bound of column.

        Returns
        -------
        float

        """
    def getObjCoeff(self) -> float:
        """
        Gets objective value coefficient of a column.

        Returns
        -------
        float

        """
    def getAge(self) -> int:
        """
        Gets the age of the column, i.e., the total number of successive times a column was in the LP
        and was 0.0 in the solution.

        Returns
        -------
        int

        """
    @override
    def __hash__(self) -> int: ...

class Row:
    data: object
    @property
    def name(self) -> str: ...
    def getLhs(self) -> float:
        """
        Returns the left hand side of row.

        Returns
        -------
        float

        """
    def getRhs(self) -> float:
        """
        Returns the right hand side of row.

        Returns
        -------
        float

        """
    def getConstant(self) -> float:
        """
        Gets constant shift of row.

        Returns
        -------
        float

        """
    def getLPPos(self) -> int:
        """
        Gets position of row in current LP, or -1 if it is not in LP.

        Returns
        -------
        int

        """
    def getBasisStatus(self) -> L["lower", "basic", "upper"]:
        """
        Gets the basis status of a row in the LP solution.

        Returns
        -------
        str
            Possible values are "lower", "basic", and "upper"

        Raises
        ------
        Exception
            If SCIP returns an unknown or "zero" basis status

        Notes
        -----
        Returns basis status "basic" for rows not in the current SCIP LP.

        """
    def isIntegral(self) -> bool:
        """
        Returns TRUE iff the activity of the row (without the row's constant)
        is always integral in a feasible solution.

        Returns
        -------
        bool

        """
    def isLocal(self) -> bool:
        """
        Returns TRUE iff the row is only valid locally.

        Returns
        -------
        bool

        """
    def isModifiable(self) -> bool:
        """
        Returns TRUE iff row is modifiable during node processing (subject to column generation).

        Returns
        -------
        bool

        """
    def isRemovable(self) -> bool:
        """
        Returns TRUE iff row is removable from the LP (due to aging or cleanup).

        Returns
        -------
        bool

        """
    def isInGlobalCutpool(self) -> bool:
        """
        Return TRUE iff row is a member of the global cut pool.

        Returns
        -------
        bool

        """
    def getOrigintype(self) -> PY_SCIP_ROWORIGINTYPE:
        """
        Returns type of origin that created the row.

        Returns
        -------
        PY_SCIP_ROWORIGINTYPE

        """
    def getConsOriginConshdlrtype(self) -> str:
        """
        Returns type of constraint handler that created the row.

        Returns
        -------
        str

        """
    def getNNonz(self) -> int:
        """
        Get number of nonzero entries in row vector.

        Returns
        -------
        int

        """
    def getNLPNonz(self) -> int:
        """
        Get number of nonzero entries in row vector that correspond to columns currently in the SCIP LP.

        Returns
        -------
        int

        """
    def getCols(self) -> list[Column]:
        """
        Gets list with columns of nonzero entries

        Returns
        -------
        list of Column

        """
    def getVals(self) -> list[float]:
        """
        Gets list with coefficients of nonzero entries.

        Returns
        -------
        list of int

        """
    def getAge(self) -> int:
        """
        Gets the age of the row. (The consecutive times the row has been non-active in the LP).

        Returns
        -------
        int

        """
    def getNorm(self) -> float:
        """
        Gets Euclidean norm of row vector.

        Returns
        -------
        float

        """
    @override
    def __hash__(self) -> int: ...

class NLRow:
    data: object
    @property
    def name(self) -> str: ...
    def getConstant(self) -> float:
        """
        Returns the constant of a nonlinear row.

        Returns
        -------
        float

        """
    def getLinearTerms(self) -> list[tuple[Variable, float]]:
        """
        Returns a list of tuples (var, coef) representing the linear part of a nonlinear row.

        Returns
        -------
        list of tuple

        """
    def getLhs(self) -> float:
        """
        Returns the left hand side of a nonlinear row.

        Returns
        -------
        float

        """
    def getRhs(self) -> float:
        """
        Returns the right hand side of a nonlinear row.

        Returns
        -------
        float

        """
    def getDualsol(self) -> float:
        """
        Gets the dual NLP solution of a nonlinear row.

        Returns
        -------
        float

        """
    @override
    def __hash__(self) -> int: ...

class Solution:
    data: object
    def __init__(self, raise_error: bool = False) -> None: ...
    def __getitem__(self, /, expr: Expr) -> float: ...
    def __setitem__(self, /, var: Variable, value: float) -> None: ...
    def getOrigin(self) -> PY_SCIP_SOLORIGIN:
        """
        Returns origin of solution: where to retrieve uncached elements.

        Returns
        -------
        PY_SCIP_SOLORIGIN
        """
    def retransform(self) -> None:
        """retransforms solution to original problem space"""
    def translate(self, target: Model) -> Solution:
        """
        translate solution to a target model solution

        Parameters
        ----------
        target : Model

        Returns
        -------
        targetSol: Solution
        """

class BoundChange:
    def getNewBound(self) -> float:
        """
        Returns the new value of the bound in the bound change.

        Returns
        -------
        float

        """
    def getVar(self) -> Variable:
        """
        Returns the variable of the bound change.

        Returns
        -------
        Variable

        """
    # TODO: enum? (0 = branching, 1 = consinfer, 2 = propinfer)
    def getBoundchgtype(self) -> int:
        """
        Returns the bound change type of the bound change.

        Returns
        -------
        int
            (0 = branching, 1 = consinfer, 2 = propinfer)

        """
    # TODO: enum? (0 = lower, 1 = upper)
    def getBoundtype(self) -> int:
        """
        Returns the bound type of the bound change.

        Returns
        -------
        int
            (0 = lower, 1 = upper)

        """
    def isRedundant(self) -> bool:
        """
        Returns whether the bound change is redundant due to a more global bound that is at least as strong.

        Returns
        -------
        bool

        """

class DomainChanges:
    def getBoundchgs(self) -> list[BoundChange]:
        """
        Returns the bound changes in the domain change.

        Returns
        -------
        list of BoundChange

        """

class Node:
    data: object
    def getParent(self) -> Node | None:
        """
        Retrieve parent node (or None if the node has no parent node).

        Returns
        -------
        Node

        """
    def getNumber(self) -> int:
        """
        Retrieve number of node.

        Returns
        -------
        int

        """
    def getDepth(self) -> int:
        """
        Retrieve depth of node.

        Returns
        -------
        int

        """
    def getType(self) -> PY_SCIP_NODETYPE:
        """
        Retrieve type of node.

        Returns
        -------
        PY_SCIP_NODETYPE

        """
    def getLowerbound(self) -> float:
        """
        Retrieve lower bound of node.

        Returns
        -------
        float

        """
    def getEstimate(self) -> float:
        """
        Retrieve the estimated value of the best feasible solution in subtree of the node.

        Returns
        -------
        float

        """
    def getAddedConss(self) -> list[Constraint]:
        """
        Retrieve all constraints added at this node.

        Returns
        -------
        list of Constraint

        """
    def getNAddedConss(self) -> int:
        """
        Retrieve number of added constraints at this node.

        Returns
        -------
        int

        """
    def isActive(self) -> bool:
        """
        Is the node in the path to the current node?

        Returns
        -------
        bool

        """
    def isPropagatedAgain(self) -> bool:
        """
        Is the node marked to be propagated again?

        Returns
        -------
        bool

        """
    def getNParentBranchings(self) -> int:
        """
        Retrieve the number of variable branchings that were performed in the parent node to create this node.

        Returns
        -------
        int

        """
    # TODO: the ints are SCIP_BOUNDTYPEs
    def getParentBranchings(
        self,
    ) -> tuple[list[Variable], list[float], list[int]] | None:
        """
        Retrieve the set of variable branchings that were performed in the parent node to create this node.

        Returns
        -------
        list of Variable
        list of float
        list of int

        """
    def getNDomchg(self) -> tuple[int, int, int]:
        """
        Retrieve the number of bound changes due to branching, constraint propagation, and propagation.

        Returns
        -------
        nbranchings : int
        nconsprop : int
        nprop : int

        """
    def getDomchg(self) -> DomainChanges | None:
        """
        Retrieve domain changes for this node.

        Returns
        -------
        DomainChanges

        """
    @override
    def __hash__(self) -> int: ...

class Variable(Expr):
    data: object
    @property
    def name(self) -> str: ...
    def ptr(self) -> int:
        """ """
    def vtype(self) -> _VTypesLong:
        """
        Retrieve the variables type (BINARY, INTEGER, IMPLINT or CONTINUOUS)

        Returns
        -------
        str
            "BINARY", "INTEGER", "CONTINUOUS", or "IMPLINT"

        """
    def isOriginal(self) -> bool:
        """
        Retrieve whether the variable belongs to the original problem

        Returns
        -------
        bool

        """
    def isInLP(self) -> bool:
        """
        Retrieve whether the variable is a COLUMN variable that is member of the current LP.

        Returns
        -------
        bool

        """
    def getIndex(self) -> int:
        """
        Retrieve the unique index of the variable.

        Returns
        -------
        int

        """
    def getCol(self) -> Column:
        """
        Retrieve column of COLUMN variable.

        Returns
        -------
        Column

        """
    def getLbOriginal(self) -> float:
        """
        Retrieve original lower bound of variable.

        Returns
        -------
        float

        """
    def getUbOriginal(self) -> float:
        """
        Retrieve original upper bound of variable.

        Returns
        -------
        float

        """
    def getLbGlobal(self) -> float:
        """
        Retrieve global lower bound of variable.

        Returns
        -------
        float

        """
    def getUbGlobal(self) -> float:
        """
        Retrieve global upper bound of variable.

        Returns
        -------
        float

        """
    def getLbLocal(self) -> float:
        """
        Retrieve current lower bound of variable.

        Returns
        -------
        float

        """
    def getUbLocal(self) -> float:
        """
        Retrieve current upper bound of variable.

        Returns
        -------
        float

        """
    def getObj(self) -> float:
        """
        Retrieve current objective value of variable.

        Returns
        -------
        float

        """
    def getLPSol(self) -> float:
        """
        Retrieve the current LP solution value of variable.

        Returns
        -------
        float

        """
    def getAvgSol(self) -> float:
        """
        Get the weighted average solution of variable in all feasible primal solutions found.

        Returns
        -------
        float

        """
    def varMayRound(self, direction: L["down", "up"] = "down") -> bool:
        """
        Checks whether it is possible to round variable up / down and stay feasible for the relaxation.

        Parameters
        ----------
        direction : str
            "up" or "down"

        Returns
        -------
        bool

        """

# TODO: make Constraint generic over type of `data`
# This can't be done only in the stubs as the Constraint class
# is not generic and thus can't be indexed by a type variable.
# Attempted in commit 5897e49
class Constraint:
    data: Any
    @property
    def name(self) -> str: ...
    def isOriginal(self) -> bool:
        """
        Retrieve whether the constraint belongs to the original problem.

        Returns
        -------
        bool

        """
    def isInitial(self) -> bool:
        """
        Returns True if the relaxation of the constraint should be in the initial LP.

        Returns
        -------
        bool

        """
    def isSeparated(self) -> bool:
        """
        Returns True if constraint should be separated during LP processing.

        Returns
        -------
        bool

        """
    def isEnforced(self) -> bool:
        """
        Returns True if constraint should be enforced during node processing.

        Returns
        -------
        bool

        """
    def isChecked(self) -> bool:
        """
        Returns True if constraint should be checked for feasibility.

        Returns
        -------
        bool

        """
    def isPropagated(self) -> bool:
        """
        Returns True if constraint should be propagated during node processing.

        Returns
        -------
        bool

        """
    def isLocal(self) -> bool:
        """
        Returns True if constraint is only locally valid or not added to any (sub)problem.

        Returns
        -------
        bool

        """
    def isModifiable(self) -> bool:
        """
        Returns True if constraint is modifiable (subject to column generation).

        Returns
        -------
        bool

        """
    def isDynamic(self) -> bool:
        """
        Returns True if constraint is subject to aging.

        Returns
        -------
        bool

        """
    def isRemovable(self) -> bool:
        """
        Returns True if constraint's relaxation should be removed from the LP due to aging or cleanup.

        Returns
        -------
        bool

        """
    def isStickingAtNode(self) -> bool:
        """
        Returns True if constraint is only locally valid or not added to any (sub)problem.

        Returns
        -------
        bool

        """
    def isActive(self) -> bool:
        """
        Returns True iff constraint is active in the current node.

        Returns
        -------
        bool

        """
    def isLinear(self) -> bool:
        """
        Returns True if constraint is linear

        Returns
        -------
        bool

        """
    def isNonlinear(self) -> bool:
        """
        Returns True if constraint is nonlinear.

        Returns
        -------
        bool

        """
    def getConshdlrName(self) -> str:
        """
        Return the constraint handler's name.

        Returns
        -------
        str

        """
    @override
    def __hash__(self) -> int: ...

class Model:
    data: Any
    def __init__(
        self,
        problemName: str = "model",
        defaultPlugins: bool = True,
        sourceModel: Model | None = None,
        origcopy: bool = False,
        flobalcopy: bool = True,
        enablepricing: bool = False,
        createscip: bool = True,
        threadsafe: bool = False,
    ) -> None: ...
    def attachEventHandlerCallback(
        self,
        callback: Callable[[Model, Event], None],
        events: Iterable[PY_SCIP_EVENTTYPE],
        name: str = "eventhandler",
        description: str = "",
    ) -> None:
        """
        Attach an event handler to the model using a callback function.

        Parameters
        ----------
        callback : callable
            The callback function to be called when an event occurs.
            The callback function should have the following signature:
            callback(model, event)
        events : list of SCIP_EVENTTYPE
            List of event types to attach the event handler to.
        name : str, optional
            Name of the event handler. If not provided, a unique default name will be generated.
        description : str, optional
            Description of the event handler. If not provided, an empty string will be used.
        """
    @override
    def __hash__(self) -> int: ...
    @staticmethod
    def from_ptr(capsule: CapsuleType, take_ownership: bool) -> Model:
        """
        Create a Model from a given pointer.

        Parameters
        ----------
        capsule
            The PyCapsule containing the SCIP pointer under the name "scip"
        take_ownership : bool
            Whether the newly created Model assumes ownership of the
            underlying Scip pointer (see ``_freescip``)

        Returns
        -------
        Model

        """
    def to_ptr(self, give_ownership: bool) -> CapsuleType:
        """
        Return the underlying Scip pointer to the current Model.

        Parameters
        ----------
        give_ownership : bool
            Whether the current Model gives away ownership of the
            underlying Scip pointer (see ``_freescip``)

        Returns
        -------
        capsule
            The underlying pointer to the current Model, wrapped in a
            PyCapsule under the name "scip".

        """
    def includeDefaultPlugins(self) -> None:
        """Includes all default plug-ins into SCIP."""
    def createProbBasic(self, problemName: str = "model") -> None:
        """
        Create new problem instance with given name.

        Parameters
        ----------
        problemName : str, optional
            name of model or problem (Default value = 'model')

        """
    def freeProb(self) -> None:
        """Frees problem and solution process data."""
    def freeTransform(self) -> None:
        """Frees all solution process data including presolving and
        transformed problem, only original problem is kept."""
    def version(self) -> float:
        """
        Retrieve SCIP version.

        Returns
        -------
        float

        """
    def printVersion(self) -> None:
        """Print version, copyright information and compile mode."""
    def printExternalCodeVersions(self) -> None:
        """Print external code versions, e.g. symmetry, non-linear solver, lp solver."""
    def getProbName(self) -> str:
        """
        Retrieve problem name.

        Returns
        -------
        str

        """
    def getTotalTime(self) -> float:
        """
        Retrieve the current total SCIP time in seconds,
        i.e. the total time since the SCIP instance has been created.

        Returns
        -------
        float

        """
    def getSolvingTime(self) -> float:
        """
        Retrieve the current solving time in seconds.

        Returns
        -------
        float

        """
    def getReadingTime(self) -> float:
        """
        Retrieve the current reading time in seconds.

        Returns
        -------
        float

        """
    def getPresolvingTime(self) -> float:
        """
        Returns the current presolving time in seconds.

        Returns
        -------
        float

        """
    def getNLPIterations(self) -> int:
        """
        Returns the total number of LP iterations so far.

        Returns
        -------
        int

        """
    def getNNodes(self) -> int:
        """
        Gets number of processed nodes in current run, including the focus node.

        Returns
        -------
        int

        """
    def getNTotalNodes(self) -> int:
        """
        Gets number of processed nodes in all runs, including the focus node.

        Returns
        -------
        int

        """
    def getNFeasibleLeaves(self) -> int:
        """
        Retrieve number of leaf nodes processed with feasible relaxation solution.

        Returns
        -------
        int

        """
    def getNInfeasibleLeaves(self) -> int:
        """
        Gets number of infeasible leaf nodes processed.

        Returns
        -------
        int

        """
    def getNLeaves(self) -> int:
        """
        Gets number of leaves in the tree.

        Returns
        -------
        int

        """
    def getNChildren(self) -> int:
        """
        Gets number of children of focus node.

        Returns
        -------
        int

        """
    def getNSiblings(self) -> int:
        """
        Gets number of siblings of focus node.

        Returns
        -------
        int

        """
    def getCurrentNode(self) -> Node:
        """
        Retrieve current node.

        Returns
        -------
        Node

        """
    def getGap(self) -> float:
        """
        Retrieve the gap,
        i.e. abs((primalbound - dualbound)/min(abs(primalbound),abs(dualbound)))

        Returns
        -------
        float

        """
    def getDepth(self) -> int:
        """
        Retrieve the depth of the current node.

        Returns
        -------
        int

        """
    def infinity(self) -> float:
        """
        Retrieve SCIP's infinity value.

        Returns
        -------
        int

        """
    def epsilon(self) -> float:
        """
        Retrieve epsilon for e.g. equality checks.

        Returns
        -------
        float

        """
    def feastol(self) -> float:
        """
        Retrieve feasibility tolerance.

        Returns
        -------
        float

        """
    def feasFrac(self, value: float) -> float:
        """
        Returns fractional part of value, i.e. x - floor(x) in feasible tolerance: x - floor(x+feastol).

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
    def frac(self, value: float) -> float:
        """
        Returns fractional part of value, i.e. x - floor(x) in epsilon tolerance: x - floor(x+eps).

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
    def feasFloor(self, value: float) -> float:
        """
        Rounds value + feasibility tolerance down to the next integer.

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
    def feasCeil(self, value: float) -> float:
        """
        Rounds value - feasibility tolerance up to the next integer.

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
    def feasRound(self, value: float) -> float:
        """
        Rounds value to the nearest integer in feasibility tolerance.

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
    def isZero(self, value: float) -> bool:
        """
        Returns whether abs(value) < eps.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
    def isFeasZero(self, value: float) -> bool:
        """
        Returns whether abs(value) < feastol.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
    def isInfinity(self, value: float) -> bool:
        """
        Returns whether value is SCIP's infinity.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
    def isFeasNegative(self, value: float) -> bool:
        """
        Returns whether value < -feastol.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
    def isFeasIntegral(self, value: float) -> bool:
        """
        Returns whether value is integral within the LP feasibility bounds.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
    def isEQ(self, val1: float, val2: float) -> bool:
        """
        Checks, if values are in range of epsilon.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
    def isFeasEQ(self, val1: float, val2: float) -> bool:
        """
        Checks, if relative difference of values is in range of feasibility tolerance.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
    def isLE(self, val1: float, val2: float) -> bool:
        """
        Returns whether val1 <= val2 + eps.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
    def isLT(self, val1: float, val2: float) -> bool:
        """
        Returns whether val1 < val2 - eps.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
    def isGE(self, val1: float, val2: float) -> bool:
        """
        Returns whether val1 >= val2 - eps.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
    def isGT(self, val1: float, val2: float) -> bool:
        """
        Returns whether val1 > val2 + eps.

        Parameters
        ----------
        val1 : float
        val2 : foat

        Returns
        -------
        bool

        """
    def getCondition(self, exact: bool = False) -> float:
        """
        Get the current LP's condition number.

        Parameters
        ----------
        exact : bool, optional
            whether to get an estimate or the exact value (Default value = False)

        Returns
        -------
        float

        """
    def enableReoptimization(self, enable: bool = True) -> None:
        """
        Include specific heuristics and branching rules for reoptimization.

        Parameters
        ----------
        enable : bool, optional
            True to enable and False to disable

        """
    def lpiGetIterations(self) -> int:
        """
        Get the iteration count of the last solved LP.

        Returns
        -------
        int

        """
    def setMinimize(self) -> None:
        """Set the objective sense to minimization."""
    def setMaximize(self) -> None:
        """Set the objective sense to maximization."""
    def setObjlimit(self, objlimit: float) -> None:
        """
        Set a limit on the objective function.
        Only solutions with objective value better than this limit are accepted.

        Parameters
        ----------
        objlimit : float
            limit on the objective function

        """
    def getObjlimit(self) -> float:
        """
        Returns current limit on objective function.

        Returns
        -------
        float

        """
    def setObjective(
        self,
        expr: Expr | SupportsFloat,
        sense: L["minimize", "maximize"] = "minimize",
        clear: bool | L["true"] = "true",  # TODO: typo?
    ) -> None:
        """
        Establish the objective function as a linear expression.

        Parameters
        ----------
        expr : Expr or float
            the objective function SCIP Expr, or constant value
        sense : str, optional
            the objective sense ("minimize" or "maximize") (Default value = 'minimize')
        clear : bool, optional
            set all other variables objective coefficient to zero (Default value = 'true')

        """
    def getObjective(self) -> Expr:
        """
        Retrieve objective function as Expr.

        Returns
        -------
        Expr

        """
    def addObjoffset(self, offset: float, solutions: bool = False) -> None:
        """
        Add constant offset to objective.

        Parameters
        ----------
        offset : float
            offset to add
        solutions : bool, optional
            add offset also to existing solutions (Default value = False)

        """
    def getObjoffset(self, original: bool = True) -> float:
        """
        Retrieve constant objective offset

        Parameters
        ----------
        original : bool, optional
            offset of original or transformed problem (Default value = True)

        Returns
        -------
        float

        """
    def setObjIntegral(self) -> None:
        """Informs SCIP that the objective value is always integral in every feasible solution.

        Notes
        -----
        This function should be used to inform SCIP that the objective function is integral,
        helping to improve the performance. This is useful when using column generation.
        If no column generation (pricing) is used, SCIP automatically detects whether the objective
        function is integral or can be scaled to be integral. However, in any case, the user has to
        make sure that no variable is added during the solving process that destroys this property.
        """
    def getLocalEstimate(self, original: bool = False) -> float:
        """
        Gets estimate of best primal solution w.r.t. original or transformed problem contained in current subtree.

        Parameters
        ----------
        original : bool, optional
            get estimate of original or transformed problem (Default value = False)

        Returns
        -------
        float

        """
    def setPresolve(self, setting: PY_SCIP_PARAMSETTING) -> None:
        """
        Set presolving parameter settings.


        Parameters
        ----------
        setting : SCIP_PARAMSETTING
            the parameter settings, e.g. SCIP_PARAMSETTING.OFF

        """
    def setProbName(self, name: str) -> None:
        """
        Set problem name.

        Parameters
        ----------
        name : str

        """
    def setSeparating(self, setting: PY_SCIP_PARAMSETTING) -> None:
        """
        Set separating parameter settings.

        Parameters
        ----------
        setting : SCIP_PARAMSETTING
            the parameter settings, e.g. SCIP_PARAMSETTING.OFF

        """
    def setHeuristics(self, setting: PY_SCIP_PARAMSETTING) -> None:
        """
        Set heuristics parameter settings.

        Parameters
        ----------
        setting : SCIP_PARAMSETTING
            the parameter settings, e.g. SCIP_PARAMSETTING.OFF

        """
    def setHeurTiming(self, heurname: str, heurtiming: PY_SCIP_HEURTIMING) -> None:
        """
        Set the timing of a heuristic

        Parameters
        ----------
        heurname : string, name of the heuristic
        heurtiming : PY_SCIP_HEURTIMING
                   positions in the node solving loop where heuristic should be executed
        """
    def getHeurTiming(self, heurname: str) -> PY_SCIP_HEURTIMING:
        """
        Get the timing of a heuristic

        Parameters
        ----------
        heurname : string, name of the heuristic

        Returns
        -------
        PY_SCIP_HEURTIMING
                   positions in the node solving loop where heuristic should be executed
        """
    def disablePropagation(self, onlyroot: bool = False) -> None:
        """
        Disables propagation in SCIP to avoid modifying the original problem during transformation.

        Parameters
        ----------
        onlyroot : bool, optional
            use propagation when root processing is finished (Default value = False)

        """
    def printProblem(
        self, ext: str = ".cip", trans: bool = False, genericnames: bool = False
    ) -> None:
        """
        Write current model/problem to standard output.

        Parameters
        ----------
        ext   : str, optional
            the extension to be used (Default value = '.cip').
            Should have an extension corresponding to one of the readable file formats,
            described in https://www.scipopt.org/doc/html/group__FILEREADERS.php.
        trans : bool, optional
            indicates whether the transformed problem is written to file (Default value = False)
        genericnames : bool, optional
            indicates whether the problem should be written with generic variable
            and constraint names (Default value = False)
        """
    def writeProblem(
        self,
        filename: str | os.PathLike[str] = "model.cip",
        trans: bool = False,
        genericnames: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Write current model/problem to a file.

        Parameters
        ----------
        filename : str, optional
            the name of the file to be used (Default value = 'model.cip').
            Should have an extension corresponding to one of the readable file formats,
            described in https://www.scipopt.org/doc/html/group__FILEREADERS.php.
        trans : bool, optional
            indicates whether the transformed problem is written to file (Default value = False)
        genericnames : bool, optional
            indicates whether the problem should be written with generic variable
            and constraint names (Default value = False)
        verbose : bool, optional
            indicates whether a success message should be printed

        """
    def addVar(
        self,
        /,
        name: str = "",
        vtype: _VTypes = "C",
        lb: float | None = 0.0,
        ub: float | None = None,
        obj: float | None = 0.0,
        pricedVar: bool = False,
        pricedVarScore: float = 1.0,
    ) -> Variable:
        """
        Create a new variable. Default variable is non-negative and continuous.

        Parameters
        ----------
        name : str, optional
            name of the variable, generic if empty (Default value = '')
        vtype : str, optional
            type of the variable: 'C' continuous, 'I' integer, 'B' binary, and 'M' implicit integer
            (Default value = 'C')
        lb : float or None, optional
            lower bound of the variable, use None for -infinity (Default value = 0.0)
        ub : float or None, optional
            upper bound of the variable, use None for +infinity (Default value = None)
        obj : float, optional
            objective value of variable (Default value = 0.0)
        pricedVar : bool, optional
            is the variable a pricing candidate? (Default value = False)
        pricedVarScore : float, optional
            score of variable in case it is priced, the higher the better (Default value = 1.0)

        Returns
        -------
        Variable

        """
    def getTransformedVar(self, var: Variable) -> Variable:
        """
        Retrieve the transformed variable.

        Parameters
        ----------
        var : Variable
            original variable to get the transformed of

        Returns
        -------
        Variable

        """
    def addVarLocks(self, var: Variable, nlocksdown: int, nlocksup: int) -> None:
        """
        Adds given values to lock numbers of variable for rounding.

        Parameters
        ----------
        var : Variable
            variable to adjust the locks for
        nlocksdown : int
            new number of down locks
        nlocksup : int
            new number of up locks

        """
    def fixVar(self, var: Variable, val: float) -> tuple[bool, bool]:
        """
        Fixes the variable var to the value val if possible.

        Parameters
        ----------
        var : Variable
            variable to fix
        val : float
            the fix value

        Returns
        -------
        infeasible : bool
            Is the fixing infeasible?
        fixed : bool
            Was the fixing performed?

        """
    def delVar(self, var: Variable) -> bool:
        """
        Delete a variable.

        Parameters
        ----------
        var : Variable
            the variable which shall be deleted

        Returns
        -------
        deleted : bool
            Whether deleting was successfull

        """
    def tightenVarLb(
        self, var: Variable, lb: float, force: bool = False
    ) -> tuple[bool, bool]:
        """
        Tighten the lower bound in preprocessing or current node, if the bound is tighter.

        Parameters
        ----------
        var : Variable
            SCIP variable
        lb : float
            possible new lower bound
        force : bool, optional
            force tightening even if below bound strengthening tolerance (default = False)

        Returns
        -------
        infeasible : bool
            Whether new domain is empty
        tightened : bool
            Whether the bound was tightened

        """
    def tightenVarUb(
        self, var: Variable, ub: float, force: bool = False
    ) -> tuple[bool, bool]:
        """
        Tighten the upper bound in preprocessing or current node, if the bound is tighter.

        Parameters
        ----------
        var : Variable
            SCIP variable
        ub : float
            possible new upper bound
        force : bool, optional
            force tightening even if below bound strengthening tolerance

        Returns
        -------
        infeasible : bool
            Whether new domain is empty
        tightened : bool
            Whether the bound was tightened

        """
    def tightenVarUbGlobal(
        self, var: Variable, ub: float, force: bool = False
    ) -> tuple[bool, bool]:
        """
        Tighten the global upper bound, if the bound is tighter.

        Parameters
        ----------
        var : Variable
            SCIP variable
        ub : float
            possible new upper bound
        force : bool, optional
            force tightening even if below bound strengthening tolerance

        Returns
        -------
        infeasible : bool
            Whether new domain is empty
        tightened : bool
            Whether the bound was tightened

        """
    def tightenVarLbGlobal(
        self, var: Variable, lb: float, force: bool = False
    ) -> tuple[bool, bool]:
        """Tighten the global lower bound, if the bound is tighter.

        Parameters
        ----------
        var : Variable
            SCIP variable
        lb : float
            possible new lower bound
        force : bool, optional
            force tightening even if below bound strengthening tolerance

        Returns
        -------
        infeasible : bool
            Whether new domain is empty
        tightened : bool
            Whether the bound was tightened

        """
    def chgVarLb(self, var: Variable, lb: float | None) -> None:
        """
        Changes the lower bound of the specified variable.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new lower bound (set to None for -infinity)

        """
    def chgVarUb(self, var: Variable, ub: float | None) -> None:
        """Changes the upper bound of the specified variable.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new upper bound (set to None for +infinity)

        """
    def chgVarLbGlobal(self, var: Variable, lb: float | None) -> None:
        """Changes the global lower bound of the specified variable.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new lower bound (set to None for -infinity)

        """
    def chgVarUbGlobal(self, var: Variable, ub: float | None) -> None:
        """Changes the global upper bound of the specified variable.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new upper bound (set to None for +infinity)

        """
    def chgVarLbNode(self, node: Node, var: Variable, lb: float | None) -> None:
        """Changes the lower bound of the specified variable at the given node.

        Parameters
        ----------
        node : Node
            Node at which the variable bound will be changed
        var : Variable
            variable to change bound of
        lb : float or None
            new lower bound (set to None for -infinity)

        """
    def chgVarUbNode(self, node: Node, var: Variable, ub: float | None) -> None:
        """Changes the upper bound of the specified variable at the given node.

        Parameters
        ----------
        node : Node
            Node at which the variable bound will be changed
        var : Variable
            variable to change bound of
        lb : float or None
            new upper bound (set to None for +infinity)

        """
    def chgVarType(self, var: Variable, vtype: _VTypes) -> None:
        """
        Changes the type of a variable.

        Parameters
        ----------
        var : Variable
            variable to change type of
        vtype : str
            new variable type. 'C' or "CONTINUOUS", 'I' or "INTEGER",
            'B' or "BINARY", and 'M' "IMPLINT".

        """
    def getVars(self, transformed: bool = False) -> list[Variable]:
        """
        Retrieve all variables.

        Parameters
        ----------
        transformed : bool, optional
            Get transformed variables instead of original (Default value = False)

        Returns
        -------
        list of Variable

        """
    def getNVars(self, transformed: bool = True) -> int:
        """
        Retrieve number of variables in the problems.

        Parameters
        ----------
        transformed : bool, optional
            Get transformed variables instead of original (Default value = True)

        Returns
        -------
        int

        """
    def getNIntVars(self) -> int:
        """
        Gets number of integer active problem variables.

        Returns
        -------
        int

        """
    def getNBinVars(self) -> int:
        """
        Gets number of binary active problem variables.

        Returns
        -------
        int

        """
    def getNImplVars(self) -> int:
        """
        Gets number of implicit integer active problem variables.

        Returns
        -------
        int

        """
    def getNContVars(self) -> int:
        """
        Gets number of continuous active problem variables.

        Returns
        -------
        int

        """
    def getVarDict(self, transformed: bool = False) -> dict[str, float]:
        """
        Gets dictionary with variables names as keys and current variable values as items.

        Parameters
        ----------
        transformed : bool, optional
            Get transformed variables instead of original (Default value = False)

        Returns
        -------
        dict of str to float

        """
    def updateNodeLowerbound(self, node: Node, lb: float) -> None:
        """
        If given value is larger than the node's lower bound (in transformed problem),
        sets the node's lower bound to the new value.

        Parameters
        ----------
        node : Node
            the node to update
        lb : float
            new bound (if greater) for the node

        """
    def relax(self) -> None:
        """Relaxes the integrality restrictions of the model."""
    def getBestChild(self) -> Node | None:
        """
        Gets the best child of the focus node w.r.t. the node selection strategy.

        Returns
        -------
        Node

        """
    def getBestSibling(self) -> Node | None:
        """
        Gets the best sibling of the focus node w.r.t. the node selection strategy.

        Returns
        -------
        Node

        """
    def getPrioChild(self) -> Node | None:
        """
        Gets the best child of the focus node w.r.t. the node selection priority
        assigned by the branching rule.

        Returns
        -------
        Node

        """
    def getPrioSibling(self) -> Node | None:
        """Gets the best sibling of the focus node w.r.t.
        the node selection priority assigned by the branching rule.

        Returns
        -------
        Node

        """
    def getBestLeaf(self) -> Node | None:
        """Gets the best leaf from the node queue w.r.t. the node selection strategy.

        Returns
        -------
        Node

        """
    def getBestNode(self) -> Node | None:
        """Gets the best node from the tree (child, sibling, or leaf) w.r.t. the node selection strategy.

        Returns
        -------
        Node

        """
    def getBestboundNode(self) -> Node | None:
        """Gets the node with smallest lower bound from the tree (child, sibling, or leaf).

        Returns
        -------
        Node

        """
    def getOpenNodes(self) -> tuple[list[Node], list[Node], list[Node]]:
        """
        Access to all data of open nodes (leaves, children, and siblings).

        Returns
        -------
        leaves : list of Node
            list of all open leaf nodes
        children : list of Node
            list of all open children nodes
        siblings : list of Node
            list of all open sibling nodes

        """
    def repropagateNode(self, node: Node) -> None:
        """Marks the given node to be propagated again the next time a node of its subtree is processed."""
    def getLPSolstat(self) -> PY_SCIP_LPSOLSTAT:
        """
        Gets solution status of current LP.

        Returns
        -------
        SCIP_LPSOLSTAT

        """
    def constructLP(self) -> bool:
        """
        Makes sure that the LP of the current node is loaded and
        may be accessed through the LP information methods.


        Returns
        -------
        cutoff : bool
            Can the node be cutoff?

        """
    def getLPObjVal(self) -> float:
        """
        Gets objective value of current LP (which is the sum of column and loose objective value).

        Returns
        -------
        float

        """
    def getLPColsData(self) -> list[Column]:
        """
        Retrieve current LP columns.

        Returns
        -------
        list of Column

        """
    def getLPRowsData(self) -> list[Row]:
        """
        Retrieve current LP rows.

        Returns
        -------
        list of Row

        """
    def getNLPRows(self) -> int:
        """
        Retrieve the number of rows currently in the LP.

        Returns
        -------
        int

        """
    def getNLPCols(self) -> int:
        """
        Retrieve the number of columns currently in the LP.

        Returns
        -------
        int

        """
    def getLPBasisInd(self) -> list[int]:
        """
        Gets all indices of basic columns and rows:
        index i >= 0 corresponds to column i, index i < 0 to row -i-1

        Returns
        -------
        list of int

        """
    def getLPBInvRow(self, row: int) -> list[float]:
        """
        Gets a row from the inverse basis matrix B^-1

        Parameters
        ----------
        row : int
            The row index of the inverse basis matrix

        Returns
        -------
        list of float

        """
    def getLPBInvARow(self, row: int) -> list[float]:
        """
        Gets a row from B^-1 * A.

        Parameters
        ----------
        row : int
            The row index of the inverse basis matrix multiplied by the coefficient matrix

        Returns
        -------
        list of float

        """
    def isLPSolBasic(self) -> bool:
        """
        Returns whether the current LP solution is basic, i.e. is defined by a valid simplex basis.

        Returns
        -------
        bool

        """
    def allColsInLP(self) -> bool:
        """
        Checks if all columns, i.e. every variable with non-empty column is present in the LP.
        This is not True when performing pricing for instance.

        Returns
        -------
        bool

        """
    def getColRedCost(self, col: Column) -> float:
        """
        Gets the reduced cost of the column in the current LP.

        Parameters
        ----------
        col : Column

        Returns
        -------
        float

        """
    def createEmptyRowSepa(
        self,
        sepa: Sepa,
        name: str = "row",
        lhs: float | None = 0.0,
        rhs: float | None = None,
        local: bool = True,
        modifiable: bool = False,
        removable: bool = True,
    ) -> Row:
        """
        Creates and captures an LP row without any coefficients from a separator.

        Parameters
        ----------
        sepa : Sepa
            separator that creates the row
        name : str, optional
            name of row (Default value = "row")
        lhs : float or None, optional
            left hand side of row (Default value = 0)
        rhs : float or None, optional
            right hand side of row (Default value = None)
        local : bool, optional
            is row only valid locally? (Default value = True)
        modifiable : bool, optional
            is row modifiable during node processing (subject to column generation)? (Default value = False)
        removable : bool, optional
            should the row be removed from the LP due to aging or cleanup? (Default value = True)

        Returns
        -------
        Row

        """
    def createEmptyRowUnspec(
        self,
        name: str = "row",
        lhs: float | None = 0.0,
        rhs: float | None = None,
        local: bool = True,
        modifiable: bool = False,
        removable: bool = True,
    ) -> Row:
        """
        Creates and captures an LP row without any coefficients from an unspecified source.

        Parameters
        ----------
        name : str, optional
            name of row (Default value = "row")
        lhs : float or None, optional
            left hand side of row (Default value = 0)
        rhs : float or None, optional
            right hand side of row (Default value = None)
        local : bool, optional
            is row only valid locally? (Default value = True)
        modifiable : bool, optional
            is row modifiable during node processing (subject to column generation)? (Default value = False)
        removable : bool, optional
            should the row be removed from the LP due to aging or cleanup? (Default value = True)

        Returns
        -------
        Row

        """
    def getRowActivity(self, row: Row) -> float:
        """
        Returns the activity of a row in the last LP or pseudo solution.

        Parameters
        ----------
        row : Row

        Returns
        -------
        float

        """
    def getRowLPActivity(self, row: Row) -> float:
        """
        Returns the activity of a row in the last LP solution.

        Parameters
        ----------
        row : Row

        Returns
        -------
        float

        """
    def releaseRow(self, row: Row) -> None:
        """
        Decreases usage counter of LP row, and frees memory if necessary.

        Parameters
        ----------
        row : Row

        """
    def cacheRowExtensions(self, row: Row) -> None:
        """
        Informs row that all subsequent additions of variables to the row
        should be cached and not directly applied;
        after all additions were applied, flushRowExtensions() must be called;
        while the caching of row extensions is activated, information methods of the
        row give invalid results; caching should be used, if a row is build with addVarToRow()
        calls variable by variable to increase the performance.

        Parameters
        ----------
        row : Row

        """
    def flushRowExtensions(self, row: Row) -> None:
        """
        Flushes all cached row extensions after a call of cacheRowExtensions()
        and merges coefficients with equal columns into a single coefficient

        Parameters
        ----------
        row : Row

        """
    def addVarToRow(self, row: Row, var: Variable, value: float) -> None:
        """
        Resolves variable to columns and adds them with the coefficient to the row.

        Parameters
        ----------
        row : Row
            Row in which the variable will be added
        var : Variable
            Variable which will be added to the row
        value : float
            Coefficient on the variable when placed in the row

        """
    def printRow(self, row: Row) -> None:
        """
        Prints row.

        Parameters
        ----------
        row : Row

        """
    def getRowNumIntCols(self, row: Row) -> int:
        """
        Returns number of intergal columns in the row.

        Parameters
        ----------
        row : Row

        Returns
        -------
        int

        """
    def getRowObjParallelism(self, row: Row) -> float:
        """
        Returns 1 if the row is parallel, and 0 if orthogonal.

        Parameters
        ----------
        row : Row

        Returns
        -------
        float

        """
    def getRowParallelism(
        self, row1: Row, row2: Row, orthofunc: L["d", "e", 100, 101] = 101
    ) -> float:
        """
        Returns the degree of parallelism between hyplerplanes. 1 if perfectly parallel, 0 if orthogonal.
        For two row vectors v, w the parallelism is calculated as: abs(v*w)/(abs(v)*abs(w)).
        101 in this case is an 'e' (euclidean) in ASCII. The other acceptable input is 100 (d for discrete).

        Parameters
        ----------
        row1 : Row
        row2 : Row
        orthofunc : int, optional
            101 (default) is an 'e' (euclidean) in ASCII. Alternate value is 100 (d for discrete)

        Returns
        -------
        float

        """
    def getRowDualSol(self, row: Row) -> float:
        """
        Gets the dual LP solution of a row.

        Parameters
        ----------
        row : Row

        Returns
        -------
        float

        """
    def addPoolCut(self, row: Row) -> None:
        """
        If not already existing, adds row to global cut pool.

        Parameters
        ----------
        row : Row

        """
    def getCutEfficacy(self, cut: Row, sol: Solution | None = None) -> float:
        """
        Returns efficacy of the cut with respect to the given primal solution or the
        current LP solution: e = -feasibility/norm

        Parameters
        ----------
        cut : Row
        sol : Solution or None, optional

        Returns
        -------
        float

        """
    def isCutEfficacious(self, cut: Row, sol: Solution | None = None) -> bool:
        """
        Returns whether the cut's efficacy with respect to the given primal solution or the
        current LP solution is greater than the minimal cut efficacy.

        Parameters
        ----------
        cut : Row
        sol : Solution or None, optional

        Returns
        -------
        float

        """
    def getCutLPSolCutoffDistance(self, cut: Row, sol: Solution) -> float:
        """
        Returns row's cutoff distance in the direction of the given primal solution.

        Parameters
        ----------
        cut : Row
        sol : Solution

        Returns
        -------
        float

        """
    def addCut(self, cut: Row, forcecut: bool = False) -> bool:
        """
        Adds cut to separation storage and returns whether cut has been detected to be infeasible for local bounds.

        Parameters
        ----------
        cut : Row
            The cut that will be added
        forcecut : bool, optional
            Whether the cut should be forced or not, i.e., selected no matter what

        Returns
        -------
        infeasible : bool
            Whether the cut has been detected to be infeasible from local bounds

        """
    def getNCuts(self) -> int:
        """
        Retrieve total number of cuts in storage.

        Returns
        -------
        int

        """
    def getNCutsApplied(self) -> int:
        """
        Retrieve number of currently applied cuts.

        Returns
        -------
        int

        """
    def getNSepaRounds(self) -> int:
        """
        Retrieve the number of separation rounds that have been performed
        at the current node.

        Returns
        -------
        int

        """
    def separateSol(
        self,
        sol: Solution | None = None,
        pretendroot: bool = False,
        allowlocal: bool = True,
        onlydelayed: bool = False,
    ) -> tuple[bool, bool]:
        """
        Separates the given primal solution or the current LP solution by calling
        the separators and constraint handlers' separation methods;
        the generated cuts are stored in the separation storage and can be accessed
        with the methods SCIPgetCuts() and SCIPgetNCuts();
        after evaluating the cuts, you have to call SCIPclearCuts() in order to remove the cuts from the
        separation storage; it is possible to call SCIPseparateSol() multiple times with
        different solutions and evaluate the found cuts afterwards.

        Parameters
        ----------
        sol : Solution or None, optional
            solution to separate, None to use current lp solution (Default value = None)
        pretendroot : bool, optional
            should the cut separators be called as if we are at the root node? (Default value = "False")
        allowlocal : bool, optional
            should the separator be asked to separate local cuts (Default value = True)
        onlydelayed : bool, optional
            should only separators be called that were delayed in the previous round? (Default value = False)

        Returns
        -------
        delayed : bool
            whether a separator was delayed
        cutoff : bool
            whether the node can be cut off

        """
    def createConsFromExpr(
        self,
        cons: ExprCons,
        name: str = "",
        initial: bool = True,
        separate: bool = True,
        enforce: bool = True,
        check: bool = True,
        propagate: bool = True,
        local: bool = False,
        modifiable: bool = False,
        dynamic: bool = False,
        removable: bool = False,
        stickingatnode: bool = False,
    ) -> Constraint:
        """
        Create a linear or nonlinear constraint without adding it to the SCIP problem.
        This is useful for creating disjunction constraints without also enforcing the individual constituents.
        Currently, this can only be used as an argument to `.addConsElemDisjunction`. To add
        an individual linear/nonlinear constraint, prefer `.addCons()`.

        Parameters
        ----------
        cons : ExprCons
            The expression constraint that is not yet an actual constraint
        name : str, optional
            the name of the constraint, generic name if empty (Default value = '')
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        modifiable : bool, optional
            is the constraint modifiable (subject to column generation)? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be  moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The created Constraint object.

        """
    def addCons(
        self,
        cons: ExprCons,
        name: str = "",
        initial: bool = True,
        separate: bool = True,
        enforce: bool = True,
        check: bool = True,
        propagate: bool = True,
        local: bool = False,
        modifiable: bool = False,
        dynamic: bool = False,
        removable: bool = False,
        stickingatnode: bool = False,
    ) -> Constraint:
        """
        Add a linear or nonlinear constraint.

        Parameters
        ----------
        cons : ExprCons
            The expression constraint that is not yet an actual constraint
        name : str, optional
            the name of the constraint, generic name if empty (Default value = "")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        modifiable : bool, optional
            is the constraint modifiable (subject to column generation)? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraints always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The created and added Constraint object.

        """
    def addConss(
        self,
        conss: Iterable[ExprCons],
        name: str | Iterable[str] = "",
        initial: bool | Iterable[bool] = True,
        separate: bool | Iterable[bool] = True,
        enforce: bool | Iterable[bool] = True,
        check: bool | Iterable[bool] = True,
        propagate: bool | Iterable[bool] = True,
        local: bool | Iterable[bool] = False,
        modifiable: bool | Iterable[bool] = False,
        dynamic: bool | Iterable[bool] = False,
        removable: bool | Iterable[bool] = False,
        stickingatnode: bool | Iterable[bool] = False,
    ) -> list[Constraint]:
        """Adds multiple linear or quadratic constraints.

        Each of the constraints is added to the model using Model.addCons().

        For all parameters, except `conss`, this method behaves differently depending on the
        type of the passed argument:
        1. If the value is iterable, it must be of the same length as `conss`. For each
        constraint, Model.addCons() will be called with the value at the corresponding index.
        2. Else, the (default) value will be applied to all of the constraints.

        Parameters
        ----------
        conss : iterable of ExprCons
            An iterable of constraint objects. Any iterable will be converted into a list before further processing.
        name : str or iterable of str, optional
            the name of the constraint, generic name if empty (Default value = '')
        initial : bool or iterable of bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool or iterable of bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool or iterable of bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool or iterable of bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool or iterable of bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool or iterable of bool, optional
            is the constraint only valid locally? (Default value = False)
        modifiable : bool or iterable of bool, optional
            is the constraint modifiable (subject to column generation)? (Default value = False)
        dynamic : bool or iterable of bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool or iterable of bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool or iterable of bool, optional
            should the constraints always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        list of Constraint
            The created and added Constraint objects.

        """
    def addConsDisjunction(
        self,
        conss: Iterable[ExprCons],
        name: str = "",
        initial: bool = True,
        relaxcons: Constraint | None = None,
        enforce: bool = True,
        check: bool = True,
        local: bool = False,
        modifiable: bool = False,
        dynamic: bool = False,
    ) -> Constraint:
        """
        Add a disjunction constraint.

        Parameters
        ----------
        conss : iterable of ExprCons
            An iterable of constraint objects to be included initially in the disjunction.
            Currently, these must be expressions.
        name : str, optional
            the name of the disjunction constraint.
        initial : bool, optional
            should the LP relaxation of disjunction constraint be in the initial LP? (Default value = True)
        relaxcons : None, optional
            a conjunction constraint containing the linear relaxation of the disjunction constraint, or None.
            NOT YET SUPPORTED. (Default value = None)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        modifiable : bool, optional
            is the constraint modifiable (subject to column generation)? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)

        Returns
        -------
        Constraint
            The created disjunction constraint

        """
    def addConsElemDisjunction(
        self, disj_cons: Constraint, cons: Constraint
    ) -> Constraint:
        """
        Appends a constraint to a disjunction.

        Parameters
        ----------
        disj_cons : Constraint
             the disjunction constraint to append to.
        cons : Constraint
            the constraint to append

        Returns
        -------
        disj_cons : Constraint
            The disjunction constraint with `cons` appended.

        """
    def getConsNVars(self, constraint: Constraint) -> int:
        """
        Gets number of variables in a constraint.

        Parameters
        ----------
        constraint : Constraint
            Constraint to get the number of variables from.

        Returns
        -------
        int

        Raises
        ------
        TypeError
            If the associated constraint handler does not have this functionality

        """
    def getConsVars(self, constraint: Constraint) -> list[Variable]:
        """
        Gets variables in a constraint.

        Parameters
        ----------
        constraint : Constraint
            Constraint to get the variables from.

        Returns
        -------
        list of Variable

        """
    def printCons(self, constraint: Constraint) -> None:
        """
        Print the constraint

        Parameters
        ----------
        constraint : Constraint

        """
    def addExprNonlinear(
        self, cons: Constraint, expr: Expr | GenExpr[Any], coef: float
    ) -> None:
        """
        Add coef*expr to nonlinear constraint.

        Parameters
        ----------
        cons : Constraint
        expr : Expr or GenExpr
        coef : float

        """
    def addConsCoeff(self, cons: Constraint, var: Variable, coeff: float) -> None:
        """
        Add coefficient to the linear constraint (if non-zero).

        Parameters
        ----------
        cons : Constraint
            Constraint to be changed
        var : Variable
            variable to be added
        coeff : float
            coefficient of new variable

        """
    def addConsNode(
        self, node: Node, cons: Constraint, validnode: Node | None = None
    ) -> None:
        """
        Add a constraint to the given node.

        Parameters
        ----------
        node : Node
            node at which the constraint will be added
        cons : Constraint
            the constraint to add to the node
        validnode : Node or None, optional
            more global node where cons is also valid. (Default=None)

        """
    def addConsLocal(self, cons: Constraint, validnode: Node | None = None) -> None:
        """
        Add a constraint to the current node.

        Parameters
        ----------
        cons : Constraint
            the constraint to add to the current node
        validnode : Node or None, optional
            more global node where cons is also valid. (Default=None)

        """
    def addConsSOS1(
        self,
        vars: Sequence[Variable],
        weights: Sequence[float] | None = None,
        name: str = "SOS1cons",
        initial: bool = True,
        separate: bool = True,
        enforce: bool = True,
        check: bool = True,
        propagate: bool = True,
        local: bool = False,
        dynamic: bool = False,
        removable: bool = False,
        stickingatnode: bool = False,
    ) -> Constraint:
        """
        Add an SOS1 constraint.

        :param vars: list of variables to be included
        :param weights: list of weights (Default value = None)
        :param name: name of the constraint (Default value = "SOS1cons")
        :param initial: should the LP relaxation of constraint be in the initial LP? (Default value = True)
        :param separate: should the constraint be separated during LP processing? (Default value = True)
        :param enforce: should the constraint be enforced during node processing? (Default value = True)
        :param check: should the constraint be checked for feasibility? (Default value = True)
        :param propagate: should the constraint be propagated during node processing? (Default value = True)
        :param local: is the constraint only valid locally? (Default value = False)
        :param dynamic: is the constraint subject to aging? (Default value = False)
        :param removable: should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        :param stickingatnode: should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (Default value = False)


        Parameters
        ----------
        vars : list of Variable
            list of variables to be included
        weights : list of float or None, optional
            list of weights (Default value = None)
        name : str, optional
            name of the constraint (Default value = "SOS1cons")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created SOS1 constraint

        """
    def addConsSOS2(
        self,
        vars: Sequence[Variable],
        weights: Sequence[float] | None = None,
        name: str = "SOS2cons",
        initial: bool = True,
        separate: bool = True,
        enforce: bool = True,
        check: bool = True,
        propagate: bool = True,
        local: bool = False,
        dynamic: bool = False,
        removable: bool = False,
        stickingatnode: bool = False,
    ) -> Constraint:
        """
        Add an SOS2 constraint.

        Parameters
        ----------
        vars : list of Variable
            list of variables to be included
        weights : list of float or None, optional
            list of weights (Default value = None)
        name : str, optional
            name of the constraint (Default value = "SOS2cons")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created SOS2 constraint

        """
    def addConsAnd(
        self,
        vars: Sequence[Variable],
        resvar: Variable,
        name: str = "ANDcons",
        initial: bool = True,
        separate: bool = True,
        enforce: bool = True,
        check: bool = True,
        propagate: bool = True,
        local: bool = False,
        modifiable: bool = False,
        dynamic: bool = False,
        removable: bool = False,
        stickingatnode: bool = False,
    ) -> Constraint:
        """
        Add an AND-constraint.

        Parameters
        ----------
        vars : list of Variable
            list of BINARY variables to be included (operators)
        resvar : Variable
            BINARY variable (resultant)
        name : str, optional
            name of the constraint (Default value = "ANDcons")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created AND constraint

        """
    def addConsOr(
        self,
        vars: Sequence[Variable],
        resvar: Variable,
        name: str = "ORcons",
        initial: bool = True,
        separate: bool = True,
        enforce: bool = True,
        check: bool = True,
        propagate: bool = True,
        local: bool = False,
        modifiable: bool = False,
        dynamic: bool = False,
        removable: bool = False,
        stickingatnode: bool = False,
    ) -> Constraint:
        """
        Add an OR-constraint.

        Parameters
        ----------
        vars : list of Variable
            list of BINARY variables to be included (operators)
        resvar : Variable
            BINARY variable (resultant)
        name : str, optional
            name of the constraint (Default value = "ORcons")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created OR constraint

        """
    def addConsXor(
        self,
        vars: Sequence[Variable],
        rhsvar: bool,
        name: str = "XORcons",
        initial: bool = True,
        separate: bool = True,
        enforce: bool = True,
        check: bool = True,
        propagate: bool = True,
        local: bool = False,
        modifiable: bool = False,
        dynamic: bool = False,
        removable: bool = False,
        stickingatnode: bool = False,
    ) -> Constraint:
        """
        Add a XOR-constraint.

        Parameters
        ----------
        vars : list of Variable
            list of binary variables to be included (operators)
        rhsvar : bool
            BOOLEAN value, explicit True, False or bool(obj) is needed (right-hand side)
        name : str, optional
            name of the constraint (Default value = "XORcons")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created XOR constraint

        """
    def addConsCardinality(
        self,
        consvars: Sequence[Variable],
        cardval: int,
        indvars: Sequence[Variable] | None = None,
        weights: Sequence[float] | None = None,
        name: str = "CardinalityCons",
        initial: bool = True,
        separate: bool = True,
        enforce: bool = True,
        check: bool = True,
        propagate: bool = True,
        local: bool = False,
        dynamic: bool = False,
        removable: bool = False,
        stickingatnode: bool = False,
    ) -> Constraint:
        """
        Add a cardinality constraint that allows at most 'cardval' many nonzero variables.

        Parameters
        ----------
        consvars : list of Variable
            list of variables to be included
        cardval : int
            nonnegative integer
        indvars : list of Variable or None, optional
            indicator variables indicating which variables may be treated as nonzero in
            cardinality constraint, or None if new indicator variables should be
            introduced automatically (Default value = None)
        weights : list of float or None, optional
            weights determining the variable order, or None if variables should be ordered
            in the same way they were added to the constraint (Default value = None)
        name : str, optional
            name of the constraint (Default value = "CardinalityCons")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created Cardinality constraint

        """
    def addConsIndicator(
        self,
        cons: ExprCons,
        binvar: Variable | None = None,
        activeone: bool = True,
        name: str = "",
        initial: bool = True,
        separate: bool = True,
        enforce: bool = True,
        check: bool = True,
        propagate: bool = True,
        local: bool = False,
        dynamic: bool = False,
        removable: bool = False,
        stickingatnode: bool = False,
    ) -> Constraint:
        """Add an indicator constraint for the linear inequality `cons`.

        The `binvar` argument models the redundancy of the linear constraint. A solution for which
        `binvar` is 1 must satisfy the constraint.

        Parameters
        ----------
        cons : ExprCons
            a linear inequality of the form "<="
        binvar : Variable, optional
            binary indicator variable, or None if it should be created (Default value = None)
        activeone : bool, optional
            constraint should active if binvar is 1 (0 if activeone = False)
        name : str, optional
            name of the constraint (Default value = "")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created Indicator constraint

        """
    def getSlackVarIndicator(self, cons: Constraint) -> Variable:
        """
        Get slack variable of an indicator constraint.


        Parameters
        ----------
        cons : Constraint
            The indicator constraint

        Returns
        -------
        Variable

        """
    def addPyCons(self, cons: Constraint) -> None:
        """
        Adds a customly created cons.

        Parameters
        ----------
        cons : Constraint
            constraint to add

        """
    def addVarSOS1(self, cons: Constraint, var: Variable, weight: float) -> None:
        """
        Add variable to SOS1 constraint.

        Parameters
        ----------
        cons : Constraint
            SOS1 constraint
        var : Variable
            new variable
        weight : weight
            weight of new variable

        """
    def appendVarSOS1(self, cons: Constraint, var: Variable) -> None:
        """
        Append variable to SOS1 constraint.

        Parameters
        ----------
        cons : Constraint
            SOS1 constraint
        var : Variable
            variable to append

        """
    def addVarSOS2(self, cons: Constraint, var: Variable, weight: float) -> None:
        """
        Add variable to SOS2 constraint.

        Parameters
        ----------
        cons : Constraint
            SOS2 constraint
        var : Variable
            new variable
        weight : weight
            weight of new variable

        """
    def appendVarSOS2(self, cons: Constraint, var: Variable) -> None:
        """
        Append variable to SOS2 constraint.

        Parameters
        ----------
        cons : Constraint
            SOS2 constraint
        var : Variable
            variable to append

        """
    def setInitial(self, cons: Constraint, newInit: bool) -> None:
        """
        Set "initial" flag of a constraint.

        Parameters
        ----------
        cons : Constraint
        newInit : bool

        """
    def setRemovable(self, cons: Constraint, newRem: bool) -> None:
        """
        Set "removable" flag of a constraint.

        Parameters
        ----------
        cons : Constraint
        newRem : bool

        """
    def setEnforced(self, cons: Constraint, newEnf: bool) -> None:
        """
        Set "enforced" flag of a constraint.

        Parameters
        ----------
        cons : Constraint
        newEnf : bool

        """
    def setCheck(self, cons: Constraint, newCheck: bool) -> None:
        """
        Set "check" flag of a constraint.

        Parameters
        ----------
        cons : Constraint
        newCheck : bool

        """
    def chgRhs(self, cons: Constraint, rhs: float | None) -> None:
        """
        Change right-hand side value of a constraint.

        Parameters
        ----------
        cons : Constraint
            linear or quadratic constraint
        rhs : float or None
            new right-hand side (set to None for +infinity)

        """
    def chgLhs(self, cons: Constraint, lhs: float | None) -> None:
        """
        Change left-hand side value of a constraint.

        Parameters
        ----------
        cons : Constraint
            linear or quadratic constraint
        lhs : float or None
            new left-hand side (set to None for -infinity)

        """
    def getRhs(self, cons: Constraint) -> float:
        """
        Retrieve right-hand side value of a constraint.

        Parameters
        ----------
        cons : Constraint
            linear or quadratic constraint

        Returns
        -------
        float

        """
    def getLhs(self, cons: Constraint) -> None:
        """
        Retrieve left-hand side value of a constraint.

        Parameters
        ----------
        cons : Constraint
            linear or quadratic constraint

        Returns
        -------
        float

        """
    def chgCoefLinear(self, cons: Constraint, var: Variable, value: float) -> None:
        """
        Changes coefficient of variable in linear constraint;
        deletes the variable if coefficient is zero; adds variable if not yet contained in the constraint
        This method may only be called during problem creation stage for an original constraint and variable.
        This method requires linear time to search for occurences of the variable in the constraint data.

        Parameters
        ----------
        cons : Constraint
            linear constraint
        var : Variable
            variable of constraint entry
        value : float
            new coefficient of constraint entry

        """
    def delCoefLinear(self, cons: Constraint, var: Variable) -> None:
        """
        Deletes variable from linear constraint
        This method may only be called during problem creation stage for an original constraint and variable.
        This method requires linear time to search for occurrences of the variable in the constraint data.

        Parameters
        ----------
        cons : Constraint
            linear constraint
        var : Variable
            variable of constraint entry

        """
    def addCoefLinear(self, cons: Constraint, var: Variable, value: float) -> None:
        """
        Adds coefficient to linear constraint (if it is not zero)

        Parameters
        ----------
        cons : Constraint
            linear constraint
        var : Variable
            variable of constraint entry
        value : float
            coefficient of constraint entry

        """
    def getActivity(self, cons: Constraint, sol: Solution | None = None) -> float:
        """
        Retrieve activity of given constraint.
        Can only be called after solving is completed.

        Parameters
        ----------
        cons : Constraint
            linear or quadratic constraint
        sol : Solution or None, optional
            solution to compute activity of, None to use current node's solution (Default value = None)

        Returns
        -------
        float

        """
    def getSlack(
        self,
        cons: Constraint,
        sol: Solution | None = None,
        side: L["lhs", "rhs"] | None = None,
    ) -> float:
        """
        Retrieve slack of given constraint.
        Can only be called after solving is completed.

        Parameters
        ----------
        cons : Constraint
            linear or quadratic constraint
        sol : Solution or None, optional
            solution to compute slack of, None to use current node's solution (Default value = None)
        side : str or None, optional
            whether to use 'lhs' or 'rhs' for ranged constraints, None to return minimum (Default value = None)

        Returns
        -------
        float

        """
    def getTransformedCons(self, cons: Constraint) -> Constraint:
        """
        Retrieve transformed constraint.

        Parameters
        ----------
        cons : Constraint

        Returns
        -------
        Constraint

        """
    def isNLPConstructed(self) -> bool:
        """
        Returns whether SCIP's internal NLP has been constructed.

        Returns
        -------
        bool

        """
    def getNNlRows(self) -> int:
        """
        Gets current number of nonlinear rows in SCIP's internal NLP.

        Returns
        -------
        int

        """
    def getNlRows(self) -> list[NLRow]:
        """
        Returns a list with the nonlinear rows in SCIP's internal NLP.

        Returns
        -------
        list of NLRow

        """
    def getNlRowSolActivity(self, nlrow: NLRow, sol: Solution | None = None) -> float:
        """
        Gives the activity of a nonlinear row for a given primal solution.

        Parameters
        ----------
        nlrow : NLRow
        sol : Solution or None, optional
            a primal solution, if None, then the current LP solution is used

        Returns
        -------
        float

        """
    def getNlRowSolFeasibility(
        self, nlrow: NLRow, sol: Solution | None = None
    ) -> float:
        """
        Gives the feasibility of a nonlinear row for a given primal solution

        Parameters
        ----------
        nlrow : NLRow
        sol : Solution or None, optional
            a primal solution, if None, then the current LP solution is used

        Returns
        -------
        bool

        """
    def getNlRowActivityBounds(self, nlrow: NLRow) -> tuple[float, float]:
        """
        Gives the minimal and maximal activity of a nonlinear row w.r.t. the variable's bounds.

        Parameters
        ----------
        nlrow : NLRow

        Returns
        -------
        tuple of float

        """
    def printNlRow(self, nlrow: NLRow) -> None:
        """
        Prints nonlinear row.

        Parameters
        ----------
        nlrow : NLRow

        """
    def checkQuadraticNonlinear(self, cons: Constraint) -> bool:
        """
        Returns if the given constraint is quadratic.

        Parameters
        ----------
        cons : Constraint

        Returns
        -------
        bool

        """
    def getTermsQuadratic(
        self, cons: Constraint
    ) -> tuple[
        list[tuple[Variable, Variable, float]],
        list[tuple[Variable, float, float]],
        list[tuple[Variable, float]],
    ]:
        """
        Retrieve bilinear, quadratic, and linear terms of a quadratic constraint.

        Parameters
        ----------
        cons : Constraint

        Returns
        -------
        bilinterms : list of tuple
        quadterms : list of tuple
        linterms : list of tuple

        """
    def setRelaxSolVal(self, var: Variable, val: float) -> None:
        """
        Sets the value of the given variable in the global relaxation solution.

        Parameters
        ----------
        var : Variable
        val : float

        """
    def getConss(self, transformed: bool = True) -> list[Constraint]:
        """
        Retrieve all constraints.

        Parameters
        ----------
        transformed : bool, optional
            get transformed variables instead of original (Default value = True)

        Returns
        -------
        list of Constraint

        """
    def getNConss(self, transformed: bool = True) -> int:
        """
        Retrieve number of all constraints.

        Parameters
        ----------
        transformed : bool, optional
            get number of transformed variables instead of original (Default value = True)

        Returns
        -------
        int

        """
    def delCons(self, cons: Constraint) -> None:
        """
        Delete constraint from the model

        Parameters
        ----------
        cons : Constraint
            constraint to be deleted

        """
    def delConsLocal(self, cons: Constraint) -> None:
        """
        Delete constraint from the current node and its children.

        Parameters
        ----------
        cons : Constraint
            constraint to be deleted

        """
    def getValsLinear(self, cons: Constraint) -> dict[str, float]:
        """
        Retrieve the coefficients of a linear constraint

        Parameters
        ----------
        cons : Constraint
            linear constraint to get the coefficients of

        Returns
        -------
        dict of str to float

        """
    def getRowLinear(self, cons: Constraint) -> Row:
        """
        Retrieve the linear relaxation of the given linear constraint as a row.
        may return NULL if no LP row was yet created; the user must not modify the row!

        Parameters
        ----------
        cons : Constraint
            linear constraint to get the coefficients of

        Returns
        -------
        Row

        """
    def getDualsolLinear(self, cons: Constraint) -> float:
        """
        Retrieve the dual solution to a linear constraint.

        Parameters
        ----------
        cons : Constraint
            linear constraint

        Returns
        -------
        float

        """
    @deprecated(
        "model.getDualMultiplier(cons) is deprecated: please use model.getDualsolLinear(cons)"
    )
    def getDualMultiplier(self, cons: Constraint) -> float:
        """
        DEPRECATED: Retrieve the dual solution to a linear constraint.

        Parameters
        ----------
        cons : Constraint
            linear constraint

        Returns
        -------
        float

        """
    def getDualfarkasLinear(self, cons: Constraint) -> float:
        """
        Retrieve the dual farkas value to a linear constraint.

        Parameters
        ----------
        cons : Constraint
            linear constraint

        Returns
        -------
        float

        """
    def getVarRedcost(self, var: Variable) -> float:
        """
        Retrieve the reduced cost of a variable.

        Parameters
        ----------
        var : Variable
            variable to get the reduced cost of

        Returns
        -------
        float

        """
    def getDualSolVal(self, cons: Constraint, boundconstraint: bool = False) -> float:
        """
        Returns dual solution value of a constraint.

        Parameters
        ----------
        cons : Constraint
            constraint to get the dual solution value of
        boundconstraint : bool, optional
            Decides whether to store a bool if the constraint is a bound constraint
            (default = False)

        Returns
        -------
        float

        """
    def optimize(self) -> None:
        """Optimize the problem."""
    def optimizeNogil(self) -> None:
        """Optimize the problem without GIL."""
    def solveConcurrent(self) -> None:
        """Transforms, presolves, and solves problem using additional solvers which emphasize on
        finding solutions.
        WARNING: This feature is still experimental and prone to some errors."""
    def presolve(self) -> None:
        """Presolve the problem."""
    def initBendersDefault(self, subproblems: Model | dict[Any, Model]) -> None:
        """
        Initialises the default Benders' decomposition with a dictionary of subproblems.

        Parameters
        ----------
        subproblems : Model or dict of object to Model
            a single Model instance or dictionary of Model instances

        """
    def computeBestSolSubproblems(self) -> None:
        """Solves the subproblems with the best solution to the master problem.
        Afterwards, the best solution from each subproblem can be queried to get
        the solution to the original problem.
        If the user wants to resolve the subproblems, they must free them by
        calling freeBendersSubproblems()
        """
    def freeBendersSubproblems(self) -> None:
        """Calls the free subproblem function for the Benders' decomposition.
        This will free all subproblems for all decompositions."""
    def updateBendersLowerbounds(
        self, lowerbounds: dict[int, float], benders: Benders | None = None
    ) -> None:
        """
        Updates the subproblem lower bounds for benders using
        the lowerbounds dict. If benders is None, then the default
        Benders' decomposition is updated.

        Parameters
        ----------
        lowerbounds : dict of int to float
        benders : Benders or None, optional

        """
    def activateBenders(self, benders: Benders, nsubproblems: int) -> None:
        """
        Activates the Benders' decomposition plugin with the input name.

        Parameters
        ----------
        benders : Benders
            the Benders' decomposition to which the subproblem belongs to
        nsubproblems : int
            the number of subproblems in the Benders' decomposition

        """
    def addBendersSubproblem(self, benders: Benders, subproblem: Model) -> None:
        """
        Adds a subproblem to the Benders' decomposition given by the input
        name.

        Parameters
        ----------
        benders : Benders
            the Benders' decomposition to which the subproblem belongs to
        subproblem : Model
            the subproblem to add to the decomposition

        """
    def setBendersSubproblemIsConvex(
        self, benders: Benders, probnumber: int, isconvex: bool = True
    ) -> None:
        """
        Sets a flag indicating whether the subproblem is convex.

        Parameters
        ----------
        benders : Benders
            the Benders' decomposition which contains the subproblem
        probnumber : int
            the problem number of the subproblem that the convexity will be set for
        isconvex : bool, optional
            flag to indicate whether the subproblem is convex (default=True)

        """
    def setupBendersSubproblem(
        self,
        probnumber: int,
        benders: Benders | None = None,
        solution: Solution | None = None,
        checktype: PY_SCIP_BENDERSENFOTYPE = PY_SCIP_BENDERSENFOTYPE.LP,
    ) -> None:
        """
        Sets up the Benders' subproblem given the master problem solution.

        Parameters
        ----------
        probnumber : int
            the index of the problem that is to be set up
        benders : Benders or None, optional
            the Benders' decomposition to which the subproblem belongs to
        solution : Solution or None, optional
            the master problem solution that is used for the set up, if None, then the LP solution is used
        checktype : PY_SCIP_BENDERSENFOTYPE
            the type of solution check that prompted the solving of the Benders' subproblems, either
            PY_SCIP_BENDERSENFOTYPE: LP, RELAX, PSEUDO or CHECK. Default is LP.

        """
    def solveBendersSubproblem(
        self,
        probnumber: int,
        solvecip: bool,
        benders: Benders | None = None,
        solution: Solution | None = None,
    ) -> tuple[bool, float | None]:
        """
        Solves the Benders' decomposition subproblem. The convex relaxation will be solved unless
        the parameter solvecip is set to True.

        Parameters
        ----------
        probnumber : int
            the index of the problem that is to be set up
        solvecip : bool
            whether the CIP of the subproblem should be solved. If False, then only the convex relaxation is solved.
        benders : Benders or None, optional
            the Benders' decomposition to which the subproblem belongs to
        solution : Solution or None, optional
            the master problem solution that is used for the set up, if None, then the LP solution is used

        Returns
        -------
        infeasible : bool
            returns whether the current subproblem is infeasible
        objective : float or None
            the objective function value of the subproblem, can be None

        """
    def getBendersSubproblem(
        self, probnumber: int, benders: Benders | None = None
    ) -> Model:
        """
        Returns a Model object that wraps around the SCIP instance of the subproblem.
        NOTE: This Model object is just a place holder and SCIP instance will not be
        freed when the object is destroyed.

        Parameters
        ----------
        probnumber : int
            the problem number for subproblem that is required
        benders : Benders or None, optional
            the Benders' decomposition object that the subproblem belongs to (Default = None)

        Returns
        -------
        Model

        """
    def getBendersVar(
        self, var: Variable, benders: Benders | None = None, probnumber: int = -1
    ) -> Variable | None:
        """
        Returns the variable for the subproblem or master problem
        depending on the input probnumber.

        Parameters
        ----------
        var : Variable
            the source variable for which the target variable is requested
        benders : Benders or None, optional
            the Benders' decomposition to which the subproblem variables belong to
        probnumber : int, optional
            the problem number for which the target variable belongs, -1 for master problem

        Returns
        -------
        Variable or None

        """
    def getBendersAuxiliaryVar(
        self, probnumber: int, benders: Benders | None = None
    ) -> Variable:
        """
        Returns the auxiliary variable that is associated with the input problem number

        Parameters
        ----------
        probnumber : int
            the problem number for which the target variable belongs, -1 for master problem
        benders : Benders or None, optional
            the Benders' decomposition to which the subproblem variables belong to

        Returns
        -------
        Variable

        """
    def checkBendersSubproblemOptimality(
        self, solution: Solution, probnumber: int, benders: Benders | None = None
    ) -> bool:
        """
        Returns whether the subproblem is optimal w.r.t the master problem auxiliary variables.

        Parameters
        ----------
        solution : Solution
            the master problem solution that is being checked for optimamlity
        probnumber : int
            the problem number for which optimality is being checked
        benders : Benders or None, optional
            the Benders' decomposition to which the subproblem belongs to

        Returns
        -------
        optimal : bool
            flag to indicate whether the current subproblem is optimal for the master

        """
    def includeBendersDefaultCuts(self, benders: Benders) -> None:
        """
        Includes the default Benders' decomposition cuts to the custom Benders' decomposition plugin.

        Parameters
        ----------
        benders : Benders
            the Benders' decomposition that the default cuts will be applied to

        """
    def includeEventhdlr(self, eventhdlr: Eventhdlr, name: str, desc: str) -> None:
        """
        Include an event handler.

        Parameters
        ----------
        eventhdlr : Eventhdlr
            event handler
        name : str
            name of event handler
        desc : str
            description of event handler

        """
    def includePricer(
        self,
        pricer: Pricer,
        name: str,
        desc: str,
        priority: int = 1,
        delay: bool = True,
    ) -> None:
        """
        Include a pricer.

        Parameters
        ----------
        pricer : Pricer
            pricer
        name : str
            name of pricer
        desc : str
            description of pricer
        priority : int, optional
            priority of pricer (Default value = 1)
        delay : bool, optional
            should the pricer be delayed until no other pricers or already existing problem variables
            with negative reduced costs are found? (Default value = True)

        """
    def includeConshdlr(
        self,
        conshdlr: Conshdlr,
        name: str,
        desc: str,
        sepapriority: int = 0,
        enfopriority: int = 0,
        chckpriority: int = 0,
        sepafreq: int = -1,
        propfreq: int = -1,
        eagerfreq: int = 100,
        maxprerounds: int = -1,
        delaysepa: bool = False,
        delayprop: bool = False,
        needscons: bool = True,
        proptiming: PY_SCIP_PROPTIMING = PY_SCIP_PROPTIMING.BEFORELP,
        presoltiming: PY_SCIP_PRESOLTIMING = PY_SCIP_PRESOLTIMING.MEDIUM,
    ) -> None:
        """
        Include a constraint handler.

        Parameters
        ----------
        conshdlr : Conshdlr
            constraint handler
        name : str
            name of constraint handler
        desc : str
            description of constraint handler
        sepapriority : int, optional
            priority for separation (Default value = 0)
        enfopriority : int, optional
            priority for constraint enforcing (Default value = 0)
        chckpriority : int, optional
            priority for checking feasibility (Default value = 0)
        sepafreq : int, optional
            frequency for separating cuts; 0 = only at root node (Default value = -1)
        propfreq : int, optional
            frequency for propagating domains; 0 = only preprocessing propagation (Default value = -1)
        eagerfreq : int, optional
            frequency for using all instead of only the useful constraints in separation,
             propagation and enforcement; -1 = no eager evaluations, 0 = first only
             (Default value = 100)
        maxprerounds : int, optional
            maximal number of presolving rounds the constraint handler participates in (Default value = -1)
        delaysepa : bool, optional
            should separation method be delayed, if other separators found cuts? (Default value = False)
        delayprop : bool, optional
            should propagation method be delayed, if other propagators found reductions? (Default value = False)
        needscons : bool, optional
            should the constraint handler be skipped, if no constraints are available? (Default value = True)
        proptiming : PY_SCIP_PROPTIMING
            positions in the node solving loop where propagation method of constraint handlers
             should be executed (Default value = SCIP_PROPTIMING.BEFORELP)
        presoltiming : PY_SCIP_PRESOLTIMING
            timing mask of the constraint handler's presolving method (Default value = SCIP_PRESOLTIMING.MEDIUM)

        """
    def copyLargeNeighborhoodSearch(
        self, to_fix: Sequence[Variable], fix_vals: Sequence[float]
    ) -> Model:
        """
        Creates a configured copy of the transformed problem and applies provided fixings intended for LNS heuristics.

        Parameters
        ----------
        to_fix : List[Variable]
            A list of variables to fix in the copy
        fix_vals : List[Real]
            A list of the values to which to fix the variables in the copy (care their order)

        Returns
        -------
        model : Model
            A model containing the created copy
        """
    def translateSubSol(self, sub_model: Model, sol: Solution, heur: Heur) -> Solution:
        """
        Translates a solution of a model copy into a solution of the main model

        Parameters
        ----------
        sub_model : Model
                The python-wrapper of the subscip
        sol : Solution
                The python-wrapper of the solution of the subscip
        heur : Heur
                The python-wrapper of the heuristic that found the solution

        Returns
        -------
        solution : Solution
                The corresponding solution in the main model
        """
    def createCons(
        self,
        conshdlr: Conshdlr,
        name: str,
        initial: bool = True,
        separate: bool = True,
        enforce: bool = True,
        check: bool = True,
        propagate: bool = True,
        local: bool = False,
        modifiable: bool = False,
        dynamic: bool = False,
        removable: bool = False,
        stickingatnode: bool = False,
    ) -> Constraint:
        """
        Create a constraint of a custom constraint handler.

        Parameters
        ----------
        conshdlr : Conshdlr
            constraint handler
        name : str
            name of constraint handler
        initial : bool, optional
            (Default value = True)
        separate : bool, optional
            (Default value = True)
        enforce : bool, optional
            (Default value = True)
        check : bool, optional
            (Default value = True)
        propagate : bool, optional
            (Default value = True)
        local : bool, optional
            (Default value = False)
        modifiable : bool, optional
            (Default value = False)
        dynamic : bool, optional
            (Default value = False)
        removable : bool, optional
            (Default value = False)
        stickingatnode : bool, optional
            (Default value = False)

        Returns
        -------
        Constraint

        """
    def includePresol(
        self,
        presol: Presol,
        name: str,
        desc: str,
        priority: int,
        maxrounds: int,
        timing: PY_SCIP_PRESOLTIMING = PY_SCIP_PRESOLTIMING.FAST,
    ) -> None:
        """
        Include a presolver.

        Parameters
        ----------
        presol : Presol
            presolver
        name : str
            name of presolver
        desc : str
            description of presolver
        priority : int
            priority of the presolver (>= 0: before, < 0: after constraint handlers)
        maxrounds : int
            maximal number of presolving rounds the presolver participates in (-1: no limit)
        timing : PY_SCIP_PRESOLTIMING, optional
             timing mask of presolver (Default value = SCIP_PRESOLTIMING_FAST)

        """
    def includeSepa(
        self,
        sepa: Sepa,
        name: str,
        desc: str,
        priority: int = 0,
        freq: int = 10,
        maxbounddist: float = 1.0,
        usessubscip: bool = False,
        delay: bool = False,
    ) -> None:
        """
        Include a separator

        :param Sepa sepa: separator
        :param name: name of separator
        :param desc: description of separator
        :param priority: priority of separator (>= 0: before, < 0: after constraint handlers)
        :param freq: frequency for calling separator
        :param maxbounddist: maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separation
        :param usessubscip: does the separator use a secondary SCIP instance? (Default value = False)
        :param delay: should separator be delayed, if other separators found cuts? (Default value = False)


        Parameters
        ----------
        sepa : Sepa
            separator
        name : str
            name of separator
        desc : str
            description of separator
        priority : int, optional
            priority of separator (>= 0: before, < 0: after constraint handlers) (default=0)
        freq : int, optional
            frequency for calling separator (default=10)
        maxbounddist : float, optional
            maximal relative distance from current node's dual bound to primal
            bound compared to best node's dual bound for applying separation.
            (default = 1.0)
        usessubscip : bool, optional
            does the separator use a secondary SCIP instance? (Default value = False)
        delay : bool, optional
            should separator be delayed if other separators found cuts? (Default value = False)

        """
    def includeReader(self, reader: Reader, name: str, desc: str, ext: str) -> None:
        """
        Include a reader.

        Parameters
        ----------
        reader : Reader
            reader
        name : str
            name of reader
        desc : str
            description of reader
        ext : str
            file extension of reader

        """
    def includeProp(
        self,
        prop: Prop,
        name: str,
        desc: str,
        presolpriority: int,
        presolmaxrounds: int,
        proptiming: PY_SCIP_PROPTIMING,
        presoltiming: PY_SCIP_PRESOLTIMING = PY_SCIP_PRESOLTIMING.FAST,
        priority: int = 1,
        freq: int = 1,
        delay: bool = True,
    ) -> None:
        """
        Include a propagator.

        Parameters
        ----------
        prop : Prop
            propagator
        name : str
            name of propagator
        desc : str
            description of propagator
        presolpriority : int
            presolving priority of the propgator (>= 0: before, < 0: after constraint handlers)
        presolmaxrounds : int
            maximal number of presolving rounds the propagator participates in (-1: no limit)
        proptiming : SCIP_PROPTIMING
            positions in the node solving loop where propagation method of constraint handlers should be executed
        presoltiming : PY_SCIP_PRESOLTIMING, optional
            timing mask of the constraint handler's presolving method (Default value = SCIP_PRESOLTIMING_FAST)
        priority : int, optional
            priority of the propagator (Default value = 1)
        freq : int, optional
            frequency for calling propagator (Default value = 1)
        delay : bool, optional
            should propagator be delayed if other propagators have found reductions? (Default value = True)

        """
    def includeHeur(
        self,
        heur: Heur,
        name: str,
        desc: str,
        dispchar: str,
        priority: int = 10000,
        freq: int = 1,
        freqofs: int = 0,
        maxdepth: int = -1,
        timingmask: PY_SCIP_HEURTIMING = PY_SCIP_HEURTIMING.BEFORENODE,
        usessubscip: bool = False,
    ) -> None:
        """
        Include a primal heuristic.

        Parameters
        ----------
        heur : Heur
            heuristic
        name : str
            name of heuristic
        desc : str
            description of heuristic
        dispchar : str
            display character of heuristic. Please use a single length string.
        priority : int. optional
            priority of the heuristic (Default value = 10000)
        freq : int, optional
            frequency for calling heuristic (Default value = 1)
        freqofs : int. optional
            frequency offset for calling heuristic (Default value = 0)
        maxdepth : int, optional
            maximal depth level to call heuristic at (Default value = -1)
        timingmask : PY_SCIP_HEURTIMING, optional
            positions in the node solving loop where heuristic should be executed
            (Default value = SCIP_HEURTIMING_BEFORENODE)
        usessubscip : bool, optional
            does the heuristic use a secondary SCIP instance? (Default value = False)

        """
    def includeRelax(
        self, relax: Relax, name: str, desc: str, priority: int = 10000, freq: int = 1
    ) -> None:
        """
        Include a relaxation handler.

        Parameters
        ----------
        relax : Relax
            relaxation handler
        name : str
            name of relaxation handler
        desc : str
            description of relaxation handler
        priority : int, optional
            priority of the relaxation handler (negative: after LP, non-negative: before LP, Default value = 10000)
        freq : int, optional
            frequency for calling relaxation handler

        """
    def includeCutsel(
        self, cutsel: Cutsel, name: str, desc: str, priority: int
    ) -> None:
        """
        Include a cut selector.

        Parameters
        ----------
        cutsel : Cutsel
            cut selector
        name : str
            name of cut selector
        desc : str
            description of cut selector
        priority : int
            priority of the cut selector

        """
    def includeBranchrule(
        self,
        branchrule: Branchrule,
        name: str,
        desc: str,
        priority: int,
        maxdepth: int,
        maxbounddist: float,
    ) -> None:
        """
        Include a branching rule.

        Parameters
        ----------
        branchrule : Branchrule
            branching rule
        name : str
            name of branching rule
        desc : str
            description of branching rule
        priority : int
            priority of branching rule
        maxdepth : int
            maximal depth level up to which this branching rule should be used (or -1)
        maxbounddist : float
            maximal relative distance from current node's dual bound to primal bound
            compared to best node's dual bound for applying branching rule
            (0.0: only on current best node, 1.0: on all nodes)

        """
    def includeNodesel(
        self,
        nodesel: Nodesel,
        name: str,
        desc: str,
        stdpriority: int,
        memsavepriority: int,
    ) -> None:
        """
        Include a node selector.

        Parameters
        ----------
        nodesel : Nodesel
            node selector
        name : str
            name of node selector
        desc : str
            description of node selector
        stdpriority : int
            priority of the node selector in standard mode
        memsavepriority : int
            priority of the node selector in memory saving mode

        """
    def includeBenders(
        self,
        benders: Benders,
        name: str,
        desc: str,
        priority: int = 1,
        cutlp: bool = True,
        cutpseudo: bool = True,
        cutrelax: bool = True,
        shareaux: bool = False,
    ) -> None:
        """
        Include a Benders' decomposition.

        Parameters
        ----------
        benders : Benders
            the Benders decomposition
        name : str
            the name
        desc : str
            the description
        priority : int, optional
            priority of the Benders' decomposition
        cutlp : bool, optional
            should Benders' cuts be generated from LP solutions
        cutpseudo : bool, optional
            should Benders' cuts be generated from pseudo solutions
        cutrelax : bool, optional
            should Benders' cuts be generated from relaxation solutions
        shareaux : bool, optional
            should the Benders' decomposition share the auxiliary variables of the
            highest priority Benders' decomposition

        """
    def includeBenderscut(
        self,
        benders: Benders,
        benderscut: Benderscut,
        name: str,
        desc: str,
        priority: int = 1,
        islpcut: bool = True,
    ) -> None:
        """
        Include a Benders' decomposition cutting method

        Parameters
        ----------
        benders : Benders
            the Benders' decomposition that this cutting method is attached to
        benderscut : Benderscut
            the Benders' decomposition cutting method
        name : str
            the name
        desc : str
            the description
        priority : int. optional
            priority of the Benders' decomposition (Default = 1)
        islpcut : bool, optional
            is this cutting method suitable for generating cuts for convex relaxations?
            (Default = True)

        """
    def getLPBranchCands(
        self,
    ) -> tuple[list[Variable], list[float], list[float], int, int, int]:
        """
        Gets branching candidates for LP solution branching (fractional variables) along with solution values,
        fractionalities, and number of branching candidates; The number of branching candidates does NOT account
        for fractional implicit integer variables which should not be used for branching decisions. Fractional
        implicit integer variables are stored at the positions nlpcands to nlpcands + nfracimplvars - 1
        branching rules should always select the branching candidate among the first npriolpcands of the candidate list

        Returns
        -------
        list of Variable
            list of variables of LP branching candidates
        list of float
            list of LP candidate solution values
        list of float
            list of LP candidate fractionalities
        int
            number of LP branching candidates
        int
            number of candidates with maximal priority
        int
            number of fractional implicit integer variables

        """
    def getPseudoBranchCands(self) -> tuple[list[Variable], int, int]:
        """
        Gets branching candidates for pseudo solution branching (non-fixed variables)
        along with the number of candidates.

        Returns
        -------
        list of Variable
            list of variables of pseudo branching candidates
        int
            number of pseudo branching candidates
        int
            number of candidates with maximal priority

        """
    def branchVar(self, variable: Variable) -> tuple[Node, Node | None, Node]:
        """
        Branch on a non-continuous variable.

        Parameters
        ----------
        variable : Variable
            Variable to branch on

        Returns
        -------
        Node
            Node created for the down (left) branch
        Node or None
            Node created for the equal child (middle child). Only exists if branch variable is integer
        Node
            Node created for the up (right) branch

        """
    def branchVarVal(
        self, variable: Variable, value: float
    ) -> tuple[Node, Node | None, Node]:
        """
        Branches on variable using a value which separates the domain of the variable.

        Parameters
        ----------
        variable : Variable
            Variable to branch on
        value : float
            value to branch on

        Returns
        -------
        Node
            Node created for the down (left) branch
        Node or None
            Node created for the equal child (middle child). Only exists if the branch variable is integer
        Node
            Node created for the up (right) branch

        """
    def calcNodeselPriority(
        self, variable: Variable, branchdir: PY_SCIP_BRANCHDIR, targetvalue: float
    ) -> int:
        """
        Calculates the node selection priority for moving the given variable's LP value
        to the given target value;
        this node selection priority can be given to the SCIPcreateChild() call.

        Parameters
        ----------
        variable : Variable
            variable on which the branching is applied
        branchdir : PY_SCIP_BRANCHDIR
            type of branching that was performed
        targetvalue : float
            new value of the variable in the child node

        Returns
        -------
        int
            node selection priority for moving the given variable's LP value to the given target value

        """
    def calcChildEstimate(self, variable: Variable, targetvalue: float) -> float:
        """
        Calculates an estimate for the objective of the best feasible solution
        contained in the subtree after applying the given branching;
        this estimate can be given to the SCIPcreateChild() call.

        Parameters
        ----------
        variable : Variable
            Variable to compute the estimate for
        targetvalue : float
            new value of the variable in the child node

        Returns
        -------
        float
            objective estimate of the best solution in the subtree after applying the given branching

        """
    def createChild(self, nodeselprio: int, estimate: float) -> Node:
        """
        Create a child node of the focus node.

        Parameters
        ----------
        nodeselprio : int
            node selection priority of new node
        estimate : float
            estimate for (transformed) objective value of best feasible solution in subtree

        Returns
        -------
        Node
            the child which was created

        """
    def startDive(self) -> None:
        """Initiates LP diving.
        It allows the user to change the LP in several ways, solve, change again, etc,
        without affecting the actual LP. When endDive() is called,
        SCIP will undo all changes done and recover the LP it had before startDive."""
    def endDive(self) -> None:
        """Quits probing and resets bounds and constraints to the focus node's environment."""
    def chgVarObjDive(self, var: Variable, newobj: float) -> None:
        """
        Changes (column) variable's objective value in current dive.

        Parameters
        ----------
        var : Variable
        newobj : float

        """
    def chgVarLbDive(self, var: Variable, newbound: float) -> None:
        """
        Changes variable's current lb in current dive.

        Parameters
        ----------
        var : Variable
        newbound : float

        """
    def chgVarUbDive(self, var: Variable, newbound: float) -> None:
        """
        Changes variable's current ub in current dive.

        Parameters
        ----------
        var : Variable
        newbound : float

        """
    def getVarLbDive(self, var: Variable) -> float:
        """
        Returns variable's current lb in current dive.

        Parameters
        ----------
        var : Variable

        Returns
        -------
        float

        """
    def getVarUbDive(self, var: Variable) -> float:
        """
        Returns variable's current ub in current dive.

        Parameters
        ----------
        var : Variable

        Returns
        -------
        float

        """
    def chgRowLhsDive(self, row: Row, newlhs: float) -> None:
        """
        Changes row lhs in current dive, change will be undone after diving
        ends, for permanent changes use SCIPchgRowLhs().

        Parameters
        ----------
        row : Row
        newlhs : float

        """
    def chgRowRhsDive(self, row: Row, newrhs: float) -> None:
        """
        Changes row rhs in current dive, change will be undone after diving
        ends. For permanent changes use SCIPchgRowRhs().

        Parameters
        ----------
        row : Row
        newrhs : float

        """
    def addRowDive(self, row: Row) -> None:
        """
        Adds a row to the LP in current dive.

        Parameters
        ----------
        row : Row

        """
    def solveDiveLP(self, itlim: int = -1) -> tuple[bool, bool]:
        """
        Solves the LP of the current dive. No separation or pricing is applied.

        Parameters
        ----------
        itlim : int, optional
            maximal number of LP iterations to perform (Default value = -1, that is, no limit)

        Returns
        -------
        lperror : bool
            whether an unresolved lp error occured
        cutoff : bool
            whether the LP was infeasible or the objective limit was reached

        """
    def inRepropagation(self) -> bool:
        """
        Returns if the current node is already solved and only propagated again.

        Returns
        -------
        bool

        """
    def startProbing(self) -> None:
        """Initiates probing, making methods SCIPnewProbingNode(), SCIPbacktrackProbing(), SCIPchgVarLbProbing(),
        SCIPchgVarUbProbing(), SCIPfixVarProbing(), SCIPpropagateProbing(), SCIPsolveProbingLP(), etc available.
        """
    def endProbing(self) -> None:
        """Quits probing and resets bounds and constraints to the focus node's environment."""
    def newProbingNode(self) -> None:
        """Creates a new probing sub node, whose changes can be undone by backtracking to a higher node in the
        probing path with a call to backtrackProbing().
        """
    def backtrackProbing(self, probingdepth: int) -> None:
        """
        Undoes all changes to the problem applied in probing up to the given probing depth.

        Parameters
        ----------
        probingdepth : int
            probing depth of the node in the probing path that should be reactivated

        """
    def getProbingDepth(self) -> int:
        """Returns the current probing depth."""
    def chgVarObjProbing(self, var: Variable, newobj: float) -> None:
        """Changes (column) variable's objective value during probing mode."""
    def chgVarLbProbing(self, var: Variable, lb: float | None) -> None:
        """
        Changes the variable lower bound during probing mode.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new lower bound (set to None for -infinity)

        """
    def chgVarUbProbing(self, var: Variable, ub: float | None) -> None:
        """
        Changes the variable upper bound during probing mode.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        ub : float or None
            new upper bound (set to None for +infinity)

        """
    def fixVarProbing(self, var: Variable, fixedval: float) -> None:
        """
        Fixes a variable at the current probing node.

        Parameters
        ----------
        var : Variable
        fixedval : float

        """
    def isObjChangedProbing(self) -> bool:
        """
        Returns whether the objective function has changed during probing mode.

        Returns
        -------
        bool

        """
    def inProbing(self) -> bool:
        """
        Returns whether we are in probing mode;
        probing mode is activated via startProbing() and stopped via endProbing().

        Returns
        -------
        bool

        """
    def solveProbingLP(self, itlim: int = -1) -> tuple[bool, bool]:
        """
        Solves the LP at the current probing node (cannot be applied at preprocessing stage)
        no separation or pricing is applied.

        Parameters
        ----------
        itlim : int
            maximal number of LP iterations to perform (Default value = -1, that is, no limit)

        Returns
        -------
        lperror : bool
            if an unresolved lp error occured
        cutoff : bool
            whether the LP was infeasible or the objective limit was reached

        """
    def applyCutsProbing(self) -> bool:
        """
        Applies the cuts in the separation storage to the LP and clears the storage afterwards;
        this method can only be applied during probing; the user should resolve the probing LP afterwards
        in order to get a new solution.
        returns:

        Returns
        -------
        cutoff : bool
            whether an empty domain was created

        """
    def propagateProbing(self, maxproprounds: int) -> tuple[bool, int]:
        """
        Applies domain propagation on the probing sub problem, that was changed after SCIPstartProbing() was called;
        the propagated domains of the variables can be accessed with the usual bound accessing calls SCIPvarGetLbLocal()
        and SCIPvarGetUbLocal(); the propagation is only valid locally, i.e. the local bounds as well as the changed
        bounds due to SCIPchgVarLbProbing(), SCIPchgVarUbProbing(), and SCIPfixVarProbing() are used for propagation.

        Parameters
        ----------
        maxproprounds : int
            maximal number of propagation rounds (Default value = -1, that is, no limit)

        Returns
        -------
        cutoff : bool
            whether the probing node can be cutoff
        ndomredsfound : int
            number of domain reductions found

        """
    def interruptSolve(self) -> None:
        """Interrupt the solving process as soon as possible."""
    def restartSolve(self) -> None:
        """Restarts the solving process as soon as possible."""
    def writeLP(self, filename: str | os.PathLike[str] = "LP.lp") -> None:
        """
        Writes current LP to a file.

        Parameters
        ----------
        filename : str, optional
            file name (Default value = "LP.lp")

        """
    def createSol(self, heur: Heur | None = None, initlp: bool = False) -> Solution:
        """
        Create a new primal solution in the transformed space.

        Parameters
        ----------
        heur : Heur or None, optional
            heuristic that found the solution (Default value = None)
        initlp : bool, optional
            Should the created solution be initialised to the current LP solution instead of all zeros

        Returns
        -------
        Solution

        """
    def createPartialSol(self, heur: Heur | None = None) -> Solution:
        """
        Create a partial primal solution, initialized to unknown values.

        Parameters
        ----------
        heur : Heur or None, optional
            heuristic that found the solution (Default value = None)

        Returns
        -------
        Solution

        """
    def createOrigSol(self, heur: Heur | None = None) -> Solution:
        """
        Create a new primal solution in the original space.

        Parameters
        ----------
        heur : Heur or None, optional
            heuristic that found the solution (Default value = None)

        Returns
        -------
        Solution

        """
    def printBestSol(self, write_zeros: bool = False) -> None:
        """
        Prints the best feasible primal solution.

        Parameters
        ----------
        write_zeros : bool, optional
            include variables that are set to zero (Default = False)

        """
    def printSol(
        self, solution: Solution | None = None, write_zeros: bool = False
    ) -> None:
        """
        Print the given primal solution.

        Parameters
        ----------
        solution : Solution or None, optional
            solution to print (default = None)
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """
    def writeBestSol(
        self,
        filename: str | bytes | os.PathLike[AnyStr] = "origprob.sol",
        write_zeros: bool = False,
    ) -> None:
        """
        Write the best feasible primal solution to a file.

        Parameters
        ----------
        filename : str, optional
            name of the output file (Default="origprob.sol")
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """
    def writeBestTransSol(
        self,
        filename: str | bytes | os.PathLike[AnyStr] = "transprob.sol",
        write_zeros: bool = False,
    ) -> None:
        """
        Write the best feasible primal solution for the transformed problem to a file.

        Parameters
        ----------
        filename : str, optional
            name of the output file (Default="transprob.sol")
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """
    def writeSol(
        self,
        solution: Solution,
        filename: str | bytes | os.PathLike[AnyStr] = "origprob.sol",
        write_zeros: bool = False,
    ) -> None:
        """
        Write the given primal solution to a file.

        Parameters
        ----------
        solution : Solution
            solution to write
        filename : str, optional
            name of the output file (Default="origprob.sol")
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """
    def writeTransSol(
        self,
        solution: Solution,
        filename: str | bytes | os.PathLike[AnyStr] = "transprob.sol",
        write_zeros: bool = False,
    ) -> None:
        """
        Write the given transformed primal solution to a file.

        Parameters
        ----------
        solution : Solution
            transformed solution to write
        filename : str, optional
            name of the output file (Default="transprob.sol")
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """
    def readSol(self, filename: str | os.PathLike[str]) -> None:
        """
        Reads a given solution file, problem has to be transformed in advance.

        Parameters
        ----------
        filename : str
            name of the input file

        """
    def readSolFile(self, filename: str | os.PathLike[str]) -> Solution:
        """
        Reads a given solution file.

        Solution is created but not added to storage/the model.
        Use 'addSol' OR 'trySol' to add it.

        Parameters
        ----------
        filename : str
            name of the input file

        Returns
        -------
        Solution

        """
    def setSolVal(self, solution: Solution, var: Variable, val: float) -> None:
        """
        Set a variable in a solution.

        Parameters
        ----------
        solution : Solution
            solution to be modified
        var : Variable
            variable in the solution
        val : float
            value of the specified variable

        """
    def trySol(
        self,
        solution: Solution,
        printreason: bool = True,
        completely: bool = False,
        checkbounds: bool = True,
        checkintegrality: bool = True,
        checklprows: bool = True,
        free: bool = True,
    ) -> bool:
        """
        Check given primal solution for feasibility and try to add it to the storage.

        Parameters
        ----------
        solution : Solution
            solution to store
        printreason : bool, optional
            should all reasons of violations be printed? (Default value = True)
        completely : bool, optional
            should all violation be checked? (Default value = False)
        checkbounds : bool, optional
            should the bounds of the variables be checked? (Default value = True)
        checkintegrality : bool, optional
            does integrality have to be checked? (Default value = True)
        checklprows : bool, optional
            do current LP rows (both local and global) have to be checked? (Default value = True)
        free : bool, optional
            should solution be freed? (Default value = True)

        Returns
        -------
        stored : bool
            whether given solution was feasible and good enough to keep

        """
    def checkSol(
        self,
        solution: Solution,
        printreason: bool = True,
        completely: bool = False,
        checkbounds: bool = True,
        checkintegrality: bool = True,
        checklprows: bool = True,
        original: bool = False,
    ) -> bool:
        """
        Check given primal solution for feasibility without adding it to the storage.

        Parameters
        ----------
        solution : Solution
            solution to store
        printreason : bool, optional
            should all reasons of violations be printed? (Default value = True)
        completely : bool, optional
            should all violation be checked? (Default value = False)
        checkbounds : bool, optional
            should the bounds of the variables be checked? (Default value = True)
        checkintegrality : bool, optional
            has integrality to be checked? (Default value = True)
        checklprows : bool, optional
            have current LP rows (both local and global) to be checked? (Default value = True)
        original : bool, optional
            must the solution be checked against the original problem (Default value = False)

        Returns
        -------
        feasible : bool
            whether the given solution was feasible or not

        """
    def addSol(self, solution: Solution, free: bool = True) -> bool:
        """
        Try to add a solution to the storage.

        Parameters
        ----------
        solution : Solution
            solution to store
        free : bool, optional
            should solution be freed afterwards? (Default value = True)

        Returns
        -------
        stored : bool
            stores whether given solution was good enough to keep

        """
    def freeSol(self, solution: Solution) -> None:
        """
        Free given solution

        Parameters
        ----------
        solution : Solution
            solution to be freed

        """
    def getNSols(self) -> int:
        """
        Gets number of feasible primal solutions stored in the solution storage in case the problem is transformed;
        in case the problem stage is SCIP_STAGE_PROBLEM, the number of solution in the original solution candidate
        storage is returned.

        Returns
        -------
        int

        """
    def getNSolsFound(self) -> int:
        """
        Gets number of feasible primal solutions found so far.

        Returns
        -------
        int

        """
    def getNLimSolsFound(self) -> int:
        """
        Gets number of feasible primal solutions respecting the objective limit found so far.

        Returns
        -------
        int

        """
    def getNBestSolsFound(self) -> int:
        """
        Gets number of feasible primal solutions found so far,
        that improved the primal bound at the time they were found.

        Returns
        -------
        int

        """
    def getSols(self) -> list[Solution]:
        """
        Retrieve list of all feasible primal solutions stored in the solution storage.

        Returns
        -------
        list of Solution

        """
    def getBestSol(self) -> Solution | None:
        """
        Retrieve currently best known feasible primal solution.

        Returns
        -------
        Solution or None

        """
    def getSolObjVal(self, sol: Solution | None, original: bool = True) -> float:
        """
        Retrieve the objective value of the solution.

        Parameters
        ----------
        sol : Solution
        original : bool, optional
            objective value in original space (Default value = True)

        Returns
        -------
        float

        """
    def getSolTime(self, sol: Solution) -> float:
        """
        Get clock time when this solution was found.

        Parameters
        ----------
        sol : Solution

        Returns
        -------
        float

        """
    def getObjVal(self, original: bool = True) -> float:
        """
        Retrieve the objective value of the best solution.

        Parameters
        ----------
        original : bool, optional
            objective value in original space (Default value = True)

        Returns
        -------
        float

        """
    def getSolVal(self, sol: Solution | None, expr: Expr) -> float:
        """
        Retrieve value of given variable or expression in the given solution or in
        the LP/pseudo solution if sol == None

        Parameters
        ----------
        sol : Solution
        expr : Expr
            polynomial expression to query the value of

        Returns
        -------
        float

        Notes
        -----
        A variable is also an expression.

        """
    def getVal(self, expr: Expr) -> float:
        """
        Retrieve the value of the given variable or expression in the best known solution.
        Can only be called after solving is completed.

        Parameters
        ----------
        expr : Expr
            polynomial expression to query the value of

        Returns
        -------
        float

        Notes
        -----
        A variable is also an expression.

        """
    def hasPrimalRay(self) -> bool:
        """
        Returns whether a primal ray is stored that proves unboundedness of the LP relaxation.

        Returns
        -------
        bool

        """
    def getPrimalRayVal(self, var: Variable) -> float:
        """
        Gets value of given variable in primal ray causing unboundedness of the LP relaxation.

        Parameters
        ----------
        var : Variable

        Returns
        -------
        float

        """
    def getPrimalRay(self) -> list[float]:
        """
        Gets primal ray causing unboundedness of the LP relaxation.

        Returns
        -------
        list of float

        """
    def getPrimalbound(self) -> float:
        """
        Retrieve the best primal bound.

        Returns
        -------
        float

        """
    def getDualbound(self) -> float:
        """
        Retrieve the best dual bound.

        Returns
        -------
        float

        """
    def getDualboundRoot(self) -> float:
        """
        Retrieve the best root dual bound.

        Returns
        -------
        float

        """
    def writeName(self, var: Variable) -> None:
        """
        Write the name of the variable to the std out.

        Parameters
        ----------
        var : Variable

        """
    def getStage(self) -> PY_SCIP_STAGE:
        """
        Retrieve current SCIP stage.

        Returns
        -------
        int

        """
    def getStageName(self) -> str:
        """
        Returns name of current stage as string.

        Returns
        -------
        str

        """
    def getStatus(
        self,
    ) -> L[
        "optimal",
        "timelimit",
        "infeasible",
        "unbounded",
        "userinterrupt",
        "inforunbd",
        "nodelimit",
        "totalnodelimit",
        "stallnodelimit",
        "gaplimit",
        "memlimit",
        "sollimit",
        "bestsollimit",
        "restartlimit",
        "primallimit",
        "duallimit",
        "unknown",
    ]:
        """
        Retrieve solution status.

        Returns
        -------
        str
            The status of SCIP.

        """
    def getObjectiveSense(self) -> L["maximize", "minimize", "unknown"]:
        """
        Retrieve objective sense.

        Returns
        -------
        str

        """
    def catchEvent(self, eventtype: PY_SCIP_EVENTTYPE, eventhdlr: Eventhdlr) -> None:
        """
        Catches a global (not variable or row dependent) event.

        Parameters
        ----------
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
    def dropEvent(self, eventtype: PY_SCIP_EVENTTYPE, eventhdlr: Eventhdlr) -> None:
        """
        Drops a global event (stops tracking the event).

        Parameters
        ----------
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
    def catchVarEvent(
        self, var: Variable, eventtype: PY_SCIP_EVENTTYPE, eventhdlr: Eventhdlr
    ) -> None:
        """
        Catches an objective value or domain change event on the given transformed variable.

        Parameters
        ----------
        var : Variable
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
    def dropVarEvent(
        self, var: Variable, eventtype: PY_SCIP_EVENTTYPE, eventhdlr: Eventhdlr
    ) -> None:
        """
        Drops an objective value or domain change event (stops tracking the event) on the given transformed variable.

        Parameters
        ----------
        var : Variable
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
    def catchRowEvent(
        self, row: Row, eventtype: PY_SCIP_EVENTTYPE, eventhdlr: Eventhdlr
    ) -> None:
        """
        Catches a row coefficient, constant, or side change event on the given row.

        Parameters
        ----------
        row : Row
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
    def dropRowEvent(
        self, row: Row, eventtype: PY_SCIP_EVENTTYPE, eventhdlr: Eventhdlr
    ) -> None:
        """
        Drops a row coefficient, constant, or side change event (stops tracking the event) on the given row.

        Parameters
        ----------
        row : Row
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
    def printStatistics(self) -> None:
        """Print statistics."""
    def writeStatistics(
        self, filename: str | bytes | os.PathLike[AnyStr] = "origprob.stats"
    ) -> None:
        """
        Write statistics to a file.

        Parameters
        ----------
        filename : str, optional
            name of the output file (Default = "origprob.stats")

        """
    def getNLPs(self) -> int:
        """
        Gets total number of LPs solved so far.

        Returns
        -------
        int

        """
    def hideOutput(self, quiet: bool = True) -> None:
        """
        Hide the output.

        Parameters
        ----------
        quiet : bool, optional
            hide output? (Default value = True)

        """
    def redirectOutput(self) -> None:
        """Send output to python instead of terminal."""
    def setLogfile(self, path: str | None) -> None:
        """
        Sets the log file name for the currently installed message handler.

        Parameters
        ----------
        path : str or None
            name of log file, or None (no log)

        """
    def setBoolParam(self, name: str, value: float) -> None:
        """
        Set a boolean-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : bool
            value of parameter

        """
    def setIntParam(self, name: str, value: int) -> None:
        """
        Set an int-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : int
            value of parameter

        """
    def setLongintParam(self, name: str, value: int) -> None:
        """
        Set a long-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : int
            value of parameter

        """
    def setRealParam(self, name: str, value: float) -> None:
        """
        Set a real-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : float
            value of parameter

        """
    def setCharParam(self, name: str, value: str) -> None:
        """
        Set a char-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : str
            value of parameter

        """
    def setStringParam(self, name: str, value: str) -> None:
        """
        Set a string-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : str
            value of parameter

        """
    def setParam(self, name: str, value: object) -> None:
        """Set a parameter with value in int, bool, real, long, char or str.

        Parameters
        ----------
        name : str
            name of parameter
        value : object
            value of parameter

        """
    def getParam(self, name: str) -> bool | float | str:
        """
        Get the value of a parameter of type
        int, bool, real, long, char or str.

        Parameters
        ----------
        name : str
            name of parameter

        Returns
        -------
        object

        """
    def getParams(self) -> dict[str, bool | float | str]:
        """
        Gets the values of all parameters as a dict mapping parameter names
        to their values.

        Returns
        -------
        dict of str to object
            dict mapping parameter names to their values.

        """
    def setParams(self, params: Mapping[str, bool | float | str]) -> None:
        """
        Sets multiple parameters at once.

        Parameters
        ----------
        params : dict of str to object
            dict mapping parameter names to their values.

        """
    def readParams(self, file: str | os.PathLike[str]) -> None:
        """
        Read an external parameter file.

        Parameters
        ----------
        file : str
            file to read

        """
    def writeParams(
        self,
        filename: str | os.PathLike[str] = "param.set",
        comments: bool = True,
        onlychanged: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Write parameter settings to an external file.

        Parameters
        ----------
        filename : str, optional
            file to be written (Default value = 'param.set')
        comments : bool, optional
            write parameter descriptions as comments? (Default value = True)
        onlychanged : bool, optional
            write only modified parameters (Default value = True)
        verbose : bool, optional
            indicates whether a success message should be printed

        """
    def resetParam(self, name: str) -> None:
        """
        Reset parameter setting to its default value

        Parameters
        ----------
        name : str
            parameter to reset

        """
    def resetParams(self) -> None:
        """Reset parameter settings to their default values."""
    def setEmphasis(
        self, paraemphasis: PY_SCIP_PARAMEMPHASIS, quiet: bool = True
    ) -> None:
        """
        Set emphasis settings

        Parameters
        ----------
        paraemphasis : PY_SCIP_PARAMEMPHASIS
            emphasis to set
        quiet : bool, optional
            hide output? (Default value = True)

        """
    def readProblem(
        self, filename: str | os.PathLike[str], extension: str | None = None
    ) -> None:
        """
        Read a problem instance from an external file.

        Parameters
        ----------
        filename : str
            problem file name
        extension : str or None
            specify file extension/type (Default value = None)

        """
    def count(self) -> None:
        """Counts the number of feasible points of problem."""
    def getNReaders(self) -> int:
        """
        Get number of currently available readers.

        Returns
        -------
        int

        """
    def getNCountedSols(self) -> int:
        """
        Get number of feasible solution.

        Returns
        -------
        int

        """
    def setParamsCountsols(self) -> None:
        """Sets SCIP parameters such that a valid counting process is possible."""
    def freeReoptSolve(self) -> None:
        """Frees all solution process data and prepares for reoptimization."""
    def chgReoptObjective(
        self, coeffs: Expr, sense: L["minimize", "maximize"] = "minimize"
    ) -> None:
        """
        Establish the objective function as a linear expression.

        Parameters
        ----------
        coeffs : list of float
            the coefficients
        sense : str
            the objective sense (Default value = 'minimize')

        """
    def chgVarBranchPriority(self, var: Variable, priority: int) -> None:
        """
        Sets the branch priority of the variable.
        Variables with higher branch priority are always preferred to variables with
        lower priority in selection of branching variable.

        Parameters
        ----------
        var : Variable
            variable to change priority of
        priority : int
            the new priority of the variable (the default branching priority is 0)

        """
    def startStrongbranch(self) -> None:
        """Start strong branching. Needs to be called before any strong branching. Must also later end strong branching.
        TODO: Propagation option has currently been disabled via Python.
        If propagation is enabled then strong branching is not done on the LP, but on additionally created nodes
        (has some overhead)."""
    def endStrongbranch(self) -> None:
        """End strong branching. Needs to be called if startStrongBranching was called previously.
        Between these calls the user can access all strong branching functionality."""
    def getVarStrongbranchLast(
        self, var: Variable
    ) -> tuple[float, float, bool, bool, float, float]:
        """
        Get the results of the last strong branching call on this variable (potentially was called
        at another node).

        Parameters
        ----------
        var : Variable
            variable to get the previous strong branching information from

        Returns
        -------
        down : float
            The dual bound of the LP after branching down on the variable
        up : float
            The dual bound of the LP after branchign up on the variable
        downvalid : bool
            stores whether the returned down value is a valid dual bound, or NULL
        upvalid : bool
            stores whether the returned up value is a valid dual bound, or NULL
        solval : float
            The solution value of the variable at the last strong branching call
        lpobjval : float
            The LP objective value at the time of the last strong branching call

        """
    def getVarStrongbranchNode(self, var: Variable) -> int:
        """
        Get the node number from the last time strong branching was called on the variable.

        Parameters
        ----------
        var : Variable
            variable to get the previous strong branching node from

        Returns
        -------
        int

        """
    def getVarStrongbranch(
        self,
        var: Variable,
        itlim: int,
        idempotent: bool = False,
        integral: bool = False,
    ) -> tuple[float, float, bool, bool, bool, bool, bool, bool, bool]:
        """
        Strong branches and gets information on column variable.

        Parameters
        ----------
        var : Variable
            Variable to get strong branching information on
        itlim : int
            LP iteration limit for total strong branching calls
        idempotent : bool, optional
            Should SCIP's state remain the same after the call?
        integral : bool, optional
            Boolean on whether the variable is currently integer.

        Returns
        -------
        down : float
            The dual bound of the LP after branching down on the variable
        up : float
            The dual bound of the LP after branchign up on the variable
        downvalid : bool
            stores whether the returned down value is a valid dual bound, or NULL
        upvalid : bool
            stores whether the returned up value is a valid dual bound, or NULL
        downinf : bool
            store whether the downwards branch is infeasible
        upinf : bool
            store whether the upwards branch is infeasible
        downconflict : bool
            store whether a conflict constraint was created for an infeasible downwards branch
        upconflict : bool
            store whether a conflict constraint was created for an infeasible upwards branch
        lperror : bool
            whether an unresolved LP error occurred in the solving process

        """
    def updateVarPseudocost(
        self, var: Variable, valdelta: float, objdelta: float, weight: float
    ) -> None:
        """
        Updates the pseudo costs of the given variable and the global pseudo costs after a change of valdelta
        in the variable's solution value and resulting change of objdelta in the LP's objective value.
        Update is ignored if objdelts is infinite. Weight is in range (0, 1], and affects how it updates
        the global weighted sum.

        Parameters
        ----------
        var : Variable
            Variable whos pseudo cost will be updated
        valdelta : float
            The change in variable value (e.g. the fractional amount removed or added by branching)
        objdelta : float
            The change in objective value of the LP after valdelta change of the variable
        weight : float
            the weight in range (0,1] of how the update affects the stored weighted sum.

        """
    def getBranchScoreMultiple(self, var: Variable, gains: list[float]) -> float:
        """
        Calculates the branching score out of the gain predictions for a branching with
        arbitrarily many children.

        Parameters
        ----------
        var : Variable
            variable to calculate the score for
        gains : list of float
            list of gains for each child.

        Returns
        -------
        float

        """
    def getTreesizeEstimation(self) -> float:
        """
        Get an estimate of the final tree size.

        Returns
        -------
        float

        """
    def getBipartiteGraphRepresentation(
        self,
        prev_col_features: Sequence[Sequence[float]] | None = None,
        prev_edge_features: Sequence[Sequence[float]] | None = None,
        prev_row_features: Sequence[Sequence[float]] | None = None,
        static_only: bool = False,
        suppress_warnings: bool = False,
    ) -> tuple[
        list[list[float | None]],
        list[list[float]],
        list[list[float]],
        dict[str, dict[str, int]],
    ]:
        """
        This function generates the bipartite graph representation of an LP, which was first used in
        the following paper:
        @inproceedings{conf/nips/GasseCFCL19,
        title={Exact Combinatorial Optimization with Graph Convolutional Neural Networks},
        author={Gasse, Maxime and Chételat, Didier and Ferroni, Nicola and Charlin, Laurent and Lodi, Andrea},
        booktitle={Advances in Neural Information Processing Systems 32},
        year={2019}
        }
        The exact features have been modified compared to the original implementation.
        This function is used mainly in the machine learning community for MIP.
        A user can only call it during the solving process, when there is an LP object. This means calling it
        from some user defined plugin on the Python side.
        An example plugin is a branching rule or an event handler, which is exclusively created to call this function.
        The user must then make certain to return the appropriate SCIP_RESULT (e.g. DIDNOTRUN)

        Parameters
        ----------
        prev_col_features : list of list or None, optional
            The list of column features previously returned by this function
        prev_edge_features : list of list or None, optional
            The list of edge features previously returned by this function
        prev_row_features : list of list or None, optional
            The list of row features previously returned by this function
        static_only : bool, optional
            Whether exclusively static features should be generated
        suppress_warnings : bool, optional
            Whether warnings should be suppressed

        Returns
        -------
        col_features : list of list
        edge_features : list of list
        row_features : list of list
        dict
            The feature mappings for the columns, edges, and rows

        """

@dataclasses.dataclass
class Statistics:
    status: str
    total_time: float
    solving_time: float
    presolving_time: float
    reading_time: float
    copying_time: float
    problem_name: str
    presolved_problem_name: str
    _variables: dict[str, int]
    _presolved_variables: dict[str, int]
    _constraints: dict[str, int]
    _presolved_constraints: dict[str, int]
    n_runs: int | None = None
    n_nodes: int | None = None
    n_solutions_found: int = -1
    first_solution: float | None = None
    primal_bound: float | None = None
    dual_bound: float | None = None
    gap: float | None = None
    primal_dual_integral: float | None = None
    @property
    def n_vars(self) -> int: ...
    @property
    def n_binary_vars(self) -> int: ...
    @property
    def n_integer_vars(self) -> int: ...
    @property
    def n_implicit_integer_vars(self) -> int: ...
    @property
    def n_continuous_vars(self) -> int: ...
    @property
    def n_presolved_vars(self) -> int: ...
    @property
    def n_presolved_binary_vars(self) -> int: ...
    @property
    def n_presolved_integer_vars(self) -> int: ...
    @property
    def n_presolved_implicit_integer_vars(self) -> int: ...
    @property
    def n_presolved_continuous_vars(self) -> int: ...
    @property
    def n_conss(self) -> int: ...
    @property
    def n_maximal_cons(self) -> int: ...
    @property
    def n_presolved_conss(self) -> int: ...
    @property
    def n_presolved_maximal_cons(self) -> int: ...

def readStatistics(filename: str | bytes | os.PathLike[AnyStr]) -> Statistics:
    """
    Given a .stats file of a solved model, reads it and returns an instance of the Statistics class
    holding some statistics.

    Parameters
    ----------
    filename : str
        name of the input file

    Returns
    -------
    Statistics

    """

def is_memory_freed() -> bool: ...
def print_memory_in_use() -> None: ...

__test__: dict[Any, Any]
