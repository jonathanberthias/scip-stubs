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

import numpy as np
from typing_extensions import (
    CapsuleType,
    Never,
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

_Shape: TypeAlias = tuple[Any, ...]
_ObjectArray: TypeAlias = np.ndarray[_Shape, np.dtype[np.object_]]
_NumberArray: TypeAlias = (
    np.ndarray[_Shape, np.dtype[np.integer[Any]]]
    | np.ndarray[_Shape, np.dtype[np.floating[Any]]]
)

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
def buildGenExprObj(expr: Expr) -> SumExpr: ...
@overload
def buildGenExprObj(expr: GenExpr[_OpT]) -> GenExpr[_OpT]: ...
@overload
def buildGenExprObj(expr: MatrixExpr) -> _ObjectArray: ...  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
@overload
def buildGenExprObj(expr: SupportsFloat) -> Constant: ...

# This case is valid at runtime if expr is the string repr of a real number
# (i.e., float(expr) does not raise), but expr is not converted to a float
# so the returned value is essentially unusable.
# @overload
# def buildGenExprObj(expr: str) -> Constant: ...

class Expr:
    terms: dict[Term, float]
    def __init__(self, /, terms: dict[Term, float] | None = None) -> None: ...
    @overload
    def __getitem__(self, index: Variable, /) -> float: ...
    @overload
    def __getitem__(self, index: Term, /) -> float: ...
    def __iter__(self, /) -> Iterator[Term]: ...
    def __abs__(self, /) -> UnaryExpr[L["abs"]]: ...
    @overload
    def __add__(self, other: Expr | float, /) -> Expr: ...
    @overload
    def __add__(self, other: GenExpr[Any], /) -> SumExpr: ...
    @overload
    def __iadd__(self, other: Expr | float, /) -> Self: ...
    @overload
    def __iadd__(self, other: GenExpr[Any], /) -> SumExpr: ...
    @overload
    def __mul__(self, other: Expr | float, /) -> Expr: ...
    @overload
    def __mul__(self, other: GenExpr[Any], /) -> ProdExpr: ...
    @overload
    def __truediv__(self, other: float, /) -> Expr: ...
    @overload
    def __truediv__(self, other: Expr | GenExpr[Any], /) -> ProdExpr: ...
    def __rtruediv__(self, other: SupportsFloat, /) -> Expr: ...
    @overload
    def __pow__(self, other: L[0], mod: object = None) -> L[1]: ...  # type: ignore[overload-overlap]
    @overload
    def __pow__(
        self, other: SupportsFloat, mod: object = None, /
    ) -> Expr | PowExpr: ...
    def __neg__(self, /) -> Expr: ...
    @overload
    def __sub__(self, other: Expr | float, /) -> Expr: ...
    @overload
    def __sub__(self, other: GenExpr[Any], /) -> SumExpr: ...
    def __radd__(self, other: SupportsFloat, /) -> Expr: ...
    def __rmul__(self, other: SupportsFloat, /) -> Expr: ...
    def __rsub__(self, other: SupportsFloat, /) -> Expr: ...
    @override  # type: ignore[override]
    @overload
    def __eq__(self, other: MatrixExpr, /) -> MatrixExprCons: ...  # type: ignore[overload-overlap]
    @overload
    def __eq__(self, other: Expr | SupportsFloat | GenExpr[Any], /) -> ExprCons: ...  # pyright: ignore[reportOverlappingOverload]
    @overload
    def __ge__(self, other: MatrixExpr, /) -> MatrixExprCons: ...  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
    @overload
    def __ge__(self, other: Expr | SupportsFloat | GenExpr[Any], /) -> ExprCons: ...
    @overload
    def __le__(self, other: MatrixExpr, /) -> MatrixExprCons: ...  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
    @overload
    def __le__(self, other: Expr | SupportsFloat | GenExpr[Any], /) -> ExprCons: ...
    def normalize(self, /) -> None: ...
    def degree(self, /) -> int: ...

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
    def normalize(self, /) -> None: ...
    def __ge__(self, other: SupportsFloat, /) -> ExprCons: ...
    def __le__(self, other: SupportsFloat, /) -> ExprCons: ...

@overload
def quicksum(termlist: Iterable[Expr | SupportsFloat]) -> Expr: ...  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
@overload
def quicksum(termlist: Iterable[Expr | SupportsFloat | GenExpr[Any]]) -> SumExpr: ...
@overload
def quickprod(termlist: Iterable[Expr | SupportsFloat]) -> Expr: ...  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
@overload
def quickprod(termlist: Iterable[Expr | SupportsFloat | GenExpr[Any]]) -> ProdExpr: ...

class GenExpr(Generic[_OpT]):
    _op: _OpT
    children: list[GenExpr[Any]]
    def __init__(self, /) -> None: ...
    def __abs__(self, /) -> UnaryExpr[L["abs"]]: ...
    def __add__(self, other: Expr | float | GenExpr[Any], /) -> SumExpr: ...
    def __mul__(self, other: Expr | float | GenExpr[Any], /) -> ProdExpr: ...
    def __pow__(self, other: SupportsFloat, mod: object = None, /) -> PowExpr: ...
    def __truediv__(self, other: Expr | float | GenExpr[Any], /) -> ProdExpr: ...
    def __rtruediv__(self, other: float, /) -> ProdExpr: ...
    def __neg__(self, /) -> ProdExpr: ...
    def __sub__(self, other: Expr | float | GenExpr[Any], /) -> SumExpr: ...
    def __radd__(self, other: float, /) -> SumExpr: ...
    def __rmul__(self, other: float, /) -> SumExpr: ...
    def __rsub__(self, other: float, /) -> SumExpr: ...
    @override  # type: ignore[override]
    @overload
    def __eq__(self, other: MatrixExpr, /) -> MatrixExprCons: ...  # type: ignore[overload-overlap]
    @overload
    def __eq__(self, other: Expr | SupportsFloat | GenExpr[Any], /) -> ExprCons: ...  # pyright: ignore[reportOverlappingOverload]
    @overload
    def __ge__(self, other: MatrixExpr, /) -> MatrixExprCons: ...  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
    @overload
    def __ge__(self, other: Expr | SupportsFloat | GenExpr[Any], /) -> ExprCons: ...
    @overload
    def __le__(self, other: MatrixExpr, /) -> MatrixExprCons: ...  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
    @overload
    def __le__(self, other: Expr | SupportsFloat | GenExpr[Any], /) -> ExprCons: ...
    def degree(self, /) -> float: ...
    def getOp(self, /) -> _OpT: ...

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

@overload
def exp(expr: MatrixExpr) -> MatrixGenExpr: ...  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
@overload
def exp(expr: Expr | SupportsFloat | GenExpr[Any]) -> UnaryExpr[L["exp"]]: ...
@overload
def log(expr: MatrixExpr) -> MatrixGenExpr: ...  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
@overload
def log(expr: Expr | SupportsFloat | GenExpr[Any]) -> UnaryExpr[L["log"]]: ...
@overload
def sqrt(expr: MatrixExpr) -> MatrixGenExpr: ...  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
@overload
def sqrt(expr: Expr | SupportsFloat | GenExpr[Any]) -> UnaryExpr[L["sqrt"]]: ...
@overload
def sin(expr: MatrixExpr) -> MatrixGenExpr: ...  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
@overload
def sin(expr: Expr | SupportsFloat | GenExpr[Any]) -> UnaryExpr[L["sin"]]: ...
@overload
def cos(expr: MatrixExpr) -> MatrixGenExpr: ...  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
@overload
def cos(expr: Expr | SupportsFloat | GenExpr[Any]) -> UnaryExpr[L["cos"]]: ...

_Node: TypeAlias = tuple[str, list[Variable | float]]

def expr_to_nodes(expr: GenExpr[Any]) -> list[_Node]: ...
def value_to_array(val: float, nodes: list[_Node]) -> int: ...
def expr_to_array(expr: GenExpr[Any], nodes: list[Node]) -> int: ...

########
# lp.pxi
########

class LP:
    def __init__(
        self, name: str = "LP", sense: L["minimize", "maximize"] = "minimize"
    ) -> None: ...
    @property
    def name(self, /) -> str: ...
    def writeLP(self, /, filename: bytes) -> None: ...
    def readLP(self, /, filename: bytes) -> None: ...
    def infinity(self, /) -> float: ...
    def isInfinity(self, /, val: SupportsFloat) -> bool: ...
    def addCol(
        self,
        entries: Sequence[tuple[int, float]],
        obj: float = 0.0,
        lb: float = 0.0,
        ub: float | None = None,
    ) -> None: ...
    def addCols(
        self,
        entrieslist: Sequence[Sequence[tuple[int, float]]],
        objs: Sequence[float] | None = None,
        lbs: Sequence[float] | None = None,
        ubs: Sequence[float] | None = None,
    ) -> None: ...
    def delCols(self, firstcol: int, lastcol: int) -> None: ...
    def addRow(
        self,
        entries: Sequence[tuple[int, float]],
        lhs: float = 0.0,
        rhs: float | None = None,
    ) -> None: ...
    def addRows(
        self,
        entrieslist: Sequence[Sequence[tuple[int, float]]],
        lhss: Sequence[float] | None = None,
        rhss: Sequence[float] | None = None,
    ) -> None: ...
    def delRows(self, firstrow: int, lastrow: int) -> None: ...
    def getBounds(
        self, firstcol: int = 0, lastcol: int | None = None
    ) -> tuple[list[float], list[float]] | None: ...
    def getSides(
        self, firstrow: int = 0, lastrow: float | None = None
    ) -> tuple[list[float], list[float]] | None: ...
    def chgObj(self, col: int, obj: float) -> None: ...
    def chgCoef(self, row: int, col: int, newval: float) -> None: ...
    def chgBound(self, col: int, lb: float, ub: float) -> None: ...
    def chgSide(self, row: int, lhs: float, rhs: float) -> None: ...
    def clear(self) -> None: ...
    def nrows(self) -> int: ...
    def ncols(self) -> int: ...
    def solve(self, dual: bool = True) -> float: ...
    def getPrimal(self) -> list[float]: ...
    def isPrimalFeasible(self) -> bool: ...
    def getDual(self) -> list[float]: ...
    def isDualFeasible(self) -> bool: ...
    def getPrimalRay(self) -> list[float] | None: ...
    def getDualRay(self) -> list[float] | None: ...
    def getNIterations(self) -> int: ...
    def getRedcost(self) -> list[float]: ...
    def getBasisInds(self) -> list[int]: ...

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
    def bendersfree(self) -> None: ...
    def bendersinit(self) -> None: ...
    def bendersexit(self) -> None: ...
    def bendersinitpre(self) -> None: ...
    def bendersexitpre(self) -> None: ...
    def bendersinitsol(self) -> None: ...
    def bendersexitsol(self) -> None: ...
    def benderscreatesub(self, probnumber: int) -> None: ...
    def benderspresubsolve(
        self,
        solution: Solution | None,
        enfotype: PY_SCIP_BENDERSENFOTYPE,
        checkint: bool,
    ) -> BendersPresubsolveRes: ...
    def benderssolvesubconvex(
        self, solution: Solution | None, probnumber: int, onlyconvex: bool
    ) -> BendersSolvesubRes: ...
    def benderssolvesub(
        self, solution: Solution | None, probnumber: int
    ) -> BendersSolvesubRes: ...
    def benderspostsolve(
        self,
        solution: Solution | None,
        enfotype: PY_SCIP_BENDERSENFOTYPE,
        mergecandidates: list[int],
        npriomergecands: int,
        checkint: bool,
        infeasible: bool,
    ) -> BendersPostsolveRes: ...
    def bendersfreesub(self, probnumber: int) -> None: ...
    def bendersgetvar(
        self, variable: Variable, probnumber: int
    ) -> BendersGetvarRes: ...

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
    def branchfree(self) -> None: ...
    def branchinit(self) -> None: ...
    def branchexit(self) -> None: ...
    def branchinitsol(self) -> None: ...
    def branchexitsol(self) -> None: ...
    def branchexeclp(self, allowaddcons: L[True]) -> BranchRuleExecTD: ...
    def branchexecext(self, allowaddcons: L[True]) -> BranchRuleExecTD: ...
    def branchexecps(self, allowaddcons: L[True]) -> BranchRuleExecPsTD: ...

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
    def consfree(self) -> None: ...
    def consinit(self, constraints: list[Constraint]) -> None: ...
    def consexit(self, constraints: list[Constraint]) -> None: ...
    def consinitpre(self, constraints: list[Constraint]) -> None: ...
    def consexitpre(self, constraints: list[Constraint]) -> None: ...
    def consinitsol(self, constraints: list[Constraint]) -> None: ...
    def consexitsol(self, constraints: list[Constraint], restart: bool) -> None: ...
    def consdelete(self, constraint: Constraint) -> None: ...
    def constrans(self, sourceconstraint: Constraint) -> ConshdlrConsTransRes: ...
    def consinitlp(self, constraints: list[Constraint]) -> ConshdlrConsInitLpRes: ...
    def conssepalp(self, constraints: list[Constraint], nusefulconss: int) -> None: ...
    def conssepasol(
        self, constraints: list[Constraint], nusefulconss: int, solution: Solution
    ) -> ConshdlrConsSepaRes: ...
    def consenfolp(
        self, constraints: list[Constraint], nusefulconss: int, solinfeasible: bool
    ) -> ConshdlrEnfoRes: ...
    def consenforelax(
        self,
        solution: Solution,
        constraints: list[Constraint],
        nusefulconss: int,
        solinfeasible: bool,
    ) -> ConshdlrEnfoRes: ...
    def consenfops(
        self,
        constraints: list[Constraint],
        nusefulconss: int,
        solinfeasible: bool,
        objinfeasible: bool,
    ) -> ConshdlrEnfoPsRes: ...
    def conscheck(
        self,
        constraints: list[Constraint],
        solution: Solution,
        checkintegrality: bool,
        checklprows: bool,
        printreason: bool,
        completely: bool,
    ) -> ConshdlrConsCheckRes: ...
    def consprop(
        self,
        constraints: list[Constraint],
        nusefulconss: int,
        nmarkedconss: int,
        proptiming: PY_SCIP_PROPTIMING,
    ) -> ConshdlrConsPropRes: ...
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
    ) -> None: ...
    def consresprop(self) -> None: ...
    def conslock(
        self,
        constraint: Constraint | None,
        # 0 == LockType.MODEL
        # 1 == LockType.CONFLICT
        # The enum is not available in PySCIPOpt
        locktype: L[0, 1],
        nlockspos: int,
        nlocksneg: int,
    ) -> None: ...
    def consactive(self, constraint: Constraint) -> None: ...
    def consdeactive(self, constraint: Constraint) -> None: ...
    def consenable(self, constraint: Constraint) -> None: ...
    def consdisable(self, constraint: Constraint) -> None: ...
    def consdelvars(self, constraints: list[Constraint]) -> None: ...
    def consprint(self, constraint: Constraint) -> None: ...
    def conscopy(self) -> None: ...
    def consparse(self) -> None: ...
    def consgetvars(self, constraint: Constraint) -> None: ...
    def consgetnvars(self, constraint: Constraint) -> ConshdlrConsGetnVarsRes: ...
    def consgetdivebdchgs(self) -> None: ...
    def consgetpermsymgraph(self) -> None: ...
    def consgetsignedpermsymgraph(self) -> None: ...

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
    def cutselfree(self) -> None: ...
    def cutselinit(self) -> None: ...
    def cutselexit(self) -> None: ...
    def cutselinitsol(self) -> None: ...
    def cutselexitsol(self) -> None: ...
    def cutselselect(
        self, cuts: list[Row], forcedcuts: list[Row], root: bool, maxnselectedcuts: int
    ) -> CutSelSelectReturnTD: ...

###########
# event.pxi
###########

class Eventhdlr:
    model: Model
    name: str
    def eventcopy(self) -> None: ...
    def eventfree(self) -> None: ...
    def eventinit(self) -> None: ...
    def eventexit(self) -> None: ...
    def eventinitsol(self) -> None: ...
    def eventexitsol(self) -> None: ...
    def eventdelete(self) -> None: ...
    def eventexec(self, event: Event) -> None: ...

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
    def heurfree(self) -> None: ...
    def heurinit(self) -> None: ...
    def heurexit(self) -> None: ...
    def heurinitsol(self) -> None: ...
    def heurexitsol(self) -> None: ...
    def heurexec(
        self, heurtiming: PY_SCIP_HEURTIMING, nodeinfeasible: bool
    ) -> HeurExecResultTD: ...

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
    def presolfree(self) -> None: ...
    def presolinit(self) -> None: ...
    def presolexit(self) -> None: ...
    def presolinitpre(self) -> None: ...
    def presolexitpre(self) -> None: ...
    def presolexec(
        self, nrounds: int, presoltiming: PY_SCIP_PRESOLTIMING
    ) -> PresolExecRes: ...

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
    def pricerfree(self) -> None: ...
    def pricerinit(self) -> None: ...
    def pricerexit(self) -> None: ...
    def pricerinitsol(self) -> None: ...
    def pricerexitsol(self) -> None: ...
    def pricerredcost(self) -> PricerRedcostRes: ...
    def pricerfarkas(self) -> PricerFarkasRes: ...

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
    def propfree(self) -> None: ...
    def propinit(self) -> None: ...
    def propexit(self) -> None: ...
    def propinitsol(self) -> None: ...
    def propexitsol(self, restart: bool) -> None: ...
    def propinitpre(self) -> None: ...
    def propexitpre(self) -> None: ...
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
    ) -> None: ...
    def propexec(self, proptiming: PY_SCIP_PROPTIMING) -> PropExecRes: ...
    def propresprop(
        self,
        confvar: Variable,
        inferinfo: int,
        # 0 == SCIP_BOUNDTYPE_LOWER
        # 1 == SCIP_BOUNDTYPE_UPPER
        bdtype: L[0, 1],
        relaxedbd: float,
    ) -> PropResPropRes: ...

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
    def sepafree(self) -> None: ...
    def sepainit(self) -> None: ...
    def sepaexit(self) -> None: ...
    def sepainitsol(self) -> None: ...
    def sepaexitsol(self) -> None: ...
    def sepaexeclp(self) -> SepaExecResultTD: ...
    def sepaexecsol(self, solution: Solution) -> SepaExecResultTD: ...

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
    def readerfree(self) -> None: ...
    def readerread(self, filename: str) -> ReaderRes: ...
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
    ) -> ReaderRes: ...

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
    def relaxfree(self) -> None: ...
    def relaxinit(self) -> None: ...
    def relaxexit(self) -> None: ...
    def relaxinitsol(self) -> None: ...
    def relaxexitsol(self) -> None: ...
    def relaxexec(self) -> RelaxExecRes: ...

#############
# nodesel.pxi
#############

@type_check_only
class NodeselNodeselectTD(TypedDict):
    selnode: Node

class Nodesel:
    model: Model
    def nodefree(self) -> None: ...
    def nodeinit(self) -> None: ...
    def nodeexit(self) -> None: ...
    def nodeinitsol(self) -> None: ...
    def nodeexitsol(self) -> None: ...
    def nodeselect(self) -> NodeselNodeselectTD: ...
    def nodecomp(self, node1: Node, node2: Node) -> int: ...

############
# matrix.pxi
############

_MatrixCompRhs: TypeAlias = SupportsFloat | Variable | _NumberArray
_MatrixOpRhs: TypeAlias = (
    SupportsFloat | Expr | GenExpr[Any] | MatrixExpr | _NumberArray
)
_MatrixRmulRhs: TypeAlias = float

class MatrixExpr(_ObjectArray):
    # Only the initial argument makes sense, all the other arguments will most likely
    # lead to an error
    @override
    def sum(  # type: ignore[override]
        self, *, initial: SupportsFloat | None = None, **kwargs: Never
    ) -> Expr: ...
    @override
    def __le__(self, other: _MatrixCompRhs) -> MatrixExprCons: ...  # type: ignore[override]
    @override
    def __ge__(self, other: _MatrixCompRhs) -> MatrixExprCons: ...  # type: ignore[override]
    @override
    def __eq__(self, other: _MatrixCompRhs) -> MatrixExprCons: ...  # type: ignore[override]
    @override
    def __add__(self, other: _MatrixOpRhs) -> MatrixExpr: ...  # type: ignore[override]
    # Re noqa: this __iadd__ can change the type of self, e.g. adding a float to a MatrixVariable
    # changes self to a MatrixExpr, so Self is not a correct return type
    # However, it causes errors at the call-site unless allow-redefinition is set
    @override
    def __iadd__(self, other: _MatrixOpRhs) -> MatrixExpr: ...  # type: ignore[override]  # noqa: PYI034
    @override
    def __mul__(self, other: _MatrixOpRhs) -> MatrixExpr: ...  # type: ignore[override]
    @override
    def __truediv__(self, other: _MatrixOpRhs) -> MatrixExpr: ...  # type: ignore[override]
    @override
    def __rtruediv__(self, other: _MatrixRmulRhs) -> MatrixExpr: ...  # type: ignore[override]
    @override
    def __pow__(self, other: SupportsFloat | _NumberArray) -> MatrixExpr: ...  # type: ignore[override]
    @override
    def __sub__(self, other: _MatrixOpRhs) -> MatrixExpr: ...  # type: ignore[override]
    @override
    def __radd__(self, other: _MatrixOpRhs) -> MatrixExpr: ...  # type: ignore[override]
    @override
    def __rmul__(self, other: _MatrixRmulRhs) -> MatrixExpr: ...  # type: ignore[override]
    @override
    def __rsub__(self, other: _MatrixOpRhs) -> MatrixExpr: ...  # type: ignore[override]

class MatrixGenExpr(MatrixExpr): ...
class MatrixExprCons(_ObjectArray): ...

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
    def getType(self) -> PY_SCIP_EVENTTYPE: ...
    def getName(self) -> str: ...
    def getNewBound(self) -> float: ...
    def getOldBound(self) -> float: ...
    def getVar(self) -> Variable: ...
    def getNode(self) -> Node: ...
    def getRow(self) -> Row: ...
    @override
    def __hash__(self) -> int: ...

class Column:
    data: object
    def __init__(self) -> None: ...
    def getLPPos(self) -> int: ...
    def getBasisStatus(self) -> L["lower", "basic", "upper", "zero"]: ...
    def isIntegral(self) -> bool: ...
    def getVar(self) -> Variable: ...
    def getPrimsol(self) -> float: ...
    def getLb(self) -> float: ...
    def getUb(self) -> float: ...
    def getObjCoeff(self) -> float: ...
    def getAge(self) -> int: ...
    @override
    def __hash__(self) -> int: ...

class Row:
    data: object
    @property
    def name(self) -> str: ...
    def getLhs(self) -> float: ...
    def getRhs(self) -> float: ...
    def getConstant(self) -> float: ...
    def getLPPos(self) -> int: ...
    def getBasisStatus(self) -> L["lower", "basic", "upper"]: ...
    def isIntegral(self) -> bool: ...
    def isLocal(self) -> bool: ...
    def isModifiable(self) -> bool: ...
    def isRemovable(self) -> bool: ...
    def isInGlobalCutpool(self) -> bool: ...
    def getOrigintype(self) -> PY_SCIP_ROWORIGINTYPE: ...
    def getConsOriginConshdlrtype(self) -> str: ...
    def getNNonz(self) -> int: ...
    def getNLPNonz(self) -> int: ...
    def getCols(self) -> list[Column]: ...
    def getVals(self) -> list[float]: ...
    def getAge(self) -> int: ...
    def getNorm(self) -> float: ...
    @override
    def __hash__(self) -> int: ...

class NLRow:
    data: object
    @property
    def name(self) -> str: ...
    def getConstant(self) -> float: ...
    def getLinearTerms(self) -> list[tuple[Variable, float]]: ...
    def getLhs(self) -> float: ...
    def getRhs(self) -> float: ...
    def getDualsol(self) -> float: ...
    @override
    def __hash__(self) -> int: ...

class Solution:
    data: object
    def __init__(self, raise_error: bool = False) -> None: ...
    def __getitem__(self, /, expr: Expr) -> float: ...
    def __setitem__(self, /, var: Variable, value: float) -> None: ...
    def getOrigin(self) -> PY_SCIP_SOLORIGIN: ...
    def retransform(self) -> None: ...
    def translate(self, target: Model) -> Solution: ...

class BoundChange:
    def getNewBound(self) -> float: ...
    def getVar(self) -> Variable: ...
    # TODO: enum? (0 = branching, 1 = consinfer, 2 = propinfer)
    def getBoundchgtype(self) -> int: ...
    # TODO: enum? (0 = lower, 1 = upper)
    def getBoundtype(self) -> int: ...
    def isRedundant(self) -> bool: ...

class DomainChanges:
    def getBoundchgs(self) -> list[BoundChange]: ...

class Node:
    data: object
    def getParent(self) -> Node | None: ...
    def getNumber(self) -> int: ...
    def getDepth(self) -> int: ...
    def getType(self) -> PY_SCIP_NODETYPE: ...
    def getLowerbound(self) -> float: ...
    def getEstimate(self) -> float: ...
    def getAddedConss(self) -> list[Constraint]: ...
    def getNAddedConss(self) -> int: ...
    def isActive(self) -> bool: ...
    def isPropagatedAgain(self) -> bool: ...
    def getNParentBranchings(self) -> int: ...
    # TODO: the ints are SCIP_BOUNDTYPEs
    def getParentBranchings(
        self,
    ) -> tuple[list[Variable], list[float], list[int]] | None: ...
    def getNDomchg(self) -> tuple[int, int, int]: ...
    def getDomchg(self) -> DomainChanges | None: ...
    @override
    def __hash__(self) -> int: ...

class Variable(Expr):
    data: object
    @property
    def name(self) -> str: ...
    def ptr(self) -> int: ...
    def vtype(self) -> _VTypesLong: ...
    def isOriginal(self) -> bool: ...
    def isInLP(self) -> bool: ...
    def getIndex(self) -> int: ...
    def getCol(self) -> Column: ...
    def getLbOriginal(self) -> float: ...
    def getUbOriginal(self) -> float: ...
    def getLbGlobal(self) -> float: ...
    def getUbGlobal(self) -> float: ...
    def getLbLocal(self) -> float: ...
    def getUbLocal(self) -> float: ...
    def getObj(self) -> float: ...
    def getLPSol(self) -> float: ...
    def getAvgSol(self) -> float: ...
    def varMayRound(self, direction: L["down", "up"] = "down") -> bool: ...

# none of the defined methods work in v5.4.1
# pretend they don't exist
class MatrixVariable(MatrixExpr):
    ...
    # def vtype(self) -> Incomplete:
    #     """
    #     Retrieve the matrix variables type (BINARY, INTEGER, IMPLINT or CONTINUOUS)

    #     Returns
    #     -------
    #     np.ndarray
    #         A matrix containing "BINARY", "INTEGER", "CONTINUOUS", or "IMPLINT"

    #     """
    # def isInLP(self) -> Incomplete:
    #     """
    #     Retrieve whether the matrix variable is a COLUMN variable that is member of the current LP.

    #     Returns
    #     -------
    #     np.ndarray
    #         An array of bools

    #     """
    # def getIndex(self) -> Incomplete:
    #     """
    #     Retrieve the unique index of the matrix variable.

    #     Returns
    #     -------
    #     np.ndarray
    #         An array of integers. No two should be the same
    #     """
    # def getCol(self) -> Incomplete:
    #     """
    #     Retrieve matrix of columns of COLUMN variables.

    #     Returns
    #     -------
    #     np.ndarray
    #         An array of Column objects
    #     """
    # def getLbOriginal(self) -> Incomplete:
    #     """
    #     Retrieve original lower bound of matrix variable.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def getUbOriginal(self) -> Incomplete:
    #     """
    #     Retrieve original upper bound of matrixvariable.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def getLbGlobal(self) -> Incomplete:
    #     """
    #     Retrieve global lower bound of matrix variable.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def getUbGlobal(self) -> Incomplete:
    #     """
    #     Retrieve global upper bound of matrix variable.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def getLbLocal(self) -> Incomplete:
    #     """
    #     Retrieve current lower bound of matrix variable.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def getUbLocal(self) -> Incomplete:
    #     """
    #     Retrieve current upper bound of matrix variable.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def getObj(self) -> Incomplete:
    #     """
    #     Retrieve current objective value of matrix variable.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def getLPSol(self) -> Incomplete:
    #     """
    #     Retrieve the current LP solution value of matrix variable.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def getAvgSol(self) -> Incomplete:
    #     """
    #     Get the weighted average solution of matrix variable in all feasible primal solutions found.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def varMayRound(self, direction: Incomplete = "down") -> Incomplete:
    #     """
    #     Checks whether it is possible to round variable up / down and stay feasible for the relaxation.

    #     Parameters
    #     ----------
    #     direction : str
    #         "up" or "down"

    #     Returns
    #     -------
    #     np.ndarray
    #         An array of bools

    #     """

# TODO: make Constraint generic over type of `data`
# This can't be done only in the stubs as the Constraint class
# is not generic and thus can't be indexed by a type variable.
# Attempted in commit 5897e49
class Constraint:
    data: Any
    @property
    def name(self) -> str: ...
    def isOriginal(self) -> bool: ...
    def isInitial(self) -> bool: ...
    def isSeparated(self) -> bool: ...
    def isEnforced(self) -> bool: ...
    def isChecked(self) -> bool: ...
    def isPropagated(self) -> bool: ...
    def isLocal(self) -> bool: ...
    def isModifiable(self) -> bool: ...
    def isDynamic(self) -> bool: ...
    def isRemovable(self) -> bool: ...
    def isStickingAtNode(self) -> bool: ...
    def isActive(self) -> bool: ...
    def isLinear(self) -> bool: ...
    def isNonlinear(self) -> bool: ...
    def getConshdlrName(self) -> str: ...
    @override
    def __hash__(self) -> int: ...

# none of the defined methods work in v5.4.1
# pretend they don't exist
class MatrixConstraint(_ObjectArray):
    ...
    # def isInitial(self) -> Incomplete:
    #     """
    #     Returns True if the relaxation of the constraint should be in the initial LP.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def isSeparated(self) -> Incomplete:
    #     """
    #     Returns True if constraint should be separated during LP processing.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def isEnforced(self) -> Incomplete:
    #     """
    #     Returns True if constraint should be enforced during node processing.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def isChecked(self) -> Incomplete:
    #     """
    #     Returns True if constraint should be checked for feasibility.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def isPropagated(self) -> Incomplete:
    #     """
    #     Returns True if constraint should be propagated during node processing.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def isLocal(self) -> Incomplete:
    #     """
    #     Returns True if constraint is only locally valid or not added to any (sub)problem.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def isModifiable(self) -> Incomplete:
    #     """
    #     Returns True if constraint is modifiable (subject to column generation).

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def isDynamic(self) -> Incomplete:
    #     """
    #     Returns True if constraint is subject to aging.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def isRemovable(self) -> Incomplete:
    #     """
    #     Returns True if constraint's relaxation should be removed from the LP due to aging or cleanup.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def isStickingAtNode(self) -> Incomplete:
    #     """
    #     Returns True if constraint is only locally valid or not added to any (sub)problem.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def isActive(self) -> Incomplete:
    #     """
    #     Returns True iff constraint is active in the current node.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def isLinear(self) -> Incomplete:
    #     """
    #     Returns True if constraint is linear

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def isNonlinear(self) -> Incomplete:
    #     """
    #     Returns True if constraint is nonlinear.

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    # def getConshdlrName(self) -> Incomplete:
    #     """
    #     Return the constraint handler's name.

    #     Returns
    #     -------
    #     np.ndarray

    #     """

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
    ) -> None: ...
    @override
    def __hash__(self) -> int: ...
    @staticmethod
    def from_ptr(capsule: CapsuleType, take_ownership: bool) -> Model: ...
    def to_ptr(self, give_ownership: bool) -> CapsuleType: ...
    def includeDefaultPlugins(self) -> None: ...
    def createProbBasic(self, problemName: str = "model") -> None: ...
    def freeProb(self) -> None: ...
    def freeTransform(self) -> None: ...
    def version(self) -> float: ...
    def getMajorVersion(self) -> int: ...
    def getMinorVersion(self) -> int: ...
    def getTechVersion(self) -> int: ...
    def printVersion(self) -> None: ...
    def printExternalCodeVersions(self) -> None: ...
    def getProbName(self) -> str: ...
    def getTotalTime(self) -> float: ...
    def getSolvingTime(self) -> float: ...
    def getReadingTime(self) -> float: ...
    def getPresolvingTime(self) -> float: ...
    def getNLPIterations(self) -> int: ...
    def getNNodes(self) -> int: ...
    def getNTotalNodes(self) -> int: ...
    def getNFeasibleLeaves(self) -> int: ...
    def getNInfeasibleLeaves(self) -> int: ...
    def getNLeaves(self) -> int: ...
    def getNChildren(self) -> int: ...
    def getNSiblings(self) -> int: ...
    def getCurrentNode(self) -> Node: ...
    def getGap(self) -> float: ...
    def getDepth(self) -> int: ...
    def cutoffNode(self, node: Node) -> None: ...
    def infinity(self) -> float: ...
    def epsilon(self) -> float: ...
    def feastol(self) -> float: ...
    def feasFrac(self, value: float) -> float: ...
    def frac(self, value: float) -> float: ...
    def feasFloor(self, value: float) -> float: ...
    def feasCeil(self, value: float) -> float: ...
    def feasRound(self, value: float) -> float: ...
    def isZero(self, value: float) -> bool: ...
    def isFeasZero(self, value: float) -> bool: ...
    def isInfinity(self, value: float) -> bool: ...
    def isFeasNegative(self, value: float) -> bool: ...
    def isFeasIntegral(self, value: float) -> bool: ...
    def isEQ(self, val1: float, val2: float) -> bool: ...
    def isFeasEQ(self, val1: float, val2: float) -> bool: ...
    def isLE(self, val1: float, val2: float) -> bool: ...
    def isLT(self, val1: float, val2: float) -> bool: ...
    def isGE(self, val1: float, val2: float) -> bool: ...
    def isGT(self, val1: float, val2: float) -> bool: ...
    def getCondition(self, exact: bool = False) -> float: ...
    def enableReoptimization(self, enable: bool = True) -> None: ...
    def lpiGetIterations(self) -> int: ...
    def setMinimize(self) -> None: ...
    def setMaximize(self) -> None: ...
    def setObjlimit(self, objlimit: float) -> None: ...
    def getObjlimit(self) -> float: ...
    def setObjective(
        self,
        expr: Expr | SupportsFloat,
        sense: L["minimize", "maximize"] = "minimize",
        clear: bool | L["true"] = "true",  # TODO: typo?
    ) -> None: ...
    def getObjective(self) -> Expr: ...
    def addObjoffset(self, offset: float, solutions: bool = False) -> None: ...
    def getObjoffset(self, original: bool = True) -> float: ...
    def setObjIntegral(self) -> None: ...
    def getLocalEstimate(self, original: bool = False) -> float: ...
    def setPresolve(self, setting: PY_SCIP_PARAMSETTING) -> None: ...
    def setProbName(self, name: str) -> None: ...
    def setSeparating(self, setting: PY_SCIP_PARAMSETTING) -> None: ...
    def setHeuristics(self, setting: PY_SCIP_PARAMSETTING) -> None: ...
    def setHeurTiming(self, heurname: str, heurtiming: PY_SCIP_HEURTIMING) -> None: ...
    def getHeurTiming(self, heurname: str) -> PY_SCIP_HEURTIMING: ...
    def disablePropagation(self, onlyroot: bool = False) -> None: ...
    def printProblem(
        self, ext: str = ".cip", trans: bool = False, genericnames: bool = False
    ) -> None: ...
    def writeProblem(
        self,
        filename: str | os.PathLike[str] = "model.cip",
        trans: bool = False,
        genericnames: bool = False,
        verbose: bool = True,
    ) -> None: ...
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
    ) -> Variable: ...
    def addMatrixVar(
        self,
        shape: int | tuple,  # type: ignore[type-arg]
        name: str | np.ndarray = "",  # type: ignore[type-arg]
        vtype: str | np.ndarray = "C",  # type: ignore[type-arg]
        lb: float | np.ndarray | None = 0.0,  # type: ignore[type-arg]
        ub: float | np.ndarray | None = None,  # type: ignore[type-arg]
        obj: float | np.ndarray | None = 0.0,  # type: ignore[type-arg]
        pricedVar: bool | np.ndarray = False,  # type: ignore[type-arg]
        pricedVarScore: float | np.ndarray | None = 1.0,  # type: ignore[type-arg]
    ) -> MatrixVariable: ...
    def getTransformedVar(self, var: Variable) -> Variable: ...
    def addVarLocks(self, var: Variable, nlocksdown: int, nlocksup: int) -> None: ...
    def fixVar(self, var: Variable, val: float) -> tuple[bool, bool]: ...
    def delVar(self, var: Variable) -> bool: ...
    def tightenVarLb(
        self, var: Variable, lb: float, force: bool = False
    ) -> tuple[bool, bool]: ...
    def tightenVarUb(
        self, var: Variable, ub: float, force: bool = False
    ) -> tuple[bool, bool]: ...
    def tightenVarUbGlobal(
        self, var: Variable, ub: float, force: bool = False
    ) -> tuple[bool, bool]: ...
    def tightenVarLbGlobal(
        self, var: Variable, lb: float, force: bool = False
    ) -> tuple[bool, bool]: ...
    def chgVarLb(self, var: Variable, lb: float | None) -> None: ...
    def chgVarUb(self, var: Variable, ub: float | None) -> None: ...
    def chgVarLbGlobal(self, var: Variable, lb: float | None) -> None: ...
    def chgVarUbGlobal(self, var: Variable, ub: float | None) -> None: ...
    def chgVarLbNode(self, node: Node, var: Variable, lb: float | None) -> None: ...
    def chgVarUbNode(self, node: Node, var: Variable, ub: float | None) -> None: ...
    def chgVarType(self, var: Variable, vtype: _VTypes) -> None: ...
    def getVars(self, transformed: bool = False) -> list[Variable]: ...
    def getNVars(self, transformed: bool = True) -> int: ...
    def getNIntVars(self) -> int: ...
    def getNBinVars(self) -> int: ...
    def getNImplVars(self) -> int: ...
    def getNContVars(self) -> int: ...
    def getVarDict(self, transformed: bool = False) -> dict[str, float]: ...
    def updateNodeLowerbound(self, node: Node, lb: float) -> None: ...
    def relax(self) -> None: ...
    def getBestChild(self) -> Node | None: ...
    def getBestSibling(self) -> Node | None: ...
    def getPrioChild(self) -> Node | None: ...
    def getPrioSibling(self) -> Node | None: ...
    def getBestLeaf(self) -> Node | None: ...
    def getBestNode(self) -> Node | None: ...
    def getBestboundNode(self) -> Node | None: ...
    def getOpenNodes(self) -> tuple[list[Node], list[Node], list[Node]]: ...
    def repropagateNode(self, node: Node) -> None: ...
    def getLPSolstat(self) -> PY_SCIP_LPSOLSTAT: ...
    def constructLP(self) -> bool: ...
    def getLPObjVal(self) -> float: ...
    def getLPColsData(self) -> list[Column]: ...
    def getLPRowsData(self) -> list[Row]: ...
    def getNLPRows(self) -> int: ...
    def getNLPCols(self) -> int: ...
    def getLPBasisInd(self) -> list[int]: ...
    def getLPBInvRow(self, row: int) -> list[float]: ...
    def getLPBInvARow(self, row: int) -> list[float]: ...
    def isLPSolBasic(self) -> bool: ...
    def allColsInLP(self) -> bool: ...
    def getColRedCost(self, col: Column) -> float: ...
    def createEmptyRowSepa(
        self,
        sepa: Sepa,
        name: str = "row",
        lhs: float | None = 0.0,
        rhs: float | None = None,
        local: bool = True,
        modifiable: bool = False,
        removable: bool = True,
    ) -> Row: ...
    def createEmptyRowUnspec(
        self,
        name: str = "row",
        lhs: float | None = 0.0,
        rhs: float | None = None,
        local: bool = True,
        modifiable: bool = False,
        removable: bool = True,
    ) -> Row: ...
    def getRowActivity(self, row: Row) -> float: ...
    def getRowLPActivity(self, row: Row) -> float: ...
    def releaseRow(self, row: Row) -> None: ...
    def cacheRowExtensions(self, row: Row) -> None: ...
    def flushRowExtensions(self, row: Row) -> None: ...
    def addVarToRow(self, row: Row, var: Variable, value: float) -> None: ...
    def printRow(self, row: Row) -> None: ...
    def getRowNumIntCols(self, row: Row) -> int: ...
    def getRowObjParallelism(self, row: Row) -> float: ...
    def getRowParallelism(
        self, row1: Row, row2: Row, orthofunc: L["d", "e", 100, 101] = 101
    ) -> float: ...
    def getRowDualSol(self, row: Row) -> float: ...
    def addPoolCut(self, row: Row) -> None: ...
    def getCutEfficacy(self, cut: Row, sol: Solution | None = None) -> float: ...
    def isCutEfficacious(self, cut: Row, sol: Solution | None = None) -> bool: ...
    def getCutLPSolCutoffDistance(self, cut: Row, sol: Solution) -> float: ...
    def addCut(self, cut: Row, forcecut: bool = False) -> bool: ...
    def getNCuts(self) -> int: ...
    def getNCutsApplied(self) -> int: ...
    def getNSepaRounds(self) -> int: ...
    def separateSol(
        self,
        sol: Solution | None = None,
        pretendroot: bool = False,
        allowlocal: bool = True,
        onlydelayed: bool = False,
    ) -> tuple[bool, bool]: ...
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
    ) -> Constraint: ...
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
    ) -> Constraint: ...
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
    ) -> list[Constraint]: ...
    def addMatrixCons(
        self,
        cons: MatrixExprCons,
        name: str | np.ndarray = "",  # type: ignore[type-arg]
        initial: bool | np.ndarray = True,  # type: ignore[type-arg]
        separate: bool | np.ndarray = True,  # type: ignore[type-arg]
        enforce: bool | np.ndarray = True,  # type: ignore[type-arg]
        check: bool | np.ndarray = True,  # type: ignore[type-arg]
        propagate: bool | np.ndarray = True,  # type: ignore[type-arg]
        local: bool | np.ndarray = False,  # type: ignore[type-arg]
        modifiable: bool | np.ndarray = False,  # type: ignore[type-arg]
        dynamic: bool | np.ndarray = False,  # type: ignore[type-arg]
        removable: bool | np.ndarray = False,  # type: ignore[type-arg]
        stickingatnode: bool | np.ndarray = False,  # type: ignore[type-arg]
    ) -> MatrixConstraint: ...
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
    ) -> Constraint: ...
    def addConsElemDisjunction(
        self, disj_cons: Constraint, cons: Constraint
    ) -> Constraint: ...
    def getConsNVars(self, constraint: Constraint) -> int: ...
    def getConsVars(self, constraint: Constraint) -> list[Variable]: ...
    def printCons(self, constraint: Constraint) -> None: ...
    def addExprNonlinear(
        self, cons: Constraint, expr: Expr | GenExpr[Any], coef: float
    ) -> None: ...
    def addConsCoeff(self, cons: Constraint, var: Variable, coeff: float) -> None: ...
    def addConsNode(
        self, node: Node, cons: Constraint, validnode: Node | None = None
    ) -> None: ...
    def addConsLocal(self, cons: Constraint, validnode: Node | None = None) -> None: ...
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
    ) -> Constraint: ...
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
    ) -> Constraint: ...
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
    ) -> Constraint: ...
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
    ) -> Constraint: ...
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
    ) -> Constraint: ...
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
    ) -> Constraint: ...
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
    ) -> Constraint: ...
    def getSlackVarIndicator(self, cons: Constraint) -> Variable: ...
    def addPyCons(self, cons: Constraint) -> None: ...
    def addVarSOS1(self, cons: Constraint, var: Variable, weight: float) -> None: ...
    def appendVarSOS1(self, cons: Constraint, var: Variable) -> None: ...
    def addVarSOS2(self, cons: Constraint, var: Variable, weight: float) -> None: ...
    def appendVarSOS2(self, cons: Constraint, var: Variable) -> None: ...
    def setInitial(self, cons: Constraint, newInit: bool) -> None: ...
    def setRemovable(self, cons: Constraint, newRem: bool) -> None: ...
    def setEnforced(self, cons: Constraint, newEnf: bool) -> None: ...
    def setCheck(self, cons: Constraint, newCheck: bool) -> None: ...
    def chgRhs(self, cons: Constraint, rhs: float | None) -> None: ...
    def chgLhs(self, cons: Constraint, lhs: float | None) -> None: ...
    def getRhs(self, cons: Constraint) -> float: ...
    def getLhs(self, cons: Constraint) -> None: ...
    def chgCoefLinear(self, cons: Constraint, var: Variable, value: float) -> None: ...
    def delCoefLinear(self, cons: Constraint, var: Variable) -> None: ...
    def addCoefLinear(self, cons: Constraint, var: Variable, value: float) -> None: ...
    def getActivity(self, cons: Constraint, sol: Solution | None = None) -> float: ...
    def getSlack(
        self,
        cons: Constraint,
        sol: Solution | None = None,
        side: L["lhs", "rhs"] | None = None,
    ) -> float: ...
    def getTransformedCons(self, cons: Constraint) -> Constraint: ...
    def isNLPConstructed(self) -> bool: ...
    def getNNlRows(self) -> int: ...
    def getNlRows(self) -> list[NLRow]: ...
    def getNlRowSolActivity(
        self, nlrow: NLRow, sol: Solution | None = None
    ) -> float: ...
    def getNlRowSolFeasibility(
        self, nlrow: NLRow, sol: Solution | None = None
    ) -> float: ...
    def getNlRowActivityBounds(self, nlrow: NLRow) -> tuple[float, float]: ...
    def printNlRow(self, nlrow: NLRow) -> None: ...
    def checkQuadraticNonlinear(self, cons: Constraint) -> bool: ...
    def getTermsQuadratic(
        self, cons: Constraint
    ) -> tuple[
        list[tuple[Variable, Variable, float]],
        list[tuple[Variable, float, float]],
        list[tuple[Variable, float]],
    ]: ...
    def setRelaxSolVal(self, var: Variable, val: float) -> None: ...
    def getConss(self, transformed: bool = True) -> list[Constraint]: ...
    def getNConss(self, transformed: bool = True) -> int: ...
    def delCons(self, cons: Constraint) -> None: ...
    def delConsLocal(self, cons: Constraint) -> None: ...
    def getValsLinear(self, cons: Constraint) -> dict[str, float]: ...
    def getRowLinear(self, cons: Constraint) -> Row: ...
    def getDualsolLinear(self, cons: Constraint) -> float: ...
    @deprecated(
        "model.getDualMultiplier(cons) is deprecated: please use model.getDualsolLinear(cons)"
    )
    def getDualMultiplier(self, cons: Constraint) -> float: ...
    def getDualfarkasLinear(self, cons: Constraint) -> float: ...
    def getVarRedcost(self, var: Variable) -> float: ...
    def getDualSolVal(
        self, cons: Constraint, boundconstraint: bool = False
    ) -> float: ...
    def optimize(self) -> None: ...
    def optimizeNogil(self) -> None: ...
    def solveConcurrent(self) -> None: ...
    def presolve(self) -> None: ...
    def initBendersDefault(self, subproblems: Model | dict[Any, Model]) -> None: ...
    def computeBestSolSubproblems(self) -> None: ...
    def freeBendersSubproblems(self) -> None: ...
    def updateBendersLowerbounds(
        self, lowerbounds: dict[int, float], benders: Benders | None = None
    ) -> None: ...
    def activateBenders(self, benders: Benders, nsubproblems: int) -> None: ...
    def addBendersSubproblem(self, benders: Benders, subproblem: Model) -> None: ...
    def setBendersSubproblemIsConvex(
        self, benders: Benders, probnumber: int, isconvex: bool = True
    ) -> None: ...
    def setupBendersSubproblem(
        self,
        probnumber: int,
        benders: Benders | None = None,
        solution: Solution | None = None,
        checktype: PY_SCIP_BENDERSENFOTYPE = PY_SCIP_BENDERSENFOTYPE.LP,
    ) -> None: ...
    def solveBendersSubproblem(
        self,
        probnumber: int,
        solvecip: bool,
        benders: Benders | None = None,
        solution: Solution | None = None,
    ) -> tuple[bool, float | None]: ...
    def getBendersSubproblem(
        self, probnumber: int, benders: Benders | None = None
    ) -> Model: ...
    def getBendersVar(
        self, var: Variable, benders: Benders | None = None, probnumber: int = -1
    ) -> Variable | None: ...
    def getBendersAuxiliaryVar(
        self, probnumber: int, benders: Benders | None = None
    ) -> Variable: ...
    def checkBendersSubproblemOptimality(
        self, solution: Solution, probnumber: int, benders: Benders | None = None
    ) -> bool: ...
    def includeBendersDefaultCuts(self, benders: Benders) -> None: ...
    def includeEventhdlr(self, eventhdlr: Eventhdlr, name: str, desc: str) -> None: ...
    def includePricer(
        self,
        pricer: Pricer,
        name: str,
        desc: str,
        priority: int = 1,
        delay: bool = True,
    ) -> None: ...
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
    ) -> None: ...
    def copyLargeNeighborhoodSearch(
        self, to_fix: Sequence[Variable], fix_vals: Sequence[float]
    ) -> Model: ...
    def translateSubSol(
        self, sub_model: Model, sol: Solution, heur: Heur
    ) -> Solution: ...
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
    ) -> Constraint: ...
    def includePresol(
        self,
        presol: Presol,
        name: str,
        desc: str,
        priority: int,
        maxrounds: int,
        timing: PY_SCIP_PRESOLTIMING = PY_SCIP_PRESOLTIMING.FAST,
    ) -> None: ...
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
    ) -> None: ...
    def includeReader(self, reader: Reader, name: str, desc: str, ext: str) -> None: ...
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
    ) -> None: ...
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
    ) -> None: ...
    def includeRelax(
        self, relax: Relax, name: str, desc: str, priority: int = 10000, freq: int = 1
    ) -> None: ...
    def includeCutsel(
        self, cutsel: Cutsel, name: str, desc: str, priority: int
    ) -> None: ...
    def includeBranchrule(
        self,
        branchrule: Branchrule,
        name: str,
        desc: str,
        priority: int,
        maxdepth: int,
        maxbounddist: float,
    ) -> None: ...
    def includeNodesel(
        self,
        nodesel: Nodesel,
        name: str,
        desc: str,
        stdpriority: int,
        memsavepriority: int,
    ) -> None: ...
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
    ) -> None: ...
    def includeBenderscut(
        self,
        benders: Benders,
        benderscut: Benderscut,
        name: str,
        desc: str,
        priority: int = 1,
        islpcut: bool = True,
    ) -> None: ...
    def getLPBranchCands(
        self,
    ) -> tuple[list[Variable], list[float], list[float], int, int, int]: ...
    def getPseudoBranchCands(self) -> tuple[list[Variable], int, int]: ...
    def branchVar(self, variable: Variable) -> tuple[Node, Node | None, Node]: ...
    def branchVarVal(
        self, variable: Variable, value: float
    ) -> tuple[Node, Node | None, Node]: ...
    def calcNodeselPriority(
        self, variable: Variable, branchdir: PY_SCIP_BRANCHDIR, targetvalue: float
    ) -> int: ...
    def calcChildEstimate(self, variable: Variable, targetvalue: float) -> float: ...
    def createChild(self, nodeselprio: int, estimate: float) -> Node: ...
    def startDive(self) -> None: ...
    def endDive(self) -> None: ...
    def chgVarObjDive(self, var: Variable, newobj: float) -> None: ...
    def chgVarLbDive(self, var: Variable, newbound: float) -> None: ...
    def chgVarUbDive(self, var: Variable, newbound: float) -> None: ...
    def getVarLbDive(self, var: Variable) -> float: ...
    def getVarUbDive(self, var: Variable) -> float: ...
    def chgRowLhsDive(self, row: Row, newlhs: float) -> None: ...
    def chgRowRhsDive(self, row: Row, newrhs: float) -> None: ...
    def addRowDive(self, row: Row) -> None: ...
    def solveDiveLP(self, itlim: int = -1) -> tuple[bool, bool]: ...
    def inRepropagation(self) -> bool: ...
    def startProbing(self) -> None: ...
    def endProbing(self) -> None: ...
    def newProbingNode(self) -> None: ...
    def backtrackProbing(self, probingdepth: int) -> None: ...
    def getProbingDepth(self) -> int: ...
    def chgVarObjProbing(self, var: Variable, newobj: float) -> None: ...
    def chgVarLbProbing(self, var: Variable, lb: float | None) -> None: ...
    def chgVarUbProbing(self, var: Variable, ub: float | None) -> None: ...
    def fixVarProbing(self, var: Variable, fixedval: float) -> None: ...
    def isObjChangedProbing(self) -> bool: ...
    def inProbing(self) -> bool: ...
    def solveProbingLP(self, itlim: int = -1) -> tuple[bool, bool]: ...
    def applyCutsProbing(self) -> bool: ...
    def propagateProbing(self, maxproprounds: int) -> tuple[bool, int]: ...
    def interruptSolve(self) -> None: ...
    def restartSolve(self) -> None: ...
    def writeLP(self, filename: str | os.PathLike[str] = "LP.lp") -> None: ...
    def createSol(self, heur: Heur | None = None, initlp: bool = False) -> Solution: ...
    def createPartialSol(self, heur: Heur | None = None) -> Solution: ...
    def createOrigSol(self, heur: Heur | None = None) -> Solution: ...
    def printBestSol(self, write_zeros: bool = False) -> None: ...
    def printSol(
        self, solution: Solution | None = None, write_zeros: bool = False
    ) -> None: ...
    def writeBestSol(
        self,
        filename: str | bytes | os.PathLike[AnyStr] = "origprob.sol",
        write_zeros: bool = False,
    ) -> None: ...
    def writeBestTransSol(
        self,
        filename: str | bytes | os.PathLike[AnyStr] = "transprob.sol",
        write_zeros: bool = False,
    ) -> None: ...
    def writeSol(
        self,
        solution: Solution,
        filename: str | bytes | os.PathLike[AnyStr] = "origprob.sol",
        write_zeros: bool = False,
    ) -> None: ...
    def writeTransSol(
        self,
        solution: Solution,
        filename: str | bytes | os.PathLike[AnyStr] = "transprob.sol",
        write_zeros: bool = False,
    ) -> None: ...
    def readSol(self, filename: str | os.PathLike[str]) -> None: ...
    def readSolFile(self, filename: str | os.PathLike[str]) -> Solution: ...
    def setSolVal(self, solution: Solution, var: Variable, val: float) -> None: ...
    def trySol(
        self,
        solution: Solution,
        printreason: bool = True,
        completely: bool = False,
        checkbounds: bool = True,
        checkintegrality: bool = True,
        checklprows: bool = True,
        free: bool = True,
    ) -> bool: ...
    def checkSol(
        self,
        solution: Solution,
        printreason: bool = True,
        completely: bool = False,
        checkbounds: bool = True,
        checkintegrality: bool = True,
        checklprows: bool = True,
        original: bool = False,
    ) -> bool: ...
    def addSol(self, solution: Solution, free: bool = True) -> bool: ...
    def freeSol(self, solution: Solution) -> None: ...
    def getNSols(self) -> int: ...
    def getNSolsFound(self) -> int: ...
    def getNLimSolsFound(self) -> int: ...
    def getNBestSolsFound(self) -> int: ...
    def getSols(self) -> list[Solution]: ...
    def getBestSol(self) -> Solution | None: ...
    def getSolObjVal(self, sol: Solution | None, original: bool = True) -> float: ...
    def getSolTime(self, sol: Solution) -> float: ...
    def getObjVal(self, original: bool = True) -> float: ...
    def getSolVal(self, sol: Solution | None, expr: Expr) -> float: ...
    def getVal(self, expr: Expr) -> float: ...
    def hasPrimalRay(self) -> bool: ...
    def getPrimalRayVal(self, var: Variable) -> float: ...
    def getPrimalRay(self) -> list[float]: ...
    def getPrimalbound(self) -> float: ...
    def getDualbound(self) -> float: ...
    def getDualboundRoot(self) -> float: ...
    def writeName(self, var: Variable) -> None: ...
    def getStage(self) -> PY_SCIP_STAGE: ...
    def getStageName(self) -> str: ...
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
    ]: ...
    def getObjectiveSense(self) -> L["maximize", "minimize", "unknown"]: ...
    def catchEvent(
        self, eventtype: PY_SCIP_EVENTTYPE, eventhdlr: Eventhdlr
    ) -> None: ...
    def dropEvent(self, eventtype: PY_SCIP_EVENTTYPE, eventhdlr: Eventhdlr) -> None: ...
    def catchVarEvent(
        self, var: Variable, eventtype: PY_SCIP_EVENTTYPE, eventhdlr: Eventhdlr
    ) -> None: ...
    def dropVarEvent(
        self, var: Variable, eventtype: PY_SCIP_EVENTTYPE, eventhdlr: Eventhdlr
    ) -> None: ...
    def catchRowEvent(
        self, row: Row, eventtype: PY_SCIP_EVENTTYPE, eventhdlr: Eventhdlr
    ) -> None: ...
    def dropRowEvent(
        self, row: Row, eventtype: PY_SCIP_EVENTTYPE, eventhdlr: Eventhdlr
    ) -> None: ...
    def printStatistics(self) -> None: ...
    def writeStatistics(
        self, filename: str | bytes | os.PathLike[AnyStr] = "origprob.stats"
    ) -> None: ...
    def getNLPs(self) -> int: ...
    def hideOutput(self, quiet: bool = True) -> None: ...
    def redirectOutput(self) -> None: ...
    def setLogfile(self, path: str | None) -> None: ...
    def setBoolParam(self, name: str, value: float) -> None: ...
    def setIntParam(self, name: str, value: int) -> None: ...
    def setLongintParam(self, name: str, value: int) -> None: ...
    def setRealParam(self, name: str, value: float) -> None: ...
    def setCharParam(self, name: str, value: str) -> None: ...
    def setStringParam(self, name: str, value: str) -> None: ...
    def setParam(self, name: str, value: object) -> None: ...
    def getParam(self, name: str) -> bool | float | str: ...
    def getParams(self) -> dict[str, bool | float | str]: ...
    def setParams(self, params: Mapping[str, bool | float | str]) -> None: ...
    def readParams(self, file: str | os.PathLike[str]) -> None: ...
    def writeParams(
        self,
        filename: str | os.PathLike[str] = "param.set",
        comments: bool = True,
        onlychanged: bool = True,
        verbose: bool = True,
    ) -> None: ...
    def resetParam(self, name: str) -> None: ...
    def resetParams(self) -> None: ...
    def setEmphasis(
        self, paraemphasis: PY_SCIP_PARAMEMPHASIS, quiet: bool = True
    ) -> None: ...
    def readProblem(
        self, filename: str | os.PathLike[str], extension: str | None = None
    ) -> None: ...
    def count(self) -> None: ...
    def getNReaders(self) -> int: ...
    def getNCountedSols(self) -> int: ...
    def setParamsCountsols(self) -> None: ...
    def freeReoptSolve(self) -> None: ...
    def chgReoptObjective(
        self, coeffs: Expr, sense: L["minimize", "maximize"] = "minimize"
    ) -> None: ...
    def chgVarBranchPriority(self, var: Variable, priority: int) -> None: ...
    def startStrongbranch(self) -> None: ...
    def endStrongbranch(self) -> None: ...
    def getVarStrongbranchLast(
        self, var: Variable
    ) -> tuple[float, float, bool, bool, float, float]: ...
    def getVarStrongbranchNode(self, var: Variable) -> int: ...
    def getVarStrongbranch(
        self,
        var: Variable,
        itlim: int,
        idempotent: bool = False,
        integral: bool = False,
    ) -> tuple[float, float, bool, bool, bool, bool, bool, bool, bool]: ...
    def updateVarPseudocost(
        self, var: Variable, valdelta: float, objdelta: float, weight: float
    ) -> None: ...
    def getBranchScoreMultiple(self, var: Variable, gains: list[float]) -> float: ...
    def getTreesizeEstimation(self) -> float: ...
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
    ]: ...

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

def readStatistics(filename: str | bytes | os.PathLike[AnyStr]) -> Statistics: ...
def is_memory_freed() -> bool: ...
def print_memory_in_use() -> None: ...

__test__: dict[Any, Any]
