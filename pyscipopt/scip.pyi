import dataclasses
from typing import (
    Any,
    ClassVar,
    Generic,
    Iterable,
    Iterator,
    Sequence,
    SupportsFloat,
    TypeVar,
    overload,
)

from _typeshed import Incomplete
from typing_extensions import Literal as L
from typing_extensions import Self, TypeAlias, override

_VTypes: TypeAlias = L[
    "C", "CONTINUOUS",
    "B", "BINARY",
    "I", "INTEGER",
    "M", "IMPLINT"
]  # fmt: skip

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
    def __hash__(self, /) -> int: ...
    @override
    def __eq__(self, other: Term, /) -> bool: ...
    def __len__(self, /) -> int: ...
    def __add__(self, other: Term, /) -> Term: ...

CONST: Term

@overload
def buildGenExprObj(expr: Expr) -> SumExpr: ...
@overload
def buildGenExprObj(expr: GenExpr[_OpT]) -> GenExpr[_OpT]: ...
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
    def __abs__(self, /) -> UnaryExpr: ...
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
    def __rtruediv__(self, other: SupportsFloat, /) -> Expr: ...
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
    def __eq__(self, other: Expr | SupportsFloat | GenExpr[Any], /) -> ExprCons: ...
    def __ge__(self, other: Expr | SupportsFloat | GenExpr[Any], /) -> ExprCons: ...
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
def quicksum(termlist: Iterable[Expr | SupportsFloat]) -> Expr: ...
@overload
def quicksum(termlist: Iterable[Expr | SupportsFloat | GenExpr]) -> SumExpr: ...
@overload
def quickprod(termlist: Iterable[Expr | SupportsFloat]) -> Expr: ...
@overload
def quickprod(termlist: Iterable[Expr | SupportsFloat | GenExpr]) -> ProdExpr: ...

class GenExpr(Generic[_OpT]):
    _op: _OpT
    children: list[GenExpr[Any]]
    def __init__(self, /) -> None: ...
    def __abs__(self, /) -> UnaryExpr: ...
    def __add__(self, other: Expr | float | GenExpr[Any], /) -> SumExpr: ...
    def __mul__(self, other: Expr | float | GenExpr[Any], /) -> ProdExpr: ...
    def __pow__(self, other: SupportsFloat, mod: Any = None, /) -> PowExpr: ...
    def __truediv__(self, other: Expr | float | GenExpr[Any], /) -> ProdExpr: ...
    def __rtruediv__(self, other: float, /) -> ProdExpr: ...
    def __neg__(self, /) -> ProdExpr: ...
    def __sub__(self, other: Expr | float | GenExpr[Any], /) -> SumExpr: ...
    def __radd__(self, other: float, /) -> SumExpr: ...
    def __rmul__(self, other: float, /) -> SumExpr: ...
    def __rsub__(self, other: float, /) -> SumExpr: ...
    @override
    def __eq__(self, other: Expr | SupportsFloat | GenExpr[Any], /) -> ExprCons: ...
    def __ge__(self, other: Expr | SupportsFloat | GenExpr[Any], /) -> ExprCons: ...
    def __le__(self, other: Expr | SupportsFloat | GenExpr[Any], /) -> ExprCons: ...
    def degree(self, /) -> float: ...
    def getOp(self, /) -> _OpT: ...

class SumExpr(GenExpr[Operator.add]):
    constant: float
    coefs: list[float]
    def __init__(self, /) -> None: ...

class ProdExpr(GenExpr[Operator.prod]):
    constant: float
    def __init__(self, /) -> None: ...

class VarExpr(GenExpr[Operator.varidx]):
    var: Variable
    children: list[Variable]
    def __init__(self, /, var: Variable) -> None: ...

class PowExpr(GenExpr[Operator.power]):
    expo: float
    def __init__(self, /) -> None: ...

_UnaryOpT = TypeVar("_UnaryOpT", bound=_UnaryOp)

class UnaryExpr(GenExpr[_UnaryOpT]):
    def __init__(self, op: _UnaryOpT, expr: GenExpr[Any]) -> None: ...

class Constant(GenExpr[Operator.const]):
    number: float
    def __init__(self, /, number: float) -> None: ...

def exp(expr: Expr | SupportsFloat | GenExpr[Any]) -> UnaryExpr[L["exp"]]: ...
def log(expr: Expr | SupportsFloat | GenExpr[Any]) -> UnaryExpr[L["log"]]: ...
def sqrt(expr: Expr | SupportsFloat | GenExpr[Any]) -> UnaryExpr[L["sqrt"]]: ...
def sin(expr: Expr | SupportsFloat | GenExpr[Any]) -> UnaryExpr[L["sin"]]: ...
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

class Benders:
    model: Model
    name: str
    def benderscreatesub(self, probnumber: int) -> Incomplete: ...
    def bendersexit(self) -> None: ...
    def bendersexitpre(self) -> None: ...
    def bendersexitsol(self) -> None: ...
    def bendersfree(self) -> None: ...
    def bendersfreesub(self, probnumber: int) -> Incomplete: ...
    def bendersgetvar(self, variable: Variable, probnumber: int) -> Variable: ...
    def bendersinit(self) -> None: ...
    def bendersinitpre(self) -> None: ...
    def bendersinitsol(self) -> None: ...
    def benderspostsolve(
        self,
        solution: Incomplete,
        enfotype: Incomplete,
        mergecandidates: Incomplete,
        npriomergecands: Incomplete,
        checkint: Incomplete,
        infeasible: Incomplete,
    ) -> None: ...
    def benderspresubsolve(
        self, solution: Incomplete, enfotype: Incomplete, checkint: Incomplete
    ) -> Incomplete: ...
    def benderssolvesub(self, solution: Incomplete, probnumber: int) -> Incomplete: ...
    def benderssolvesubconvex(
        self, solution: Incomplete, probnumber: Incomplete, onlyconvex: Incomplete
    ) -> Incomplete: ...

################
# benderscut.pxi
################

class Benderscut:
    benders: Benders
    model: Model
    name: str
    def benderscutexec(
        self, solution: Incomplete, probnumber: Incomplete, enfotype: Incomplete
    ) -> Incomplete: ...
    def benderscutexit(self) -> None: ...
    def benderscutexitsol(self) -> None: ...
    def benderscutfree(self) -> None: ...
    def benderscutinit(self) -> None: ...
    def benderscutinitsol(self) -> None: ...

################
# branchrule.pxi
################

class Branchrule:
    model: Model
    def branchexecext(self, allowaddcons: bool) -> Incomplete: ...
    def branchexeclp(self, allowaddcons: bool) -> Incomplete: ...
    def branchexecps(self, allowaddcons: bool) -> Incomplete: ...
    def branchexit(self) -> None: ...
    def branchexitsol(self) -> None: ...
    def branchfree(self) -> None: ...
    def branchinit(self) -> None: ...
    def branchinitsol(self) -> None: ...

##############
# conshdlr.pxi
##############

class Conshdlr:
    model: Model
    name: str
    def consactive(self, constraint: Constraint) -> None: ...
    def conscheck(
        self,
        constraints: Incomplete,
        solution: Incomplete,
        checkintegrality: Incomplete,
        checklprows: Incomplete,
        printreason: Incomplete,
        completely: Incomplete,
    ) -> Incomplete: ...
    def conscopy(self) -> Incomplete: ...
    def consdeactive(self, constraint: Constraint) -> None: ...
    def consdelete(self, constraint: Constraint) -> None: ...
    def consdelvars(self, constraints: Incomplete) -> None: ...
    def consdisable(self, constraint: Constraint) -> None: ...
    def consenable(self, constraint: Constraint) -> None: ...
    def consenfolp(
        self,
        constraints: Incomplete,
        nusefulconss: Incomplete,
        solinfeasible: Incomplete,
    ) -> None: ...
    def consenfops(
        self,
        constraints: Incomplete,
        nusefulconss: Incomplete,
        solinfeasible: Incomplete,
        objinfeasible: Incomplete,
    ) -> None: ...
    def consenforelax(
        self,
        solution: Incomplete,
        constraints: Incomplete,
        nusefulconss: Incomplete,
        solinfeasible: Incomplete,
    ) -> None: ...
    def consexit(self, constraints: Incomplete) -> None: ...
    def consexitpre(self, constraints: Incomplete) -> None: ...
    def consexitsol(self, constraints: Incomplete, restart: Incomplete) -> None: ...
    def consfree(self) -> None: ...
    def consgetdivebdchgs(self) -> None: ...
    def consgetnvars(self, constraint: Constraint) -> int: ...
    def consgetpermsymgraph(self) -> None: ...
    def consgetsignedpermsymgraph(self) -> None: ...
    def consgetvars(self, constraint: Constraint) -> None: ...
    def consinit(self, constraints: Incomplete) -> None: ...
    def consinitlp(self, constraints: Incomplete) -> None: ...
    def consinitsol(self, constraints: Incomplete) -> None: ...
    def conslock(
        self,
        constraint: Constraint,
        locktype: Incomplete,
        nlockspos: Incomplete,
        nlocksneg: Incomplete,
    ) -> None: ...
    def consparse(self) -> None: ...
    def conspresol(
        self,
        constraints: Incomplete,
        nrounds: Incomplete,
        presoltiming: Incomplete,
        nnewfixedvars: Incomplete,
        nnewaggrvars: Incomplete,
        nnewchgvartypes: Incomplete,
        nnewchgbds: Incomplete,
        nnewholes: Incomplete,
        nnewdelconss: Incomplete,
        nnewaddconss: Incomplete,
        nnewupgdconss: Incomplete,
        nnewchgcoefs: Incomplete,
        nnewchgsides: Incomplete,
        result_dict: Incomplete,
    ) -> None: ...
    def consprint(self, constraint: Constraint) -> None: ...
    def consprop(
        self,
        constraints: Incomplete,
        nusefulconss: Incomplete,
        nmarkedconss: Incomplete,
        proptiming: Incomplete,
    ) -> None: ...
    def consresprop(self) -> None: ...
    def conssepalp(self, constraints: Incomplete, nusefulconss: Incomplete) -> None: ...
    def conssepasol(
        self, constraints: Incomplete, nusefulconss: Incomplete, solution: Incomplete
    ) -> None: ...
    def constrans(self, sourceconstraint: Constraint) -> None: ...
    def consinitpre(self, constraints: Incomplete) -> None: ...

############
# cutsel.pxi
############

class Cutsel:
    model: Incomplete
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def cutselexit(self):
        """executed before the transformed problem is freed"""
    def cutselexitsol(self):
        """executed before the branch-and-bound process is freed"""
    def cutselfree(self):
        """frees memory of cut selector"""
    def cutselinit(self):
        """executed after the problem is transformed. use this call to initialize cut selector data."""
    def cutselinitsol(self):
        """executed when the presolving is finished and the branch-and-bound process is about to begin"""
    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        """first method called in each iteration in the main solving loop."""

###########
# event.pxi
###########

class Eventhdlr:
    model: Incomplete
    name: Incomplete
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def eventcopy(self):
        """sets copy callback for all events of this event handler"""
    def eventdelete(self):
        """sets callback to free specific event data"""
    def eventexec(self, event):
        """calls execution method of event handler"""
    def eventexit(self):
        """calls exit method of event handler"""
    def eventexitsol(self):
        """informs event handler that the branch and bound process data is being freed"""
    def eventfree(self):
        """calls destructor and frees memory of event handler"""
    def eventinit(self):
        """initializes event handler"""
    def eventinitsol(self):
        """informs event handler that the branch and bound process is being started"""

###############
# heuristic.pxi
###############

class Heur:
    model: Incomplete
    name: Incomplete
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def heurexec(self, heurtiming, nodeinfeasible):
        """should the heuristic the executed at the given depth, frequency, timing,..."""
    def heurexit(self):
        """calls exit method of primal heuristic"""
    def heurexitsol(self):
        """informs primal heuristic that the branch and bound process data is being freed"""
    def heurfree(self):
        """calls destructor and frees memory of primal heuristic"""
    def heurinit(self):
        """initializes primal heuristic"""
    def heurinitsol(self):
        """informs primal heuristic that the branch and bound process is being started"""

############
# presol.pxi
############

class Presol:
    model: Incomplete
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def presolexec(self, nrounds, presoltiming):
        """executes presolver"""
    def presolexit(self):
        """deinitializes presolver"""
    def presolexitpre(self):
        """informs presolver that the presolving process is finished"""
    def presolfree(self):
        """frees memory of presolver"""
    def presolinit(self):
        """initializes presolver"""
    def presolinitpre(self):
        """informs presolver that the presolving process is being started"""

############
# pricer.pxi
############

class Pricer:
    model: Incomplete
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def pricerexit(self):
        """calls exit method of variable pricer"""
    def pricerexitsol(self):
        """informs variable pricer that the branch and bound process data is being freed"""
    def pricerfarkas(self):
        """calls Farkas pricing method of variable pricer"""
    def pricerfree(self):
        """calls destructor and frees memory of variable pricer"""
    def pricerinit(self):
        """initializes variable pricer"""
    def pricerinitsol(self):
        """informs variable pricer that the branch and bound process is being started"""
    def pricerredcost(self):
        """calls reduced cost pricing method of variable pricer"""

################
# propagator.pxi
################

class Prop:
    model: Incomplete
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def propexec(self, proptiming):
        """calls execution method of propagator"""
    def propexit(self):
        """calls exit method of propagator"""
    def propexitpre(self):
        """informs propagator that the presolving process is finished"""
    def propexitsol(self, restart):
        """informs propagator that the prop and bound process data is being freed"""
    def propfree(self):
        """calls destructor and frees memory of propagator"""
    def propinit(self):
        """initializes propagator"""
    def propinitpre(self):
        """informs propagator that the presolving process is being started"""
    def propinitsol(self):
        """informs propagator that the prop and bound process is being started"""
    def proppresol(self, nrounds, presoltiming, result_dict):
        """executes presolving method of propagator"""
    def propresprop(self, confvar, inferinfo, bdtype, relaxedbd):
        """resolves the given conflicting bound, that was reduced by the given propagator"""

##########
# sepa.pxi
##########

class Sepa:
    model: Incomplete
    name: Incomplete
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def sepaexeclp(self):
        """calls LP separation method of separator"""
    def sepaexecsol(self, solution):
        """calls primal solution separation method of separator"""
    def sepaexit(self):
        """calls exit method of separator"""
    def sepaexitsol(self):
        """informs separator that the branch and bound process data is being freed"""
    def sepafree(self):
        """calls destructor and frees memory of separator"""
    def sepainit(self):
        """initializes separator"""
    def sepainitsol(self):
        """informs separator that the branch and bound process is being started"""

############
# reader.pxi
############

class Reader:
    model: Incomplete
    name: Incomplete
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def readerfree(self):
        """calls destructor and frees memory of reader"""
    def readerread(self, filename):
        """calls read method of reader"""
    def readerwrite(
        self,
        file,
        name,
        transformed,
        objsense,
        objscale,
        objoffset,
        binvars,
        intvars,
        implvars,
        contvars,
        fixedvars,
        startnvars,
        conss,
        maxnconss,
        startnconss,
        genericnames,
    ):
        """calls write method of reader"""

###########
# relax.pxi
###########

class Relax:
    model: Incomplete
    name: Incomplete
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def relaxexec(self):
        """callls execution method of relaxation handler"""
    def relaxexit(self):
        """calls exit method of relaxation handler"""
    def relaxexitsol(self):
        """informs relaxation handler that the branch and bound process data is being freed"""
    def relaxfree(self):
        """calls destructor and frees memory of relaxation handler"""
    def relaxinit(self):
        """initializes relaxation handler"""
    def relaxinitsol(self):
        """informs relaxaton handler that the branch and bound process is being started"""

#############
# nodesel.pxi
#############

class Nodesel:
    model: Incomplete
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def nodecomp(self, node1, node2):
        """
        compare two leaves of the current branching tree

        It should return the following values:

          value < 0, if node 1 comes before (is better than) node 2
          value = 0, if both nodes are equally good
          value > 0, if node 1 comes after (is worse than) node 2.
        """
    def nodeexit(self):
        """executed before the transformed problem is freed"""
    def nodeexitsol(self):
        """executed before the branch-and-bound process is freed"""
    def nodefree(self):
        """frees memory of node selector"""
    def nodeinit(self):
        """executed after the problem is transformed. use this call to initialize node selector data."""
    def nodeinitsol(self):
        """executed when the presolving is finished and the branch-and-bound process is about to begin"""
    def nodeselect(self):
        """first method called in each iteration in the main solving loop."""

##########
# scip.pxi
##########

MAJOR: int
MINOR: int
PATCH: int
EventNames: dict
StageNames: dict
__test__: dict

def is_memory_freed(): ...
def print_memory_in_use(): ...
def readStatistics(filename): ...
def str_conversion(x): ...
def PY_SCIP_CALL(rc): ...

class BoundChange:
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getBoundchgtype(self):
        """
        Returns the bound change type of the bound change.

        Returns
        -------
        int
            (0 = branching, 1 = consinfer, 2 = propinfer)

        """
    def getBoundtype(self):
        """
        Returns the bound type of the bound change.

        Returns
        -------
        int
            (0 = lower, 1 = upper)

        """
    def getNewBound(self):
        """
        Returns the new value of the bound in the bound change.

        Returns
        -------
        float

        """
    def getVar(self):
        """
        Returns the variable of the bound change.

        Returns
        -------
        Variable

        """
    def isRedundant(self):
        """
        Returns whether the bound change is redundant due to a more global bound that is at least as strong.

        Returns
        -------
        bool

        """

class Column:
    data: object
    def __init__(self) -> None: ...
    def getAge(self) -> int: ...
    def getBasisStatus(self) -> L["lower", "basic", "upper", "zero"]: ...
    def getLPPos(self) -> int: ...
    def getLb(self) -> float: ...
    def getObjCoeff(self) -> float: ...
    def getPrimsol(self) -> float: ...
    def getUb(self) -> float: ...
    def getVar(self) -> Variable: ...
    def isIntegral(self) -> bool: ...
    @override
    def __hash__(self) -> int: ...
    @override
    def __eq__(self, other: object) -> bool: ...
    @override
    def __ne__(self, other: object) -> bool: ...

class Constraint:
    data: Incomplete
    name: Incomplete
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getConshdlrName(self):
        """
        Return the constraint handler's name.

        Returns
        -------
        str

        """
    def isActive(self):
        """
        Returns True iff constraint is active in the current node.

        Returns
        -------
        bool

        """
    def isChecked(self):
        """
        Returns True if constraint should be checked for feasibility.

        Returns
        -------
        bool

        """
    def isDynamic(self):
        """
        Returns True if constraint is subject to aging.

        Returns
        -------
        bool

        """
    def isEnforced(self):
        """
        Returns True if constraint should be enforced during node processing.

        Returns
        -------
        bool

        """
    def isInitial(self):
        """
        Returns True if the relaxation of the constraint should be in the initial LP.

        Returns
        -------
        bool

        """
    def isLinear(self):
        """
        Returns True if constraint is linear

        Returns
        -------
        bool

        """
    def isLocal(self):
        """
        Returns True if constraint is only locally valid or not added to any (sub)problem.

        Returns
        -------
        bool

        """
    def isModifiable(self):
        """
        Returns True if constraint is modifiable (subject to column generation).

        Returns
        -------
        bool

        """
    def isNonlinear(self):
        """
        Returns True if constraint is nonlinear.

        Returns
        -------
        bool

        """
    def isOriginal(self):
        """
        Retrieve whether the constraint belongs to the original problem.

        Returns
        -------
        bool

        """
    def isPropagated(self):
        """
        Returns True if constraint should be propagated during node processing.

        Returns
        -------
        bool

        """
    def isRemovable(self):
        """
        Returns True if constraint's relaxation should be removed from the LP due to aging or cleanup.

        Returns
        -------
        bool

        """
    def isSeparated(self):
        """
        Returns True if constraint should be separated during LP processing.

        Returns
        -------
        bool

        """
    def isStickingAtNode(self):
        """
        Returns True if constraint is only locally valid or not added to any (sub)problem.

        Returns
        -------
        bool

        """
    @override
    def __eq__(self, other: object) -> bool:
        """Return self==value."""
    def __ge__(self, other: object) -> bool:
        """Return self>=value."""
    def __gt__(self, other: object) -> bool:
        """Return self>value."""
    @override
    def __hash__(self) -> int:
        """Return hash(self)."""
    def __le__(self, other: object) -> bool:
        """Return self<=value."""
    def __lt__(self, other: object) -> bool:
        """Return self<value."""
    @override
    def __ne__(self, other: object) -> bool:
        """Return self!=value."""

class DomainChanges:
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getBoundchgs(self):
        """
        Returns the bound changes in the domain change.

        Returns
        -------
        list of BoundChange

        """

class Event:
    data: Incomplete
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getName(self):
        """
        Gets name of event.

        Returns
        -------
        str

        """
    def getNewBound(self):
        """
        Gets new bound for a bound change event.

        Returns
        -------
        float

        """
    def getNode(self):
        """
        Gets node for a node or LP event.

        Returns
        -------
        Node

        """
    def getOldBound(self):
        """
        Gets old bound for a bound change event.

        Returns
        -------
        float

        """
    def getRow(self):
        """
        Gets row for a row event.

        Returns
        -------
        Row

        """
    def getType(self):
        """
        Gets type of event.

        Returns
        -------
        PY_SCIP_EVENTTYPE

        """
    def getVar(self):
        """
        Gets variable for a variable event (var added, var deleted, var fixed,
        objective value or domain change, domain hole added or removed).

        Returns
        -------
        Variable

        """
    @override
    def __eq__(self, other: object) -> bool:
        """Return self==value."""
    def __ge__(self, other: object) -> bool:
        """Return self>=value."""
    def __gt__(self, other: object) -> bool:
        """Return self>value."""
    @override
    def __hash__(self) -> int:
        """Return hash(self)."""
    def __le__(self, other: object) -> bool:
        """Return self<=value."""
    def __lt__(self, other: object) -> bool:
        """Return self<value."""
    @override
    def __ne__(self, other: object) -> bool:
        """Return self!=value."""

class Model:
    data: Incomplete
    def __init__(
        self,
        problemName: str = ...,
        defaultPlugins: bool = True,
        sourceModel: Model | None = None,
        origcopy: bool = False,
        flobalcopy: bool = True,
        enablepricing: bool = False,
        createscip: bool = True,
        threadsafe: bool = False,
    ) -> None:
        """
        Main class holding a pointer to SCIP for managing most interactions

        Parameters
        ----------
        problemName : str, optional
            name of the problem (default 'model')
        defaultPlugins : bool, optional
            use default plugins? (default True)
        sourceModel : Model or None, optional
            create a copy of the given Model instance (default None)
        origcopy : bool, optional
            whether to call copy or copyOrig (default False)
        globalcopy : bool, optional
            whether to create a global or a local copy (default True)
        enablepricing : bool, optional
            whether to enable pricing in copy (default False)
        createscip : bool, optional
            initialize the Model object and creates a SCIP instance (default True)
        threadsafe : bool, optional
            False if data can be safely shared between the source and target problem (default False)

        """
    def activateBenders(self, benders, nsubproblems):
        """
        Activates the Benders' decomposition plugin with the input name.

        Parameters
        ----------
        benders : Benders
            the Benders' decomposition to which the subproblem belongs to
        nsubproblems : int
            the number of subproblems in the Benders' decomposition

        """
    def addBendersSubproblem(self, benders, subproblem):
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
    def addCoefLinear(self, cons, var, value):
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
    def addCons(
        self,
        cons,
        name=...,
        initial=...,
        separate=...,
        enforce=...,
        check=...,
        propagate=...,
        local=...,
        modifiable=...,
        dynamic=...,
        removable=...,
        stickingatnode=...,
    ):
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
    def addConsAnd(
        self,
        vars,
        resvar,
        name=...,
        initial=...,
        separate=...,
        enforce=...,
        check=...,
        propagate=...,
        local=...,
        modifiable=...,
        dynamic=...,
        removable=...,
        stickingatnode=...,
    ):
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
    def addConsCardinality(
        self,
        consvars,
        cardval,
        indvars=...,
        weights=...,
        name=...,
        initial=...,
        separate=...,
        enforce=...,
        check=...,
        propagate=...,
        local=...,
        dynamic=...,
        removable=...,
        stickingatnode=...,
    ):
        """
        Add a cardinality constraint that allows at most \'cardval\' many nonzero variables.

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
    def addConsCoeff(self, cons, var, coeff):
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
    def addConsDisjunction(
        self,
        conss,
        name=...,
        initial=...,
        relaxcons=...,
        enforce=...,
        check=...,
        local=...,
        modifiable=...,
        dynamic=...,
    ):
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
    def addConsElemDisjunction(self, disj_cons, cons):
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
    def addConsIndicator(
        self,
        cons,
        binvar=...,
        activeone=...,
        name=...,
        initial=...,
        separate=...,
        enforce=...,
        check=...,
        propagate=...,
        local=...,
        dynamic=...,
        removable=...,
        stickingatnode=...,
    ):
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
    def addConsLocal(self, cons, validnode=...):
        """
        Add a constraint to the current node.

        Parameters
        ----------
        cons : Constraint
            the constraint to add to the current node
        validnode : Node or None, optional
            more global node where cons is also valid. (Default=None)

        """
    def addConsNode(self, node, cons, validnode=...):
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
    def addConsOr(
        self,
        vars,
        resvar,
        name=...,
        initial=...,
        separate=...,
        enforce=...,
        check=...,
        propagate=...,
        local=...,
        modifiable=...,
        dynamic=...,
        removable=...,
        stickingatnode=...,
    ):
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
    def addConsSOS1(
        self,
        vars,
        weights=...,
        name=...,
        initial=...,
        separate=...,
        enforce=...,
        check=...,
        propagate=...,
        local=...,
        dynamic=...,
        removable=...,
        stickingatnode=...,
    ):
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
        vars,
        weights=...,
        name=...,
        initial=...,
        separate=...,
        enforce=...,
        check=...,
        propagate=...,
        local=...,
        dynamic=...,
        removable=...,
        stickingatnode=...,
    ):
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
    def addConsXor(
        self,
        vars,
        rhsvar,
        name=...,
        initial=...,
        separate=...,
        enforce=...,
        check=...,
        propagate=...,
        local=...,
        modifiable=...,
        dynamic=...,
        removable=...,
        stickingatnode=...,
    ):
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
    def addConss(
        self,
        conss,
        name=...,
        initial=...,
        separate=...,
        enforce=...,
        check=...,
        propagate=...,
        local=...,
        modifiable=...,
        dynamic=...,
        removable=...,
        stickingatnode=...,
    ):
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
    def addCut(self, cut, forcecut=...):
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
    def addExprNonlinear(self, cons, expr, coef):
        """
        Add coef*expr to nonlinear constraint.

        Parameters
        ----------
        cons : Constraint
        expr : Expr or GenExpr
        coef : float

        """
    def addObjoffset(self, offset, solutions=...):
        """
        Add constant offset to objective.

        Parameters
        ----------
        offset : float
            offset to add
        solutions : bool, optional
            add offset also to existing solutions (Default value = False)

        """
    def addPoolCut(self, row):
        """
        If not already existing, adds row to global cut pool.

        Parameters
        ----------
        row : Row

        """
    def addPyCons(self, cons):
        """
        Adds a customly created cons.

        Parameters
        ----------
        cons : Constraint
            constraint to add

        """
    def addRowDive(self, row):
        """
        Adds a row to the LP in current dive.

        Parameters
        ----------
        row : Row

        """
    def addSol(self, solution, free=...):
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
    def addVarLocks(self, var, nlocksdown, nlocksup):
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
    def addVarSOS1(self, cons, var, weight):
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
    def addVarSOS2(self, cons, var, weight):
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
    def addVarToRow(self, row, var, value):
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
    def allColsInLP(self):
        """
        Checks if all columns, i.e. every variable with non-empty column is present in the LP.
        This is not True when performing pricing for instance.

        Returns
        -------
        bool

        """
    def appendVarSOS1(self, cons, var):
        """
        Append variable to SOS1 constraint.

        Parameters
        ----------
        cons : Constraint
            SOS1 constraint
        var : Variable
            variable to append

        """
    def appendVarSOS2(self, cons, var):
        """
        Append variable to SOS2 constraint.

        Parameters
        ----------
        cons : Constraint
            SOS2 constraint
        var : Variable
            variable to append

        """
    def applyCutsProbing(self):
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
    def attachEventHandlerCallback(self, callback, events, name=..., description=...):
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
    def backtrackProbing(self, probingdepth):
        """
        Undoes all changes to the problem applied in probing up to the given probing depth.

        Parameters
        ----------
        probingdepth : int
            probing depth of the node in the probing path that should be reactivated

        """
    def branchVar(self, variable):
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
    def branchVarVal(self, variable, value):
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
    def cacheRowExtensions(self, row):
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
    def calcChildEstimate(self, variable, targetvalue):
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
    def calcNodeselPriority(self, variable, branchdir, targetvalue):
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
    def catchEvent(self, eventtype, eventhdlr):
        """
        Catches a global (not variable or row dependent) event.

        Parameters
        ----------
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
    def catchRowEvent(self, row, eventtype, eventhdlr):
        """
        Catches a row coefficient, constant, or side change event on the given row.

        Parameters
        ----------
        row : Row
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
    def catchVarEvent(self, var, eventtype, eventhdlr):
        """
        Catches an objective value or domain change event on the given transformed variable.

        Parameters
        ----------
        var : Variable
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
    def checkBendersSubproblemOptimality(self, solution, probnumber, benders=...):
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
    def checkQuadraticNonlinear(self, cons):
        """
        Returns if the given constraint is quadratic.

        Parameters
        ----------
        cons : Constraint

        Returns
        -------
        bool

        """
    def checkSol(
        self,
        solution,
        printreason=...,
        completely=...,
        checkbounds=...,
        checkintegrality=...,
        checklprows=...,
        original=...,
    ):
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
    def chgCoefLinear(self, cons, var, value):
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
    def chgLhs(self, cons, lhs):
        """
        Change left-hand side value of a constraint.

        Parameters
        ----------
        cons : Constraint
            linear or quadratic constraint
        lhs : float or None
            new left-hand side (set to None for -infinity)

        """
    def chgReoptObjective(self, coeffs, sense=...):
        """
        Establish the objective function as a linear expression.

        Parameters
        ----------
        coeffs : list of float
            the coefficients
        sense : str
            the objective sense (Default value = 'minimize')

        """
    def chgRhs(self, cons, rhs):
        """
        Change right-hand side value of a constraint.

        Parameters
        ----------
        cons : Constraint
            linear or quadratic constraint
        rhs : float or None
            new right-hand side (set to None for +infinity)

        """
    def chgRowLhsDive(self, row, newlhs):
        """
        Changes row lhs in current dive, change will be undone after diving
        ends, for permanent changes use SCIPchgRowLhs().

        Parameters
        ----------
        row : Row
        newlhs : float

        """
    def chgRowRhsDive(self, row, newrhs):
        """
        Changes row rhs in current dive, change will be undone after diving
        ends. For permanent changes use SCIPchgRowRhs().

        Parameters
        ----------
        row : Row
        newrhs : float

        """
    def chgVarBranchPriority(self, var, priority):
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
    def chgVarLb(self, var, lb):
        """
        Changes the lower bound of the specified variable.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new lower bound (set to None for -infinity)

        """
    def chgVarLbDive(self, var, newbound):
        """
        Changes variable's current lb in current dive.

        Parameters
        ----------
        var : Variable
        newbound : float

        """
    def chgVarLbGlobal(self, var, lb):
        """Changes the global lower bound of the specified variable.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new lower bound (set to None for -infinity)

        """
    def chgVarLbNode(self, node, var, lb):
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
    def chgVarLbProbing(self, var, lb):
        """
        Changes the variable lower bound during probing mode.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new lower bound (set to None for -infinity)

        """
    def chgVarObjDive(self, var, newobj):
        """
        Changes (column) variable's objective value in current dive.

        Parameters
        ----------
        var : Variable
        newobj : float

        """
    def chgVarObjProbing(self, var, newobj):
        """Changes (column) variable's objective value during probing mode."""
    def chgVarType(self, var, vtype):
        """
        Changes the type of a variable.

        Parameters
        ----------
        var : Variable
            variable to change type of
        vtype : str
            new variable type. \'C\' or "CONTINUOUS", \'I\' or "INTEGER",
            \'B\' or "BINARY", and \'M\' "IMPLINT".

        """
    def chgVarUb(self, var, ub):
        """Changes the upper bound of the specified variable.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new upper bound (set to None for +infinity)

        """
    def chgVarUbDive(self, var, newbound):
        """
        Changes variable's current ub in current dive.

        Parameters
        ----------
        var : Variable
        newbound : float

        """
    def chgVarUbGlobal(self, var, ub):
        """Changes the global upper bound of the specified variable.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new upper bound (set to None for +infinity)

        """
    def chgVarUbNode(self, node, var, ub):
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
    def chgVarUbProbing(self, var, ub):
        """
        Changes the variable upper bound during probing mode.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        ub : float or None
            new upper bound (set to None for +infinity)

        """
    def computeBestSolSubproblems(self):
        """Solves the subproblems with the best solution to the master problem.
        Afterwards, the best solution from each subproblem can be queried to get
        the solution to the original problem.
        If the user wants to resolve the subproblems, they must free them by
        calling freeBendersSubproblems()
        """
    def constructLP(self):
        """
        Makes sure that the LP of the current node is loaded and
        may be accessed through the LP information methods.


        Returns
        -------
        cutoff : bool
            Can the node be cutoff?

        """
    def copyLargeNeighborhoodSearch(self, to_fix, fix_vals):
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
    def count(self):
        """Counts the number of feasible points of problem."""
    def createChild(self, nodeselprio, estimate):
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
    def createCons(
        self,
        conshdlr,
        name,
        initial=...,
        separate=...,
        enforce=...,
        check=...,
        propagate=...,
        local=...,
        modifiable=...,
        dynamic=...,
        removable=...,
        stickingatnode=...,
    ):
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
    def createConsFromExpr(
        self,
        cons,
        name=...,
        initial=...,
        separate=...,
        enforce=...,
        check=...,
        propagate=...,
        local=...,
        modifiable=...,
        dynamic=...,
        removable=...,
        stickingatnode=...,
    ):
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
    def createEmptyRowSepa(
        self, sepa, name=..., lhs=..., rhs=..., local=..., modifiable=..., removable=...
    ):
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
        self, name=..., lhs=..., rhs=..., local=..., modifiable=..., removable=...
    ):
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
    def createOrigSol(self, heur=...):
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
    def createPartialSol(self, heur=...):
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
    def createProbBasic(self, problemName=...):
        """
        Create new problem instance with given name.

        Parameters
        ----------
        problemName : str, optional
            name of model or problem (Default value = 'model')

        """
    def createSol(self, heur=..., initlp=...):
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
    def delCoefLinear(self, cons, var):
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
    def delCons(self, cons):
        """
        Delete constraint from the model

        Parameters
        ----------
        cons : Constraint
            constraint to be deleted

        """
    def delConsLocal(self, cons):
        """
        Delete constraint from the current node and its children.

        Parameters
        ----------
        cons : Constraint
            constraint to be deleted

        """
    def delVar(self, var):
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
    def disablePropagation(self, onlyroot=...):
        """
        Disables propagation in SCIP to avoid modifying the original problem during transformation.

        Parameters
        ----------
        onlyroot : bool, optional
            use propagation when root processing is finished (Default value = False)

        """
    def dropEvent(self, eventtype, eventhdlr):
        """
        Drops a global event (stops tracking the event).

        Parameters
        ----------
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
    def dropRowEvent(self, row, eventtype, eventhdlr):
        """
        Drops a row coefficient, constant, or side change event (stops tracking the event) on the given row.

        Parameters
        ----------
        row : Row
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
    def dropVarEvent(self, var, eventtype, eventhdlr):
        """
        Drops an objective value or domain change event (stops tracking the event) on the given transformed variable.

        Parameters
        ----------
        var : Variable
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
    def enableReoptimization(self, enable=...):
        """
        Include specific heuristics and branching rules for reoptimization.

        Parameters
        ----------
        enable : bool, optional
            True to enable and False to disable

        """
    def endDive(self):
        """Quits probing and resets bounds and constraints to the focus node's environment."""
    def endProbing(self):
        """Quits probing and resets bounds and constraints to the focus node's environment."""
    def endStrongbranch(self):
        """End strong branching. Needs to be called if startStrongBranching was called previously.
        Between these calls the user can access all strong branching functionality."""
    def epsilon(self):
        """
        Retrieve epsilon for e.g. equality checks.

        Returns
        -------
        float

        """
    def feasCeil(self, value):
        """
        Rounds value - feasibility tolerance up to the next integer.

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
    def feasFloor(self, value):
        """
        Rounds value + feasibility tolerance down to the next integer.

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
    def feasFrac(self, value):
        """
        Returns fractional part of value, i.e. x - floor(x) in feasible tolerance: x - floor(x+feastol).

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
    def feasRound(self, value):
        """
        Rounds value to the nearest integer in feasibility tolerance.

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
    def feastol(self):
        """
        Retrieve feasibility tolerance.

        Returns
        -------
        float

        """
    def fixVar(self, var, val):
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
    def fixVarProbing(self, var, fixedval):
        """
        Fixes a variable at the current probing node.

        Parameters
        ----------
        var : Variable
        fixedval : float

        """
    def flushRowExtensions(self, row):
        """
        Flushes all cached row extensions after a call of cacheRowExtensions()
        and merges coefficients with equal columns into a single coefficient

        Parameters
        ----------
        row : Row

        """
    def frac(self, value):
        """
        Returns fractional part of value, i.e. x - floor(x) in epsilon tolerance: x - floor(x+eps).

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
    def freeBendersSubproblems(self):
        """Calls the free subproblem function for the Benders' decomposition.
        This will free all subproblems for all decompositions."""
    def freeProb(self):
        """Frees problem and solution process data."""
    def freeReoptSolve(self):
        """Frees all solution process data and prepares for reoptimization."""
    def freeSol(self, solution):
        """
        Free given solution

        Parameters
        ----------
        solution : Solution
            solution to be freed

        """
    def freeTransform(self):
        """Frees all solution process data including presolving and
        transformed problem, only original problem is kept."""
    @staticmethod
    def from_ptr(capsule, take_ownership):
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
    def getActivity(self, cons, sol=...):
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
    def getBendersAuxiliaryVar(self, probnumber, benders=...):
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
    def getBendersSubproblem(self, probnumber, benders=...):
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
    def getBendersVar(self, var, benders=..., probnumber=...):
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
    def getBestChild(self):
        """
        Gets the best child of the focus node w.r.t. the node selection strategy.

        Returns
        -------
        Node

        """
    def getBestLeaf(self):
        """Gets the best leaf from the node queue w.r.t. the node selection strategy.

        Returns
        -------
        Node

        """
    def getBestNode(self):
        """Gets the best node from the tree (child, sibling, or leaf) w.r.t. the node selection strategy.

        Returns
        -------
        Node

        """
    def getBestSibling(self):
        """
        Gets the best sibling of the focus node w.r.t. the node selection strategy.

        Returns
        -------
        Node

        """
    def getBestSol(self):
        """
        Retrieve currently best known feasible primal solution.

        Returns
        -------
        Solution or None

        """
    def getBestboundNode(self):
        """Gets the node with smallest lower bound from the tree (child, sibling, or leaf).

        Returns
        -------
        Node

        """
    def getBipartiteGraphRepresentation(
        self,
        prev_col_features=...,
        prev_edge_features=...,
        prev_row_features=...,
        static_only=...,
        suppress_warnings=...,
    ):
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
    def getBranchScoreMultiple(self, var, gains):
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
    def getColRedCost(self, col):
        """
        Gets the reduced cost of the column in the current LP.

        Parameters
        ----------
        col : Column

        Returns
        -------
        float

        """
    def getCondition(self, exact=...):
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
    def getConsNVars(self, constraint):
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
    def getConsVars(self, constraint):
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
    def getConss(self, transformed=...):
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
    def getCurrentNode(self):
        """
        Retrieve current node.

        Returns
        -------
        Node

        """
    def getCutEfficacy(self, cut, sol=...):
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
    def getCutLPSolCutoffDistance(self, cut, sol):
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
    def getDepth(self):
        """
        Retrieve the depth of the current node.

        Returns
        -------
        int

        """
    def getDualMultiplier(self, cons):
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
    def getDualSolVal(self, cons, boundconstraint=...):
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
    def getDualbound(self):
        """
        Retrieve the best dual bound.

        Returns
        -------
        float

        """
    def getDualboundRoot(self):
        """
        Retrieve the best root dual bound.

        Returns
        -------
        float

        """
    def getDualfarkasLinear(self, cons):
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
    def getDualsolLinear(self, cons):
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
    def getGap(self):
        """
        Retrieve the gap,
        i.e. abs((primalbound - dualbound)/min(abs(primalbound),abs(dualbound)))

        Returns
        -------
        float

        """
    def getHeurTiming(self, heurname):
        """
                Get the timing of a heuristic

                Parameters
                ----------
                heurname : string, name of the heuristic

                Returns
                -------
                PY_SCIP_HEURTIMING
        \t\t   positions in the node solving loop where heuristic should be executed
        """
    def getLPBInvARow(self, row):
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
    def getLPBInvRow(self, row):
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
    def getLPBasisInd(self):
        """
        Gets all indices of basic columns and rows:
        index i >= 0 corresponds to column i, index i < 0 to row -i-1

        Returns
        -------
        list of int

        """
    def getLPBranchCands(self):
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
    def getLPColsData(self):
        """
        Retrieve current LP columns.

        Returns
        -------
        list of Column

        """
    def getLPObjVal(self):
        """
        Gets objective value of current LP (which is the sum of column and loose objective value).

        Returns
        -------
        float

        """
    def getLPRowsData(self):
        """
        Retrieve current LP rows.

        Returns
        -------
        list of Row

        """
    def getLPSolstat(self):
        """
        Gets solution status of current LP.

        Returns
        -------
        SCIP_LPSOLSTAT

        """
    def getLhs(self, cons):
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
    def getLocalEstimate(self, original=...):
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
    def getNBestSolsFound(self):
        """
        Gets number of feasible primal solutions found so far,
        that improved the primal bound at the time they were found.

        Returns
        -------
        int

        """
    def getNBinVars(self):
        """
        Gets number of binary active problem variables.

        Returns
        -------
        int

        """
    def getNChildren(self):
        """
        Gets number of children of focus node.

        Returns
        -------
        int

        """
    def getNConss(self, transformed=...):
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
    def getNContVars(self):
        """
        Gets number of continuous active problem variables.

        Returns
        -------
        int

        """
    def getNCountedSols(self):
        """
        Get number of feasible solution.

        Returns
        -------
        int

        """
    def getNCuts(self):
        """
        Retrieve total number of cuts in storage.

        Returns
        -------
        int

        """
    def getNCutsApplied(self):
        """
        Retrieve number of currently applied cuts.

        Returns
        -------
        int

        """
    def getNFeasibleLeaves(self):
        """
        Retrieve number of leaf nodes processed with feasible relaxation solution.

        Returns
        -------
        int

        """
    def getNImplVars(self):
        """
        Gets number of implicit integer active problem variables.

        Returns
        -------
        int

        """
    def getNInfeasibleLeaves(self):
        """
        Gets number of infeasible leaf nodes processed.

        Returns
        -------
        int

        """
    def getNIntVars(self):
        """
        Gets number of integer active problem variables.

        Returns
        -------
        int

        """
    def getNLPCols(self):
        """
        Retrieve the number of columns currently in the LP.

        Returns
        -------
        int

        """
    def getNLPIterations(self):
        """
        Returns the total number of LP iterations so far.

        Returns
        -------
        int

        """
    def getNLPRows(self):
        """
        Retrieve the number of rows currently in the LP.

        Returns
        -------
        int

        """
    def getNLPs(self):
        """
        Gets total number of LPs solved so far.

        Returns
        -------
        int

        """
    def getNLeaves(self):
        """
        Gets number of leaves in the tree.

        Returns
        -------
        int

        """
    def getNLimSolsFound(self):
        """
        Gets number of feasible primal solutions respecting the objective limit found so far.

        Returns
        -------
        int

        """
    def getNNlRows(self):
        """
        Gets current number of nonlinear rows in SCIP's internal NLP.

        Returns
        -------
        int

        """
    def getNNodes(self):
        """
        Gets number of processed nodes in current run, including the focus node.

        Returns
        -------
        int

        """
    def getNReaders(self):
        """
        Get number of currently available readers.

        Returns
        -------
        int

        """
    def getNSepaRounds(self):
        """
        Retrieve the number of separation rounds that have been performed
        at the current node.

        Returns
        -------
        int

        """
    def getNSiblings(self):
        """
        Gets number of siblings of focus node.

        Returns
        -------
        int

        """
    def getNSols(self):
        """
        Gets number of feasible primal solutions stored in the solution storage in case the problem is transformed;
        in case the problem stage is SCIP_STAGE_PROBLEM, the number of solution in the original solution candidate
        storage is returned.

        Returns
        -------
        int

        """
    def getNSolsFound(self):
        """
        Gets number of feasible primal solutions found so far.

        Returns
        -------
        int

        """
    def getNTotalNodes(self):
        """
        Gets number of processed nodes in all runs, including the focus node.

        Returns
        -------
        int

        """
    def getNVars(self, transformed=...):
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
    def getNlRowActivityBounds(self, nlrow):
        """
        Gives the minimal and maximal activity of a nonlinear row w.r.t. the variable's bounds.

        Parameters
        ----------
        nlrow : NLRow

        Returns
        -------
        tuple of float

        """
    def getNlRowSolActivity(self, nlrow, sol=...):
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
    def getNlRowSolFeasibility(self, nlrow, sol=...):
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
    def getNlRows(self):
        """
        Returns a list with the nonlinear rows in SCIP's internal NLP.

        Returns
        -------
        list of NLRow

        """
    def getObjVal(self, original=...):
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
    def getObjective(self):
        """
        Retrieve objective function as Expr.

        Returns
        -------
        Expr

        """
    def getObjectiveSense(self):
        """
        Retrieve objective sense.

        Returns
        -------
        str

        """
    def getObjlimit(self):
        """
        Returns current limit on objective function.

        Returns
        -------
        float

        """
    def getObjoffset(self, original=...):
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
    def getOpenNodes(self):
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
    def getParam(self, name):
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
    def getParams(self):
        """
        Gets the values of all parameters as a dict mapping parameter names
        to their values.

        Returns
        -------
        dict of str to object
            dict mapping parameter names to their values.

        """
    def getPresolvingTime(self):
        """
        Returns the current presolving time in seconds.

        Returns
        -------
        float

        """
    def getPrimalRay(self):
        """
        Gets primal ray causing unboundedness of the LP relaxation.

        Returns
        -------
        list of float

        """
    def getPrimalRayVal(self, var):
        """
        Gets value of given variable in primal ray causing unboundedness of the LP relaxation.

        Parameters
        ----------
        var : Variable

        Returns
        -------
        float

        """
    def getPrimalbound(self):
        """
        Retrieve the best primal bound.

        Returns
        -------
        float

        """
    def getPrioChild(self):
        """
        Gets the best child of the focus node w.r.t. the node selection priority
        assigned by the branching rule.

        Returns
        -------
        Node

        """
    def getPrioSibling(self):
        """Gets the best sibling of the focus node w.r.t.
        the node selection priority assigned by the branching rule.

        Returns
        -------
        Node

        """
    def getProbName(self):
        """
        Retrieve problem name.

        Returns
        -------
        str

        """
    def getProbingDepth(self):
        """Returns the current probing depth."""
    def getPseudoBranchCands(self):
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
    def getReadingTime(self):
        """
        Retrieve the current reading time in seconds.

        Returns
        -------
        float

        """
    def getRhs(self, cons):
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
    def getRowActivity(self, row):
        """
        Returns the activity of a row in the last LP or pseudo solution.

        Parameters
        ----------
        row : Row

        Returns
        -------
        float

        """
    def getRowDualSol(self, row):
        """
        Gets the dual LP solution of a row.

        Parameters
        ----------
        row : Row

        Returns
        -------
        float

        """
    def getRowLPActivity(self, row):
        """
        Returns the activity of a row in the last LP solution.

        Parameters
        ----------
        row : Row

        Returns
        -------
        float

        """
    def getRowLinear(self, cons):
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
    def getRowNumIntCols(self, row):
        """
        Returns number of intergal columns in the row.

        Parameters
        ----------
        row : Row

        Returns
        -------
        int

        """
    def getRowObjParallelism(self, row):
        """
        Returns 1 if the row is parallel, and 0 if orthogonal.

        Parameters
        ----------
        row : Row

        Returns
        -------
        float

        """
    def getRowParallelism(self, row1, row2, orthofunc=...):
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
    def getSlack(self, cons, sol=..., side=...):
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
    def getSlackVarIndicator(self, cons):
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
    def getSolObjVal(self, sol, original=...):
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
    def getSolTime(self, sol):
        """
        Get clock time when this solution was found.

        Parameters
        ----------
        sol : Solution

        Returns
        -------
        float

        """
    def getSolVal(self, sol, expr):
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
    def getSols(self):
        """
        Retrieve list of all feasible primal solutions stored in the solution storage.

        Returns
        -------
        list of Solution

        """
    def getSolvingTime(self):
        """
        Retrieve the current solving time in seconds.

        Returns
        -------
        float

        """
    def getStage(self):
        """
        Retrieve current SCIP stage.

        Returns
        -------
        int

        """
    def getStageName(self):
        """
        Returns name of current stage as string.

        Returns
        -------
        str

        """
    def getStatus(self):
        """
        Retrieve solution status.

        Returns
        -------
        str
            The status of SCIP.

        """
    def getTermsQuadratic(self, cons):
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
    def getTotalTime(self):
        """
        Retrieve the current total SCIP time in seconds,
        i.e. the total time since the SCIP instance has been created.

        Returns
        -------
        float

        """
    def getTransformedCons(self, cons):
        """
        Retrieve transformed constraint.

        Parameters
        ----------
        cons : Constraint

        Returns
        -------
        Constraint

        """
    def getTransformedVar(self, var):
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
    def getTreesizeEstimation(self):
        """
        Get an estimate of the final tree size.

        Returns
        -------
        float

        """
    def getVal(self, expr):
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
    def getValsLinear(self, cons):
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
    def getVarDict(self, transformed=...):
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
    def getVarLbDive(self, var):
        """
        Returns variable's current lb in current dive.

        Parameters
        ----------
        var : Variable

        Returns
        -------
        float

        """
    def getVarRedcost(self, var):
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
    def getVarStrongbranch(self, var, itlim, idempotent=..., integral=...):
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
    def getVarStrongbranchLast(self, var):
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
    def getVarStrongbranchNode(self, var):
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
    def getVarUbDive(self, var):
        """
        Returns variable's current ub in current dive.

        Parameters
        ----------
        var : Variable

        Returns
        -------
        float

        """
    def getVars(self, transformed=...):
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
    def hasPrimalRay(self):
        """
        Returns whether a primal ray is stored that proves unboundedness of the LP relaxation.

        Returns
        -------
        bool

        """
    def hideOutput(self, quiet=...):
        """
        Hide the output.

        Parameters
        ----------
        quiet : bool, optional
            hide output? (Default value = True)

        """
    def inProbing(self):
        """
        Returns whether we are in probing mode;
        probing mode is activated via startProbing() and stopped via endProbing().

        Returns
        -------
        bool

        """
    def inRepropagation(self):
        """
        Returns if the current node is already solved and only propagated again.

        Returns
        -------
        bool

        """
    def includeBenders(
        self,
        benders,
        name,
        desc,
        priority=...,
        cutlp=...,
        cutpseudo=...,
        cutrelax=...,
        shareaux=...,
    ):
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
    def includeBendersDefaultCuts(self, benders):
        """
        Includes the default Benders' decomposition cuts to the custom Benders' decomposition plugin.

        Parameters
        ----------
        benders : Benders
            the Benders' decomposition that the default cuts will be applied to

        """
    def includeBenderscut(
        self, benders, benderscut, name, desc, priority=..., islpcut=...
    ):
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
    def includeBranchrule(
        self, branchrule, name, desc, priority, maxdepth, maxbounddist
    ):
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
    def includeConshdlr(
        self,
        conshdlr,
        name,
        desc,
        sepapriority=...,
        enfopriority=...,
        chckpriority=...,
        sepafreq=...,
        propfreq=...,
        eagerfreq=...,
        maxprerounds=...,
        delaysepa=...,
        delayprop=...,
        needscons=...,
        proptiming=...,
        presoltiming=...,
    ):
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
    def includeCutsel(self, cutsel, name, desc, priority):
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
    def includeDefaultPlugins(self):
        """Includes all default plug-ins into SCIP."""
    def includeEventhdlr(self, eventhdlr, name, desc):
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
    def includeHeur(
        self,
        heur,
        name,
        desc,
        dispchar,
        priority=...,
        freq=...,
        freqofs=...,
        maxdepth=...,
        timingmask=...,
        usessubscip=...,
    ):
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
    def includeNodesel(self, nodesel, name, desc, stdpriority, memsavepriority):
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
    def includePresol(self, presol, name, desc, priority, maxrounds, timing=...):
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
    def includePricer(self, pricer, name, desc, priority=..., delay=...):
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
    def includeProp(
        self,
        prop,
        name,
        desc,
        presolpriority,
        presolmaxrounds,
        proptiming,
        presoltiming=...,
        priority=...,
        freq=...,
        delay=...,
    ):
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
    def includeReader(self, reader, name, desc, ext):
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
    def includeRelax(self, relax, name, desc, priority=..., freq=...):
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
    def includeSepa(
        self,
        sepa,
        name,
        desc,
        priority=...,
        freq=...,
        maxbounddist=...,
        usessubscip=...,
        delay=...,
    ):
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
    def infinity(self):
        """
        Retrieve SCIP's infinity value.

        Returns
        -------
        int

        """
    def initBendersDefault(self, subproblems):
        """
        Initialises the default Benders' decomposition with a dictionary of subproblems.

        Parameters
        ----------
        subproblems : Model or dict of object to Model
            a single Model instance or dictionary of Model instances

        """
    def interruptSolve(self):
        """Interrupt the solving process as soon as possible."""
    def isCutEfficacious(self, cut, sol=...):
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
    def isEQ(self, val1, val2):
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
    def isFeasEQ(self, val1, val2):
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
    def isFeasIntegral(self, value):
        """
        Returns whether value is integral within the LP feasibility bounds.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
    def isFeasNegative(self, value):
        """
        Returns whether value < -feastol.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
    def isFeasZero(self, value):
        """
        Returns whether abs(value) < feastol.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
    def isGE(self, val1, val2):
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
    def isGT(self, val1, val2):
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
    def isInfinity(self, value):
        """
        Returns whether value is SCIP's infinity.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
    def isLE(self, val1, val2):
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
    def isLPSolBasic(self):
        """
        Returns whether the current LP solution is basic, i.e. is defined by a valid simplex basis.

        Returns
        -------
        bool

        """
    def isLT(self, val1, val2):
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
    def isNLPConstructed(self):
        """
        Returns whether SCIP's internal NLP has been constructed.

        Returns
        -------
        bool

        """
    def isObjChangedProbing(self):
        """
        Returns whether the objective function has changed during probing mode.

        Returns
        -------
        bool

        """
    def isZero(self, value):
        """
        Returns whether abs(value) < eps.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
    def lpiGetIterations(self):
        """
        Get the iteration count of the last solved LP.

        Returns
        -------
        int

        """
    def newProbingNode(self):
        """Creates a new probing sub node, whose changes can be undone by backtracking to a higher node in the
        probing path with a call to backtrackProbing().
        """
    def optimize(self):
        """Optimize the problem."""
    def optimizeNogil(self):
        """Optimize the problem without GIL."""
    def presolve(self):
        """Presolve the problem."""
    def printBestSol(self, write_zeros=...):
        """
        Prints the best feasible primal solution.

        Parameters
        ----------
        write_zeros : bool, optional
            include variables that are set to zero (Default = False)

        """
    def printCons(self, constraint):
        """
        Print the constraint

        Parameters
        ----------
        constraint : Constraint

        """
    def printExternalCodeVersions(self):
        """Print external code versions, e.g. symmetry, non-linear solver, lp solver."""
    def printNlRow(self, nlrow):
        """
        Prints nonlinear row.

        Parameters
        ----------
        nlrow : NLRow

        """
    def printProblem(self, ext=..., trans=..., genericnames=...):
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
    def printRow(self, row):
        """
        Prints row.

        Parameters
        ----------
        row : Row

        """
    def printSol(self, solution=..., write_zeros=...):
        """
        Print the given primal solution.

        Parameters
        ----------
        solution : Solution or None, optional
            solution to print (default = None)
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """
    def printStatistics(self):
        """Print statistics."""
    def printVersion(self):
        """Print version, copyright information and compile mode."""
    def propagateProbing(self, maxproprounds):
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
    def readParams(self, file):
        """
        Read an external parameter file.

        Parameters
        ----------
        file : str
            file to read

        """
    def readProblem(self, filename, extension=...):
        """
        Read a problem instance from an external file.

        Parameters
        ----------
        filename : str
            problem file name
        extension : str or None
            specify file extension/type (Default value = None)

        """
    def readSol(self, filename):
        """
        Reads a given solution file, problem has to be transformed in advance.

        Parameters
        ----------
        filename : str
            name of the input file

        """
    def readSolFile(self, filename):
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
    def redirectOutput(self):
        """Send output to python instead of terminal."""
    def relax(self):
        """Relaxes the integrality restrictions of the model."""
    def releaseRow(self, row):
        """
        Decreases usage counter of LP row, and frees memory if necessary.

        Parameters
        ----------
        row : Row

        """
    def repropagateNode(self, node):
        """Marks the given node to be propagated again the next time a node of its subtree is processed."""
    def resetParam(self, name):
        """
        Reset parameter setting to its default value

        Parameters
        ----------
        name : str
            parameter to reset

        """
    def resetParams(self):
        """Reset parameter settings to their default values."""
    def restartSolve(self):
        """Restarts the solving process as soon as possible."""
    def separateSol(self, sol=..., pretendroot=..., allowlocal=..., onlydelayed=...):
        """
        Separates the given primal solution or the current LP solution by calling
        the separators and constraint handlers\' separation methods;
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
    def setBendersSubproblemIsConvex(self, benders, probnumber, isconvex=...):
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
    def setBoolParam(self, name, value):
        """
        Set a boolean-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : bool
            value of parameter

        """
    def setCharParam(self, name, value):
        """
        Set a char-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : str
            value of parameter

        """
    def setCheck(self, cons, newCheck):
        """
        Set "check" flag of a constraint.

        Parameters
        ----------
        cons : Constraint
        newCheck : bool

        """
    def setEmphasis(self, paraemphasis, quiet=...):
        """
        Set emphasis settings

        Parameters
        ----------
        paraemphasis : PY_SCIP_PARAMEMPHASIS
            emphasis to set
        quiet : bool, optional
            hide output? (Default value = True)

        """
    def setEnforced(self, cons, newEnf):
        """
        Set "enforced" flag of a constraint.

        Parameters
        ----------
        cons : Constraint
        newEnf : bool

        """
    def setHeurTiming(self, heurname, heurtiming):
        """
                Set the timing of a heuristic

                Parameters
                ----------
                heurname : string, name of the heuristic
                heurtiming : PY_SCIP_HEURTIMING
        \t\t   positions in the node solving loop where heuristic should be executed
        """
    def setHeuristics(self, setting):
        """
        Set heuristics parameter settings.

        Parameters
        ----------
        setting : SCIP_PARAMSETTING
            the parameter settings, e.g. SCIP_PARAMSETTING.OFF

        """
    def setInitial(self, cons, newInit):
        """
        Set "initial" flag of a constraint.

        Parameters
        ----------
        cons : Constraint
        newInit : bool

        """
    def setIntParam(self, name, value):
        """
        Set an int-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : int
            value of parameter

        """
    def setLogfile(self, path):
        """
        Sets the log file name for the currently installed message handler.

        Parameters
        ----------
        path : str or None
            name of log file, or None (no log)

        """
    def setLongintParam(self, name, value):
        """
        Set a long-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : int
            value of parameter

        """
    def setMaximize(self):
        """Set the objective sense to maximization."""
    def setMinimize(self):
        """Set the objective sense to minimization."""
    def setObjIntegral(self):
        """Informs SCIP that the objective value is always integral in every feasible solution.

        Notes
        -----
        This function should be used to inform SCIP that the objective function is integral,
        helping to improve the performance. This is useful when using column generation.
        If no column generation (pricing) is used, SCIP automatically detects whether the objective
        function is integral or can be scaled to be integral. However, in any case, the user has to
        make sure that no variable is added during the solving process that destroys this property.
        """
    def setObjective(self, expr, sense=..., clear=...):
        """
        Establish the objective function as a linear expression.

        Parameters
        ----------
        expr : Expr or float
            the objective function SCIP Expr, or constant value
        sense : str, optional
            the objective sense ("minimize" or "maximize") (Default value = \'minimize\')
        clear : bool, optional
            set all other variables objective coefficient to zero (Default value = \'true\')

        """
    def setObjlimit(self, objlimit):
        """
        Set a limit on the objective function.
        Only solutions with objective value better than this limit are accepted.

        Parameters
        ----------
        objlimit : float
            limit on the objective function

        """
    def setParam(self, name, value):
        """Set a parameter with value in int, bool, real, long, char or str.

        Parameters
        ----------
        name : str
            name of parameter
        value : object
            value of parameter

        """
    def setParams(self, params):
        """
        Sets multiple parameters at once.

        Parameters
        ----------
        params : dict of str to object
            dict mapping parameter names to their values.

        """
    def setParamsCountsols(self):
        """Sets SCIP parameters such that a valid counting process is possible."""
    def setPresolve(self, setting):
        """
        Set presolving parameter settings.


        Parameters
        ----------
        setting : SCIP_PARAMSETTING
            the parameter settings, e.g. SCIP_PARAMSETTING.OFF

        """
    def setProbName(self, name):
        """
        Set problem name.

        Parameters
        ----------
        name : str

        """
    def setRealParam(self, name, value):
        """
        Set a real-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : float
            value of parameter

        """
    def setRelaxSolVal(self, var, val):
        """
        Sets the value of the given variable in the global relaxation solution.

        Parameters
        ----------
        var : Variable
        val : float

        """
    def setRemovable(self, cons, newRem):
        """
        Set "removable" flag of a constraint.

        Parameters
        ----------
        cons : Constraint
        newRem : bool

        """
    def setSeparating(self, setting):
        """
        Set separating parameter settings.

        Parameters
        ----------
        setting : SCIP_PARAMSETTING
            the parameter settings, e.g. SCIP_PARAMSETTING.OFF

        """
    def setSolVal(self, solution, var, val):
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
    def setStringParam(self, name, value):
        """
        Set a string-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : str
            value of parameter

        """
    def setupBendersSubproblem(
        self, probnumber, benders=..., solution=..., checktype=...
    ):
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
    def solveBendersSubproblem(self, probnumber, solvecip, benders=..., solution=...):
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
    def solveConcurrent(self):
        """Transforms, presolves, and solves problem using additional solvers which emphasize on
        finding solutions.
        WARNING: This feature is still experimental and prone to some errors."""
    def solveDiveLP(self, itlim=...):
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
    def solveProbingLP(self, itlim=...):
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
    def startDive(self):
        """Initiates LP diving.
        It allows the user to change the LP in several ways, solve, change again, etc,
        without affecting the actual LP. When endDive() is called,
        SCIP will undo all changes done and recover the LP it had before startDive."""
    def startProbing(self):
        """Initiates probing, making methods SCIPnewProbingNode(), SCIPbacktrackProbing(), SCIPchgVarLbProbing(),
        SCIPchgVarUbProbing(), SCIPfixVarProbing(), SCIPpropagateProbing(), SCIPsolveProbingLP(), etc available.
        """
    def startStrongbranch(self):
        """Start strong branching. Needs to be called before any strong branching. Must also later end strong branching.
        TODO: Propagation option has currently been disabled via Python.
        If propagation is enabled then strong branching is not done on the LP, but on additionally created nodes
        (has some overhead)."""
    def tightenVarLb(self, var, lb, force=...):
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
    def tightenVarLbGlobal(self, var, lb, force=...):
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
    def tightenVarUb(self, var, ub, force=...):
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
    def tightenVarUbGlobal(self, var, ub, force=...):
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
    def to_ptr(self, give_ownership):
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
    def translateSubSol(self, sub_model, sol, heur):
        """
                \t\tTranslates a solution of a model copy into a solution of the main model
        \t\t
                \t\tParameters
                \t\t----------
                \t\tsub_model : Model
                \t\t\tThe python-wrapper of the subscip
                \t\tsol : Solution
                \t\t\tThe python-wrapper of the solution of the subscip
                \t\theur : Heur
                \t\t\tThe python-wrapper of the heuristic that found the solution
        \t\t
                \t\tReturns
                \t\t-------   \t\t
                \t\tsolution : Solution
                \t\t\tThe corresponding solution in the main model
        """
    def trySol(
        self,
        solution,
        printreason=...,
        completely=...,
        checkbounds=...,
        checkintegrality=...,
        checklprows=...,
        free=...,
    ):
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
    def updateBendersLowerbounds(self, lowerbounds, benders=...):
        """
        Updates the subproblem lower bounds for benders using
        the lowerbounds dict. If benders is None, then the default
        Benders' decomposition is updated.

        Parameters
        ----------
        lowerbounds : dict of int to float
        benders : Benders or None, optional

        """
    def updateNodeLowerbound(self, node, lb):
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
    def updateVarPseudocost(self, var, valdelta, objdelta, weight):
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
    def version(self):
        """
        Retrieve SCIP version.

        Returns
        -------
        float

        """
    def writeBestSol(self, filename=..., write_zeros=...):
        """
        Write the best feasible primal solution to a file.

        Parameters
        ----------
        filename : str, optional
            name of the output file (Default="origprob.sol")
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """
    def writeBestTransSol(self, filename=..., write_zeros=...):
        """
        Write the best feasible primal solution for the transformed problem to a file.

        Parameters
        ----------
        filename : str, optional
            name of the output file (Default="transprob.sol")
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """
    def writeLP(self, filename=...):
        """
        Writes current LP to a file.

        Parameters
        ----------
        filename : str, optional
            file name (Default value = "LP.lp")

        """
    def writeName(self, var):
        """
        Write the name of the variable to the std out.

        Parameters
        ----------
        var : Variable

        """
    def writeParams(self, filename=..., comments=..., onlychanged=..., verbose=...):
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
    def writeProblem(self, filename=..., trans=..., genericnames=..., verbose=...):
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
    def writeSol(self, solution, filename=..., write_zeros=...):
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
    def writeStatistics(self, filename=...):
        """
        Write statistics to a file.

        Parameters
        ----------
        filename : str, optional
            name of the output file (Default = "origprob.stats")

        """
    def writeTransSol(self, solution, filename=..., write_zeros=...):
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
    @override
    def __eq__(self, other: object) -> bool:
        """Return self==value."""
    def __ge__(self, other: object) -> bool:
        """Return self>=value."""
    def __gt__(self, other: object) -> bool:
        """Return self>value."""
    @override
    def __hash__(self) -> int:
        """Return hash(self)."""
    def __le__(self, other: object) -> bool:
        """Return self<=value."""
    def __lt__(self, other: object) -> bool:
        """Return self<value."""
    @override
    def __ne__(self, other: object) -> bool:
        """Return self!=value."""

class NLRow:
    data: Incomplete
    name: Incomplete
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getConstant(self):
        """
        Returns the constant of a nonlinear row.

        Returns
        -------
        float

        """
    def getDualsol(self):
        """
        Gets the dual NLP solution of a nonlinear row.

        Returns
        -------
        float

        """
    def getLhs(self):
        """
        Returns the left hand side of a nonlinear row.

        Returns
        -------
        float

        """
    def getLinearTerms(self):
        """
        Returns a list of tuples (var, coef) representing the linear part of a nonlinear row.

        Returns
        -------
        list of tuple

        """
    def getRhs(self):
        """
        Returns the right hand side of a nonlinear row.

        Returns
        -------
        float

        """
    @override
    def __eq__(self, other: object) -> bool:
        """Return self==value."""
    def __ge__(self, other: object) -> bool:
        """Return self>=value."""
    def __gt__(self, other: object) -> bool:
        """Return self>value."""
    @override
    def __hash__(self) -> int:
        """Return hash(self)."""
    def __le__(self, other: object) -> bool:
        """Return self<=value."""
    def __lt__(self, other: object) -> bool:
        """Return self<value."""
    @override
    def __ne__(self, other: object) -> bool:
        """Return self!=value."""

class Node:
    data: Incomplete
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getAddedConss(self):
        """
        Retrieve all constraints added at this node.

        Returns
        -------
        list of Constraint

        """
    def getDepth(self):
        """
        Retrieve depth of node.

        Returns
        -------
        int

        """
    def getDomchg(self):
        """
        Retrieve domain changes for this node.

        Returns
        -------
        DomainChanges

        """
    def getEstimate(self):
        """
        Retrieve the estimated value of the best feasible solution in subtree of the node.

        Returns
        -------
        float

        """
    def getLowerbound(self):
        """
        Retrieve lower bound of node.

        Returns
        -------
        float

        """
    def getNAddedConss(self):
        """
        Retrieve number of added constraints at this node.

        Returns
        -------
        int

        """
    def getNDomchg(self):
        """
        Retrieve the number of bound changes due to branching, constraint propagation, and propagation.

        Returns
        -------
        nbranchings : int
        nconsprop : int
        nprop : int

        """
    def getNParentBranchings(self):
        """
        Retrieve the number of variable branchings that were performed in the parent node to create this node.

        Returns
        -------
        int

        """
    def getNumber(self):
        """
        Retrieve number of node.

        Returns
        -------
        int

        """
    def getParent(self):
        """
        Retrieve parent node (or None if the node has no parent node).

        Returns
        -------
        Node

        """
    def getParentBranchings(self):
        """
        Retrieve the set of variable branchings that were performed in the parent node to create this node.

        Returns
        -------
        list of Variable
        list of float
        list of int

        """
    def getType(self):
        """
        Retrieve type of node.

        Returns
        -------
        PY_SCIP_NODETYPE

        """
    def isActive(self):
        """
        Is the node in the path to the current node?

        Returns
        -------
        bool

        """
    def isPropagatedAgain(self):
        """
        Is the node marked to be propagated again?

        Returns
        -------
        bool

        """
    @override
    def __eq__(self, other: object) -> bool:
        """Return self==value."""
    def __ge__(self, other: object) -> bool:
        """Return self>=value."""
    def __gt__(self, other: object) -> bool:
        """Return self>value."""
    @override
    def __hash__(self) -> int:
        """Return hash(self)."""
    def __le__(self, other: object) -> bool:
        """Return self<=value."""
    def __lt__(self, other: object) -> bool:
        """Return self<value."""
    @override
    def __ne__(self, other: object) -> bool:
        """Return self!=value."""

class PY_SCIP_BENDERSENFOTYPE:
    CHECK: ClassVar[int] = ...
    LP: ClassVar[int] = ...  # noqa: F811
    PSEUDO: ClassVar[int] = ...
    RELAX: ClassVar[int] = ...
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class PY_SCIP_BRANCHDIR:
    AUTO: ClassVar[int] = ...
    DOWNWARDS: ClassVar[int] = ...
    FIXED: ClassVar[int] = ...
    UPWARDS: ClassVar[int] = ...
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class PY_SCIP_EVENTTYPE:
    BESTSOLFOUND: ClassVar[int] = ...
    BOUNDCHANGED: ClassVar[int] = ...
    BOUNDRELAXED: ClassVar[int] = ...
    BOUNDTIGHTENED: ClassVar[int] = ...
    DISABLED: ClassVar[int] = ...
    DOMCHANGED: ClassVar[int] = ...
    FIRSTLPSOLVED: ClassVar[int] = ...
    GBDCHANGED: ClassVar[int] = ...
    GHOLEADDED: ClassVar[int] = ...
    GHOLECHANGED: ClassVar[int] = ...
    GHOLEREMOVED: ClassVar[int] = ...
    GLBCHANGED: ClassVar[int] = ...
    GUBCHANGED: ClassVar[int] = ...
    HOLECHANGED: ClassVar[int] = ...
    IMPLADDED: ClassVar[int] = ...
    LBCHANGED: ClassVar[int] = ...
    LBRELAXED: ClassVar[int] = ...
    LBTIGHTENED: ClassVar[int] = ...
    LHOLEADDED: ClassVar[int] = ...
    LHOLECHANGED: ClassVar[int] = ...
    LHOLEREMOVED: ClassVar[int] = ...
    LPEVENT: ClassVar[int] = ...
    LPSOLVED: ClassVar[int] = ...
    NODEBRANCHED: ClassVar[int] = ...
    NODEDELETE: ClassVar[int] = ...
    NODEEVENT: ClassVar[int] = ...
    NODEFEASIBLE: ClassVar[int] = ...
    NODEFOCUSED: ClassVar[int] = ...
    NODEINFEASIBLE: ClassVar[int] = ...
    NODESOLVED: ClassVar[int] = ...
    OBJCHANGED: ClassVar[int] = ...
    POORSOLFOUND: ClassVar[int] = ...
    PRESOLVEROUND: ClassVar[int] = ...
    ROWADDEDLP: ClassVar[int] = ...
    ROWADDEDSEPA: ClassVar[int] = ...
    ROWCHANGED: ClassVar[int] = ...
    ROWCOEFCHANGED: ClassVar[int] = ...
    ROWCONSTCHANGED: ClassVar[int] = ...
    ROWDELETEDLP: ClassVar[int] = ...
    ROWDELETEDSEPA: ClassVar[int] = ...
    ROWEVENT: ClassVar[int] = ...
    ROWSIDECHANGED: ClassVar[int] = ...
    SOLEVENT: ClassVar[int] = ...
    SOLFOUND: ClassVar[int] = ...
    SYNC: ClassVar[int] = ...
    UBCHANGED: ClassVar[int] = ...
    UBRELAXED: ClassVar[int] = ...
    UBTIGHTENED: ClassVar[int] = ...
    VARADDED: ClassVar[int] = ...
    VARCHANGED: ClassVar[int] = ...
    VARDELETED: ClassVar[int] = ...
    VAREVENT: ClassVar[int] = ...
    VARFIXED: ClassVar[int] = ...
    VARUNLOCKED: ClassVar[int] = ...
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class PY_SCIP_HEURTIMING:
    AFTERLPLOOP: ClassVar[int] = ...
    AFTERLPNODE: ClassVar[int] = ...
    AFTERLPPLUNGE: ClassVar[int] = ...
    AFTERPROPLOOP: ClassVar[int] = ...
    AFTERPSEUDONODE: ClassVar[int] = ...
    AFTERPSEUDOPLUNGE: ClassVar[int] = ...
    BEFORENODE: ClassVar[int] = ...
    BEFOREPRESOL: ClassVar[int] = ...
    DURINGLPLOOP: ClassVar[int] = ...
    DURINGPRESOLLOOP: ClassVar[int] = ...
    DURINGPRICINGLOOP: ClassVar[int] = ...
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class PY_SCIP_LPSOLSTAT:
    ERROR: ClassVar[int] = ...
    INFEASIBLE: ClassVar[int] = ...
    ITERLIMIT: ClassVar[int] = ...
    NOTSOLVED: ClassVar[int] = ...
    OBJLIMIT: ClassVar[int] = ...
    OPTIMAL: ClassVar[int] = ...
    TIMELIMIT: ClassVar[int] = ...
    UNBOUNDEDRAY: ClassVar[int] = ...
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class PY_SCIP_NODETYPE:
    CHILD: ClassVar[int] = ...
    DEADEND: ClassVar[int] = ...
    FOCUSNODE: ClassVar[int] = ...
    FORK: ClassVar[int] = ...
    JUNCTION: ClassVar[int] = ...
    LEAF: ClassVar[int] = ...
    PROBINGNODE: ClassVar[int] = ...
    PSEUDOFORK: ClassVar[int] = ...
    REFOCUSNODE: ClassVar[int] = ...
    SIBLING: ClassVar[int] = ...
    SUBROOT: ClassVar[int] = ...
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class PY_SCIP_PARAMEMPHASIS:
    BENCHMARK: ClassVar[int] = ...
    COUNTER: ClassVar[int] = ...
    CPSOLVER: ClassVar[int] = ...
    DEFAULT: ClassVar[int] = ...
    EASYCIP: ClassVar[int] = ...
    FEASIBILITY: ClassVar[int] = ...
    HARDLP: ClassVar[int] = ...
    NUMERICS: ClassVar[int] = ...
    OPTIMALITY: ClassVar[int] = ...
    PHASEFEAS: ClassVar[int] = ...
    PHASEIMPROVE: ClassVar[int] = ...
    PHASEPROOF: ClassVar[int] = ...
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class PY_SCIP_PARAMSETTING:
    AGGRESSIVE: ClassVar[int] = ...
    DEFAULT: ClassVar[int] = ...
    FAST: ClassVar[int] = ...
    OFF: ClassVar[int] = ...
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class PY_SCIP_PRESOLTIMING:
    EXHAUSTIVE: ClassVar[int] = ...
    FAST: ClassVar[int] = ...
    MEDIUM: ClassVar[int] = ...
    NONE: ClassVar[int] = ...
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class PY_SCIP_PROPTIMING:
    AFTERLPLOOP: ClassVar[int] = ...
    AFTERLPNODE: ClassVar[int] = ...
    BEFORELP: ClassVar[int] = ...
    DURINGLPLOOP: ClassVar[int] = ...
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class PY_SCIP_RESULT:
    BRANCHED: ClassVar[int] = ...
    CONSADDED: ClassVar[int] = ...
    CONSCHANGED: ClassVar[int] = ...
    CUTOFF: ClassVar[int] = ...
    DELAYED: ClassVar[int] = ...
    DIDNOTFIND: ClassVar[int] = ...
    DIDNOTRUN: ClassVar[int] = ...
    FEASIBLE: ClassVar[int] = ...
    FOUNDSOL: ClassVar[int] = ...
    INFEASIBLE: ClassVar[int] = ...
    NEWROUND: ClassVar[int] = ...
    REDUCEDDOM: ClassVar[int] = ...
    SEPARATED: ClassVar[int] = ...
    SOLVELP: ClassVar[int] = ...
    SUCCESS: ClassVar[int] = ...
    SUSPENDED: ClassVar[int] = ...
    UNBOUNDED: ClassVar[int] = ...
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class PY_SCIP_ROWORIGINTYPE:
    CONS: ClassVar[int] = ...
    REOPT: ClassVar[int] = ...
    SEPA: ClassVar[int] = ...
    UNSPEC: ClassVar[int] = ...
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class PY_SCIP_SOLORIGIN:
    LPSOL: ClassVar[int] = ...
    NLPSOL: ClassVar[int] = ...
    ORIGINAL: ClassVar[int] = ...
    PARTIAL: ClassVar[int] = ...
    PSEUDOSOL: ClassVar[int] = ...
    RELAXSOL: ClassVar[int] = ...
    UNKNOWN: ClassVar[int] = ...
    ZERO: ClassVar[int] = ...
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class PY_SCIP_STAGE:
    EXITPRESOLVE: ClassVar[int] = ...
    EXITSOLVE: ClassVar[int] = ...
    FREE: ClassVar[int] = ...
    FREETRANS: ClassVar[int] = ...
    INIT: ClassVar[int] = ...
    INITPRESOLVE: ClassVar[int] = ...
    INITSOLVE: ClassVar[int] = ...
    PRESOLVED: ClassVar[int] = ...
    PRESOLVING: ClassVar[int] = ...
    PROBLEM: ClassVar[int] = ...
    SOLVED: ClassVar[int] = ...
    SOLVING: ClassVar[int] = ...
    TRANSFORMED: ClassVar[int] = ...
    TRANSFORMING: ClassVar[int] = ...
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class PY_SCIP_STATUS:
    BESTSOLLIMIT: ClassVar[int] = ...
    DUALLIMIT: ClassVar[int] = ...
    GAPLIMIT: ClassVar[int] = ...
    INFEASIBLE: ClassVar[int] = ...
    INFORUNBD: ClassVar[int] = ...
    MEMLIMIT: ClassVar[int] = ...
    NODELIMIT: ClassVar[int] = ...
    OPTIMAL: ClassVar[int] = ...
    PRIMALLIMIT: ClassVar[int] = ...
    RESTARTLIMIT: ClassVar[int] = ...
    SOLLIMIT: ClassVar[int] = ...
    STALLNODELIMIT: ClassVar[int] = ...
    TIMELIMIT: ClassVar[int] = ...
    TOTALNODELIMIT: ClassVar[int] = ...
    UNBOUNDED: ClassVar[int] = ...
    UNKNOWN: ClassVar[int] = ...
    USERINTERRUPT: ClassVar[int] = ...
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class Row:
    data: Incomplete
    name: Incomplete
    def __init__(self, *args) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getAge(self):
        """
        Gets the age of the row. (The consecutive times the row has been non-active in the LP).

        Returns
        -------
        int

        """
    def getBasisStatus(self):
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
    def getCols(self):
        """
        Gets list with columns of nonzero entries

        Returns
        -------
        list of Column

        """
    def getConsOriginConshdlrtype(self):
        """
        Returns type of constraint handler that created the row.

        Returns
        -------
        str

        """
    def getConstant(self):
        """
        Gets constant shift of row.

        Returns
        -------
        float

        """
    def getLPPos(self):
        """
        Gets position of row in current LP, or -1 if it is not in LP.

        Returns
        -------
        int

        """
    def getLhs(self):
        """
        Returns the left hand side of row.

        Returns
        -------
        float

        """
    def getNLPNonz(self):
        """
        Get number of nonzero entries in row vector that correspond to columns currently in the SCIP LP.

        Returns
        -------
        int

        """
    def getNNonz(self):
        """
        Get number of nonzero entries in row vector.

        Returns
        -------
        int

        """
    def getNorm(self):
        """
        Gets Euclidean norm of row vector.

        Returns
        -------
        float

        """
    def getOrigintype(self):
        """
        Returns type of origin that created the row.

        Returns
        -------
        PY_SCIP_ROWORIGINTYPE

        """
    def getRhs(self):
        """
        Returns the right hand side of row.

        Returns
        -------
        float

        """
    def getVals(self):
        """
        Gets list with coefficients of nonzero entries.

        Returns
        -------
        list of int

        """
    def isInGlobalCutpool(self):
        """
        Return TRUE iff row is a member of the global cut pool.

        Returns
        -------
        bool

        """
    def isIntegral(self):
        """
        Returns TRUE iff the activity of the row (without the row's constant)
        is always integral in a feasible solution.

        Returns
        -------
        bool

        """
    def isLocal(self):
        """
        Returns TRUE iff the row is only valid locally.

        Returns
        -------
        bool

        """
    def isModifiable(self):
        """
        Returns TRUE iff row is modifiable during node processing (subject to column generation).

        Returns
        -------
        bool

        """
    def isRemovable(self):
        """
        Returns TRUE iff row is removable from the LP (due to aging or cleanup).

        Returns
        -------
        bool

        """
    @override
    def __eq__(self, other: object) -> bool:
        """Return self==value."""
    def __ge__(self, other: object) -> bool:
        """Return self>=value."""
    def __gt__(self, other: object) -> bool:
        """Return self>value."""
    @override
    def __hash__(self) -> int:
        """Return hash(self)."""
    def __le__(self, other: object) -> bool:
        """Return self<=value."""
    def __lt__(self, other: object) -> bool:
        """Return self<value."""
    @override
    def __ne__(self, other: object) -> bool:
        """Return self!=value."""

class Solution:
    data: Incomplete
    def __init__(self, *args) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getOrigin(self):
        """
        Returns origin of solution: where to retrieve uncached elements.

        Returns
        -------
        PY_SCIP_SOLORIGIN
        """
    def retransform(self):
        """retransforms solution to original problem space"""
    def translate(self, target):
        """
        translate solution to a target model solution

        Parameters
        ----------
        target : Model

        Returns
        -------
        targetSol: Solution
        """
    def __delitem__(self, other) -> None:
        """Delete self[key]."""
    def __getitem__(self, index):
        """Return self[key]."""
    def __setitem__(self, index, object) -> None:
        """Set self[key] to value."""

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
    _variables: dict
    _presolved_variables: dict
    _constraints: dict
    _presolved_constraints: dict
    n_runs: int | None = None
    n_nodes: int | None = None
    n_solutions_found: int = -1
    first_solution: float | None = None
    primal_bound: float | None = None
    dual_bound: float | None = None
    gap: float | None = None
    primal_dual_integral: float | None = None
    @property
    def n_binary_vars(self): ...
    @property
    def n_conss(self): ...
    @property
    def n_continuous_vars(self): ...
    @property
    def n_implicit_integer_vars(self): ...
    @property
    def n_integer_vars(self): ...
    @property
    def n_maximal_cons(self): ...
    @property
    def n_presolved_binary_vars(self): ...
    @property
    def n_presolved_conss(self): ...
    @property
    def n_presolved_continuous_vars(self): ...
    @property
    def n_presolved_implicit_integer_vars(self): ...
    @property
    def n_presolved_integer_vars(self): ...
    @property
    def n_presolved_maximal_cons(self): ...
    @property
    def n_presolved_vars(self): ...
    @property
    def n_vars(self): ...

class Variable(Expr):
    data: Incomplete
    name: Incomplete
    def __init__(self) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def getAvgSol(self):
        """
        Get the weighted average solution of variable in all feasible primal solutions found.

        Returns
        -------
        float

        """
    def getCol(self) -> Column:
        """
        Retrieve column of COLUMN variable.

        Returns
        -------
        Column

        """
    def getIndex(self):
        """
        Retrieve the unique index of the variable.

        Returns
        -------
        int

        """
    def getLPSol(self) -> float: ...
    def getLbGlobal(self):
        """
        Retrieve global lower bound of variable.

        Returns
        -------
        float

        """
    def getLbLocal(self):
        """
        Retrieve current lower bound of variable.

        Returns
        -------
        float

        """
    def getLbOriginal(self):
        """
        Retrieve original lower bound of variable.

        Returns
        -------
        float

        """
    def getObj(self):
        """
        Retrieve current objective value of variable.

        Returns
        -------
        float

        """
    def getUbGlobal(self):
        """
        Retrieve global upper bound of variable.

        Returns
        -------
        float

        """
    def getUbLocal(self):
        """
        Retrieve current upper bound of variable.

        Returns
        -------
        float

        """
    def getUbOriginal(self):
        """
        Retrieve original upper bound of variable.

        Returns
        -------
        float

        """
    def isInLP(self) -> bool: ...
    def isOriginal(self):
        """
        Retrieve whether the variable belongs to the original problem

        Returns
        -------
        bool

        """
    def ptr(self):
        """ """
    def varMayRound(self, direction=...):
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
    def vtype(self):
        """
        Retrieve the variables type (BINARY, INTEGER, IMPLINT or CONTINUOUS)

        Returns
        -------
        str
            "BINARY", "INTEGER", "CONTINUOUS", or "IMPLINT"

        """
