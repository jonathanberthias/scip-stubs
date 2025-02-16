from _typeshed import Incomplete

class LP:
    name: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """
        Keyword arguments:
        name -- the name of the problem (default 'LP')
        sense -- objective sense (default minimize)
        """
    def addCol(self, *args, **kwargs):
        """Adds a single column to the LP.

        Keyword arguments:
        entries -- list of tuples, each tuple consists of a row index and a coefficient
        obj     -- objective coefficient (default 0.0)
        lb      -- lower bound (default 0.0)
        ub      -- upper bound (default infinity)
        """
    def addCols(self, *args, **kwargs):
        """Adds multiple columns to the LP.

        Keyword arguments:
        entrieslist -- list containing lists of tuples, each tuple contains a coefficient and a row index
        objs  -- objective coefficient (default 0.0)
        lbs   -- lower bounds (default 0.0)
        ubs   -- upper bounds (default infinity)
        """
    def addRow(self, *args, **kwargs):
        """Adds a single row to the LP.

        Keyword arguments:
        entries -- list of tuples, each tuple contains a coefficient and a column index
        lhs     -- left-hand side of the row (default 0.0)
        rhs     -- right-hand side of the row (default infinity)
        """
    def addRows(self, *args, **kwargs):
        """Adds multiple rows to the LP.

        Keyword arguments:
        entrieslist -- list containing lists of tuples, each tuple contains a coefficient and a column index
        lhss        -- left-hand side of the row (default 0.0)
        rhss        -- right-hand side of the row (default infinity)
        """
    def chgBound(self, *args, **kwargs):
        """Changes the lower and upper bound of a single column.

        Keyword arguments:
        col -- column to change
        lb  -- new lower bound
        ub  -- new upper bound
        """
    def chgCoef(self, *args, **kwargs):
        """Changes a single coefficient in the LP.

        Keyword arguments:
        row -- row to change
        col -- column to change
        newval -- new coefficient
        """
    def chgObj(self, *args, **kwargs):
        """Changes objective coefficient of a single column.

        Keyword arguments:
        col -- column to change
        obj -- new objective coefficient
        """
    def chgSide(self, *args, **kwargs):
        """Changes the left- and right-hand side of a single row.

        Keyword arguments:
        row -- row to change
        lhs -- new left-hand side
        rhs -- new right-hand side
        """
    def clear(self, *args, **kwargs):
        """Clears the whole LP."""
    def delCols(self, *args, **kwargs):
        """Deletes a range of columns from the LP.

        Keyword arguments:
        firstcol -- first column to delete
        lastcol  -- last column to delete
        """
    def delRows(self, *args, **kwargs):
        """Deletes a range of rows from the LP.

        Keyword arguments:
        firstrow -- first row to delete
        lastrow  -- last row to delete
        """
    def getBasisInds(self, *args, **kwargs):
        """Returns the indices of the basic columns and rows; index i >= 0 corresponds to column i, index i < 0 to row -i-1"""
    def getBounds(self, *args, **kwargs):
        """Returns all lower and upper bounds for a range of columns.

        Keyword arguments:
        firstcol -- first column (default 0)
        lastcol  -- last column (default ncols - 1)
        """
    def getDual(self, *args, **kwargs):
        """Returns the dual solution of the last LP solve."""
    def getDualRay(self, *args, **kwargs):
        """Returns a dual ray if possible, None otherwise."""
    def getNIterations(self, *args, **kwargs):
        """Returns the number of LP iterations of the last LP solve."""
    def getPrimal(self, *args, **kwargs):
        """Returns the primal solution of the last LP solve."""
    def getPrimalRay(self, *args, **kwargs):
        """Returns a primal ray if possible, None otherwise."""
    def getRedcost(self, *args, **kwargs):
        """Returns the reduced cost vector of the last LP solve."""
    def getSides(self, *args, **kwargs):
        """Returns all left- and right-hand sides for a range of rows.

        Keyword arguments:
        firstrow -- first row (default 0)
        lastrow  -- last row (default nrows - 1)
        """
    def infinity(self, *args, **kwargs):
        """Returns infinity value of the LP."""
    def isDualFeasible(self, *args, **kwargs):
        """Returns True iff LP is proven to be dual feasible."""
    def isInfinity(self, *args, **kwargs):
        """Checks if a given value is equal to the infinity value of the LP.

        Keyword arguments:
        val -- value that should be checked
        """
    def isPrimalFeasible(self, *args, **kwargs):
        """Returns True iff LP is proven to be primal feasible."""
    def ncols(self, *args, **kwargs):
        """Returns the number of columns."""
    def nrows(self, *args, **kwargs):
        """Returns the number of rows."""
    def readLP(self, *args, **kwargs):
        """Reads LP from a file.

        Keyword arguments:
        filename -- the name of the file to be used
        """
    def solve(self, *args, **kwargs):
        """Solves the current LP.

        Keyword arguments:
        dual -- use the dual or primal Simplex method (default: dual)
        """
    def writeLP(self, *args, **kwargs):
        """Writes LP to a file.

        Keyword arguments:
        filename -- the name of the file to be used
        """
    def __reduce__(self): ...
