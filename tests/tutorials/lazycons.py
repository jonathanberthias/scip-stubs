from __future__ import annotations

from typing import TYPE_CHECKING

import networkx  # type: ignore[import-untyped]
import numpy as np
from typing_extensions import override

from pyscipopt import SCIP_RESULT, Conshdlr, Constraint, Model, Variable, quicksum
from pyscipopt.scip import Constraint, Solution

if TYPE_CHECKING:
    from pyscipopt.scip import ConshdlrConsCheckRes, ConshdlrEnfoRes


# TSP Subtour Elimination Constraint Example

scip = Model()

n = 300
x: dict[int, dict[int, Variable]] = {}
c: dict[int, dict[int, float]] = {}
for i in range(n):
    x[i] = {}
    c[i] = {}
    for j in range(i + 1, n):
        x[i][j] = scip.addVar(vtype="B", name=f"x_{i}_{j}")
        c[i][j] = np.random.uniform(10)
scip.setObjective(
    quicksum(quicksum(c[i][j] * x[i][j] for j in range(i + 1, n)) for i in range(n)),
    "minimize",
)
for i in range(n):
    scip.addCons(
        quicksum(x[i][j] for j in range(i + 1, n))
        + quicksum(x[j][i] for j in range(i - 1, 0, -1))
        == 2,
        name=f"sum_in_out_{i}",
    )


# subtour elimination constraint handler
class SEC(Conshdlr):
    # method for creating a constraint of this constraint handler type
    def createCons(
        self, name: str, variables: dict[int, dict[int, Variable]]
    ) -> Constraint:
        model: Model = self.model
        cons: Constraint = model.createCons(self, name)

        # data relevant for the constraint; in this case we only need to know which
        # variables cannot form a subtour
        cons.data = {"vars": variables}
        return cons

    # find subtours in the graph induced by the edges {i,j} for which x[i][j] is positive
    # at the given solution; when solution is None, the LP solution is used
    def find_subtours(
        self, cons: Constraint, solution: Solution | None = None
    ) -> list[set[int]]:
        edges = []
        x: dict[int, dict[int, Variable]] = cons.data["vars"]

        for i in list(x.keys()):
            for j in list(x[i].keys()):
                if self.model.getSolVal(solution, x[i][j]) > 0.5:
                    edges.append((i, j))

        G = networkx.Graph()
        G.add_edges_from(edges)
        components: list[set[int]] = list(networkx.connected_components(G))

        if len(components) == 1:
            return []
        else:
            return components

    # checks whether solution is feasible
    @override
    def conscheck(
        self,
        constraints: list[Constraint],
        solution: Solution,
        checkintegrality: bool,
        checklprows: bool,
        printreason: bool,
        completely: bool,
    ) -> ConshdlrConsCheckRes:
        # check if there is a violated subtour elimination constraint
        for cons in constraints:
            if self.find_subtours(cons, solution):
                return {"result": SCIP_RESULT.INFEASIBLE}

        # no violated constriant found -> feasible
        return {"result": SCIP_RESULT.FEASIBLE}

    # enforces the LP solution: searches for subtours in the solution and adds
    # adds constraints forbidding all the found subtours
    @override
    def consenfolp(
        self, constraints: list[Constraint], nusefulconss: int, solinfeasible: bool
    ) -> ConshdlrEnfoRes:
        consadded = False

        for cons in constraints:
            subtours = self.find_subtours(cons)

            # if there are subtours
            if subtours:
                x: dict[int, dict[int, Variable]] = cons.data["vars"]

                # add subtour elimination constraint for each subtour
                for S in subtours:
                    print("Constraint added!")
                    self.model.addCons(
                        quicksum(x[i][j] for i in S for j in S if j > i) <= len(S) - 1
                    )
                    consadded = True

        if consadded:
            return {"result": SCIP_RESULT.CONSADDED}
        else:
            return {"result": SCIP_RESULT.FEASIBLE}

    # this is rather technical and not relevant for the exercise. to learn more see
    # https://scipopt.org/doc/html/CONS.php#CONS_FUNDAMENTALCALLBACKS
    @override
    def conslock(
        self,
        constraint: Constraint | None,
        locktype: int,
        nlockspos: int,
        nlocksneg: int,
    ) -> None:
        pass


# create the constraint handler
conshdlr = SEC()

# Add the constraint handler to SCIP. We set check priority < 0 so that only integer feasible solutions
# are passed to the conscheck callback
scip.includeConshdlr(
    conshdlr, "TSP", "TSP subtour eliminator", chckpriority=-10, enfopriority=-10
)

# create a subtour elimination constraint
cons: Constraint = conshdlr.createCons("no_subtour_cons", x)

# add constraint to SCIP
scip.addPyCons(cons)
