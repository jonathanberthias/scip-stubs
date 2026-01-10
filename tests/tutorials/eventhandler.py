from __future__ import annotations

from typing_extensions import override

from pyscipopt import SCIP_EVENTTYPE, Eventhdlr, Model
from pyscipopt.scip import Event

scip = Model()


# SCIP Events

# What's an Event Handler?

# Adding Event Handlers with Callbacks


def print_obj_value(model: Model, event: Event) -> None:
    print("New best solution found with objective value: {}".format(model.getObjVal()))


m = Model()
m.attachEventHandlerCallback(print_obj_value, [SCIP_EVENTTYPE.BESTSOLFOUND])
m.optimize()


# Adding Event Handlers with Classes


class BestSolCounter(Eventhdlr):
    def __init__(self, model: Model) -> None:
        self.model = model
        self.count = 0

    @override
    def eventinit(self) -> None:
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    @override
    def eventexit(self) -> None:
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    @override
    def eventexec(self, event: Event) -> None:
        self.count += 1
        print(
            "!!!![@BestSolCounter] New best solution found. Total best solutions found: {}".format(
                self.count
            )
        )


m = Model()
best_sol_counter = BestSolCounter(m)
m.includeEventhdlr(
    best_sol_counter,
    "best_sol_event_handler",
    "Event handler that counts the number of best solutions found",
)
m.optimize()
assert best_sol_counter.count == 1
