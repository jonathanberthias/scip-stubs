from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from pyscipopt import (
    SCIP_HEURTIMING,
    SCIP_LPSOLSTAT,
    SCIP_RESULT,
    Heur,
    Model,
    Variable,
)
from pyscipopt.scip import Solution

if TYPE_CHECKING:
    from pyscipopt.scip import HeurExecResultTD


scip = Model()


# What is a Heuristic?


# Simple Rounding Heuristic Example


class SimpleRoundingHeuristic(Heur):
    @override
    def heurexec(
        self, heurtiming: SCIP_HEURTIMING, nodeinfeasible: bool
    ) -> HeurExecResultTD:
        scip: Model = self.model
        result = SCIP_RESULT.DIDNOTRUN

        # This heuristic does not run if the LP status is not optimal
        lpsolstat = scip.getLPSolstat()
        if lpsolstat != SCIP_LPSOLSTAT.OPTIMAL:
            return {"result": result}  # type: ignore[typeddict-item]

        # We haven't added handling of implicit integers to this heuristic
        if scip.getNImplVars() > 0:
            return {"result": result}  # type: ignore[typeddict-item]

        # Get the current branching candidate, i.e., the current fractional variables with integer requirements
        (
            branch_cands,
            branch_cand_sols,
            branch_cand_fracs,
            ncands,
            npriocands,
            nimplcands,
        ) = scip.getLPBranchCands()

        # Ignore if there are no branching candidates
        if ncands == 0:
            return {"result": result}  # type: ignore[typeddict-item]

        # Create a solution that is initialised to the LP values
        sol: Solution = scip.createSol(self, initlp=True)

        # Now round the variables that can be rounded
        for i in range(ncands):
            old_sol_val: float = branch_cand_sols[i]
            scip_var: Variable = branch_cands[i]
            may_round_up: bool = scip_var.varMayRound(direction="up")
            may_round_down: bool = scip_var.varMayRound(direction="down")
            # If we can round in both directions then round in objective function direction
            if may_round_up and may_round_down:
                if scip_var.getObj() >= 0.0:
                    new_sol_val = scip.feasFloor(old_sol_val)
                else:
                    new_sol_val = scip.feasCeil(old_sol_val)
            elif may_round_down:
                new_sol_val = scip.feasFloor(old_sol_val)
            elif may_round_up:
                new_sol_val = scip.feasCeil(old_sol_val)
            else:
                # The variable cannot be rounded. The heuristic will fail.
                continue

            # Set the rounded new solution value
            scip.setSolVal(sol, scip_var, new_sol_val)

        # Now try the solution. Note: This will free the solution afterwards by default.
        stored = scip.trySol(sol)

        if stored:
            return {"result": SCIP_RESULT.FOUNDSOL}
        else:
            return {"result": SCIP_RESULT.DIDNOTFIND}


heuristic = SimpleRoundingHeuristic()
scip.includeHeur(
    heuristic,
    "SimpleRounding",
    "custom heuristic implemented in python",
    "Y",
    timingmask=SCIP_HEURTIMING.DURINGLPLOOP,
)
