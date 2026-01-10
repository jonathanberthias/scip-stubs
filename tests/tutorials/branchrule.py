from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from pyscipopt import SCIP_RESULT, Branchrule, Model

if TYPE_CHECKING:
    from pyscipopt.scip import BranchRuleExecTD

scip = Model()


# What is Branching

# Example Branching Rule

import numpy as np


class MostInfBranchRule(Branchrule):
    def __init__(self, scip: Model):
        self.scip = scip

    @override
    def branchexeclp(self, allowaddcons: bool) -> BranchRuleExecTD:
        # Get the branching candidates. Only consider the number of priority candidates (they are sorted to be first)
        # The implicit integer candidates in general shouldn't be branched on. Unless specified by the user
        # npriocands and ncands are the same (npriocands are variables that have been designated as priorities)
        (
            branch_cands,
            branch_cand_sols,
            branch_cand_fracs,
            ncands,
            npriocands,
            nimplcands,
        ) = self.scip.getLPBranchCands()

        # Find the variable that is most fractional
        best_cand_idx = 0
        best_dist = np.inf
        for i in range(npriocands):
            if abs(branch_cand_fracs[i] - 0.5) <= best_dist:
                best_dist = abs(branch_cand_fracs[i] - 0.5)
                best_cand_idx = i

        # Branch on the variable with the largest score
        down_child, eq_child, up_child = self.model.branchVarVal(
            branch_cands[best_cand_idx], branch_cand_sols[best_cand_idx]
        )

        return {"result": SCIP_RESULT.BRANCHED}


most_inf_branch_rule = MostInfBranchRule(scip)
scip.includeBranchrule(
    most_inf_branch_rule,
    "mostinfeasible",
    "custom most infeasible branching rule",
    priority=10000000,
    maxdepth=-1,
    maxbounddist=1,
)

# Strong Branching Information


class StrongBranchingRule(Branchrule):
    def __init__(self, scip: Model) -> None:
        self.scip = scip

    @override
    def branchexeclp(self, allowaddcons: bool) -> BranchRuleExecTD:
        (
            branch_cands,
            branch_cand_sols,
            branch_cand_fracs,
            ncands,
            npriocands,
            nimplcands,
        ) = self.scip.getLPBranchCands()

        # Initialise scores for each variable
        scores = [-self.scip.infinity() for _ in range(npriocands)]
        down_bounds: list[float | None] = [None for _ in range(npriocands)]
        up_bounds: list[float | None] = [None for _ in range(npriocands)]

        # Initialise placeholder values
        num_nodes = self.scip.getNNodes()
        lpobjval = self.scip.getLPObjVal()
        lperror = False
        best_cand_idx = 0

        # Start strong branching and iterate over the branching candidates
        self.scip.startStrongbranch()
        for i in range(npriocands):
            # Check the case that the variable has already been strong branched on at this node.
            # This case occurs when events happen in the node that should be handled immediately.
            # When processing the node again (because the event did not remove it), there's no need to duplicate work.
            if self.scip.getVarStrongbranchNode(branch_cands[i]) == num_nodes:
                down, up, downvalid, upvalid, _, lastlpobjval = (
                    self.scip.getVarStrongbranchLast(branch_cands[i])
                )
                if downvalid:
                    down_bounds[i] = down
                if upvalid:
                    up_bounds[i] = up
                downgain = max([down - lastlpobjval, 0])
                upgain = max([up - lastlpobjval, 0])
                scores[i] = self.scip.getBranchScoreMultiple(
                    branch_cands[i], [downgain, upgain]
                )
                continue

            # Strong branch!
            (
                down,
                up,
                downvalid,
                upvalid,
                downinf,
                upinf,
                downconflict,
                upconflict,
                lperror,
            ) = self.scip.getVarStrongbranch(branch_cands[i], 200, idempotent=False)

            # In the case of an LP error handle appropriately (for this example we just break the loop)
            if lperror:
                break

            # In the case of both infeasible sub-problems cutoff the node
            if downinf and upinf:
                return {"result": SCIP_RESULT.CUTOFF}

            # Calculate the gains for each up and down node that strong branching explored
            if not downinf and downvalid:
                down_bounds[i] = down
                downgain = max([down - lpobjval, 0])
            else:
                downgain = 0
            if not upinf and upvalid:
                up_bounds[i] = up
                upgain = max([up - lpobjval, 0])
            else:
                upgain = 0

            # Update the pseudo-costs
            lpsol = branch_cands[i].getLPSol()
            if not downinf and downvalid:
                self.scip.updateVarPseudocost(
                    branch_cands[i], -self.scip.frac(lpsol), downgain, 1
                )
            if not upinf and upvalid:
                self.scip.updateVarPseudocost(
                    branch_cands[i], 1 - self.scip.frac(lpsol), upgain, 1
                )

            scores[i] = self.scip.getBranchScoreMultiple(
                branch_cands[i], [downgain, upgain]
            )
            if scores[i] > scores[best_cand_idx]:
                best_cand_idx = i

        # End strong branching
        self.scip.endStrongbranch()

        # In the case of an LP error
        if lperror:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # Branch on the variable with the largest score
        down_child, eq_child, up_child = self.model.branchVarVal(
            branch_cands[best_cand_idx], branch_cands[best_cand_idx].getLPSol()
        )

        # Update the bounds of the down node and up node. Some cols might not exist due to pricing
        if self.scip.allColsInLP():
            # TODO: can down_child and up_child be None?
            down_bound = down_bounds[best_cand_idx]
            up_bound = up_bounds[best_cand_idx]
            if down_child is not None and down_bound is not None:  # type: ignore[redundant-expr]
                self.scip.updateNodeLowerbound(down_child, down_bound)
            if up_child is not None and up_bound is not None:  # type: ignore[redundant-expr]
                self.scip.updateNodeLowerbound(up_child, up_bound)

        return {"result": SCIP_RESULT.BRANCHED}
