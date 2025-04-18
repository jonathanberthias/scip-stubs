from __future__ import annotations
from typing_extensions import override

from typing import TYPE_CHECKING

from pyscipopt import Model, SCIP_RESULT
from pyscipopt.scip import Cutsel, Row

if TYPE_CHECKING:
    from pyscipopt.scip import CutSelSelectReturnTD

scip = Model()


# What is a Cut Selector?

# Cut Selector Structure


class DummyCutsel(Cutsel):
    @override
    def cutselselect(
        self, cuts: list[Row], forcedcuts: list[Row], root: bool, maxnselectedcuts: int
    ) -> CutSelSelectReturnTD:
        """
        :param cuts: the cuts which we want to select from. Is a list of scip Rows
        :param forcedcuts: the cuts which we must add. Is a list of scip Rows
        :param root: boolean indicating whether weare at the root node
        :param maxnselectedcuts: int which is the maximum amount of cuts that can be selected
        :return: sorted cuts and forcedcuts
        """
        sorted_cuts = cuts
        n = len(sorted_cuts)
        return {"cuts": sorted_cuts, "nselectedcuts": n, "result": SCIP_RESULT.SUCCESS}


cutsel = DummyCutsel()
scip.includeCutsel(cutsel, "name", "description", 5000000)

# Example Cut Selector


class MaxEfficacyCutsel(Cutsel):
    @override
    def cutselselect(
        self, cuts: list[Row], forcedcuts: list[Row], root: bool, maxnselectedcuts: int
    ) -> CutSelSelectReturnTD:
        """
        Selects the 10 cuts with largest efficacy.
        """

        scip = self.model

        scores: list[float] = [0] * len(cuts)
        for i in range(len(scores)):
            scores[i] = scip.getCutEfficacy(cuts[i])

        rankings = sorted(range(len(cuts)), key=lambda x: scores[x], reverse=True)

        sorted_cuts = [cuts[rank] for rank in rankings]

        assert len(sorted_cuts) == len(cuts)

        return {
            "cuts": sorted_cuts,
            "nselectedcuts": min(maxnselectedcuts, len(cuts), 10),
            "result": SCIP_RESULT.SUCCESS,
        }
