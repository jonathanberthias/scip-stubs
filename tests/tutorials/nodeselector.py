from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from pyscipopt import Model, Nodesel
from pyscipopt.scip import Node

if TYPE_CHECKING:
    from pyscipopt.scip import NodeselNodeselectTD

scip = Model()

# What is a Node Selector?


# Example Node Selector


# Depth First Search Node Selector
class DFS(Nodesel):
    def __init__(self, scip: Model) -> None:
        self.scip = scip

    @override
    def nodeselect(self) -> NodeselNodeselectTD:
        """Decide which of the leaves from the branching tree to process next"""
        selnode: Node | None = self.scip.getPrioChild()
        if selnode is None:
            selnode = self.scip.getPrioSibling()
            if selnode is None:
                selnode = self.scip.getBestLeaf()

        assert selnode is not None
        return {"selnode": selnode}

    @override
    def nodecomp(self, node1: Node, node2: Node) -> int:
        """
        compare two leaves of the current branching tree

        It should return the following values:

        value < 0, if node 1 comes before (is better than) node 2
        value = 0, if both nodes are equally good
        value > 0, if node 1 comes after (is worse than) node 2.
        """
        depth_1: int = node1.getDepth()
        depth_2: int = node2.getDepth()
        if depth_1 > depth_2:
            return -1
        elif depth_1 < depth_2:
            return 1
        else:
            lb_1: float = node1.getLowerbound()
            lb_2: float = node2.getLowerbound()
            if lb_1 < lb_2:
                return -1
            elif lb_1 > lb_2:
                return 1
            else:
                return 0


dfs_node_sel = DFS(scip)
scip.includeNodesel(
    dfs_node_sel, "DFS", "Depth First Search Nodesel.", 1000000, 1000000
)
