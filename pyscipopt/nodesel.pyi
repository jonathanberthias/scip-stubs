from _typeshed import Incomplete

class Nodesel:
    model: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def nodecomp(self, *args, **kwargs):
        """
        compare two leaves of the current branching tree

        It should return the following values:

          value < 0, if node 1 comes before (is better than) node 2
          value = 0, if both nodes are equally good
          value > 0, if node 1 comes after (is worse than) node 2.
        """
    def nodeexit(self, *args, **kwargs):
        """executed before the transformed problem is freed"""
    def nodeexitsol(self, *args, **kwargs):
        """executed before the branch-and-bound process is freed"""
    def nodefree(self, *args, **kwargs):
        """frees memory of node selector"""
    def nodeinit(self, *args, **kwargs):
        """executed after the problem is transformed. use this call to initialize node selector data."""
    def nodeinitsol(self, *args, **kwargs):
        """executed when the presolving is finished and the branch-and-bound process is about to begin"""
    def nodeselect(self, *args, **kwargs):
        """first method called in each iteration in the main solving loop."""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
