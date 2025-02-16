from _typeshed import Incomplete

class Cutsel:
    model: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def cutselexit(self, *args, **kwargs):
        """executed before the transformed problem is freed"""
    def cutselexitsol(self, *args, **kwargs):
        """executed before the branch-and-bound process is freed"""
    def cutselfree(self, *args, **kwargs):
        """frees memory of cut selector"""
    def cutselinit(self, *args, **kwargs):
        """executed after the problem is transformed. use this call to initialize cut selector data."""
    def cutselinitsol(self, *args, **kwargs):
        """executed when the presolving is finished and the branch-and-bound process is about to begin"""
    def cutselselect(self, *args, **kwargs):
        """first method called in each iteration in the main solving loop."""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
