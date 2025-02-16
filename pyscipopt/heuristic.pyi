from _typeshed import Incomplete

class Heur:
    model: Incomplete
    name: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def heurexec(self, *args, **kwargs):
        """should the heuristic the executed at the given depth, frequency, timing,..."""
    def heurexit(self, *args, **kwargs):
        """calls exit method of primal heuristic"""
    def heurexitsol(self, *args, **kwargs):
        """informs primal heuristic that the branch and bound process data is being freed"""
    def heurfree(self, *args, **kwargs):
        """calls destructor and frees memory of primal heuristic"""
    def heurinit(self, *args, **kwargs):
        """initializes primal heuristic"""
    def heurinitsol(self, *args, **kwargs):
        """informs primal heuristic that the branch and bound process is being started"""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
