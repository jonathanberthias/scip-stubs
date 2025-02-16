from _typeshed import Incomplete

class Branchrule:
    model: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def branchexecext(self, *args, **kwargs):
        """executes branching rule for external branching candidates"""
    def branchexeclp(self, *args, **kwargs):
        """executes branching rule for fractional LP solution"""
    def branchexecps(self, *args, **kwargs):
        """executes branching rule for not completely fixed pseudo solution"""
    def branchexit(self, *args, **kwargs):
        """deinitializes branching rule"""
    def branchexitsol(self, *args, **kwargs):
        """informs branching rule that the branch and bound process data is being freed"""
    def branchfree(self, *args, **kwargs):
        """frees memory of branching rule"""
    def branchinit(self, *args, **kwargs):
        """initializes branching rule"""
    def branchinitsol(self, *args, **kwargs):
        """informs branching rule that the branch and bound process is being started"""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
