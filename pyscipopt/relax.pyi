from _typeshed import Incomplete

class Relax:
    model: Incomplete
    name: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def relaxexec(self, *args, **kwargs):
        """callls execution method of relaxation handler"""
    def relaxexit(self, *args, **kwargs):
        """calls exit method of relaxation handler"""
    def relaxexitsol(self, *args, **kwargs):
        """informs relaxation handler that the branch and bound process data is being freed"""
    def relaxfree(self, *args, **kwargs):
        """calls destructor and frees memory of relaxation handler"""
    def relaxinit(self, *args, **kwargs):
        """initializes relaxation handler"""
    def relaxinitsol(self, *args, **kwargs):
        """informs relaxaton handler that the branch and bound process is being started"""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
