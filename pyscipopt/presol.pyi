from _typeshed import Incomplete

class Presol:
    model: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def presolexec(self, *args, **kwargs):
        """executes presolver"""
    def presolexit(self, *args, **kwargs):
        """deinitializes presolver"""
    def presolexitpre(self, *args, **kwargs):
        """informs presolver that the presolving process is finished"""
    def presolfree(self, *args, **kwargs):
        """frees memory of presolver"""
    def presolinit(self, *args, **kwargs):
        """initializes presolver"""
    def presolinitpre(self, *args, **kwargs):
        """informs presolver that the presolving process is being started"""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
