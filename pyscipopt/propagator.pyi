from _typeshed import Incomplete

class Prop:
    model: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def propexec(self, *args, **kwargs):
        """calls execution method of propagator"""
    def propexit(self, *args, **kwargs):
        """calls exit method of propagator"""
    def propexitpre(self, *args, **kwargs):
        """informs propagator that the presolving process is finished"""
    def propexitsol(self, *args, **kwargs):
        """informs propagator that the prop and bound process data is being freed"""
    def propfree(self, *args, **kwargs):
        """calls destructor and frees memory of propagator"""
    def propinit(self, *args, **kwargs):
        """initializes propagator"""
    def propinitpre(self, *args, **kwargs):
        """informs propagator that the presolving process is being started"""
    def propinitsol(self, *args, **kwargs):
        """informs propagator that the prop and bound process is being started"""
    def proppresol(self, *args, **kwargs):
        """executes presolving method of propagator"""
    def propresprop(self, *args, **kwargs):
        """resolves the given conflicting bound, that was reduced by the given propagator"""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
