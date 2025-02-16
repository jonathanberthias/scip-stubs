from _typeshed import Incomplete

class Sepa:
    model: Incomplete
    name: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def sepaexeclp(self, *args, **kwargs):
        """calls LP separation method of separator"""
    def sepaexecsol(self, *args, **kwargs):
        """calls primal solution separation method of separator"""
    def sepaexit(self, *args, **kwargs):
        """calls exit method of separator"""
    def sepaexitsol(self, *args, **kwargs):
        """informs separator that the branch and bound process data is being freed"""
    def sepafree(self, *args, **kwargs):
        """calls destructor and frees memory of separator"""
    def sepainit(self, *args, **kwargs):
        """initializes separator"""
    def sepainitsol(self, *args, **kwargs):
        """informs separator that the branch and bound process is being started"""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
