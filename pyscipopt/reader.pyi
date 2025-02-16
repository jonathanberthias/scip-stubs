from _typeshed import Incomplete

class Reader:
    model: Incomplete
    name: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def readerfree(self, *args, **kwargs):
        """calls destructor and frees memory of reader"""
    def readerread(self, *args, **kwargs):
        """calls read method of reader"""
    def readerwrite(self, *args, **kwargs):
        """calls write method of reader"""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
