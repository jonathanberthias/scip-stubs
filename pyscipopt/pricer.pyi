from _typeshed import Incomplete

class Pricer:
    model: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def pricerexit(self, *args, **kwargs):
        """calls exit method of variable pricer"""
    def pricerexitsol(self, *args, **kwargs):
        """informs variable pricer that the branch and bound process data is being freed"""
    def pricerfarkas(self, *args, **kwargs):
        """calls Farkas pricing method of variable pricer"""
    def pricerfree(self, *args, **kwargs):
        """calls destructor and frees memory of variable pricer"""
    def pricerinit(self, *args, **kwargs):
        """initializes variable pricer"""
    def pricerinitsol(self, *args, **kwargs):
        """informs variable pricer that the branch and bound process is being started"""
    def pricerredcost(self, *args, **kwargs):
        """calls reduced cost pricing method of variable pricer"""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
