from _typeshed import Incomplete

class Eventhdlr:
    model: Incomplete
    name: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def eventcopy(self, *args, **kwargs):
        """sets copy callback for all events of this event handler"""
    def eventdelete(self, *args, **kwargs):
        """sets callback to free specific event data"""
    def eventexec(self, *args, **kwargs):
        """calls execution method of event handler"""
    def eventexit(self, *args, **kwargs):
        """calls exit method of event handler"""
    def eventexitsol(self, *args, **kwargs):
        """informs event handler that the branch and bound process data is being freed"""
    def eventfree(self, *args, **kwargs):
        """calls destructor and frees memory of event handler"""
    def eventinit(self, *args, **kwargs):
        """initializes event handler"""
    def eventinitsol(self, *args, **kwargs):
        """informs event handler that the branch and bound process is being started"""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
