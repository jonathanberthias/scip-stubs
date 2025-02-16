from _typeshed import Incomplete

class Benders:
    model: Incomplete
    name: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def benderscreatesub(self, *args, **kwargs):
        """creates the subproblems and registers it with the Benders decomposition struct"""
    def bendersexit(self, *args, **kwargs):
        """calls exit method of Benders decomposition"""
    def bendersexitpre(self, *args, **kwargs):
        """informs the Benders decomposition that the presolving process has been completed"""
    def bendersexitsol(self, *args, **kwargs):
        """informs Benders decomposition that the branch and bound process data is being freed"""
    def bendersfree(self, *args, **kwargs):
        """calls destructor and frees memory of Benders decomposition"""
    def bendersfreesub(self, *args, **kwargs):
        """frees the subproblems"""
    def bendersgetvar(self, *args, **kwargs):
        """Returns the corresponding master or subproblem variable for the given variable. This provides a call back for the variable mapping between the master and subproblems."""
    def bendersinit(self, *args, **kwargs):
        """initializes Benders deconposition"""
    def bendersinitpre(self, *args, **kwargs):
        """informs the Benders decomposition that the presolving process is being started"""
    def bendersinitsol(self, *args, **kwargs):
        """informs Benders decomposition that the branch and bound process is being started"""
    def benderspostsolve(self, *args, **kwargs):
        """sets post-solve callback of Benders decomposition"""
    def benderspresubsolve(self, *args, **kwargs):
        """sets the pre subproblem solve callback of Benders decomposition"""
    def benderssolvesub(self, *args, **kwargs):
        """sets solve callback of Benders decomposition"""
    def benderssolvesubconvex(self, *args, **kwargs):
        """sets convex solve callback of Benders decomposition"""
    def __reduce__(self): ...
