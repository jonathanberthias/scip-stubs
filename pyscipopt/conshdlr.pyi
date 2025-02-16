from _typeshed import Incomplete

class Conshdlr:
    model: Incomplete
    name: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def consactive(self, *args, **kwargs):
        """sets activation notification method of constraint handler"""
    def conscheck(self, *args, **kwargs):
        """calls feasibility check method of constraint handler"""
    def conscopy(self, *args, **kwargs):
        """sets copy method of both the constraint handler and each associated constraint"""
    def consdeactive(self, *args, **kwargs):
        """sets deactivation notification method of constraint handler"""
    def consdelete(self, *args, **kwargs):
        """sets method of constraint handler to free specific constraint data"""
    def consdelvars(self, *args, **kwargs):
        """calls variable deletion method of constraint handler"""
    def consdisable(self, *args, **kwargs):
        """sets disabling notification method of constraint handler"""
    def consenable(self, *args, **kwargs):
        """sets enabling notification method of constraint handler"""
    def consenfolp(self, *args, **kwargs):
        """calls enforcing method of constraint handler for LP solution for all constraints added"""
    def consenfops(self, *args, **kwargs):
        """calls enforcing method of constraint handler for pseudo solution for all constraints added"""
    def consenforelax(self, *args, **kwargs):
        """calls enforcing method of constraint handler for a relaxation solution for all constraints added"""
    def consexit(self, *args, **kwargs):
        """calls exit method of constraint handler"""
    def consexitpre(self, *args, **kwargs):
        """informs constraint handler that the presolving is finished"""
    def consexitsol(self, *args, **kwargs):
        """informs constraint handler that the branch and bound process data is being freed"""
    def consfree(self, *args, **kwargs):
        """calls destructor and frees memory of constraint handler"""
    def consgetdivebdchgs(self, *args, **kwargs):
        """calls diving solution enforcement callback of constraint handler, if it exists"""
    def consgetnvars(self, *args, **kwargs):
        """sets constraint variable number getter method of constraint handler"""
    def consgetpermsymgraph(self, *args, **kwargs):
        """permutation symmetry detection graph getter callback, if it exists"""
    def consgetsignedpermsymgraph(self, *args, **kwargs):
        """signed permutation symmetry detection graph getter callback, if it exists"""
    def consgetvars(self, *args, **kwargs):
        """sets constraint variable getter method of constraint handler"""
    def consinit(self, *args, **kwargs):
        """calls initialization method of constraint handler"""
    def consinitlp(self, *args, **kwargs):
        """calls LP initialization method of constraint handler to separate all initial active constraints"""
    def consinitpre(self, *args, **kwargs):
        """informs constraint handler that the presolving process is being started"""
    def consinitsol(self, *args, **kwargs):
        """informs constraint handler that the branch and bound process is being started"""
    def conslock(self, *args, **kwargs):
        """variable rounding lock method of constraint handler"""
    def consparse(self, *args, **kwargs):
        """sets constraint parsing method of constraint handler"""
    def conspresol(self, *args, **kwargs):
        """calls presolving method of constraint handler"""
    def consprint(self, *args, **kwargs):
        """sets constraint display method of constraint handler"""
    def consprop(self, *args, **kwargs):
        """calls propagation method of constraint handler"""
    def consresprop(self, *args, **kwargs):
        """sets propagation conflict resolving method of constraint handler"""
    def conssepalp(self, *args, **kwargs):
        """calls separator method of constraint handler to separate LP solution"""
    def conssepasol(self, *args, **kwargs):
        """calls separator method of constraint handler to separate given primal solution"""
    def constrans(self, *args, **kwargs):
        """sets method of constraint handler to transform constraint data into data belonging to the transformed problem"""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
