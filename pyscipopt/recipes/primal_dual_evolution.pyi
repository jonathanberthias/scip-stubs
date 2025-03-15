import pyscipopt.scip
from pyscipopt import SCIP_EVENTTYPE as SCIP_EVENTTYPE
from pyscipopt.scip import Eventhdlr as Eventhdlr
from pyscipopt.scip import Model as Model

def attach_primal_dual_evolution_eventhdlr(model: pyscipopt.scip.Model):
    """
    Attaches an event handler to a given SCIP model that collects primal and dual solutions,
    along with the solving time when they were found.
    The data is saved in model.data["primal_log"] and model.data["dual_log"]. They consist of
    a list of tuples, each tuple containing the solving time and the corresponding solution.

    A usage example can be found in examples/finished/plot_primal_dual_evolution.py. The
    example takes the information provided by this recipe and uses it to plot the evolution
    of the dual and primal bounds over time.
    """
