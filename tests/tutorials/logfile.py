from __future__ import annotations

from pathlib import Path

from pyscipopt import Model

# SCIP Log Files

scip = Model()

# How to Read SCIP Output

# Presolve Information

# Branch-and-Bound Information

# Final Summarised Information

# How to Redirect SCIP Output

scip.hideOutput()

scip.redirectOutput()

path_to_file = Path("out.log")
scip.setLogfile(path_to_file)


# SCIP Statistics

scip.printStatistics()
scip.writeStatistics(filename=path_to_file)
