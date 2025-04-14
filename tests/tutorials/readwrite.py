from __future__ import annotations

# Read and Write Files
# Model File Formats
# Write a Model
from pyscipopt import Model

scip = Model()
scip.writeProblem(filename="example_file.mps", trans=False, genericnames=False)

# Read a Model

scip.readProblem(filename="example_file.mps")
