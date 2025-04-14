from pyscipopt import Constraint, Model, Variable, cos, exp, log, sin, sqrt

scip = Model()

# Non-Linear Objectives


x: Variable = scip.addVar(vtype="I", name="x")
y: Variable = scip.addVar(vtype="I", name="y")
z: Variable = scip.addVar(vtype="I", name="z")
cons_1: Constraint = scip.addCons(x + y >= 5, name="cons_1")
cons_2: Constraint = scip.addCons(x + 1.3 * y <= 10, name="cons_2")
cons_3: Constraint = scip.addCons(z >= x * x + y, name="cons_3")
scip.setObjective(z)


# Polynomials


# TODO: can we make the types exact here?
lhs = 3 * (x**2) + ((y**3) * (z**2)) + ((2 * x) + (3 * z)) ** 2
lhs = lhs / (2 * x * z)
cons_4: Constraint = scip.addCons(lhs <= x * y * z, name="poly_cons")


# Square Root (sqrt)


cons_5 = scip.addCons(sqrt(x) <= y, name="sqrt_cons")

# Absolute (Abs)


cons_6 = scip.addCons(abs(x) <= y + 5, name="abs_cons")

# Exponential (exp) and Log

cons_7 = scip.addCons((1 / (1 + exp(-x))) == y, name="exp_cons")
cons_8 = scip.addCons(log(x) <= z, name="log_cons")


# Sin and Cosine (cos)


cons_9 = scip.addCons(sin(x) == y, name="sin_cons")
cons_10 = scip.addCons(cos(y) <= 0.5, name="cos_cons")
