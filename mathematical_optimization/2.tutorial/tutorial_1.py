import pulp

problem = pulp.LpProblem("SLE", pulp.LpMaximize)

x = pulp.LpVariable("x", cat="Continuous")
y = pulp.LpVariable("y", cat="Continuous")

problem += 120 * x + 150 * y == 1440
problem += x + y == 10

status = problem.solve()

print("status: ", pulp.LpStatus[status])
print("x=", x.value(), "y=", y.value())
