# 📦 Imports
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus
import matplotlib.pyplot as plt
import numpy as np

# 📊 Problem Setup
# Product A: $20 profit, 2 machine hrs, 3 labor hrs
# Product B: $30 profit, 4 machine hrs, 1 labor hr
# Constraints: 100 machine hours, 90 labor hours

# 1. Define the problem
model = LpProblem(name="product-mix", sense=LpMaximize)

# 2. Decision variables
x = LpVariable(name="A_units", lowBound=0, cat='Integer')
y = LpVariable(name="B_units", lowBound=0, cat='Integer')

# 3. Objective function
model += 20 * x + 30 * y, "Total_Profit"

# 4. Constraints
model += 2 * x + 4 * y <= 100, "Machine_Hours"
model += 3 * x + 1 * y <= 90, "Labor_Hours"

# 5. Solve the model
status = model.solve()

# 6. Results
print(f"Status: {LpStatus[model.status]}")
print(f"Produce {x.value()} units of Product A")
print(f"Produce {y.value()} units of Product B")
print(f"Maximum Profit: ${model.objective.value()}")

# 7. Visualizing the Constraints
x_vals = np.linspace(0, 60, 400)
y1 = (100 - 2 * x_vals) / 4  # Machine constraint
y2 = (90 - 3 * x_vals) / 1  # Labor constraint

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y1, label="Machine Constraint")
plt.plot(x_vals, y2, label="Labor Constraint")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim((0, 50))
plt.ylim((0, 50))
plt.xlabel("Product A units")
plt.ylabel("Product B units")

# Feasible region
plt.fill_between(x_vals, np.minimum(y1, y2), alpha=0.3)

# Optimal point
plt.plot(x.value(), y.value(), 'ro', label="Optimal Solution")
plt.title("Feasible Region and Optimal Solution")
plt.legend()
plt.grid(True)
plt.show()
