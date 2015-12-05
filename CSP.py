__author__ = 'chris'
from constraint import *

class CSP:

    def __init__(self):
        self.problem = Problem()

    def addVariable(self, newVariable, valueRange):
        self.problem.addVariables(newVariable, valueRange)

    def addConstraint(self, lambdaFn, variables):
        self.problem.addConstraint(lambdaFn, variables)

    def solve(self):
        solutions = self.problem.getSolutions()
        return solutions

def compareFn(row1, row2):
    return row1 != row2

csp = CSP()
numpieces = 3
rows = range(numpieces)
cols = range(numpieces)

print rows, cols
csp.addVariable(cols, rows)

for col1 in cols:
    for col2 in cols:
        if col1 < col2:
            csp.addConstraint(compareFn, (col1, col2))
print csp.solve()
