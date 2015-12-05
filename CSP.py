__author__ = 'chris'
from constraint import *
import util

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

def cuisineCompare(biz1, biz2):
    category1 = biz1.categories[0]
    category2 = biz2.categories[0]
    if category1 == "Food" or category2 == "Food":
        return True
    elif category1 == category2:
        return False
    else:
        return True

def reduceBizs(filteredBizs, constraints):
    numVariables = constraints["numRecs"]
    csp = CSP()

    for i in range(numVariables):
        csp.addVariable("rec" + i, filteredBizs)

    # add constraint that no 2 cuisines are the same
    for i in range(numVariables):
        for j in range(i + 1, numVariables):
            csp.addConstraint(cuisineCompare, ("rec" + i, "rec" + j))

    print csp.solve()

json1 = {"business_id": "qCPBS-m_4uDO0EgIYGtoxw", "full_address": "1689 S SanTan Village Pkwy\nGilbert, AZ 85295", "hours": {"Monday": {"close": "00:00", "open": "09:00"}, "Tuesday": {"close": "00:00", "open": "09:00"}, "Friday": {"close": "02:00", "open": "09:00"}, "Wednesday": {"close": "00:00", "open": "09:00"}, "Thursday": {"close": "00:00", "open": "09:00"}, "Sunday": {"close": "00:00", "open": "09:00"}, "Saturday": {"close": "02:00", "open": "09:00"}}, "open": True, "categories": ["Bars", "Golf", "Nightlife", "Active Life", "American (New)", "Restaurants"], "city": "Gilbert", "review_count": 28, "name": "Topgolf", "neighborhoods": [], "longitude": -111.7425148, "state": "AZ", "stars": 4.0, "latitude": 33.319839600000002, "attributes": {"Alcohol": "full_bar", "Noise Level": "loud", "Has TV": True, "Attire": "casual", "Ambience": {"romantic": False, "intimate": False, "classy": False, "hipster": False, "divey": False, "touristy": False, "trendy": False, "upscale": False, "casual": False}, "Good for Kids": True, "Price Range": 3, "Good For Dancing": False, "Delivery": False, "Coat Check": False, "Accepts Credit Cards": True, "Take-out": False, "Happy Hour": False, "Outdoor Seating": True, "Takes Reservations": False, "Waiter Service": True, "Wi-Fi": "free", "Caters": True, "Good For": {"dessert": False, "latenight": False, "lunch": False, "dinner": False, "breakfast": False, "brunch": False}, "Parking": {"garage": False, "street": False, "validated": False, "lot": False, "valet": False}, "Music": {"dj": False}, "Good For Groups": True}, "type": "business"}
biz1 = util.Biz(json1)
print biz1.categories