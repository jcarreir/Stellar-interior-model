import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config.config import K, Na, max_err, M_total, X, Y, T_central, R_total, L_total, M_test, X_test, Y_test, T_test, R_test, L_test
from src.star_class import StellarModel, Results


# We create an instance of the class StellarModel with the parameters of the test case
test = StellarModel(M_test, X_test, Y_test, T_test, R_test, L_test)
test.complete_model()
test_initial_results = Results(test.R, test.P, test.T, test.L, test.M)
print("Total error: ", test.error)

# We build the range of values for the central temperature to optimize the calculation
T_values = np.arange(test.T_central - 0.5, test.T_central + 0.5, 0.01)
test.optimal_temperature_calculation(T_values, plot=True)
# We make a thinner range of values for the central temperature to optimize the calculation
T_values = np.arange(test.T_central - 0.05, test.T_central + 0.05, 0.001)
test.optimal_temperature_calculation(T_values, plot=True)
test_optimal_temperature_results = Results(test)

# We create a grid of values for the total luminosity and the total radius to optimize the calculation
L_values = np.arange(test.L_total - 0.5, test.L_total + 0.5, 0.01)
R_values = np.arange(test.R_total - 0.5, test.R_total + 0.5, 0.01)