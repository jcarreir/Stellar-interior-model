import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config.config import K, Na, max_err, M_total, X, Y, T_central, R_total, L_total, M_test, X_test, Y_test, T_test, R_test, L_test
from src.star_class import StellarModel

# We create an instance of the class StellarModel with the parameters of the star
star = StellarModel(M_total, X, Y, T_central, R_total, L_total)
star.complete_model()
print("Total error: ", star.error)

# We compute the extra variables and plot them all
star.extra_variables_calculation()
star.plot_normalized_variables()

# We build the range of values for the central temperature to optimize the calculation
T_values = np.arange(star.T_central - 0.5, star.T_central + 0.5, 0.01)
array_error = star.optimal_temperature_calculation(T_values)
star.plot_array_error(T_values, array_error)

# We make a thinning of the array of values to optimize the calculation
T_values = np.arange(star.T_central - 0.05, star.T_central + 0.05, 0.001)
array_error = star.optimal_temperature_calculation(T_values)
star.plot_array_error(T_values, array_error)

# We compute the extra variables and plot them all for the optimized model
star.extra_variables_calculation()
star.plot_normalized_variables()

# We build the range of values for the total radius and luminosity to optimize the calculation
R_values = np.linspace(11.5, 12.5, 11)
L_values = np.linspace(40, 50, 11)
matrix_error = star.optimal_grid_calculation(R_values, L_values, T_values)
star.plot_matrix_error(matrix_error, R_values, L_values)

# We make a thinning of the matrix of values to optimize the calculation
R_values = np.linspace(11.52, 11.78, 11)
L_values = np.linspace(44.25, 46.75, 11)
matrix_error = star.optimal_grid_calculation(R_values, L_values, T_values)
star.plot_matrix_error(matrix_error, R_values, L_values)

# A last thinning of the matrix of values to optimize the calculation with a smaller step for the temperature
R_values = np.linspace(11.62, 11.68, 11)
L_values = np.linspace(45.3, 45.7, 11)
T_values = np.arange(star.T_central - 0.02, star.T_central + 0.02, 0.0001)
matrix_error = star.optimal_grid_calculation(R_values, L_values, T_values)
star.plot_matrix_error(matrix_error, R_values, L_values)

# We can see now variables are smooth
star.extra_variables_calculation()
star.plot_normalized_variables(vertical_line=True)

# We save the results in a csv file
star.save_data("final_model.csv")