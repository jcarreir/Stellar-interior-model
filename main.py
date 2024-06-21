import numpy as np
from src.star_class import StellarModel
from config.config import M_total, X, Y, R_total, L_total, T_central, K, Na, max_err

"""
def run_model():
    L_total = 40  # luminosidad total, ejemplo
    R_total = 12  # radio total, ejemplo
    T_central = 1.5e7  # temperatura central, ejemplo

    reversal('surface')
    R, k_polytrope = radiative_envelope(R_total, L_total)
    T = convective_core(0, k_polytrope, 0, 'surface')

    # Guardar o mostrar los resultados
    print("Radii:", R)
    print("Temperatures:", T)

if __name__ == "__main__":
    run_model()

"""

M = 5
X = 0.75
Y = 0.2
T_central = 1.5
R_total = 12
L_total = 40

star = StellarModel(M, X, Y, T_central, R_total, L_total)
T_values = np.arange(star.T_central - 0.5, star.T_central + 0.5, 0.01)
star.optimal_temperature_calculation(T_values, plot=True)
T_values = np.arange(star.T_central - 0.05, star.T_central + 0.05, 0.001)
star.optimal_temperature_calculation(T_values, plot=True)

R_values = np.linspace(11.52, 11.78, 11)
L_values = np.linspace(44.25, 46.75, 11)
matrix_error = star.optimal_grid_calculation(R_values, L_values, T_values)
star.plot_matrix_error(matrix_error, R_values, L_values, "contour")
star.plot_matrix_error(matrix_error, R_values, L_values, "pixels")