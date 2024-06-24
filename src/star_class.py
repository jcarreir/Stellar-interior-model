import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config.config import K, Na, max_err

class StellarModel:
    def __init__(self, M_total, X, Y, T_central, R_total, L_total):
        self.M_total = M_total
        self.X = X
        self.Y = Y
        self.T_central = T_central
        self.R_total = R_total
        self.L_total = L_total
        self.error = "Not computed yet"
        self.initialize_parameters()
        self.initialize_arrays()


    def initialize_parameters(self):
        """
        Initializes the parameters and constants used in the stellar model calculations.
        """

        # Compute dependent variables
        self.Z = 1 - self.X - self.Y
        self.mu = 1 / (2 * self.X + 0.75 * self.Y + 0.5 * self.Z)

        # Constants adjusted to the unit system (CGS)
        self.C_m = 0.01523 * self.mu
        self.C_p = 8.084 * self.mu
        self.C_t_rad = 0.01679 * self.Z * (1 + self.X) * self.mu**2
        self.C_t_conv = 3.234 * self.mu

        # Constants for outer layer computations
        self.A_1 = 1.9022 * self.mu * self.M_total
        self.A_2 = 10.645 * (self.M_total / (self.mu * self.Z * (1 + self.X) * self.L_total))**(1/2)

        # Initialize radius parameters and other properties
        # Start the integration from a radius slightly below the total radius to avoid convergence issues.
        self.R_initial = 0.9 * self.R_total
        # Define the integration step as a small fraction of the initial radius.
        self.h = -0.01 * self.R_initial
        # Construct an array with the radius values in reverse order, starting near the surface.
        self.R = np.arange(self.R_initial, 0, self.h)
        # Add the center of the star to the array.
        self.R = np.append(self.R, 0)
    

    def initialize_arrays(self):
        """
        Initializes the arrays used to store the star properties and gradients during the integration.
        """
        # Initialize arrays for the star properties
        self.P = np.zeros_like(self.R)
        self.T = np.zeros_like(self.R)
        self.M = np.zeros_like(self.R)
        self.L = np.zeros_like(self.R)

        # Initialize the gradient values
        self.dP_dr = np.zeros_like(self.R)
        self.dT_dr = np.zeros_like(self.R)
        self.dM_dr = np.zeros_like(self.R)
        self.dL_dr = np.zeros_like(self.R)

        # Initialize the energy generation rate and transport parameter arrays
        self.epsilon = np.zeros_like(self.R)
        self.nu = np.zeros_like(self.R)
        self.cycle = np.zeros_like(self.R, dtype=str)
        self.C_l = np.zeros_like(self.R)
        self.transport_parameter = np.zeros_like(self.R)


    def reversal(self, state):
        """
        Reverses the arrays of star properties and gradients to integrate from the surface to the center
        or from the center to the surface.

        Parameters:
        - state: The state of the star (either 'surface' or 'center').
        """
        if (state == "surface" and self.h < 0) or (state == "center" and self.h > 0):
            return
        self.P = self.P[::-1]
        self.T = self.T[::-1]
        self.M = self.M[::-1]
        self.L = self.L[::-1]
        self.R = self.R[::-1]
        self.h = -self.h
        self.dP_dr = self.dP_dr[::-1]
        self.dT_dr = self.dT_dr[::-1]
        self.dM_dr = self.dM_dr[::-1]
        self.dL_dr = self.dL_dr[::-1]
        self.epsilon = self.epsilon[::-1]
        self.nu = self.nu[::-1]
        self.cycle = self.cycle[::-1]
        self.C_l = self.C_l[::-1]
        self.transport_parameter = self.transport_parameter[::-1]


    def delta_1(self, i, values):
        """
        Computes the first-order forward difference for a given array of values at index i.

        Parameters:
        - i: Index of the array.
        - values: Array of values.

        Returns:
        - The first-order forward difference at index i.
        """
        return self.h * (values[i] - values[i-1])


    def delta_2(self, i, values):
        """
        Computes the second-order forward difference for a given array of values at index i.

        Parameters:
        - i: Index of the array.
        - values: Array of values.

        Returns:
        - The second-order forward difference at index i.
        """
        return self.delta_1(i, values) - self.delta_1(i-1, values)
    

    def pp_chain(self, T, P):
        """
        Computes the energy generation rate for the pp chain and the associated parameters.
        
        Parameters:
        - T: Temperature (in units of 10^7 K).
        - P: Pressure (in CGS units).
        
        Returns:
        - epsilon_pp: Energy generation rate for the pp chain.
        - nu_pp: Exponent for the temperature dependence of the pp chain.
        - cycle: Type of nuclear cycle (pp chain).
        - C_l_pp: Coefficient for the energy generation rate.
        """
        if T < 0.4:
            return 0, 0, "NA", 0
        elif T < 0.6:
            nu_pp = 6
            epsilon_pp_1 = 10**(-6.84)
        elif T < 0.95:
            nu_pp = 5
            epsilon_pp_1 = 10**(-6.04)
        elif T < 1.2:
            nu_pp = 4.5
            epsilon_pp_1 = 10**(-5.56)
        elif T < 1.65:
            nu_pp = 4
            epsilon_pp_1 = 10**(-5.02)
        elif T < 2.4:
            nu_pp = 3.5
            epsilon_pp_1 = 10**(-4.40)
        else:
            nu_pp = 1
            epsilon_pp_1 = 0

        Rho = self.mu * P / (K * Na * T)
        epsilon_pp = epsilon_pp_1 * self.X * self.X * Rho * (T * 10)**nu_pp
        cycle = 'pp'
        C_l_pp = 0.01845 * epsilon_pp_1 * self.X * self.X * (10 ** nu_pp) * (self.mu ** 2)

        return epsilon_pp, nu_pp, cycle, C_l_pp
    

    def CNO_cycle(self, T, P):
        """
        Computes the energy generation rate for the CNO cycle and the associated parameters.

        Parameters:
        - T: Temperature (in units of 10^7 K).
        - P: Pressure (in CGS units).

        Returns:
        - epsilon_CNO: Energy generation rate for the CNO cycle.
        - nu_CNO: Exponent for the temperature dependence of the CNO cycle.
        - cycle: Type of nuclear cycle (CNO cycle).
        - C_l_CNO: Coefficient for the energy generation rate.
        """
        if T < 1.2:
            nu_CNO = 0
            epsilon_CNO_1 = 0
        elif T < 1.6:
            nu_CNO = 20
            epsilon_CNO_1 = 10**(-22.2)
        elif T < 2.25:
            nu_CNO = 18
            epsilon_CNO_1 = 10**(-19.8)
        elif T < 2.75:
            nu_CNO = 16
            epsilon_CNO_1 = 10**(-17.1)
        elif T < 3.6:
            nu_CNO = 15
            epsilon_CNO_1 = 10**(-15.6)
        elif T < 5:
            nu_CNO = 13
            epsilon_CNO_1 = 10**(-12.5)
        else:
            nu_CNO = 1
            epsilon_CNO_1 = 0

        Rho = self.mu * P / (K * Na * T)
        epsilon_CNO = epsilon_CNO_1 * self.X * (self.Z/3) * Rho * (T * 10)**nu_CNO
        cycle = 'CNO'
        C_l_CNO = 0.01845 * epsilon_CNO_1 * self.X * (self.Z/3) * (10 ** nu_CNO) * (self.mu ** 2)

        return epsilon_CNO, nu_CNO, cycle, C_l_CNO
    

    def energy_generation_rate(self, T, P):
        """
        Computes the energy generation rate and associated parameters based on the temperature and pressure.

        Parameters:
        - T: Temperature (in units of 10^7 K).
        - P: Pressure (in CGS units).

        Returns:
        - epsilon: Energy generation rate.
        - nu: Exponent for the temperature dependence.
        - cycle: Type of nuclear cycle.
        - C_l: Coefficient for the energy generation rate.
        """
        if T < 1.2:
            epsilon, nu, cycle, C_l = self.pp_chain(T, P)
        elif T > 2.4:
            epsilon, nu, cycle, C_l = self.CNO_cycle(T, P)
        else:
            epsilon_pp, nu_pp, cycle_pp, C_l_pp = self.pp_chain(T, P)
            epsilon_CNO, nu_CNO, cycle_CNO, C_l_CNO = self.CNO_cycle(T, P)
            if epsilon_pp > epsilon_CNO:
                epsilon = epsilon_pp
                nu = nu_pp
                cycle = cycle_pp
                C_l = C_l_pp
            else:
                epsilon = epsilon_CNO
                nu = nu_CNO
                cycle = cycle_CNO
                C_l = C_l_CNO

        return epsilon, nu, cycle, C_l
    

    def three_layers_surface(self):
        """
        Computes the properties of the first three layers of the star starting from the surface.
        
        The properties computed are the temperature, pressure, mass, luminosity, and the gradients
        of mass, pressure, luminosity, and temperature with respect to the radius.
        """
        for i in range(3):
            self.T[i] = self.A_1 * ((1 / self.R[i]) - (1 / self.R_total))
            self.P[i] = self.A_2 * (self.T[i]**4.25)
            self.M[i] = self.M_total
            self.L[i] = self.L_total

            self.dM_dr[i] = 0
            self.dP_dr[i] = -self.C_p * self.P[i] * self.M[i] / (self.T[i] * self.R[i] ** 2)
            self.dL_dr[i] = 0
            self.dT_dr[i] = -self.C_t_rad * (self.P[i] ** 2) * self.L[i] / ((self.T[i] ** 8.5) * self.R[i] ** 2)

    
    def radiative_envelope(self):
        """
        Computes the radiative envelope of the star starting from the third layer.
        
        Returns:
        - i: The index of the last layer of the radiative envelope.
        """
        i = 2 # We start at the third layer

        while True:
            est_P = self.P[i] + self.h * self.dP_dr[i] + 1/2 * self.delta_1(i, self.dP_dr) + 5/12 * self.delta_2(i, self.dP_dr)
            est_T = self.T[i] + self.h * self.dT_dr[i] + 1/2 * self.delta_1(i, self.dT_dr)

            while True:
                while True:
                    self.dM_dr[i+1] = self.C_m * est_P * self.R[i+1] ** 2 / est_T
                    cal_M = self.M[i] + self.h * self.dM_dr[i+1] - 1/2 * self.delta_1(i+1, self.dM_dr)
                    self.dP_dr[i+1] = -self.C_p * est_P * cal_M / (est_T * self.R[i+1] ** 2)
                    cal_P = self.P[i] + self.h * self.dP_dr[i+1] - 1/2 * self.delta_1(i+1, self.dP_dr)

                    if abs(cal_P - est_P) / cal_P < max_err:
                        break
                    else:
                        est_P = cal_P
            
                self.epsilon[i+1], self.nu[i+1], self.cycle[i+1], self.C_l[i+1] = self.energy_generation_rate(est_T, cal_P)
                self.dL_dr[i+1] = self.C_l[i+1] * (cal_P ** 2) * (est_T ** (self.nu[i+1] - 2)) * (self.R[i+1] ** 2)
                cal_L = self.L[i] + self.h * self.dL_dr[i+1] - 1/2 * self.delta_1(i+1, self.dL_dr) - 1/12 * self.delta_2(i+1, self.dL_dr)
                self.dT_dr[i+1] = -self.C_t_rad * cal_P ** 2 * cal_L / ((est_T ** (8.5)) * (self.R[i+1] ** 2))
                cal_T = self.T[i] + self.h * self.dT_dr[i+1] - 1/2 * self.delta_1(i+1, self.dT_dr)

                if abs(cal_T - est_T)/cal_T < max_err:
                    break
                else:
                    est_T = cal_T
                    est_P = cal_P

            self.transport_parameter[i+1] = (cal_T / cal_P) * (self.dP_dr[i+1] / self.dT_dr[i+1])
        
            self.P[i+1] = cal_P
            self.T[i+1] = cal_T
            self.M[i+1] = cal_M
            self.L[i+1] = cal_L

            if self.transport_parameter[i+1] < 2.5:
                break
            else:
                # We compute the next layer
                i += 1

        # Constant polytropic index
        self.k_polytrope = self.P[i+1]/(self.T[i+1]**(5/2))

        return i
    
    def convective_core(self, i, R_down, type):
        """
        Computes the convective core of the star starting from the last layer of the radiative envelope.

        Parameters:
        - i: The index of the last layer of the radiative envelope.
        - R_down: The radius at which the transition layer is located.
        - type: The type of integration ('surface' or 'center').

        Returns:
        - i: The index of the last layer of the convective core.
        """
        # We start at the layer i = 81 if we integrate from the surface to the center
        # We start at the layer i = 2 if we integrate from the center to the surface
        while True:
            est_T = self.T[i] + self.h * self.dT_dr[i] + 1/2 * self.delta_1(i, self.dT_dr)

            while True:
                # We compute the pressure as a polytrope
                est_P = self.k_polytrope * est_T ** (5/2)
                self.dM_dr[i+1] = self.C_m * est_P * self.R[i+1] ** 2 / est_T
                cal_M = self.M[i] + self.h * self.dM_dr[i+1] - 1/2 * self.delta_1(i+1, self.dM_dr)

                if self.R[i+1] > 0:
                    self.dT_dr[i+1] = -self.C_t_conv * cal_M / (self.R[i+1] ** 2)
                    cal_T = self.T[i] + self.h * self.dT_dr[i+1] - 1/2 * self.delta_1(i+1, self.dT_dr)
                else:
                    cal_T = est_T

                if abs((cal_T - est_T)/cal_T) < max_err:
                    break
                else:
                    est_T = cal_T

            # We compute the pressure as a polytrope
            # Caution! Now is calculated values instead of estimated values
            cal_P = self.k_polytrope * cal_T ** (5/2)
            self.epsilon[i+1], self.nu[i+1], self.cycle[i+1], self.C_l[i+1] = self.energy_generation_rate(est_T, cal_P)
            self.dL_dr[i+1] = self.C_l[i+1] * (cal_P ** 2) * (cal_T ** (self.nu[i+1] - 2)) * (self.R[i+1] ** 2) 
            cal_L = self.L[i] + self.h * self.dL_dr[i+1] - 1/2 * self.delta_1(i+1, self.dL_dr) - 1/12 * self.delta_2(i+1, self.dL_dr)

            self.P[i+1] = cal_P
            self.T[i+1] = cal_T
            self.M[i+1] = cal_M
            self.L[i+1] = cal_L

            if (self.R[i+1] <= 0) and (type == "surface"):
                i += 1
                break
            elif (self.R[i+1] > R_down) and (type == "center"):
                i += 1
                break
            else:
                i += 1

        return i
    

    def transition_layer_down(self):
        """
        Computes the transition layer of the star starting from the last layer of the convective core.
        
        Returns:
        - R_down: The radius at which the transition layer is located.
        - P_down: The pressure at which the transition layer is located.
        - T_down: The temperature at which the transition layer is located.
        - L_down: The luminosity at which the transition layer is located.
        - M_down: The mass at which the transition layer is located.
        """
        # We try to find the index of the first element that meets the condition transport_parameter > 0 and < 2.5
        indices = np.where((self.transport_parameter > 0) & (self.transport_parameter < 2.5))[0]

        # We interpolate the values of the parameters at the transition point
        R_down = np.interp(2.5, [self.transport_parameter[indices[0]], self.transport_parameter[indices[0]-1]], [self.R[indices[0]], self.R[indices[0]-1]])
        P_down = np.interp(R_down, [self.R[indices[0]], self.R[indices[0]-1]], [self.P[indices[0]], self.P[indices[0]-1]])
        T_down = np.interp(R_down, [self.R[indices[0]], self.R[indices[0]-1]], [self.T[indices[0]], self.T[indices[0]-1]])
        L_down = np.interp(R_down, [self.R[indices[0]], self.R[indices[0]-1]], [self.L[indices[0]], self.L[indices[0]-1]])
        M_down = np.interp(R_down, [self.R[indices[0]], self.R[indices[0]-1]], [self.M[indices[0]], self.M[indices[0]-1]])

        return R_down, P_down, T_down, L_down, M_down
    

    def transition_layer_up(self, R_down, j):
        """
        Computes the transition layer of the star starting from the last layer of the convective core.

        Parameters:
        - R_down: The radius at which the transition layer is located.
        - j: The index of the last layer of the convective core.

        Returns:
        - R_up: The radius at which the transition layer is located.
        - P_up: The pressure at which the transition layer is located.
        - T_up: The temperature at which the transition layer is located.
        - L_up: The luminosity at which the transition layer is located.
        - M_up: The mass at which the transition layer is located.
        """
        # We interpolate the values of the parameters at the transition point
        R_up = R_down
        P_up = np.interp(R_up, [self.R[j-1], self.R[j]], [self.P[j-1], self.P[j]])
        T_up = np.interp(R_up, [self.R[j-1], self.R[j]], [self.T[j-1], self.T[j]])
        L_up = np.interp(R_up, [self.R[j-1], self.R[j]], [self.L[j-1], self.L[j]])
        M_up = np.interp(R_up, [self.R[j-1], self.R[j]], [self.M[j-1], self.M[j]])

        return R_up, P_up, T_up, L_up, M_up
    

    def three_layers_core(self):
        """
        Computes the properties of the first three layers of the star starting from the center.
        
        The properties computed are the temperature, pressure, mass, luminosity, and the gradients
        of mass, pressure, luminosity, and temperature with respect to the radius.

        Returns:
        - i: The index of the last layer of the central layers.
        """
        # Computation of the first three layers
        for i in range(3):
            self.T[i] = self.T_central - 0.008207 * (self.mu ** 2) * (self.k_polytrope) * (self.T_central ** 1.5) * (self.R[i] ** 2)
            self.P[i] = self.k_polytrope * (self.T[i] ** 2.5)
            self.M[i] = (self.C_m / 3) * self.k_polytrope * (self.T_central ** 1.5) * (self.R[i] ** 3)
            self.epsilon[i], self.nu[i], self.cycle[i], self.C_l[i] = self.energy_generation_rate(self.T[i], self.P[i])
            self.L[i] = (self.C_l[i] / 3) * (self.k_polytrope ** 2) * (self.T_central ** (self.nu[i] + 3)) * (self.R[i] ** 3)

            # We must compute the gradient of the variables
            self.dM_dr[i] = self.C_m * self.k_polytrope * self.T[i] ** (3/2) * self.R[i] ** 2
            # dP_dr_values[i] =  Not necessary to compute the gradient of the pressure
            self.dL_dr[i] = self.C_l[i] * (self.k_polytrope ** 2) * self.T[i] ** (self.nu[i] + 3) * self.R[i] ** 2
            if self.R[i] > 0:
                self.dT_dr[i] = -self.C_t_conv * self.M[i] / (self.R[i] ** 2)
            else:    
                self.dT_dr[i] = 0

        return i
    

    def calculate_relative_errors(self, down_values, up_values):
        """
        Calculates the relative errors given 'down' and 'up' values for parameters, and
        computes the total relative error as the square root of the sum of the squares of
        individual errors.
        
        Parameters:
        - down_values: List of 'down' values [r, P, T, L, M]
        - up_values: List of 'up' values [r, P, T, L, M]
        
        Returns:
        - total_relative_error: The total relative error as a percentage.
        """
        # Calculate relative errors for each parameter
        relative_errors = [(abs(up - down) / down) * 100 for down, up in zip(down_values, up_values)]
        
        # Calculate total relative error as the square root of the sum of the squares of the individual errors
        total_relative_error = np.sqrt(sum([error**2 for error in relative_errors]))
        
        return total_relative_error
    

    def extra_layers(self):
        """
        Computes the extra layers of the star to reach the surface.

        Returns:
        - R_extra: The radius values for the extra layers.
        - P_extra: The pressure values for the extra layers.
        - T_extra: The temperature values for the extra layers.
        - L_extra: The luminosity values for the extra layers.
        - M_extra: The mass values for the extra layers.
        """
        # We make sure the step size is negative to integrate from the center to the surface
        if self.h < 0:
            h_extra = self.h
        else:
            h_extra = -self.h

        R_extra = np.arange(self.R_total, self.R_initial, h_extra)
        T_extra = np.zeros(len(R_extra))
        P_extra = np.zeros(len(R_extra))
        M_extra = np.zeros(len(R_extra))
        L_extra = np.zeros(len(R_extra))

        i = -len(R_extra)

        while i < 0:
            T_extra[i] = self.A_1*((1/R_extra[i])-(1/self.R_total))
            P_extra[i] = self.A_2*(T_extra[i]**4.25)
            M_extra[i] = self.M_total
            L_extra[i] = self.L_total
            i += 1

        return R_extra, P_extra, T_extra, L_extra, M_extra
    

    def complete_model(self):
        """
        Computes the complete stellar model by integrating the star from the surface to the center
        and from the center to the surface, computing the transition layer, and calculating the
        relative errors in the transition layer.
        """
        # We make use of the reversal function to make sure we are integrating from the surface to the center
        self.reversal("surface")
        # We compute the surface layers of the star
        self.three_layers_surface()
        # We compute the radiative envelope inwards
        i = self.radiative_envelope()
        # We compute the convective core inwards
        self.convective_core(i, 0, "surface")
        # We compute the transition layer inwards
        down_values = self.transition_layer_down()
        R_down = down_values[0]
        # We make use of the reversal function to make sure we are integrating from the center to the surface
        self.reversal("center")
        # We compute the central layers of the star
        j = self.three_layers_core()
        # We compute the convective core outwards
        m = self.convective_core(j, R_down, "center")
        # We save the value of the layer where the transition layer is located
        self.transition_layer_index = m
        # We compute the transition layer outwards
        up_values = self.transition_layer_up(R_down, m)
        # We compute the error in the transition layer
        self.error = self.calculate_relative_errors(down_values, up_values)
        # We compute the extra layers
        R_extra, P_extra, T_extra, L_extra, M_extra = self.extra_layers()
        # We append the extra layers to the arrays
        self.R = np.append(self.R, R_extra[::-1])
        self.P = np.append(self.P, P_extra[::-1])
        self.T = np.append(self.T, T_extra[::-1])
        self.M = np.append(self.M, M_extra[::-1])
        self.L = np.append(self.L, L_extra[::-1])
        # We complete transport_parameter array with zeros
        self.transport_parameter = np.append(self.transport_parameter, np.zeros(len(R_extra)))
    

    def optimal_temperature_calculation(self, T_values):
        """
        Computes the central temperature that minimizes the total relative error
        by iterating over a range of central temperatures and calculating the total relative error
        for each central temperature.

        Parameters:
        - T_values: An array of central temperatures to iterate over.

        Returns:
        - array_error: An array of total relative errors for each central temperature.
        """
        # We define an array to store the total relative error for each central temperature
        array_error = np.zeros(len(T_values))

        # We iterate over the range of central temperatures
        for i, T_central in enumerate(T_values):
            # We update the central temperature
            self.T_central = T_central
            # We reinitialize the parameters and arrays
            self.initialize_parameters()
            self.initialize_arrays()
            # We compute the total relative error for the current central temperature
            self.complete_model()
            array_error[i] = self.error

        self.T_central = T_values[np.argmin(array_error)]
        self.error = np.min(array_error)

        # Now that we have the optimal temperature, we compute the complete model
        self.initialize_parameters()
        self.initialize_arrays()
        self.complete_model()

        # Print the central temperature that minimizes the total relative error
        print("Central Temperature that minimizes the Total Relative Error (K):", T_values[np.argmin(array_error)])
        # Print the minimum total relative error
        print("Minimum Total Relative Error (%):", np.min(array_error))

        return array_error

    
    def plot_array_error(self, T_values, array_error):
        """
        Plots the total relative error as a function of the central temperature.

        Parameters:
        - T_values: An array of central temperatures to iterate over.
        - array_error: An array of total relative errors for each central temperature.
        """
        plt.figure()
        plt.plot(T_values, array_error)
        plt.axvline(x=T_values[np.argmin(array_error)], color='r', linestyle='--')
        plt.xlabel('Central Temperature ($10^7$ K)')
        plt.ylabel('Total Relative Error (%)')
        plt.grid()
        plt.legend(['Total Relative Error', 'Minimum Total Relative Error'])
        plt.show()


    def optimal_grid_calculation(self, R_values, L_values, T_values):
        """
        Computes the total radius and luminosity that minimize the total relative error
        by iterating over a grid of total radius and luminosity values and calculating the total
        relative error for each combination of total radius and luminosity.

        Parameters:
        - R_values: An array of total radius values to iterate over.
        - L_values: An array of total luminosity values to iterate over.
        - T_values: An array of central temperatures to use in the calculations.

        Returns:
        - matrix_error: A matrix of total relative errors for each combination of total radius and luminosity.
        """
        matrix_error = np.zeros((len(R_values), len(L_values)))
        matrix_temperature = np.zeros((len(R_values), len(L_values)))

        for i, R_total in enumerate(R_values):
            self.R_total = R_total
            for j, L_total in enumerate(L_values):
                self.L_total = L_total
                self.optimal_temperature_calculation(T_values)
                matrix_error[i, j] = self.error
                matrix_temperature[i, j] = self.T_central

        self.error = np.min(matrix_error)
        self.T_central = np.min(matrix_temperature)
        i, j = np.unravel_index(np.argmin(matrix_error, axis=None), matrix_error.shape)
        self.R_total = R_values[i]
        self.L_total = L_values[j]
        print("----------------------------------------------------------------------------------")
        # Print the central temperature that minimizes the total relative error
        print("Central Temperature that minimizes the Total Relative Error (K):", self.T_central)
        # Print the total radius that minimizes the total relative error
        print("Total Radius that minimizes the Total Relative Error ($10^{10}$ cm):", self.R_total)
        # Print the total luminosity that minimizes the total relative error
        print("Total Luminosity that minimizes the Total Relative Error ($10^{33}$ erg/s):", self.L_total)
        # Print the minimum total relative error
        print("Minimum Total Relative Error (%):", self.error)

        return matrix_error
    

    def plot_matrix_error(self, matrix_error, R_values, L_values):
        """
        Plots the total relative error as a function of the total radius and luminosity.

        Parameters:
        - matrix_error: A matrix of total relative errors for each combination of total radius and luminosity.
        - R_values: An array of total radius values to iterate over.
        - L_values: An array of total luminosity values to iterate over.
        - plot: A boolean value that determines whether the plot should be displayed.
        """
        # Radius in vertical axis, Luminosity in horizontal axis
        plt.figure()
        plt.imshow(matrix_error, extent=[min(L_values), max(L_values), max(R_values), min(R_values)], aspect='auto')
        plt.colorbar(label='Total Relative Error (%)')
        plt.xlabel('Total Luminosity ($10^{33}$ erg/s)')
        plt.ylabel('Total Radius ($10^{10}$ cm)')
        plt.grid(False)
        plt.show()


    def extra_variables_calculation(self):
        """
        Computes the extra variables of the star, such as the density, opacity, and energy generation rate.
        """
        # Initialize the arrays for the extra variables
        self.Rho = np.zeros_like(self.R)
        self.epsilon = np.zeros_like(self.R)
        self.nu = np.zeros_like(self.R)
        self.cycle = np.zeros_like(self.R, dtype=str)
        self.C_l = np.zeros_like(self.R)
        for i in range(len(self.T)):
            # Compute the density
            if self.T[i] == 0:
                self.Rho[i] = 0
            else:
                self.Rho[i] = self.mu * self.P[i] / (K * Na * self.T[i])
            # Compute the energy generation rate
            self.epsilon[i], self.nu[i], self.cycle[i], self.C_l[i] = self.energy_generation_rate(self.T[i], self.P[i])


    def plot_normalized_variables(self, variable = 'all', independent_variable = 'radius', vertical_line = False):
        """
        Plots the star's properties normalized by their maximum values as a function of the normalized radius.

        Parameters:
        - variable: The variable to plot ('all', 'Mass', 'Luminosity', 'Temperature', 'Pressure', 'Density', 'Epsilon').
        """
        if independent_variable == 'radius':
            independent_variable = self.R / self.R_total
            dependent_variable = self.M
            label = 'Mass'
            xlabel = "Normalized Radius"
        elif independent_variable == 'mass':
            independent_variable = self.M / self.M[-1]
            dependent_variable = self.R
            label = 'Radius'
            xlabel = "Normalized Mass"
        else:
            print("Invalid independent variable. Please choose 'radius' or 'mass'.")

        if variable == 'all':
            plt.figure()
            if  vertical_line:
                plt.axvline(x=self.R[self.transition_layer_index] / self.R[-1], color='k', linestyle='--', label = 'Transition Layer')
            plt.plot(independent_variable, dependent_variable / dependent_variable[-1], label = label)
            plt.plot(independent_variable, self.L / self.L[-1], label='Luminosity')
            plt.plot(independent_variable, self.T / self.T[0], label='Temperature')
            plt.plot(independent_variable, self.P / self.P[0], label='Pressure')
            plt.plot(independent_variable, self.Rho / self.Rho[0], label='Density')
            
            plt.xlabel(xlabel)
            plt.ylabel('Normalized Values')
            plt.legend()
            plt.grid()
            plt.show()

        elif variable == 'Mass' or variable == 'Radius':
            plt.figure()
            plt.plot(independent_variable, dependent_variable)
            plt.xlabel(xlabel)
            plt.ylabel('Mass ($10^^{33} g$)')
            plt.grid()
            plt.show()

        elif variable == 'Luminosity':
            plt.figure()
            plt.plot(independent_variable, self.L)
            plt.xlabel(xlabel)
            plt.ylabel('Luminosity ($10^{33} erg/s$)')
            plt.grid()
            plt.show()

        elif variable == 'Temperature':
            plt.figure()
            plt.plot(independent_variable, self.T)
            plt.xlabel(xlabel)
            plt.ylabel('Temperature ($10^7 K$)')
            plt.grid()
            plt.show()

        elif variable == 'Pressure':
            plt.figure()
            plt.plot(independent_variable, self.P)
            plt.xlabel(xlabel)
            plt.ylabel('Pressure ($10^{15} dyne/cm^2$)')
            plt.grid()
            plt.show()

        elif variable == 'Density':
            plt.figure()
            plt.plot(independent_variable, self.Rho)
            plt.xlabel(xlabel)
            plt.ylabel('Density ($g/cm^3$)')
            plt.grid()
            plt.show()

        elif variable == 'Epsilon':
            plt.figure()
            plt.plot(independent_variable, self.epsilon)
            plt.xlabel(xlabel)
            plt.ylabel('Energy Generation Rate ($erg/s/g$)')
            plt.grid()
            plt.show()

        else:
            print("Invalid variable. Please choose 'all', 'Mass', 'Luminosity', 'Temperature', 'Pressure', 'Density', or 'Epsilon'.")

    
    def save_data(self, filename):
        """
        Saves the star's properties to a CSV file.

        Parameters:
        - filename: The name of the file to save the data.
        """
        # Create a DataFrame with the star's properties
        df = pd.DataFrame({
            'Radius': self.R,
            'Pressure': self.P,
            'Temperature': self.T,
            'Luminosity': self.L,
            'Mass': self.M,
            'Density': self.Rho,
            'Energy Generation Rate': self.epsilon,
            'Transport Parameter': self.transport_parameter,
            'Cycle': self.cycle,
        })
        # Save the DataFrame to a CSV file
        df.to_csv(filename, index=False)