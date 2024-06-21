# config.py
# Initial parameters for the star model

# Total mass of the star in 10^33 grams
M_total = 5.0

# Chemical composition of the star
X = 0.75  # Fraction of Hydrogen
Y = 0.20  # Fraction of Helium

# Initial values for integration
R_total = 12  # Total radius of the star in 10^10 cm
L_total = 40  # Total luminosity of the star in 10^33 erg/s
T_central = 1.5  # Central temperature of the star in 10^7 K

# Physical constants
K = 1.380649e-16  # Boltzmann constant in erg/K
Na = 6.02214076e23  # Avogadro's number, dimensionless

# Numerical parameters
max_err = 1e-4  # Maximum allowed error in numerical integrations