from Reformer_model_Alberton_approach import*
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Set font to Segoe UI if available
mpl.rcParams['font.family'] = 'Segoe UI'

# Given composition values before normalization
H2 = 64.892
CH4 = 0.602
CO = 12.731
CO2 = 21.522

# Calculate the sum of the given values
total = H2 + CH4 + CO + CO2

# Normalize the values
H2_normalized = (H2 / total)
CH4_normalized = (CH4 / total)
CO_normalized = (CO / total)
CO2_normalized = (CO2 / total)

# Normalized values to plot as single points
normalized_composition = [CH4_normalized, CO_normalized, CO2_normalized, H2_normalized]

# Define reactor length array
z = np.linspace(0, Length, np.size(wi_out[0]))
Tf = np.ones(np.size(z)) * Tf
Tw = np.array(Tw_list)
# Plot 1: Temperature profile
plt.figure(figsize=(8, 6))
plt.plot(z, T_R1, label='Tg')
plt.plot(z, Tf, label='Tf')
plt.plot(z,Tw, label='Tw')
plt.xlabel('Reactor Length [m]')
plt.ylabel('T [K]')
#plt.title('Temperature Profile Along the Reactor')
plt.legend()
plt.grid(True)
plt.tight_layout()


# Plot 2: Molar fractions with normalized data points
plt.figure(figsize=(8, 6))
for i in range(n_comp - 1):
    plt.plot(z, yi_[i])
plt.scatter([z[-1]] * len(normalized_composition), normalized_composition, marker='*' ,color='k', label='Real Measurement', zorder=5)
plt.xlabel('Reactor Length [m]')
plt.ylabel('Molar Fraction')
#plt.title('Molar Dry Fractions')
plt.legend(['CH4', 'CO', 'CO2', 'H2', 'Real Measurement'])
plt.grid(True)
plt.tight_layout()



# Plot 3: Pressure profile
plt.figure(figsize=(8, 6))
plt.plot(z, P_R1)
plt.xlabel('Reactor Length [m]')
plt.ylabel('P [bar]')
#plt.title('Pressure Profile Along the Reactor')
plt.grid(True)
plt.tight_layout()
plt.show()