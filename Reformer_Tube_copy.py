
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, quad, solve_bvp
from scipy.optimize import fsolve
import sympy as sp
from Twall import solve_heat_bvp
from Physical_Properties_copy import *
from Kinetics import calculate_kinetics
from Input_Data import *

# Creating two figures: one for Tw and another for the rest of the variables
fig1, ax1 = plt.subplots()  # For Tw
fig2, ax = plt.subplots(3, 1, figsize=(10, 10))  # For T, P, and compositions
plt.ion()  # Turn on interactive mode

# Reactor function for the discretized system
def ReformerUnit_FDM(t, y, Tf, z_grid, N, Epsilon, Dp, Aint, MW, nu, R, dTube, Twin, RhoC, DHreact, Tc, Pc, F_R1, e_w,
                     lambda_s, e_s, tau, p_h, c_h, n_h, s_h, s_w, kw, mesh_size, m_gas, x_f, f_furnace):
    alpha = 0.01  # scaling factor
    n_comp = 5  # Number of components

    # Reshape y into the appropriate form for species, temperature, and pressure
    xi = np.reshape(y[0:N * n_comp], (n_comp, N))
    Tw = y[N * n_comp:N * (n_comp + 1)]  # Wall temperature
    T = y[N * (n_comp + 1):N * (n_comp + 2)]  # Reactor temperature
    P = y[N * (n_comp + 2):N * (n_comp + 3)]  # Pressure
    #print('xi\n',xi,'\nT\n',T,'\nTw\n',Tw)
    if np.any(xi) <0:
        print('Negative value xi')
    if np.any(T) < 0:
        print('Negative value T')
    if np.any(Tw) < 0:
        print('Negative value Tw')
    if np.any(P) <0:
        print('Negative value Ps')
    # Prepare arrays to store derivatives
    dxi_dz = np.zeros((n_comp, N))
    dT_dz = np.zeros(N)
    d2Tw_dz2 = np.zeros(N)
    dP_dz = np.zeros(N)
    
    d2x_dz2 = np.zeros((n_comp,N))
    d2T_dz2=np.zeros(N)

    # Loop through each internal grid point for finite difference
    for i in range(1, N - 1):
        # Compute step sizes
        dz_forward = z_grid[i + 1] - z_grid[i]  # Forward step size
        dz_backward = z_grid[i] - z_grid[i - 1]  # Backward step size
        dz_total = dz_forward + dz_backward  # Total step size for central difference

        # Gas-phase properties and flow calculation for each grid point
        xi_single = xi[:, i]
        f_gas = m_gas / (np.sum(np.multiply(xi_single, MW[:-2])))
        u, omega, MWmix, RhoGas, Pi = Auxiliary_Calc_Tube(xi_single, f_gas, MW, P[i], m_gas, F_R1, Aint, Epsilon, R,
                                                          T[i])
        u_f, omega_f, MWmix_f, RhoGas_f = Auxiliary_Calc_Furnace(x_f, f_furnace, MW, m_gas, F_R1, Aint, Epsilon, R, Tf)

        # Reaction and transport parameters
        DH_reaction, Cpmix, Cpmix_f = calculate_DH_reaction(T[i], MW, omega, DHreact, nu, omega_f)
        Uint, h_t, lambda_gas, DynVis, lambda_ax, Re, Uext, Ur, lambda_f = calculate_heat_transfer_coefficients(T[i],
                                                                                                                MW,
                                                                                                                n_comp,
                                                                                                                Tc, Pc,
                                                                                                                xi_single,
                                                                                                                Cpmix,
                                                                                                                Cpmix_f,
                                                                                                                RhoGas,
                                                                                                                RhoGas_f,
                                                                                                                u, Dp,
                                                                                                                dTube,
                                                                                                                e_w,
                                                                                                                Epsilon,
                                                                                                                lambda_s,
                                                                                                                dTube_out,
                                                                                                                u_f,
                                                                                                                D_h,
                                                                                                                x_f, Tf)
        Deff, Dax = calculate_diffusivity(T[i], P[i], n_comp, xi_single, MWmix, MW, e_s, tau, R, Epsilon, u, Dp, Re)
        rj, kr = calculate_kinetics(T[i], R, Pi, RhoC, Epsilon)
        Eta = calculate_effectiveness_factor(kr, Dp, c_h, n_h, s_h, lambda_gas, Deff, p_h)

        # Mass balance equations (for each species)
        for j in range(n_comp):
            dxi_dz[j, i] = (xi[j, i + 1] - xi[j, i - 1]) / dz_total  # Central difference using non-uniform grid
            d2x_dz2[j,i] = (xi[j, i + 1] - 2 * xi[j, i] + xi[j, i - 1]/ (dz_forward * dz_backward))
            reaction_term = Aint / (f_gas * 3600) * np.sum(np.multiply(nu[:, j], np.multiply(Eta, rj)))
            diffusive_term = Deff[j] *  d2x_dz2[j,i]
            dxi_dz[j, i] += reaction_term + diffusive_term

        # Energy balance equations
        dT_dz[i] = (T[i + 1] - T[i - 1]) / dz_total  # Central difference using non-uniform grid
        d2T_dz2[i] = (T[i + 1] - 2 * T[i] + T[i - 1]) / (dz_forward * dz_backward)
        heat_reaction = np.sum(np.multiply(DH_reaction, np.multiply(Eta, rj))) * 1e6  # J/h/m3
        energy_balance = (-Aint / ((m_gas * 3600) * Cpmix) * heat_reaction
                          + (np.pi * dTube / (m_gas * Cpmix)) * Uint * (Tw[i] - T[i])
                          + lambda_ax * Epsilon *d2T_dz2[i])
        dT_dz[i] += energy_balance

        # Wall temperature equation
        #dTw_dz[i] = (Tw[i + 1] - Tw[i - 1]) / dz_total  # Central difference using non-uniform grid
        d2Tw_dz2[i] = (Tw[i + 1] - 2 * Tw[i] + Tw[i - 1]) / (dz_forward * dz_backward)
        d2Tw_dz2[i] += (1 / (kw * s_w) * (Uint * (Tw[i] - T[i]) + Uext * (Tw[i] - Tf) + Ur * (Tw[i] ** 4 - Tf ** 4)))

        # Scale d2Tw_dz2 by alpha
        d2Tw_dz2[i] *= alpha

        # Clamp Tw to avoid divergence
        Tw[i] = np.clip(Tw[i], 300, 1200)  # Clamping Tw between 300 K and 1200 K
        
        # Pressure drop equation (momentum balance)
        dP_dz[i] = (-150 * (((1 - Epsilon) ** 2) / (Epsilon ** 3)) * DynVis * u / (Dp ** 2)
                    - 1.75 * ((1 - Epsilon) / (Epsilon ** 3)) * m_gas * u / (Dp * Aint)) / 1e5  # Pa -> bar

    # Boundary conditions (forward/backward difference)
    dxi_dz[:, 0] = (xi[:, 1] - xi[:, 0]) / (z_grid[1] - z_grid[0])  # Forward difference at inlet
    dxi_dz[:, -1] = (xi[:, -1] - xi[:, -2]) / (z_grid[-1] - z_grid[-2])  # Backward difference at outlet
    
    dT_dz[0] = (T[1] - T[0]) / (z_grid[1] - z_grid[0])
    dT_dz[-1] = (T[-1] - T[-2]) / (z_grid[-1] - z_grid[-2])
    
    d2Tw_dz2[0] = (Tw[2] - 2*Tw[1] + Tw[0]) / (z_grid[2] - z_grid[0])
    d2Tw_dz2[-1] = (Tw[-1] - 2*Tw[-2]+Tw[-3]) / (z_grid[-1] - z_grid[-3])
    
    dP_dz[0] = (P[1] - P[0]) / (z_grid[1] - z_grid[0])
    dP_dz[-1] = (P[-1] - P[-2]) / (z_grid[-1] - z_grid[-2])

    # Reshape derivatives into a single array
    dydz = np.concatenate([dxi_dz.ravel(), d2Tw_dz2, dT_dz, dP_dz])

    # LIVE PLOTTING
    # First plot Tw in a separate figure
    ax1.clear()
    ax1.plot(z_grid[:-1], Tw, label='Wall Temp (Tw)', color='r')  # Ensure same length as z_grid
    ax1.set_title('Wall Temperature (Tw)')
    ax1.set_xlabel('Reactor Length (z)')
    ax1.set_ylabel('Tw (K)')
    ax1.legend()
    plt.pause(0.01)  # Pause for a brief moment to update the plot

    # Now, plot the rest of the variables in the second figure
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()

    # Plot T (Reactor Temperature)
    ax[0].plot(z_grid[:-1], T, label='Reactor Temp (T)', color='b')  # Ensure same length as z_grid
    ax[0].set_title('Reactor Temperature (T)')
    ax[0].set_xlabel('Reactor Length (z)')
    ax[0].set_ylabel('T (K)')
    ax[0].legend()

    # Plot P (Pressure)
    ax[1].plot(z_grid[:-1], P, label='Pressure (P)', color='g')  # Ensure same length as z_grid
    ax[1].set_title('Pressure (P)')
    ax[1].set_xlabel('Reactor Length (z)')
    ax[1].set_ylabel('P (bar)')
    ax[1].legend()

    # Plot compositions of species (xi)
    species_names = ['H2', 'CO', 'CO2', 'H2O', 'CH4']  # You can adjust based on your species
    colors = ['c', 'm', 'y', 'k', 'orange']  # Different colors for each species

    for i in range(n_comp):
        ax[2].plot(z_grid[:-1], xi[i], label=f'{species_names[i]}', color=colors[i])  # Match length

    ax[2].set_title('Species Compositions')
    ax[2].set_xlabel('Reactor Length (z)')
    ax[2].set_ylabel('Concentration')
    ax[2].legend()

    plt.pause(0.01)  # Pause for a brief moment to update the plot

    return dydz

# Setup the grid and initial conditions
n1 = 10

n2 = 5
N =  n1+n2 # Number of grid points
zspan = [0, 2]  # Reactor length in meters
z_mid = 0.3  # Position where the grid changes resolution

# Initialize an empty list to hold grid points
z_grid = []

# Grid for the first section (0 to 0.5 m)
z1 = np.linspace(zspan[0], z_mid, n1, endpoint=False)  # Exclude the endpoint to avoid overlap
z_grid.extend(z1)

# Grid for the second section (0.5 to 2 m)
z2 = np.linspace(z_mid, zspan[1], n2 + 1)  # Include the endpoint at zspan[1]
z_grid.extend(z2)

# Convert the list to a numpy array
z_grid = np.array(z_grid)
#zspan = np.linspace(0,2,N)
dz = np.diff(z_grid)
yin = np.zeros((5, N))  # Initial mole fractions
for i in range(N):
    yin[:, i] = x_in_R1 # Example mole fractions

Tw_in = np.ones(N) * Tin_w # Example initial wall temperature
T_in = np.ones(N) * Tin_R1 # Example initial reactor temperature
P_in = np.ones(N) * Pin_R1  # Example initial pressure

t1 = np.ones(len(z1))
t2 = np.ones(len(z2))

for i in range(0,len(t1)):
    t1[i] = Tin_R1 - i*38.5
for i in range(0,len(t2)):
    t2[i] = Tin_R1 + i*(38.5*2)

T_in = np.concatenate((t1,t2))

t1 = np.ones(len(z1))
t2 = np.ones(len(z2))
for i in range(0,len(t1)):
    t1[i] = Tin_w - i*5
for i in range(0,len(t2)):
    t2[i] = Tin_w + i*40
Tw_in = np.concatenate((t1,t2))


# Concatenate all initial guesses into a single array
y0 = np.concatenate([yin.ravel(), Tw_in, T_in, P_in])
z = z_grid

# Solve using solve_ivp with finite difference method
sol = solve_ivp(ReformerUnit_FDM, zspan, y0, method='BDF', t_eval=z, args=(Tin_f,z_grid, N, Epsilon, Dp, Aint, MW, nu, R, dTube, Twin, RhoC, DHreact, Tc, Pc, F_R1, e_w, lambda_s, e_s, tau, p_h, c_h, n_h, s_h, s_w, kw, mesh_size, m_R1, x_f, f_furnace))

print(sol)
# Extract the results
xi_out = sol.y[0:5*N, :].reshape((5, N))  # Molar fractions
Tw_out = sol.y[5*N:N*(5+1), :]  # Wall temperature
Tf_out = sol.y[N*(5+1):N*(5+2), :]  # Furnace temperature
T_out = sol.y[N*(5+2):N*(5+3), :]  # Reactor temperature
P_out = sol.y[N*(5+3):N*(5+4), :]  # Reactor pressure
