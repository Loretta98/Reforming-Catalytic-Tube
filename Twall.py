#########################################################################################################################################
############################################### Radial Temperature along the tube calculation ###########################################
#########################################################################################################################################

import numpy as np
from scipy.integrate import solve_bvp

# Define the boundary value problem in a wrapper function
def solve_heat_bvp(y_guess,h_t, rin, T, epsilon_w, epsilon_f, sigma, h_f, Tf, sw,mesh_size):
    # Define the boundary conditions with additional parameters
    # Define the system of ODEs
    kw = 28.5 # tube thermal conductivity [W/mK]
    h_t = 210
    a1 = h_f/kw
    a2 = sigma/kw*epsilon_f
    b1 = h_t/kw
    b2 = sigma/kw*epsilon_w

    #print(h_t)

    # Define the boundary conditions
    def bc(ya, yb):
        # Boundary conditions at r = rin and r = rin + sw
        bc1 = (ya[1] - (-b1 * ya[0] + b1 * T - b2 * ya[0] ** 4))
        bc2 = (yb[1] - (-a1 * Tf + a1 * yb[0] - a2 * Tf ** 4))
        return np.array([bc1, bc2])

    # Define the heat equation with additional parameters
    def odes(r, y):
        # y[0] = x, y[1] = dx/dr
        dxdr = y[1]
        d2xdr2 = - y[1] / r  # From the original equation
        return np.vstack((dxdr, d2xdr2))

    # Set up the initial guess for the solution (linear guess)
    r = np.linspace(rin, rin + sw, mesh_size)

    # Solve the boundary value problem
    solution = solve_bvp(odes, bc, r, y_guess,tol=1e-3,max_nodes=1000)

    # Check if the solution converged
    if solution.success==0:
        print("BVP solution did not converge")
        print(solution)

    # Return the solution for plotting
    return solution
