# This is a sample Python script.
import numpy 
#  QUESTO CODICE CONTIENE MODELLO COMPLETO DI 1 REATTORE DI SMR 
#  CINETICA Xu-Froment 

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, quad, solve_bvp
from scipy.optimize import fsolve
import sympy as sp

#########################################################################################################################################
############################################### Radial Temperature along the tube calculation ###########################################
#########################################################################################################################################


# Define the boundary value problem in a wrapper function
def solve_heat_bvp(h_t, r_in, delta_z, T_g, epsilon_w, epsilon_f, sigma, h_f, T_f, s_w):
    # Define the boundary conditions with additional parameters
    def bc(Ta, Tb):
        T_in_w = Ta[0]  # Temperature at r = r_in (only first value from Ta)
        T_ext_w = Tb[0]  # Temperature at r = r_in + s_w (only first value from Tb)

        # Evaluate Q_cond_in and Q_cond_ext only at the boundary points.
        Q_conv_wg = h_t * 2 * np.pi * r_in * delta_z * (T_in_w - T_g)
        Q_rad_wg = sigma * 2 * np.pi * r_in * delta_z * epsilon_w * (T_in_w ** 4)
        Q_cond_in = Q_conv_wg + Q_rad_wg

        Q_conv_fw = h_f * 2 * np.pi * (r_in + s_w) * delta_z * (T_f - T_ext_w)
        Q_rad_fw = sigma * 2 * np.pi * (r_in + s_w) * delta_z * epsilon_f * (T_f ** 4)
        Q_cond_ext = Q_conv_fw + Q_rad_fw

        return np.array([T_in_w - T_g, T_ext_w - T_f])  # Example: Dirichlet BCs for temperatures

    # Define the heat equation with additional parameters
    def heat_eqn(r, T):
        return np.vstack([T[1], -(T[1] / r)])  # dT/dr and d²T/dr² = 0

    # Create a radial mesh and initial guess for temperature distribution
    r_mesh = np.linspace(r_in, r_in + s_w, 50)
    T_guess = np.zeros((2, r_mesh.size))  # Initial guess (constant temperature)

    # Solve the BVP
    solution = solve_bvp(heat_eqn, bc, r_mesh, T_guess)

    # Check if the solution converged
    if solution.success:
        print("BVP solution converged")
    else:
        print("BVP solution did not converge")

    # Return the solution for plotting
    return solution


#####################################################################################################################################
################################################# Tubular reactor resolution ########################################################
#####################################################################################################################################


def TubularReactor(z,y,Epsilon,Dp,m_gas,Aint,MW,nu,R,dTube,Twin,RhoC,DHreact,Tc,Pc,F_R1,e_w, lambda_s,e_s,tau,p_h,c_h,n_h,s_h,s_w):

    omega = y[0:5]
    T =     y[5]
    P =     y[6]
    Tw = Twin
    #Tw = Twin + 12.145*z + 0.011*z**2
    #Tw = 150*np.log(2*z+1) + Twin
    # Aux. Calculations
    mi = m_gas*omega                                        # Mass flowrate per tube per component [kg/s tube]
    ni = np.divide(mi,MW)                                   # Molar flowrate per tube per component [kmol/s tube]
    ntot = np.sum(ni)                                       # Molar flowrate per tube [kmol/s tube]
    yi = ni/ntot                                            # Molar fraction

    Pi = P*yi                                               # Partial Pressure
    Ppa = P * 1E5                                           # Pressure [Pa]
    Eta = 0.1                                               # effectiveness factor (Latham et al., Kumar et al.)

    MWmix = np.sum(yi*MW)                                   # Mixture

    # Estimation of physical properties with ideal mixing rules
    RhoGas = (Ppa*MWmix) / (R*T)  / 1000                                           # Gas mass density [kg/m3]
    VolFlow_R1 = m_gas / RhoGas                                                 # Volumetric flow per tube [m3/s]
    u = (F_R1*1000) * R * T / (Aint*Ppa)                                       # Superficial Gas velocity if the tube was empy (Coke at al. 2007)
    u = VolFlow_R1 / (Aint * Epsilon)                                           # Gas velocity in the tube [m/s]
    
    # Mixture massive Specific Heat calculation (Shomate Equation)                                                        # Enthalpy of the reaction at the gas temperature [J/mol]
            # CH4,          CO,             CO2,            H2,              H2O    
    c1 = np.array([-0.7030298, 25.56759,  24.99735, 33.066178 , 30.09 ])
    c2 = np.array([108.4773,   6.096130,  55.18696, -11.363417, 6.832514])/1000
    c3 = np.array([-42.52157,  4.054656, -33.69137, 11.432816 ,  6.793435])/1e6
    c4 = np.array([5.862788, -2.671301,   7.948387, -2.772874 ,  -2.534480])/1e9
    c5 = np.array([0.678565,  0.131021,  -0.136638, -0.158558 , 0.082139])*1e6
 
    Cp_mol = c1+c2*T+c3*T**2+c4*T**3+c5/T**2                        # Molar specific heat per component [J/molK]
    Cp = Cp_mol/MW*1000                                             # Mass specific heat per component [J/kgK]
    Cpmix = np.sum(Cp*omega)                                        # Mass specific heat [J/kgK]
    DH_reaction = DHreact*1000 + np.sum(nu*(c1*(T-298) + c2*(T**2-298**2)/2 + c3*(T**3-298**3)/3 + c4*(T**4-298**4)/4 - c5*(1/T-1/298)),1) #J/mol
    DH_reaction = DH_reaction*1000 #J/kmol

    ######################################################################################################
    ################################## Heat Transfer Coefficient Calculation #############################
    ######################################################################################################
   
    # Viscosity coefficients from Yaws
                 # CH4,          CO,             CO2,            H2,              H2O,           O2             N2
    A = np.array([3.844,         35.086,        11.336       ,  27.758,          -36.826,    44.224    ,       42.606   ])
    B = np.array([4.0112e-1,     5.065e-1,      4.99e-1      ,  2.12e-1,         4.29e-1,    5.62e-1   ,       4.75e-1  ])
    C = np.array([-1.4303e-4,    -1.314e-4,     -1.0876e-4   ,  -3.28e-5,        -1.62e-5,   -1.13e-4  ,       -9.88e-5 ])

    # Thermal conductivity coeffiecients from Yaws 
                # CH4,          CO,             CO2,            H2,              H2O,           O2             N2
    a = np.array([-0.00935,     0.0015 ,        -0.01183       , 0.03951 ,       0.00053 ,     0.00121  ,      0.00309    ])
    b = np.array([1.4028e-4,    8.2713e-5,      1.0174e-4      ,  4.5918e-4,     4.7093e-5,    8.6157e-5   ,   7.593e-5   ])
    c = np.array([3.318e-8,    -1.9171e-8,     -2.2242e-8   ,  -6.4933e-8,       4.9551e-8,   -1.3346e-8  ,    -1.1014e-8 ])

    # Wassiljewa equation for low-pressure gas viscosity with Mason and Saxena modification 
    k_i     = a + b*T + c*T**2                                                                      # thermal conductivity of gas [W/m/K]
    A_matrix = np.identity(n_comp)
    thermal_conductivity_array = np.zeros(n_comp)
    
    Gamma = 210*(Tc*MW**3/Pc**4)**(1/6)     # reduced inverse thermal conductivity [W/mK]^-1
    Tr = T/Tc
    k = (np.exp(0.0464*Tr)-np.exp(-0.2412*Tr))
    
    for i in range(0,n_comp-1):
        for j in range(i+1,n_comp): 
            A_matrix[i,j] = ( 1+ ((Gamma[j]*k[i])/(Gamma[i]*k[j]))**(0.5) * (MW[j]/MW[i])**(0.25) )**2 / ( 8*(1 + MW[i]/MW[j])**(0.5) )
            A_matrix[j,i] = (1 + ((Gamma[i] * k[j]) / (Gamma[j] * k[i])) ** (0.5) * (MW[i] / MW[j]) ** (0.25)) ** 2 / (
                        8 * (1 + MW[j] / MW[i]) ** (0.5))
            #A_matrix[j,i] = k_i[j]/k_i[i]*MW[i]/MW[j] * A_matrix[i,j]

    for i in range(0,n_comp):
        den = 0
        for j in range(0,n_comp): 
            den += yi[j]*A_matrix[j,i]

        num = yi[i]*k_i[i]
        thermal_conductivity_array[i] = num/den 
    lambda_gas   = sum(thermal_conductivity_array)                                                        # Thermal conductivity of the mixture [W/m/K]
    # K_gas = 0.9
    # Wilke Method for low-pressure gas viscosity   
    mu_i    = (A + B*T + C*T**2)*1e-7                                                               # viscosity of gas [micropoise]
    PHI = np.identity(n_comp)                                                                       # initialization for PHI calculation
    dynamic_viscosity_array = np.zeros(n_comp)

    for i in range(0,n_comp-1):
        for j in range(i+1,n_comp): 
            PHI[i,j] = ( 1+ (mu_i[i]/mu_i[j])**(0.5) * (MW[j]/MW[i])**(0.25) )**2 / ( 8*(1 + MW[i]/MW[j])**(0.5) )
            PHI[j,i] = mu_i[j]/mu_i[i]*MW[i]/MW[j] * PHI[i,j]

    for i in range(0,n_comp):
        den = 0
        for j in range(0,n_comp): 
            den += yi[j]*PHI[j,i]

        num = yi[i]*mu_i[i]
        dynamic_viscosity_array[i] = num/den 
    
    DynVis  = sum(dynamic_viscosity_array)                                                  # Dynamic viscosity [kg/m/s]

    Pr = Cpmix*DynVis/lambda_gas                                                            # Prandtl number
    Re = RhoGas * u * Dp / DynVis                                                           # Reynolds number []

    #Pr_f = Cpmix_f*DynVis_f/lambda_f                                                            # Prandtl number
    #Re_f = RhoGas_f * u_f * D_f / DynVis_f                                                           # Reynolds number []

    #h_t = 0.4*lambda_gas/Dp*(2.58*Re**(1/3)*Pr**(1/3)+0.094*Re**(0.8)*Pr**(0.4))                     # Convective coefficient tube side [W/m2/K] from Nandasana paper 2003
    h_t = lambda_gas/Dp*(2.58*Re**(1/3)*Pr**(1/3)+0.094*Re**(0.8)*Pr**(0.4))                            # Convective coefficient tube side [W/m2/K] from Pantoleontos 2012
    #h_f = lambda_f/Dh*(0.023*(Re_f)**0.8*Pr_f**0.3)                                                     # Convenctive coefficient furnace side [W/m2/K] (Dittus-Boeleter's equation)
    h_f = 833.77    # guess 

    # Overall transfer coefficient in packed beds, Dixon 1996 
    eps = 0.9198/((dTube/Dp)**2) + 0.3414
    aw = (1-1.5*(dTube/Dp)**(-1.5))*(lambda_gas/Dp)*(Re**0.59)*Pr**(1/3)                    # wall thermal transfer coefficient [W/m2/K]
    ars = 0.8171*e_w/(2-e_w)*(T/1000)**3                                                    # [W/m2/K]
    aru = (0.8171*(T/1000)**3) / (1+(eps/(1-eps))*(1-e_w)/e_w)
    lamba_er_o = Epsilon*(lambda_gas+0.95*aru*Dp)+0.95*(1-Epsilon)/(2/(3*lambda_s)+1/(10*lambda_gas+ars*Dp))
    lambda_er = lamba_er_o+0.11*lambda_gas*Re*Pr**(1/3)/(1+46*(Dp/dTube_out)**2)            # effective radial conductivity [W/m/K]
    Bi = aw*dTube_out/2/lambda_er
    U = 1 / ( 1/aw + dTube_out/6/lambda_er)*((Bi+3)/(Bi+4))                                 # J/m2/s/K = W/m2/K
    h_env = 0.1                                                                             # Convective coefficient external environment [W/m2/K]
    Thick = 0.01                                                                            # Tube Thickness [m]
    
#####################################################################################################
########################## Kinetic Costant and Rate of reaction (Xu-Froment)#########################
#####################################################################################################

    # CH4 + H2O -> CO + 3 H2 ; CO + H2O -> CO2 + H2 ; CH4 + 2H2O +> CO2 + 4H2

    # Equilibrium constants fitted from Nielsen in [bar]
    gamma = np.array([[-757.323e5, 997.315e3, -28.893e3, 31.29], [-646.231e5, 563.463e3, 3305.75,-3.466]]) #[bar^2], [-]
    Keq1 = np.exp(gamma[0,0]/(T**3)+gamma[0,1]/(T**2)+gamma[0,2]/T + gamma[0,3])
    Keq2 = np.exp(gamma[1,0]/(T**3)+gamma[1,1]/(T**2)+gamma[1,2]/T + gamma[1,3])
    Keq3 = Keq1*Keq2

    # Arrhenius     I,       II,        III
    Tr_a = 648                                                                          # K 
    Tr_b = 823                                                                          # K 
    k0 = np.array([1.842e-4, 7.558,     2.193e-5])                                      # pre exponential factor @648 K kmol/bar/kgcat/h
    E_a = np.array([240.1,   67.13,     243.9])                                         # activation energy [kJ/mol]
    kr = k0*np.exp(-(E_a*1000)/R*(1/T-1/Tr_a))
    kr_list.append(kr)
    # Van't Hoff    CO,     H2,         CH4,    H2O
    K0_a = np.array([40.91,   0.0296])                                      # pre exponential factor @648 K [1/bar]
    DH0_a = np.array([-70.65, -82.90])                                      # adsorption enthalpy [kJ/mol]
    K0_b = np.array([0.1791, 0.4152])                                       # pre exponential factor @823 K [1/bar, -]

    DH0_b = np.array([-38.28,  88.68])                                # adsorption enthalpy [kJ/mol]
    
    Kr_a = K0_a*np.exp(-(DH0_a*1000)/R*(1/T-1/Tr_a))
    Kr_b = K0_b*np.exp(-(DH0_b*1000)/R*(1/T-1/Tr_b))
    #   CO, H2, CH4, H2O
    Kr = np.concatenate((Kr_a,Kr_b)) # [1/bar] unless last one [-]
    # Components  [CH4, CO, CO2, H2, H2O, O2, N2]
    DEN = 1 + Kr[0]*Pi[1] + Kr[1]*Pi[3] + Kr[2]*Pi[0] + Kr[3]*Pi[4]/Pi[3]
    rj = np.array([ (kr[0]/Pi[3]**(2.5)) * (Pi[0]*Pi[4]-(Pi[3]**3)*Pi[1]/Keq1) / DEN**2 , (kr[1]/Pi[3]) * (Pi[1]*Pi[4]-Pi[3]*Pi[2]/Keq2) / DEN**2 , (kr[2]/Pi[3]**(3.5)) * (Pi[0]*(Pi[4]**2)-(Pi[3]**4)*Pi[2]/Keq3) / DEN**2 ]) * RhoC * (1-Epsilon)  # kmol/m3/h
    
    ###########################################################################################################
    ######################################## Particle balance #################################################
    ###########################################################################################################
    Dmi = np.zeros(n_comp); Dij = np.identity(n_comp); Dki = np.zeros((n_comp))
    k_boltz = 1.380649*1e-23                    # Boltzmann constant m2kg/K/s2
                    # CH4,   CO,    CO2,   H2,  H2O,  O2,  N2
    dip_mom = np.array([0.0, 0.1, 0.0, 0.0, 1.8]) #, 0.0, 0.0]) # dipole moment debyes
    #dip_mom_r = 54.46*dip_mom**2*Pc/Tc**2       # reduced dipole moment by Lucas
    
    Vb = np.array([8.884,6.557,14.94,1.468,20.36])*1e3                      # liquid molar volume flow @Tb [cm3/mol] from Aspen 
    Tb = np.array([-161.4, -190.1, -87.26, -253.3, 99.99]) + 273.15         # Normal Boiling point [K] from Aspen
    # Components  [CH4, CO, CO2, H2, H2O, O2, N2]

    ################### Diffusion coefficient calculation #########################################

    # Theoretical correlation
    # Components  [CH4, CO, CO2, H2, H2O]
    sigma = np.array([3.758, 3.690, 3.941, 2.827, 2.641])   # characteristic Lennard-Jones lenght in Angstrom 
    epsi = np.array([148.6, 91.7, 195.2, 59.7, 809.1])      # characteristic Lennard-Jones energy eps/k [K] 
    # Empirical correlation
    #sigma = 1.18*Vb**(1/3)
    #epsi = 1.15*Tb
    delta = np.array(1.94*1e3*dip_mom**2)/Vb/Tb    # Chapman-Enskog with Brokaw correction with polar gases 
    sigma = ((1.58*Vb)/(1+1.3*delta**2))**(1/3)     # Chapman-Enskog with Brokaw correction with polar gases 
    epsi = 1.18*(1+1.3*delta**2)*Tb                 # Chapman-Enskog with Brokaw correction with polar gases

    for i in range(0,n_comp-1):
        for j in range(i+1,n_comp):
            epsi_ij = (epsi[i]*epsi[j])**0.5
            T_a = T / epsi_ij
            delta_ij = (delta[i]*delta[j])**0.5  
            omega_d = 1.06036/(T_a**0.15610) + 0.193/(np.exp(0.47635*T_a)) + 1.03587/(np.exp(1.52996*T_a)) + 1.76474/(np.exp(3.89411*T_a)) # diffusion collision integral [-] lennard-jones
            omega_d = omega_d +0.19*delta_ij**2/T_a                 # Chapman-Enskog with Brokaw correction with polar gases 
            #sigma_ij = (sigma[i] + sigma[j]) / 2 
            sigma_ij = (sigma[i]*sigma[j])**0.5
            M_ij = 2* 1/ ((1/MW[i])+(1/MW[j]))
            Dij[i,j] =  ( (3.03 - (0.98/M_ij**0.5))/1000 * T**(3/2) ) / (P*M_ij**0.5 *sigma_ij**2*omega_d)  # cm2/s
            Dij[j,i] = Dij[i,j]

    pore_radius = 1e-8  # pore radius in cm

    for i in range(0,n_comp):
        for j in range(0,n_comp):
            den += np.sum(yi[j] / Dij[i,j])
        Dmi [i] = (1-yi[i])/ den               # Molecular diffusion coefficient for component with Wilke's Equation
        Dki [i] = 9700 * pore_radius * (T / MW[i]) # Knudsen diffusion [cm2/s]

    pore_diameter = 130 * 1e-8   # pore diameter in cm, average from Xu-Froment II
    Dki = pore_diameter/3*(8*R*T/(MWmix/1e3)/np.pi)**0.5            # Knudsen diffusion [cm2/s]
    Deff = 1 / ( (e_s/tau)* (1/Dmi + 1/Dki))                        # Effective diffusion [cm2/s]
    Deff_CH4 = Deff[0]*1E-4             # Effective diffusion [m2/s]
    Deff_list.append(Deff_CH4)

 
    ######################################### Effectiveness factors calculation from CFD models, based on the work of Alberton, 2009 #######################################

    # Catalyst parameters Dp (catalyst diameter); p_h (pellet height); c_h (central hole diameter); n_h (number of side holes); s_h (side holes diameter)

    A_c = np.multiply(np.array([5.22389,0.39393,5.48814]),1e-1)
    B_c = np.array([0.354836,575612,0.327636])
    C_c = np.multiply(np.array([1.39726,1.59117]),1e2)
    D_c = np.multiply(np.array([-2.63225,-1.06324]),1e1)

    Area_pellet = 2/p_h*((Dp*1000)**2-c_h**2-n_h*s_h**2)+4*((Dp*1000)+c_h+n_h*s_h) # total area of the pellet [mm2]
    Volume_pellet =  (Dp*1000)**2 - c_h**2 - n_h*s_h**2 # total volume of the pellet [mm3]

    alpha = np.zeros(3)
    alpha[0] = A_c[0]+B_c[0]* np.arctan((np.log(lambda_gas/(Deff[0]*1e-4))*np.log(Deff[0]*1e-4*(kr[0]**(0.5)))-C_c[0])/(D_c[0]))
    alpha[1] = A_c[1]+B_c[1]*Deff[0]*1e-4
    alpha[2] = A_c[2]+B_c[2]* np.arctan((np.log(lambda_gas/(Deff[0]*1e-4))*np.log(Deff[0]*1e-4*(kr[2]**(0.5)))-C_c[1])/(D_c[1]))
    specific_area = Area_pellet/Volume_pellet

    Eta = np.ones(3)

    for i in range(0,3):
        Eta[i] = alpha[i]*specific_area

    Eta_list.append(Eta)

    ############################################### Radial Tube temperature distribution #################################################################################

    # Create a radial mesh and initial guess for temperature distribution
    r_in = dTube/2                                                                  # tube internal radius [m]
    delta_z =  np.linspace(0,Length,N)                                                                     # lenght of each control volume element [m]
    epsilon_w = 0.85            # wall emissivity 
    epsilon_f = 0.3758          # furnace emissivity
    T_f = Twin
    s_k_boltz = 5.66961e-8      #Stefan Boltzmann constant  [Jm2/K/s]
    # Solve the BVP
    # Solve the BVP using the wrapper function
    solution = solve_heat_bvp(h_t, r_in, delta_z, T, epsilon_w, epsilon_f, s_k_boltz, h_f, T_f, s_w)

    # Plot the results
    if solution.success:
        r_sol = solution.x
        T_sol = solution.y[0]

    Tw = T_sol

#################################################################################################################################################
################################################################## Equations#####################################################################

    Reactor1 = Aint / (m_gas*3600) * MW[0] * np.sum(np.multiply(nu[:, 0], np.multiply(Eta, rj)))
    Reactor2 = Aint / (m_gas*3600) * MW[1] * np.sum(np.multiply(nu[:, 1], np.multiply(Eta, rj)))
    Reactor3 = Aint / (m_gas*3600) * MW[2] * np.sum(np.multiply(nu[:, 2], np.multiply(Eta, rj)))
    Reactor4 = Aint / (m_gas*3600) * MW[3] * np.sum(np.multiply(nu[:, 3], np.multiply(Eta, rj)))
    Reactor5 = Aint / (m_gas*3600) * MW[4] * np.sum(np.multiply(nu[:, 4], np.multiply(Eta, rj)))

    term_0 = np.sum(np.multiply(DH_reaction, np.multiply(Eta,rj)))
    term_1 = - Aint/ ((m_gas*3600)*Cpmix) * term_0
    term_2 = (np.pi*dTube/(m_gas*Cpmix))*U*(Tw - T)
    # thermal axial diffusion 
    #term_3 = 
    Reactor6 =  term_1 + term_2 #+ term_3
    Reactor7 = ( (-150 * (((1-Epsilon)**2)/(Epsilon**3)) * DynVis * u/ (Dp**2) - (1.75* ((1-Epsilon)/(Epsilon**3)) * m_gas*u/(Dp*Aint))  ) ) / 1e5
    
    #Reactor8 = 
    
    return np.array([Reactor1, Reactor2, Reactor3, Reactor4, Reactor5, Reactor6, Reactor7])


######################################################################################################################################################
# INPUT DATA REACTOR
n_comp = 5;                                 # total number of species
nu = np.array([ [-1, 1, 0, 3, -1],
                [0, -1, 1, 1, -1], 
                [-1, 0, 1, 4, -2]])         # SMR, WGS, reverse methanation stoichiometric coefficients

# Components  [CH4, CO, CO2, H2, H2O] O2, N2]
MW = np.array([16.04, 28.01, 44.01, 2.016, 18.01528 ]) #, 32.00, 28.01])                    # Molecular Molar weight       [kg/kmol]
Tc = np.array([-82.6, -140.3, 31.2, -240, 374]) + 273.15            # Critical Temperatures [K]
Pc = np.array([46.5, 35, 73.8, 13, 220.5])                          # Critical Pressures [bar]

# Data from FAT experimental setup 
Nt =   4                                                                                    # Number of tubes
#dTube = 0.1
dTube = 0.14142                                                                             # Tube diameter [m]
s_w = 0.06                                                                                  # Tube thickness
dTube_out = dTube+s_w                                                                       # Tube outlet diameter [m]
Length = 2                                                                                  # Length of the reactor [m]

# Catalyst particle data
Epsilon = 0.519                                                                             # Void Fraction 
RhoC = 2355.2                                                                               # Catalyst density [kg/m3]
#RhoC = 1000 # [kg/m3]
Dp = 0.015                                                                                 # Catalyst particle diameter [m]
p_h = 15                                                                                 # Pellet height [mm]
c_h = 5                                                                                  # central hole diameter [mm]
n_h = 0                                                                                     # number of side holes 
s_h = 0                                                                                     # side holes diameter [m]
tau = 3.54                                                                                  # Tortuosity 
e_s = 0.25                                                                                  # porosity of the catalyst particle [m3void/ m3cat] --> Tacchino
e_w = 0.8                                                                                   # emissivity of tube 
lambda_s = 0.3489                                                                           # thermal conductivity of the solid [W/m/K]
Twin = 850+273.15                                                                         # Tube wall temperature [K]
Eta_list = []
kr_list = []
Deff_list = []
# Input Streams Definition - Pantoleontos Data                                                                                
#f_IN = 0.00651                                                                               # input molar flowrate (kmol/s)

# Components  [CH4, CO, CO2, H2, H2O]
Tin_R1 =  785+273.15                                                                            # Inlet Temperature [K]
Pin_R1 =  15                                                                              # Inlet Pressure [Bar]
#x_in_R1 = np.array([0.22155701, 0.00, 0.01242592, 0.02248117, 0.74353591 ])                              # Inlet molar composition
Fin = np.array([0.5439,0.0001,0.3461,0.0001,2.7039])    #kmol/h

f_IN = np.sum(Fin)/Nt                                   # inlet molar flow per tube [kmol/h]
x_in_R1 = np.zeros(n_comp)
for i in range(0,n_comp):
    x_in_R1[i] = Fin[i]/np.sum(Fin)                     # inlet molar composition
MWmix = np.sum(x_in_R1*MW)
w_in = x_in_R1*MW / MWmix
m_R1 = f_IN*np.sum(np.multiply(x_in_R1,MW))/3600             # Inlet mass flow [kg/s]
f_IN_i = x_in_R1*f_IN                                       # inlet flowrate per tube
omegain_R1 = w_in                                                              # Inlet mass composition
                                                    
SC = x_in_R1[4] / x_in_R1[0]        # the steam to carbon was calculated upon the total amount of carbon, not only methane

# Thermodynamic Data
R = 8.314                                                                               # [J/molK]
################################################################################
# AUXILIARY CALCULATIONS
Aint = numpy.pi*dTube**2/4                                                              # Tube section [m2]
# Perry's data 
        # CH4,          CO,             CO2,            H2,              H2O          
dH_formation_i = np.array([-74.52, -110.53, -393.51, 0, -241.814])                                  # Enthalpy of formation [kJ/mol]       
DHreact = np.sum(np.multiply(nu,dH_formation_i),axis=1).transpose()                                 # Enthalpy of reaction              [kJ/mol]
################################################################################

################################################################################
# REACTOR INLET
Mi_R1 = m_R1 * omegain_R1 * 3600                                            # Mass flowrate per component [kg/h]
Ni_R1 = np.divide(Mi_R1, MW)                                        # Molar flowrate per component [kmol/h]
Ntot_R1 = np.sum(Ni_R1)                                             # Molar flowrate [kmol/h]
zi_R1 = Ni_R1 / Ntot_R1                                             # Inlet Molar fraction to separator
MWmix_R1 = np.sum(np.multiply(zi_R1,MW))                            # Mixture molecular weight
F_R1 = m_R1/MWmix_R1                                                # Inlet Molar flowrate [kmol/s]

# SOLVER FIRST REACTOR

zspan = np.array([0,Length])
N = 100                                             # Discretization
z = np.linspace(0,Length,N)
# Tw = 1000.4 + 12.145*z + 0.011*z**2
y0_R1  = np.concatenate([omegain_R1, [Tin_R1], [Pin_R1]])

sol = solve_ivp(TubularReactor, zspan, y0_R1, t_eval=z,
                 args=(Epsilon, Dp, m_R1, Aint, MW, nu, R, dTube, Twin, RhoC, DHreact, Tc, Pc, F_R1, e_w, lambda_s,e_s,tau, p_h,c_h,n_h,s_h,s_w))

wi_out = np.zeros( (5,np.size(sol.y[0])) )
wi_out = sol.y[0:5]                              
T_R1 = sol.y[5]
P_R1 = sol.y[6]



################################################################################
# REACTOR OUTLET
# CH4,          CO,             CO2,            H2,              H2O
Fi_out = np.zeros((n_comp,np.size(wi_out[0])))
F_tot_out = np.zeros(np.size(wi_out[0])); yi = np.zeros(np.size(wi_out[0]))
Mi_out = m_R1 * wi_out                                        # Mass flowrate per component [kg/h]
for i in range(0,n_comp):
    Fi_out[i] = Mi_out[i,:]/MW[i]                                     # Molar flowrate per component [kmol/h]
for j in range(0,np.size(wi_out[0])):
    F_tot_out[j] = np.sum(Fi_out[:,j])                                              # Molar flowrate [kmol/h]
yi = Fi_out/F_tot_out                                                           # outlet Molar fraction to separator
# MWmix_f1 = np.sum(np.multiply(zi_f1,MW))                                # Mixture molecular weight
#F_f1 = M_R1/MWmix_f1*3600                                                # Outlet Molar flowrate [kmol/h]


################################################################
# POST CALCULATION
#Tw = np.ones(np.size(wi_out[0]))*Twin
z = np.linspace(0,Length,np.size(wi_out[0]))
#Tw = Twin + 12.145*z + 0.011*z**2
Tw = 150*np.log(2*z+1)+Twin
################################################################
# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.set_xlabel('Reator Lenght [m]'); ax1.set_ylabel('T [K]')
ax1.legend(['Tg','Tf'])
ax1.plot(z,T_R1,z,Tw)

ax2.set_xlabel('Reator Lenght [m]'); ax2.set_ylabel('Molar Fraction')
for i in range(0,n_comp):
    ax2.plot(z, yi[i])
ax2.legend(['CH4', 'C0','CO2', 'H2','H2O'])

ax3.set_xlabel('Reactor Lenght [m]'); ax3.set_ylabel('P [bar]')
ax3.plot(z,P_R1)

# plt.figure()
# Eta_list = np.array(Eta_list)
# z1 = np.linspace(0,Length,np.size(Eta_list[:,0]))
# plt.xlabel('Reactor Lenght [m]'), plt.ylabel('diffusion efficiency')
# plt.plot(z1,Eta_list[:,0],label=r'$\eta1$');
# plt.plot(z1,Eta_list[:,1],label=r'$\eta2$');
# plt.plot(z1,Eta_list[:,2],label=r'$\eta3$')
# plt.legend()

# plt.figure()
# Deff_list = np.array(Deff_list)
# z1 = np.linspace(0,Length,np.size(Deff_list))
# plt.xlabel('Reactor Lenght [m]'), plt.ylabel('diffusion coefficient [m2/s]')
# plt.plot(z1,Deff_list,label='CH4')
# plt.legend()

# plt.figure()
# kr_list = np.array(kr_list)
# z1 = np.linspace(0,Length,np.size(kr_list[:,0]))
# plt.xlabel('Reactor Lenght [m]'), plt.ylabel('Rate of equation')
# plt.plot(z1,kr_list[:,0],label='kr1'); plt.plot(z1,kr_list[:,1],label='kr2'); plt.plot(z1,kr_list[:,2],label='kr3')
# plt.legend()
plt.show()

