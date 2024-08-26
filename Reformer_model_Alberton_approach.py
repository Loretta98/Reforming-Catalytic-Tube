#  QUESTO CODICE CONTIENE MODELLO COMPLETO DI 1 REATTORE DI SMR 
#  CINETICA Xu-Froment 

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, quad, solve_bvp
from scipy.optimize import fsolve
import sympy as sp
from Twall import solve_heat_bvp
from Physical_Properties import *
from Kinetics import calculate_kinetics

#####################################################################################################################################
################################################# Tubular reactor resolution ########################################################
#####################################################################################################################################


def TubularReactor(z,y,Epsilon,Dp,m_gas,Aint,MW,nu,R,dTube,Twin,RhoC,DHreact,Tc,Pc,F_R1,e_w, lambda_s,e_s,tau,p_h,c_h,n_h,s_h,s_w,mesh_size):

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
    RhoGas = (Ppa*MWmix) / (R*T)  / 1000                                        # Gas mass density [kg/m3]
    VolFlow_R1 = m_gas / RhoGas                                                 # Volumetric flow per tube [m3/s]
    u = (F_R1*1000) * R * T / (Aint*Ppa)                                        # Superficial Gas velocity if the tube was empy (Coke at al. 2007)
    u = VolFlow_R1 / (Aint * Epsilon)                                           # Gas velocity in the tube [m/s]
    
    DH_reaction,Cpmix = calculate_DH_reaction(T, MW, omega, DHreact, nu) #J/kmol
    Deff = calculate_diffusivity(T,P,n_comp,yi,MWmix,MW,e_s,tau,R)
    U, h_t, h_f, lambda_gas, DynVis = calculate_heat_transfer_coefficients(T,MW,n_comp,Tc,Pc,yi,Cpmix,RhoGas,u,Dp,dTube,e_w,Epsilon,lambda_s,dTube_out)# J/m2/s/K = W/m2/K
    rj, kr = calculate_kinetics(T, R, Pi,RhoC,Epsilon)
    Eta = calculate_effectiveness_factor(kr,Dp,c_h,n_h,s_h,lambda_gas,Deff,p_h)

    h_env = 0.1                                                                 # Convective coefficient external environment [W/m2/K]
    Thick = 0.01                                                                # Tube Thickness [m]

    Deff_CH4 = Deff[0]*1E-4                                                     # Effective diffusion [m2/s]
    Deff_list.append(Deff_CH4)
    Eta_list.append(Eta)
    h_t_list.append(h_t)
    U_list.append(U)
    ############################################### Radial Tube temperature distribution #################################################################################

    # Create a radial mesh and initial guess for temperature distribution
    r_in = dTube/2                                                                  # tube internal radius [m]
    epsilon_w = 0.85            # wall emissivity 
    epsilon_f = 0.3758          # furnace emissivity
    T_f = 850+273.15            # Furnace Temperature
    #T_f = Tw
    s_k_boltz = 5.66961e-8      #Stefan Boltzmann constant  [Jm2/K/s]

    # Solve the BVP
    r = np.linspace(dTube / 2, dTube / 2 + s_w, mesh_size)
    #y_guess_1 = np.array(Guess_list[-1])
    #y_guess_1 = []
    y_guess = np.zeros((2, r.size))
    if z == 0:
        y_guess = np.zeros((2, r.size))  # y_guess[0] = x, y_guess[1] = dx/dr
    else:
        y_guess = np.array(Guess_list[-1])
    
    # Solve the BVP using the wrapper function
    solution = solve_heat_bvp(y_guess, h_t, r_in, T, epsilon_w, epsilon_f, s_k_boltz, h_f, T_f, s_w,mesh_size)
    
    if solution.success:
        r_sol = solution.x
        T_sol = solution.y[0]
        dT_sol = solution.y[1]
        y_guess_1 = np.ones((2,r.size))
        y_guess_1[0] = T_sol
        y_guess_1[1] = dT_sol
        Guess_list.append(y_guess_1)

    Tw = T_sol[-1]
    Tw_list.append(Tw)

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
Twin = 850+273.15                                                                           # Tube wall temperature [K]

Eta_list = []
kr_list = []
Deff_list = []
Guess_list = []
h_t_list = []
Tw_list = []
U_list = []
mesh_size = 100
# Input Streams Definition - Pantoleontos Data                                                                                
#f_IN = 0.00651                                                                               # input molar flowrate (kmol/s)

# Components  [CH4, CO, CO2, H2, H2O]
Tin_R1 =  785+273.15                                                                            # Inlet Temperature [K]
Pin_R1 =  15                                                                              # Inlet Pressure [Bar]
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
Aint = np.pi*dTube**2/4                                                              # Tube section [m2]
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
                 args=(Epsilon, Dp, m_R1, Aint, MW, nu, R, dTube, Twin, RhoC, DHreact, Tc, Pc, F_R1, e_w, lambda_s,e_s,tau, p_h,c_h,n_h,s_h,s_w,mesh_size))

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
Tw = np.array(Tw_list)
z = np.linspace(0,Length,np.size(wi_out[0]))
z1 = np.linspace(0,Length,np.size(Tw))
Tf = np.ones(np.size(z1))*(850+273.15)
#Tw = Twin + 12.145*z + 0.011*z**2
#Tw = 150*np.log(2*z+1)+Twin
################################################################
# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.set_xlabel('Reator Lenght [m]'); ax1.set_ylabel('T [K]')
ax1.plot(z,T_R1,z1,Tw,z1,Tf)
ax1.legend(['Tg','Tw','Tf'])

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

plt.figure()
h_t = np.array(h_t_list)
z1 = np.linspace(0,Length,np.size(h_t_list))
U = np.array(U_list)
z2 = np.linspace(0,Length,np.size(U_list))
plt.xlabel('Reactor Lenght [m]'), plt.ylabel('tube thermal conductivity [W/m/K]')
plt.plot(z1,h_t,z2,U)
plt.legend('h_t','U')

plt.show()

