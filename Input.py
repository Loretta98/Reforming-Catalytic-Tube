import numpy as np 

# INPUT DATA FIRST REACTOR
n_comp = 5; 
nu = np.array([ [-1, 1, 0, 3, -1],
                [0, -1, 1, 1, -1], 
                [-1, 0, 1, 4, -2]])  # SMR, WGS, reverse methanation

# Components  [CH4, CO, CO2, H2, H2O] O2, N2]
MW = np.array([16.04, 28.01, 44.01, 2.016, 18.01528 ]) #, 32.00, 28.01])                    # Molecular Molar weight       [kg/kmol]
Tc = np.array([-82.6, -140.3, 31.2, -240, 374]) + 273.15            # Critical Temperatures [K]
Pc = np.array([46.5, 35, 73.8, 13, 220.5])                          # Critical Pressures [bar]
# Reactor Design Pantoleontos

# Data from FAT experimental setup 
Nt =   4                                                                                   # Number of tubes
#dTube = 0.1
dTube = 0.14142                                                                              # Tube diameter [m]
dTube_out = dTube+0.06                                                                          # Tube outlet diameter [m]
Length = 2                                                                                 # Length of the reactor [m]

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
Tin_R1 =  600+273.15                                                                            # Inlet Temperature [K]
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
