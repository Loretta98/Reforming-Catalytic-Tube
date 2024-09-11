
import numpy as np 

# INPUT DATA REACTOR
n_comp = 5                                 # total number of species
nu = np.array([ [-1, 1, 0, 3, -1],
                [0, -1, 1, 1, -1], 
                [-1, 0, 1, 4, -2]])         # SMR, WGS, reverse methanation stoichiometric coefficients

# Components  [CH4, CO, CO2, H2, H2O] O2, N2]
MW = np.array([16.04, 28.01, 44.01, 2.016, 18.01528, 32.00, 28.01])                    # Molecular Molar weight       [kg/kmol]
Tc = np.array([-82.6, -140.3, 31.2, -240, 374,-118.6,-147]) + 273.15            # Critical Temperatures [K]
Pc = np.array([46.5, 35, 73.8, 13, 220.5,50.5,34])                          # Critical Pressures [bar]

# Data from FAT experimental setup 
Nt =   4                                                                                    # Number of tubes
#dTube = 0.1
dTube = 0.14142                                                                             # Tube diameter [m]
s_w = 0.06                                                                                  # Tube thickness
dTube_out = dTube+s_w                                                                       # Tube outlet diameter [m]
Length = 2                                                                                  # Length of the reactor [m]
kw = 28.5                                                                                   # Thermal conductivity of the wall [W/m] from Quirino
kf = 2.6                                                                                    # Thermal conductivity of the refractory [W/m] from Quirino

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

# Furnace data 
f_biogas = 30                           # Nm3/h Biogas available as fuel 
f_biogas = 30/22.41                     # kmol/h 
f_air = f_biogas/0.6*2/0.21*(1+5/100)  # Ratio of air for stochiometric combustion with a 5% excess of air 
f_furnace = f_biogas + f_air            # kmol/h  
# [CH4, CO, CO2, H2, H2O, O2,N2]
x_f = np.array([0.093, 0.062, 0, 0, 0, 0.178, 0.667])   # composition at the inlet 
x_f = np.array([0, 0.06, 0.11, 0, 0.13, 0, 0.7])        # composition of the exhaust gas
p = dTube_out*(1+1)                                     # Tubes pitch [m]
D_h =  (dTube+s_w*2)*(4/np.pi*(p/(dTube/2+s_w)-1))    # Hydraulic Diameter [m]
A_f = 2*2- dTube_out**2/4*np.pi*Nt  # tranversal area of the furnace [m2] L = 2, W = 2 metri come guess
m_furnace = np.sum(np.multiply(f_furnace,np.multiply(x_f,MW))) # kg/h 

z = np.linspace(0,Length)
A = 4.6; B = 2.014 ; L = 0.5        # Constants to emulate combustion considtions
LHV = 22                            # lower heating value of biogas with CH4 = 60%, CO2 = 40% [MJ/m3]
Rho_BG = 0.8244 # [kg/m3] Aspen
LHV = LHV/Rho_BG # [MJ/kg]
Hv = 2.257*1000                          # Vapor heat of vaporization [kJ/kg]
nH2O = 2 ; nfuel = 1
Q0 = (LHV + Hv * nH2O / nfuel) * m_furnace / 1000 #[kJ/kg*kg/h]
Q0_ = 111.6*3600                          #[kW] from Aspen = kJ/h
Q = Q0 *A*B/L*(z/L)**(B-1)*np.exp(-A*(z/L)**B) 

# Input Streams Definition - Pantoleontos Data                                                                                
#f_IN = 0.00651                                                                                     # input molar flowrate (kmol/s)

# Components  [CH4, CO, CO2, H2, H2O]
Tin_R1 =  785 + 273.15                                                                              # Inlet Temperature [K]
Tin_f = 120 +273.15
Tin_w = 785 + 273.15
Pin_R1 =  15                                                                                        # Inlet Pressure [Bar]
Fin = np.array([0.5439,0.0001,0.3461,0.0001,2.7039])            #kmol/h

f_IN = np.sum(Fin)/Nt                                           # inlet molar flow per tube [kmol/h]
x_in_R1 = np.zeros(n_comp)
for i in range(0,n_comp):
    x_in_R1[i] = Fin[i]/np.sum(Fin)                             # inlet molar composition
MWmix = np.sum(x_in_R1*MW[:-2])
w_in = x_in_R1*MW[:-2] / MWmix
m_R1 = f_IN*np.sum(np.multiply(x_in_R1,MW[:-2]))/3600                # Inlet mass flow for tube [kg/s]
f_IN_i = x_in_R1*f_IN                                           # inlet flowrate per tube
omegain_R1 = w_in                                               # Inlet mass composition
                                                    
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
Ni_R1 = np.divide(Mi_R1, MW[:-2])                                        # Molar flowrate per component [kmol/h]
Ntot_R1 = np.sum(Ni_R1)                                             # Molar flowrate [kmol/h]
zi_R1 = Ni_R1 / Ntot_R1                                             # Inlet Molar fraction to separator
MWmix_R1 = np.sum(np.multiply(zi_R1,MW[:-2]))                            # Mixture molecular weight
F_R1 = m_R1/MWmix_R1                                                # Inlet Molar flowrate [kmol/s]
# Tw = 1000.4 + 12.145*z + 0.011*z**2