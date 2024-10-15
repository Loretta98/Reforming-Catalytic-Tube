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

# Components  [CH4, CO, CO2, H2, H2O] O2, N2]
MW_f = np.array([16.04, 28.01, 44.01, 2.016, 18.01528, 32.00, 28.01])                    # Molecular Molar weight       [kg/kmol]
Tc_f = np.array([-82.6, -140.3, 31.2, -240, 374,-118.6,-147]) + 273.15            # Critical Temperatures [K]
Pc_f = np.array([46.5, 35, 73.8, 13, 220.5,50.5,34])                          # Critical Pressures [bar]


# Reactor Design Pantoleontos

# Data from FAT experimental setup 
Nt =   4                                                                                   # Number of tubes
#dTube = 0.1
dTube = 0.14142                                                                              # Tube diameter [m]
dTube_out = dTube+0.06                                                                          # Tube outlet diameter [m]
Length = 2                                                                                 # Length of the reactor [m]
s_w = 0.03
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
e_w = 0.85                                                                                  # emissivity of tube from Quirino
e_f = 0.3758                                                                                # emissitivty of the furnace from Quirino
lambda_s = 0.3489                                                                           # thermal conductivity of the solid [W/m/K]
sigma = 5.67e-8 # Stefan Boltzmann constan [W/m2/K4] 
k_w = 28.5 # tube thermal conductivity [W/mK]
p = dTube_out*(1+1)                                     # Tubes pitch [m]
D_h =  (dTube+s_w*2)*(4/np.pi*(p/(dTube/2+s_w)-1))    # Hydraulic Diameter [m]
Eta_list = []
kr_list = []
Deff_list = []
Tw_list = []
# Input Streams Definition - Pantoleontos Data                                                                                
#f_IN = 0.00651                                                                               # input molar flowrate (kmol/s)


# Furnace data 
f_biogas = 30                           # Nm3/h Biogas available as fuel 
f_biogas = 30/22.41                     # kmol/h
excess = 0
f_air = ((f_biogas*0.6)*2)/0.21*(1+excess)  # Ratio of air for stochiometric combustion with a 5% excess of air
f_furnace = f_biogas + f_air            # kmol/h  
# [CH4, CO, CO2, H2, H2O, O2,N2]
x_f = np.array([0.093, 0.062, 0, 0, 0, 0.178, 0.667])   # composition at the inlet 
x_f = np.array([0, 0.06, 0.11, 0, 0.13, 0, 0.7])        # composition of the exhaust gas
p = dTube_out*(1+1)                                     # Tubes pitch [m]
D_h =  (dTube+s_w*2)*(4/np.pi*(p/(dTube/2+s_w)-1))    # Hydraulic Diameter [m]
A_f = 2*2- dTube_out**2/4*np.pi*Nt  # tranversal area of the furnace [m2] L = 2, W = 2 metri come guess
m_furnace = np.sum(np.multiply(f_furnace,np.multiply(x_f,MW_f))) # kg/h


# Components  [CH4, CO, CO2, H2, H2O]

#### INLET FROM REAL DATA !!!! ####
Twin = 600+273.15                                                                         # Tube wall temperature [K]
Tf = 900+273.15
Tin_R1 =  600+273.15                                                                            # Inlet Temperature [K]
Pin_R1 =  15                                                                              # Inlet Pressure [Bar]

#x_in_R1 = np.array([0.22155701, 0.00, 0.01242592, 0.02248117, 0.74353591 ])                              # Inlet molar composition
Fin = np.array([0.5439,0.0001,0.3461,0.0001,2.7039])    #kmol/h
f_IN = np.sum(Fin)/Nt                                   # inlet molar flow per tube [kmol/h]
x_in_R1 = np.zeros(n_comp)
for i in range(0,n_comp):
    x_in_R1[i] = Fin[i]/np.sum(Fin)                     # inlet molar composition

# if measured data is: 0.5 CO2 e 0.5 CH4 in M1, M1 = 24 kg/h and M2 = 47.8 kg/h  all water
# Mtot = 71.8 kg/h 
# [CH4, CO, CO2, H2, H2O] O2, N2]
M1 = 23.85; M2 = 47.855              #kg/h 
# Mole fractions in stream M1
x_M1_CH4 =0.57
x_M1_CO2 = 0.43
# Calculate molar flow rates (kmol/h) for M1 and M2
F1 = M1 / (x_M1_CH4 * MW[0] + x_M1_CO2 * MW[2])  # CH4 and CO2 in M1
F2 = M2 / MW[4]  # All H2O in M2

# Total molar flow rate (kmol/h)
F3 = F1 + F2

# Mole fractions at the inlet
x_in_CH4 = (F1 * x_M1_CH4) / F3
x_in_CO2 = (F1 * x_M1_CO2) / F3
x_in_H2O = F2 / F3

# Mole fraction vector for all species: [CH4, CO, CO2, H2, H2O}
x_in = np.zeros(5)
x_in[0] = x_in_CH4  # CH4
x_in[1] = 0.00001
x_in[2] = x_in_CO2  # CO2
x_in[3] = 0.00001
x_in[4] = x_in_H2O  # H2O

# Output the mole fractions of the inlet stream
print("Mole fraction of CH4 at the inlet:", x_in[0])
print("Mole fraction of CO2 at the inlet:", x_in[2])
print("Mole fraction of H2O at the inlet:", x_in[4])

f_IN = F3/Nt
x_in_R1 = x_in
# x_in_R1 = np.array([0.1488, 0.00001, 0.0992, 0.00001,0.7520])
# x_in_R1 = np.array([0.1313, 0.00001, 0.1074, 0.00001,0.7613])

Fin = F3*x_in_R1
print('Min = ', M1+M2)
MWmix = np.sum(x_in_R1*MW)
w_in = x_in_R1*MW / MWmix
m_R1 = f_IN*np.sum(np.multiply(x_in_R1,MW))/3600             # Inlet mass flow [kg/s]

f_IN_i = x_in_R1*f_IN                                       # inlet flowrate per tube

omegain_R1 = w_in                                                              # Inlet mass composition
                                                    
SC = x_in_R1[4] / x_in_R1[0]        # the steam to carbon was calculated upon the total amount of carbon, not only methane
print('Steam to Carbon ratio=', SC)
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
