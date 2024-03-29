# This is a sample Python script.
import numpy 
#  QUESTO CODICE CONTIENE MODELLO COMPLETO DI 1 REATTORE DI SMR 
#  CINETICA Xu-Froment 

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import os
import win32com.client as win32
import thermo


def GetMolarFraction(x,SN,COR):

    #x[0:2] # CO, CO2, H2

    Equation = np.zeros(3)

    Equation[0] = SN-(x[2]-x[1])/(x[0]+x[1])
    Equation[1] = COR-x[1]/(x[0]+x[1])
    Equation[2] = 1-x[0]-x[1]-x[2]

    return Equation

def TubularReactor(z,y,Epsilon,Dp,Hp,m_gas,Aint,MW,nu,R,DHreact,DynVis,K_gas,dTube,Nt,DHvap,TShell,RhoC,Tcs,Pcs,omegas):

    omega = y[0:5]
    T =     y[6]
    P =     y[7]

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

    RhoGas = (P*MWmix) / (R*T)  /1000                                           # Gas mass density [kg/m3]

    VolFlow_R1 = m_gas / RhoGas                                                 # Volumetric flow per tube [m3/s]

    u = VolFlow_R1 / (Aint * Epsilon)                                           # Gas velocity in the tube [m/s]     

    Re = RhoGas * u * dTube / DynVis                                            # Reynolds number []

    # Mixture massive Specific Heat calculation (NASA correalations)
                # CH4,          CO,             CO2,            H2,              H2O,           O2             N2
    a1 = np.array([0.748e-2,    2.715      ,    3.85       ,     3.337       ,   3.033      ,   3.282       ,  0.029e+2    ])
    a2 = np.array([1.339e-2,    2.063e-3   ,    4.414e-3   ,     -4.940e-5   ,   2.176e-3   ,   1.483e-3    ,  0.148e-02   ])
    a3 = np.array([-5.732e-6,   -9.988e-7  ,    -2.214e-6  ,     4.994e-7    ,   -1.64e-7   ,   -7.579e-7   ,  -0.0568e-5  ])
    a4 = np.array([1.222e-9,    2.3e-10    ,    5.234e-10  ,     -1.795e-10  ,   -9.704e-11 ,   2.094e-10   ,  0.1e-09     ])
    a5 = np.array([-1.018e-13,  -2.036e-14 ,    -4.72e-14  ,     2.002e-14   ,   1.682e-14  ,   -2.167e-14  ,  -0.067e-13  ])
    a6 = np.array([-9.468e+03,  -1.415e+4  ,    -4.874e+4  ,     -9.501e+2   ,   -3.00e+4   ,   -1.088e+3   ,  -0.092e+4   ])

    Cp_mol = R*(a1 + a2*T + a3*T**2 + a4*T**3 + a5*T**4)                                    # Molar specific heat per component [J/molK]
    Cp = Cp_mol/MW*1000                                                                     # Mass specific heat per component [J/kgK]

    dHf0 = R*T*(a1*np.log(T) + a2*T + a3*(T**2)/2 + a4*(T**3)/4 + a5*(T**4) /5 + a6/T)      # Enthalpy of formation of each compound [J/mol]
    dHf0 = dHf0*1000                                                                        # [J/kmol]
    Cpmix = np.sum(Cp*omega)                                                                # Mixture specific heat [J/kgK]

    # reazioni del SMR      R1                      R2                  R3                           
    DHR = np.array([ dHf0[0]+dHf0[4],               dHf0[1]+dHf0[4],    dHf0[0]+2*dHf0[4]])  # enthalpy of reagents
    DHP = np.array([ dHf0[1]+3*dHf0[3],             dHf0[2]+dHf0[3],    dHf0[2]+4*dHf0[3]])  # enthalpy of products

    DH_reaction = DHR - DHP                                                                  # Enthalpy of the reaction at the gas temperature

    ################################## Heat Transfer Coefficient Calculation ##################################################################################
    
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

    k_i     = a + b*T + c*T**2                                                              # thermal conductivity of gas [W/m/K]
    K_gas   = sum(omega*k_i)                                                                # Thermal conductivity of the mixture as of Wassiljewa equation [W/m/K]

    # Wilke Method for low-pressure gas viscosity   
    mu_i    = A + B*T + C*T**2                                                              # viscosity of gas [micropoise]
    PHI = np.identity(n_comp)                                                               # initialization for PHI calculation
    dynamic_viscosity_array = np.zeros(n_comp)

    for i in range(0,n_comp-2):
        for j in range(i+1,n_comp-1): 
            PHI[i,j] = ( 1+ (mu_i[i]/mu_i[j])**(0.5) * (MW[j]/MW[i])**(0.25) )**2 / ( 8*(1 + MW[i]/MW[j])**(0.5) )
            PHI[j,i] = mu_i[j]/mu_i[i]*MW[i]/MW[j] * PHI[i,j]

    for i in range(0,n_comp-1):
        den = 0
        for j in range(0,n_comp-1): 
            den += yi[j]*PHI[j,i]

        num = yi[i]*mu_i[i]
        dynamic_viscosity_array[i] = num/den 
    
    DynVis  = sum(dynamic_viscosity_array) * 1e-7                                           # Dynamic viscosity [kg/m/s] 

    Pr = Cpmix*DynVis/K_gas                                                                 # Prandtl number
    h_t = K_gas/Dp*(2.58*Re**(1/3)*Pr**(1/3)+0.094*Re**(0.8)+Pr**(0.4))                     # Convective coefficient tube side [W/m2/K]
    # h_t =  1463.9   # Convective coefficient tube side [W/m2/K] from Poliana's code

    h_env = 0.1                                                                             # Convective coefficient external environment [W/m2/K]
    Thick = 0.01                                                                            # Tube Thickness [m]
    
##################################################################
# Kinetic Costant and Rate of reaction (Xu-Froment)
    # CH4 + H2O -> CO + 3 H2 ; CO + H2O -> CO2 + H2 ; CH4 + 2H2O +> CO2 + 4H2
    
    # Equilibrium constants Twigg coverted to [bar]
    alpha = 1000/T - 1
    Keq1 = 1.01325 / (np.exp(alpha*(alpha*(alpha*(0.2513*alpha-0.3665)-0.58101)+27.1337)-3.277))
    Keq2 = np.exp(alpha*(alpha*(0.63508-0.29353*alpha)+4.1778)+0.31688)
    Keq3 = Keq1*Keq2
    # Arrhenius     I,       II,        III
    k0 = np.array([1.842e-4, 7.558,     2.193e-5])                                      # pre exponential factor @648 K 
    E_a = np.array([240.1,   67.13,     243.9])                                         # activation energy [kJ/mol]
    kr = k0*np.exp(-E_a/R/T)
    # Van't Hoff    CO,     H2,         CH4,    H2O
    K0 = np.array([40.91,   0.0296,     0.1791, 0.4152])                                # pre exponential factor @648,823 K
    DH0 = np.array([-70.65, -82.90,     -38.28,  88.68])                                # adsorption enthalpy [kJ/mol]
    Kr = K0*np.exp(-DH0*1000/R/T)

    # Components  [CH4, CO, CO2, H2, H2O, O2, N2]
    DEN = 1 + Kr[0]*Pi[1] + Kr[1]*Pi[3] + Kr[2]*Pi[0] + Kr[3]*Pi[4]/Pi[3]
    rj = np.array([ (kr[0]/Pi[3]**(2.5)) * (Pi[0]*Pi[4]-(Pi[3]**3)*Pi[1]/Kr[0]) / DEN**2 , (kr[1]/Pi[3]) * (Pi[1]*Pi[4]-Pi[3]*Pi[2]/Kr[1]) / DEN**2 , (kr[2]/Pi[3]**(3.5)) * (Pi[0]*(Pi[4]**2)-(Pi[3]**4)*Pi[2]/Kr[2]) / DEN**2 ]) * RhoC * (1-Epsilon)

#####################################################################
# Equations
    Reactor1 = Aint / (m_gas*3600) * MW[0] * np.sum(np.multiply(nu[:, 0], np.multiply(Eta, rj)))
    Reactor2 = Aint / (m_gas*3600) * MW[1] * np.sum(np.multiply(nu[:, 1], np.multiply(Eta, rj)))
    Reactor3 = Aint / (m_gas*3600) * MW[2] * np.sum(np.multiply(nu[:, 2], np.multiply(Eta, rj)))
    Reactor4 = Aint / (m_gas*3600) * MW[3] * np.sum(np.multiply(nu[:, 3], np.multiply(Eta, rj)))
    Reactor5 = Aint / (m_gas*3600) * MW[4] * np.sum(np.multiply(nu[:, 4], np.multiply(Eta, rj)))
    Reactor6 = - Aint/ ((m_gas*3600)*Cpmix) * np.sum(np.multiply(DH_reaction, np.multiply(Eta,rj))) + np.pi*dTube/(m_gas*Cpmix)*h_t*(Twin - T)
    Reactor7 = (-150 * (((1-Epsilon)**2)/Epsilon**3) * DynVis*u/Dp**2 -1.75 * ((1-Epsilon)/Epsilon**3) * m_gas*u/(Dp*Aint) )
    
    return np.array([Reactor1, Reactor2, Reactor3, Reactor4, Reactor5, Reactor6, Reactor7])

#######################################################################
# INPUT DATA FIRST REACTOR
n_comp = 5; 
nu = np.array([ [-1, 1, 0, 3, -1],
                [0, -1, 1, 1, -1], 
                [-1, 0, 1, 4, -2]])  # SMR, WGS, reverse methanation

# Components  [CH4, CO, CO2, H2, H2O, O2, N2]
MW = np.array([16.04, 28.01, 44.01, 2.016, 18.01528, 32.00, 28.01])                    # Molecular Molar weight       [kg/kmol]
omegas = np.array([])

# Reactor Design Poliana

Nt =   52                                                                           # Number of tubes
dTube = 0.106                                                                           # Tube diameter [m], 
Length = 9.36                                                                           # Length of the reactor [m]

Epsilon = 0.519                                                                         # Void Fraction 
RhoC = 2355.2                                                                           # Catalyst density [kg/m3] 
Dp = 0.0054                                                                             # Catalyst particle diameter [m] 

Twin = 1000 + 273.15                                                                    # Tube wall temperature [K]

# Input Streams Definition - Poliana Data
SC = 2.7                                                                                  # Steam to Carbon Ratio
M_R1 = 6.6348                                                                           # Input Mass Flowrate [kg/s]

# FirstGuess_in = np.array([0.2, 0.2, 0.6])
# root = fsolve(GetMolarFraction, x0=FirstGuess_in, args=(SN,COR))
# omegain_R1 = np.array([0.00, root[0]*MW[1]/(root[0]*MW[1]+root[1]*MW[2]+root[2]*MW[3]), root[1]*MW[2]/(root[0]*MW[1]+root[1]*MW[2]+root[2]*MW[3]), root[2]*MW[3]/(root[0]*MW[1]+root[1]*MW[2]+root[2]*MW[3]), 0.00, 0.00])                   # Inlet mass composition

Tin_R1 =  646.15                                                                            # Inlet Temperature [K]
Pin_R1 =  21.6                                                                              # Inlet Pressure [Bar]

# Thermodynamic Data
R = 8.314                                                                          # [kj/kmolK]
################################################################################
# AUXILIARY CALCULATIONS
Aint = numpy.pi*dTube**2/4                                                          # Tube section [m2]
#m_R1 = M_R1/Nt                                                                      # Mass Flowrate per tube [kg/s tube]
################################################################################

# SOLVER FIRST REACTOR

zspan = np.array([0,Length])
z = np.linspace(0,Length,401)
m_vap_0 = 0                                                                            # Steam generated
y0_R1 = np.concatenate((omegain_R1, [Tin_R1], [m_vap_0], [Pin_R1]))

sol = solve_ivp(TubularReactor, zspan, y0_R1, t_eval=z, args=(Epsilon, Dp, Hp, m_R1, Aint, MW, nu, R, DHreact, DynVis_R1, K_gas, dTube, Nt, DHvap, TShell, RhoC, Tcs, Pcs, omegas))

omega_R1 = np.zeros( (6,401) )
omega_R1 = sol.y[0:6]                              # y[6] è il metano che rimane invariato
T_R1 = sol.y[6] - 273.15
mvap = sol.y[7]
P_R1 = sol.y[8]

################################################################################
# REACTOR INLET
Mi_R1 = M_R1 * omegain_R1 * 3600                                            # Mass flowrate per component [kg/h]
Ni_R1 = np.divide(Mi_R1, MW)                                        # Molar flowrate per component [kmol/h]
Ntot_R1 = np.sum(Ni_R1)                                             # Molar flowrate [kmol/h]
zi_R1 = Ni_R1 / Ntot_R1                                             # Inlet Molar fraction to separator
MWmix_R1 = np.sum(np.multiply(zi_R1,MW))                            # Mixture molecular weight
F_R1 = M_R1/MWmix_R1*3600                                                # Inlet Molar flowrate [kmol/h]

################################################################################
# REACTOR OUTLET

Mi_f1 = M_R1 * omega_R1[:,-1] * 3600                                        # Mass flowrate per component [kg/h]
Ni_f1 = np.divide(Mi_f1, MW)                                        # Molar flowrate per component [kmol/h]
Ntot_f1 = np.sum(Ni_f1)                                             # Molar flowrate [kmol/h]
zi_f1 = Ni_f1 / Ntot_f1                                             # Inlet Molar fraction to separator
MWmix_f1 = np.sum(np.multiply(zi_f1,MW))                            # Mixture molecular weight
F_f1 = M_R1/MWmix_f1*3600                                                # Inlet Molar flowrate [kmol/h]


################################################################
# POST CALCULATION

Conv_R1 = ((Ni_R1[2]-Ni_f1[1])/Ni_R1[2])

Res = np.array([Mi_f1[0], T_R1.max(), T_R1[-1]])

# np.savetxt("C:\Users\mbozzini\OneDrive - Politecnico di Milano\Desktop\PhD\Articoli\2024_01_Articolo1\CASO BASE\prova.csv",T_R1,delimiter=",")
print(Mi_f1)
# PRODUCTION


################################################################
# Plotting
