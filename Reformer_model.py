# This is a sample Python script.
import numpy
#  QUESTO CODICE CONTIENE MODELLO COMPLETO DI 1 REATTORE DI SMR 
#  CINETICA Xu-Froment 

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def TubularandFurnaceReactor(z, y, Epsilon, Dp, m_gas, Aint, MW, nu, R, dTube, Twin, RhoC, DHreact, Tc, Pc, F_R1, e_w, lambda_s,Lf, lambda_t,sigma,DHydr, Pf, nu_c,F_gas_i_IN, DHreact_f,Rf,Aint_f, M_in_furnace):
    omega = y[0:5]          # Tube side Components  [CH4, CO, CO2, H2, H2O]
    T = y[5]
    P = y[6]
    omega_f = y[7:14]       # Furnace Side Components [CH4, CO, CO2,  H2, H2O, O2, N2]
    Tf = y[14]
    #Twext = y[15]
    # MW Components  [CH4, CO, CO2, H2, H2O, O2, N2]
    A= 1000.4
    B = 12.145
    C = 0.11
    Twext = A+B*z+C*z**2
    #################################### Tube Side ##############################################################################################
    mi = m_gas * omega              # Mass flowrate per tube per component [kg/s tube]
    ni = np.divide(mi, MW[:-2])        # Molar flowrate per tube per component [kmol/s tube]
    ntot = np.sum(ni)               # Molar flowrate per tube [kmol/s tube]
    yi = ni / ntot                  # Molar fraction

    Pi = P * yi                     # Partial Pressure
    Ppa = P * 1E5                   # Pressure [Pa]
    Eta = 0.1                       # effectiveness factor (Latham et al., Kumar et al.)

    MWmix = np.sum(yi * MW[:-2]) #Mixture

    # Estimation of physical properties with ideal mixing rules
    RhoGas = (Ppa * MWmix) / (R * T) / 1000  # Gas mass density [kg/m3]
    u = (F_R1 * 1000 / 3600) * R * T / (Aint * Ppa)  # Superficial Gas velocity if the tube was empy (Coke at al. 2007)

    #################################### Furnace Side ##############################################################################################
    m_fi = M_in_furnace * omega_f       # Mass flowrate per tube per component [kg/s]
    ni_f = np.divide(m_fi, MW)          # Molar flowrate per tube per component [kmol/s]
    nf_tot = np.sum(ni_f)                 # Molar flowrate [kmol/s]
    y_fi = ni_f / nf_tot                # Molar fraction
    
    P_fi = Pin_f * y_fi                 # Partial Pressure
    Ppa_f = Pin_f * 1E5                  # Pressure [Pa]
    MWmix_f = np.sum(y_fi * MW) #Mixture
    # Estimation of physical properties with ideal mixing rules
    RhoGas_f = (Ppa_f * MWmix_f) / (R * Tf) / 1000  # Gas mass density [kg/m3]
    u_f = (nf_tot * 1000) * R * Tf / (Aint_f * Ppa_f)  # Superficial Gas velocity if the tube was empy (Coke at al. 2007)

    ############################## Mixture massive Specific Heat calculation (Shomate Equation)    ################### Enthalpy of the reaction at the gas temperature [J/mol]

    #################################### Tube Side ##############################################################################################
        # CH4,          CO,             CO2,            H2,        H2O
    c1 = np.array([-0.7030298, 25.56759, 24.99735, 33.066178, 30.09])
    c2 = np.array([108.4773, 6.096130, 55.18696, -11.363417, 6.832514]) / 1000
    c3 = np.array([-42.52157, 4.054656, -33.69137, 11.432816, 6.793435]) / 1e6
    c4 = np.array([5.862788, -2.671301, 7.948387, -2.772874, -2.534480]) / 1e9
    c5 = np.array([0.678565, 0.131021, -0.136638, -0.158558, 0.082139]) * 1e6
    
    Cp_mol = c1 + c2 * T + c3 * T ** 2 + c4 * T ** 3 + c5 / T ** 2  # Molar specific heat per component [J/molK]
    Cp = Cp_mol / MW[:-2] * 1000  # Mass specific heat per component [J/kgK]
    Cpmix = np.sum(Cp * omega)
    DH_reaction = DHreact * 1000 + np.sum(nu * (c1 * (T - 298) + c2 * (T ** 2 - 298 ** 2) / 2 + c3 * (T ** 3 - 298 ** 3) / 3 + c4 * (T ** 4 - 298 ** 4) / 4 - c5 * (1 / T - 1 / 298)), 1)  # J/mol
    DH_reaction = DH_reaction * 1000  # J/kmol
    #################################### Furnace Side ##############################################################################################
                    # CH4,     CO,       CO2,      H2(1000-25000 K), H2O,  O2(700-2000 K),  N2(500-2000 K)
    c1_f = np.array([-0.7030298, 25.56759, 24.99735, 18.563083, 30.09,  30.03235 , 19.50583])
    c2_f = np.array([108.4773, 6.096130, 55.18696, 12.257357, 6.832514, 8.772972, 19.88705]) / 1000
    c3_f = np.array([-42.52157, 4.054656, -33.69137, -2.859786, 6.793435, -3.988133, -8.598535]) / 1e6
    c4_f = np.array([5.862788, -2.671301, 7.948387, 0.268238, -2.534480, 0.788313, 1.369784]) / 1e9
    c5_f = np.array([0.678565, 0.131021, -0.136638, 1.977990, 0.082139,-0.741599, 0.527601]) * 1e6

    Cp_mol_f = c1_f + c2_f * Tf + c3_f * Tf ** 2 + c4_f * Tf ** 3 + c5_f / Tf ** 2  # Molar specific heat per component [J/molK]
    Cp_f = Cp_mol_f / MW * 1000  # Mass specific heat per component [J/kgK]
    Cpmix_f = np.sum(Cp_f * omega_f)
    DH_reaction_f = DHreact_f * 1000 + np.sum(nu_c * (c1_f * (Tf - 298) + c2_f * (Tf ** 2 - 298 ** 2) / 2 + c3_f * (Tf ** 3 - 298 ** 3) / 3 + c4_f * (Tf ** 4 - 298 ** 4) / 4 - c5_f * (1 / Tf - 1 / 298)), 1)  # J/mol
    DH_combustion = DH_reaction_f * 1000  # J/kmol

    ################################## Heat Transfer Coefficient Calculation ##################################################################################

    # Viscosity coefficients from Yaws
    # CH4,          CO,             CO2,            H2,              H2O,           O2             N2
    A = np.array([3.844, 35.086, 11.336, 27.758, -36.826, 44.224, 42.606])
    B = np.array([4.0112e-1, 5.065e-1, 4.99e-1, 2.12e-1, 4.29e-1, 5.62e-1, 4.75e-1])
    C = np.array([-1.4303e-4, -1.314e-4, -1.0876e-4, -3.28e-5, -1.62e-5, -1.13e-4, -9.88e-5])

    # Thermal conductivity coeffiecients from Yaws
    # CH4,          CO,             CO2,            H2,              H2O,           O2             N2
    a = np.array([-0.00935, 0.0015, -0.01183, 0.03951, 0.00053, 0.00121, 0.00309])
    b = np.array([1.4028e-4, 8.2713e-5, 1.0174e-4, 4.5918e-4, 4.7093e-5, 8.6157e-5, 7.593e-5])
    c = np.array([3.318e-8, -1.9171e-8, -2.2242e-8, -6.4933e-8, 4.9551e-8, -1.3346e-8, -1.1014e-8])

    # Wassiljewa equation for low-pressure gas viscosity with Mason and Saxena modification
    k_i = a + b * T + c * T ** 2  # thermal conductivity of gas [W/m/K]
    Gamma = 210 * (Tc * MW ** 3 / Pc ** 4) ** (1 / 6)  # reduced inverse thermal conductivity [W/mK]^-1
    Tr = T / Tc
    k = (np.exp(0.0464 * Tr) - np.exp(-0.2412 * Tr))

    ########################## Tube Side ################################
    A_matrix = np.identity(n_comp)
    thermal_conductivity_array = np.zeros(n_comp)

    for i in range(0, n_comp - 1):
        for j in range(i + 1, n_comp):
            A_matrix[i, j] = (1 + ((Gamma[j] * k[i]) / (Gamma[i] * k[j])) ** (0.5) * (MW[j] / MW[i]) ** (0.25)) ** 2 / (
                        8 * (1 + MW[i] / MW[j]) ** (0.5))
            A_matrix[j, i] = (1 + ((Gamma[i] * k[j]) / (Gamma[j] * k[i])) ** (0.5) * (MW[i] / MW[j]) ** (0.25)) ** 2 / (
                    8 * (1 + MW[j] / MW[i]) ** (0.5))
            # A_matrix[j,i] = k_i[j]/k_i[i]*MW[i]/MW[j] * A_matrix[i,j]

    for i in range(0, n_comp):
        den = 0
        for j in range(0, n_comp):
            den += yi[j] * A_matrix[j, i]

        num = yi[i] * k_i[i]
        thermal_conductivity_array[i] = num / den
    lambda_gas = sum(thermal_conductivity_array)                                    # Thermal conductivity of the mixture [W/m/K]
    
    # Wilke Method for low-pressure gas viscosity
    mu_i = (A + B * T + C * T ** 2) * 1e-7                                          # viscosity of gas [micropoise]
    PHI = np.identity(n_comp)                                                       # initialization for PHI calculation
    dynamic_viscosity_array = np.zeros(n_comp)
    for i in range(0, n_comp - 1):
        for j in range(i + 1, n_comp):
            PHI[i, j] = (1 + (mu_i[i] / mu_i[j]) ** (0.5) * (MW[j] / MW[i]) ** (0.25)) ** 2 / (
                        8 * (1 + MW[i] / MW[j]) ** (0.5))
            PHI[j, i] = mu_i[j] / mu_i[i] * MW[i] / MW[j] * PHI[i, j]

    for i in range(0, n_comp):
        den = 0
        for j in range(0, n_comp):
            den += yi[j] * PHI[j, i]

        num = yi[i] * mu_i[i]
        dynamic_viscosity_array[i] = num / den

    DynVis = sum(dynamic_viscosity_array)                                                       # Dynamic viscosity [kg/m/s]
    Pr = Cpmix * DynVis / lambda_gas                                                            # Prandtl number
    Re = RhoGas * u * Dp / DynVis                                                               # Reynolds number []
    ############################ Furnace Side ################################
    n_comp_f = 7
    A_matrix_f = np.identity(n_comp_f)
    thermal_conductivity_array_f = np.zeros(n_comp_f)

    for i in range(0, n_comp_f - 1):
        for j in range(i + 1, n_comp_f):
            A_matrix_f[i, j] = (1 + ((Gamma[j] * k[i]) / (Gamma[i] * k[j])) ** (0.5) * (MW[j] / MW[i]) ** (0.25)) ** 2 / (
                        8 * (1 + MW[i] / MW[j]) ** (0.5))
            A_matrix_f[j, i] = (1 + ((Gamma[i] * k[j]) / (Gamma[j] * k[i])) ** (0.5) * (MW[i] / MW[j]) ** (0.25)) ** 2 / (
                    8 * (1 + MW[j] / MW[i]) ** (0.5))
            # A_matrix[j,i] = k_i[j]/k_i[i]*MW[i]/MW[j] * A_matrix[i,j]

    for i in range(0, n_comp_f):
        den = 0
        for j in range(0, n_comp_f):
            den += y_fi[j] * A_matrix_f[j, i]

        num = y_fi[i] * k_i[i]
        thermal_conductivity_array_f[i] = num / den
    lambda_f = sum(thermal_conductivity_array)   # Thermal conductivity of the mixture [W/m/K]     
   
    # Wilke Method for low-pressure gas viscosity
    PHI_f = np.identity(n_comp_f)                                                       # initialization for PHI calculation
    dynamic_viscosity_array_f = np.zeros(n_comp_f)

    for i in range(0, n_comp_f - 1):
        for j in range(i + 1, n_comp_f):
            PHI_f[i, j] = (1 + (mu_i[i] / mu_i[j]) ** (0.5) * (MW[j] / MW[i]) ** (0.25)) ** 2 / (
                        8 * (1 + MW[i] / MW[j]) ** (0.5))
            PHI_f[j, i] = mu_i[j] / mu_i[i] * MW[i] / MW[j] * PHI_f[i, j]

    for i in range(0, n_comp_f):
        den = 0
        for j in range(0, n_comp_f):
            den += y_fi[j] * PHI_f[j, i]

        num = y_fi[i] * mu_i[i]
        dynamic_viscosity_array_f[i] = num / den

    DynVis_f = sum(dynamic_viscosity_array_f)                                                       # Dynamic viscosity [kg/m/s]
   
    Pr_f = Cpmix_f * DynVis_f / lambda_f                                                            # Prandtl number
    Re_f = RhoGas_f * u_f * DHydr / DynVis_f                                                               # Reynolds number []
    print('Re:',Re_f); print('Pr:', Pr_f)

    # h_t = K_gas/Dp*(2.58*Re**(1/3)*Pr**(1/3)+0.094*Re**(0.8)*Pr**(0.4))                       # Convective coefficient tube side [W/m2/K]
    # Overall transfer coefficient in packed beds, Dixon 1996
    # eps = 0.9198 / ((dTube / Dp) ** 2) + 0.3414
    # aw = (1 - 1.5 * (dTube / Dp) ** (-1.5)) * (lambda_gas / Dp) * (Re ** 0.59) * Pr ** (
    #             1 / 3)  # wall thermal transfer coefficient [W/m2/K]
    # ars = 0.8171 * e_w / (2 - e_w) * (T / 1000) ** 3  # [W/m2/K]
    # aru = (0.8171 * (T / 1000) ** 3) / (1 + (eps / (1 - eps)) * (1 - e_w) / e_w)
    # lamba_er_o = Epsilon * (lambda_gas + 0.95 * aru * Dp) + 0.95 * (1 - Epsilon) / (
    #             2 / (3 * lambda_s) + 1 / (10 * lambda_gas + ars * Dp))
    # lambda_er = lamba_er_o + 0.11 * lambda_gas * Re * Pr ** (1 / 3) / (
    #             1 + 46 * (Dp / dTube_out) ** 2)  # effective radial conductivity [W/m/K]
    # Bi = aw * dTube_out / 2 / lambda_er
    # U = 1 / (1 / aw + dTube_out / 6 / lambda_er) * ((Bi + 3) / (Bi + 4))

    h_conv = lambda_f/DHydr*0.023*Re_f**(4/5)*Pr_f**(1/3)
    Fv = 1
    e_g = 0.3758
    Fe = 1/(1/e_g + 1/e_w-1)

    Tavg = (T + Twext) /2
    h_rad = 4*sigma*Fv*Fe*Tavg**3
    hf = h_rad + h_conv
    U = 1/ ( 1/hf + dTube/lambda_t*np.log(dTube_out/dTube))
    h_env = 0.1                                                                 # Convective coefficient external environment [W/m2/K]
    Thick = 0.01                                                                # Tube Thickness [m]

    ##################################################################
    # Kinetic Costant and Rate of reaction (Xu-Froment)
    # CH4 + H2O -> CO + 3 H2 ; CO + H2O -> CO2 + H2 ; CH4 + 2H2O +> CO2 + 4H2

    # Equilibrium constants fitted from Nielsen in [bar]
    gamma = np.array(
        [[-757.323e5, 997.315e3, -28.893e3, 31.29], [-646.231e5, 563.463e3, 3305.75, -3.466]])  # [bar^2], [-]
    Keq1 = np.exp(gamma[0, 0] / (T ** 3) + gamma[0, 1] / (T ** 2) + gamma[0, 2] / T + gamma[0, 3])
    Keq2 = np.exp(gamma[1, 0] / (T ** 3) + gamma[1, 1] / (T ** 2) + gamma[1, 2] / T + gamma[1, 3])
    Keq3 = Keq1 * Keq2

    # Arrhenius     I,       II,        III
    Tr_a = 648  # K
    Tr_b = 823  # K
    k0 = np.array([1.842e-4, 7.558, 2.193e-5])  # pre exponential factor @648 K kmol/bar/kgcat/h
    E_a = np.array([240.1, 67.13, 243.9])  # activation energy [kJ/mol]
    kr = k0 * np.exp(-(E_a * 1000) / R * (1 / T - 1 / Tr_a))

    # Van't Hoff    CO,     H2,         CH4,    H2O
    K0_a = np.array([40.91, 0.0296])  # pre exponential factor @648 K [1/bar]
    DH0_a = np.array([-70.65, -82.90])  # adsorption enthalpy [kJ/mol]
    K0_b = np.array([0.1791, 0.4152])  # pre exponential factor @823 K [1/bar, -]

    DH0_b = np.array([-38.28, 88.68])  # adsorption enthalpy [kJ/mol]

    Kr_a = K0_a * np.exp(-(DH0_a * 1000) / R * (1 / T - 1 / Tr_a))
    Kr_b = K0_b * np.exp(-(DH0_b * 1000) / R * (1 / T - 1 / Tr_b))
    #   CO, H2, CH4, H2O
    Kr = np.concatenate((Kr_a, Kr_b))  # [1/bar] unless last one [-]

    # Components  [CH4, CO, CO2, H2, H2O, O2, N2]
    DEN = 1 + Kr[0] * Pi[1] + Kr[1] * Pi[3] + Kr[2] * Pi[0] + Kr[3] * Pi[4] / Pi[3]
    rj = np.array([(kr[0] / Pi[3] ** (2.5)) * (Pi[0] * Pi[4] - (Pi[3] ** 3) * Pi[1] / Keq1) / DEN ** 2,
                   (kr[1] / Pi[3]) * (Pi[1] * Pi[4] - Pi[3] * Pi[2] / Keq2) / DEN ** 2, (kr[2] / Pi[3] ** (3.5)) * (
                               Pi[0] * (Pi[4] ** 2) - (Pi[3] ** 4) * Pi[2] / Keq3) / DEN ** 2]) * RhoC * (
                     1 - Epsilon)  # kmol/m3/h
    if z<Lf: 
        rj_f = np.array(2*(1-z/Lf)*(nf_tot/(np.pi*((DHydr**2)/4-(dTube_out**2)/4)*Lf)))
    else: 
        rj_f = np.array([0,0,0])

    #####################################################################
    # Tube side Equations
    Reactor1 = Aint / (m_gas * 3600) * MW[0] * np.sum(np.multiply(nu[:, 0], np.multiply(Eta, rj)))
    Reactor2 = Aint / (m_gas * 3600) * MW[1] * np.sum(np.multiply(nu[:, 1], np.multiply(Eta, rj)))
    Reactor3 = Aint / (m_gas * 3600) * MW[2] * np.sum(np.multiply(nu[:, 2], np.multiply(Eta, rj)))
    Reactor4 = Aint / (m_gas * 3600) * MW[3] * np.sum(np.multiply(nu[:, 3], np.multiply(Eta, rj)))
    Reactor5 = Aint / (m_gas * 3600) * MW[4] * np.sum(np.multiply(nu[:, 4], np.multiply(Eta, rj)))

    T6 = - Aint / ((m_gas * 3600) * Cpmix) * np.sum(np.multiply(DH_reaction, np.multiply(Eta, rj))) + (np.pi * dTube / (m_gas * Cpmix)) * U * (Twext - T)

    P7 = ((-150 * (((1 - Epsilon) ** 2) / (Epsilon ** 3)) * DynVis * u / (Dp ** 2) - (1.75 * ((1 - Epsilon) / (Epsilon ** 3)) * m_gas * u / (Dp * Aint)))) / 1e5
    
    # Furnace side Equations 
    Reactor8 =  MW[0]/M_in_furnace*np.pi*(Rf**2 - (dTube_out**2)/4)*np.sum(np.multiply(nu_c[:, 0],rj_f))
    Reactor9 = MW[1]/M_in_furnace*np.pi*(Rf**24 - (dTube_out**2)/4)*np.sum(np.multiply(nu_c[:, 1],rj_f))
    Reactor10 = MW[2]/M_in_furnace*np.pi*(Rf**2 - (dTube_out**2)/4)*np.sum(np.multiply(nu_c[:, 2],rj_f))
    Reactor11 = MW[3]/M_in_furnace*np.pi*(Rf**2 - (dTube_out**2)/4)*np.sum(np.multiply(nu_c[:, 3],rj_f))
    Reactor12 = MW[4]/M_in_furnace*np.pi*(Rf**2 - (dTube_out**2)/4)*np.sum(np.multiply(nu_c[:, 4],rj_f))
    Reactor13 = MW[5]/M_in_furnace*np.pi*(Rf**2 - (dTube_out**2)/4)*np.sum(np.multiply(nu_c[:, 5],rj_f))
    Reactor14 = MW[6]/M_in_furnace*np.pi*(Rf**2 - (dTube_out**2)/4)*np.sum(np.multiply(nu_c[:, 6],rj_f))
    T15 = ( np.pi*dTube*U*(T-Tf) + np.pi*((DHydr**2)/4-(dTube_out**2)/4)*np.sum(np.multiply(DH_combustion, np.multiply(Eta, rj))) ) / (M_in_furnace*Cpmix_f )

    # posso usare fsolve per risolvere l'equazione algebrica per Twext
    #T16 = Twin + U * dTube * np.log(dTube/dTube_out)/lambda_t*(Tf-Twin)
    return np.array([Reactor1, Reactor2, Reactor3, Reactor4, Reactor5, T6, P7, Reactor8, Reactor9, Reactor10, Reactor11, Reactor12, Reactor13, Reactor14, T15])


####################################################################################################################################################
# INPUT DATA FIRST REACTOR

# Components  [CH4, CO, CO2, H2, H2O, O2, N2]
MW = np.array([16.04, 28.01, 44.01, 2.016,
               18.01528, 32.00, 28.01])                    # Molecular Molar weight       [kg/kmol]
Tc = np.array([-82.6, -140.3, 31.2, -240, 374, -118.6, -146.9]) + 273.15  # Critical Temperatures [K]
Pc = np.array([46.5, 35, 73.8, 13, 220.5, 50.76, 34])  # Critical Pressures [bar]

########################################################### Tube Side ####################################################################################
n_comp = 5
nu = np.array([[-1, 1, 0, 3, -1],
               [0, -1, 1, 1, -1],
               [-1, 0, 1, 4, -2]])  # SMR, WGS, reverse methanation

# Reactor Design Tacchino 
Nt = 336                        # Number of tubes
dTube = 0.126                   # Tube diameter [m]
dTube_out = 0.146               # Tube outlet diameter [m]
Length = 12                     # Length of the reactor [m]
Epsilon = 0.607                 # Void Fraction
RhoC = 1100                     # Catalyst density [kg/m3]
Dp = 3.5e-3                     # Catalyst particle diameter [m]
e_w = 0.84                      # emissivity of tube
lambda_s = 0.3489               # thermal conductivity of the solid [W/m/K]
Twin = 1100.40                  # Tube wall temperature [K]

# Input Streams Definition - Tacchino Data reported for one single tube - BASE CASE
fIN_steam = 2.31    #[mol/s]
fIN_NG = 0.64       #[mol/s]
Steam_to_Carbon = fIN_steam/fIN_NG
# Additional Mixing calculation 
MIN_steam = fIN_steam*MW[4]
MIN_NG = fIN_NG*MW[0]
MIN = ( MIN_steam + MIN_NG )/1000       # [kg/s]
f_IN =  10.62/3600                      # input molar flowrate overall (kmol/s)
Tin_R1 = 842.78                         # Inlet Temperature [K]
Pin_R1 = 24.652                         # Inlet Pressure [Bar]

# Components  [CH4, CO, CO2, H2, H2O]
x_in_R1 = np.array([0.22155701, 0.0, 0.01242592, 0.02248117, 0.74353591])  # Inlet molar composition
MWmix = np.sum(x_in_R1 * MW[:-2])
w_in = x_in_R1 * MW[:-2] / MWmix
M_R1 = f_IN * np.sum(np.multiply(x_in_R1, MW[:-2]))                              # Inlet mass flow [kg/s]
f_IN_i = x_in_R1 * f_IN
win_R1 = np.multiply(f_IN_i, MW[:-2]) / M_R1                                     # Inlet mass composition
SC = x_in_R1[4] / x_in_R1[0]

# Thermodynamic Data
R = 8.314  # [J/molK]

# Auxiliary Calculations 
Aint = numpy.pi * dTube ** 2 / 4  # Tube section [m2]
m_R1 = M_R1   
# REACTOR INLET
Mi_R1 = m_R1 * win_R1 * 3600  # Mass flowrate per component [kg/h]
Ni_R1 = np.divide(Mi_R1, MW[:-2])  # Molar flowrate per component [kmol/h]
Ntot_R1 = np.sum(Ni_R1)  # Molar flowrate [kmol/h]
zi_R1 = Ni_R1 / Ntot_R1  # Inlet Molar fraction to separator
MWmix_R1 = np.sum(np.multiply(zi_R1, MW[:-2]))  # Mixture molecular weight
F_R1 = m_R1 / MWmix_R1 * 3600  # Inlet Molar flowrate [kmol/h]
                                                                    
# Perry's data
# CH4,          CO,             CO2,            H2,              H2O
dH_formation_i = np.array([-74.52, -110.53, -393.51, 0, -241.814])  # Enthalpy of formation @298 K  [kJ/mol]
DHreact = np.sum(np.multiply(nu, dH_formation_i), axis=1).transpose()  # Enthalpy of reaction              [kJ/mol]

################################################### Furnace Side ####################################################################################
# Reaction Parameter 
# Furnace Side Components [CH4, CO, CO2,  H2, H2O, O2, N2]
nu_c = np.array([[-1, 0, 1, 0, 2, 2, 0], 
                 [0, 0, 0, -1, 1, 0.5, 0], 
                 [0, -1, 1, 0, 0, 0.5, 0]])
# Perry's data
# CH4,          CO,             CO2,            H2,              H2O, O2, N2
dH_formation_i_f = np.array([-74.52, -110.53, -393.51, 0, -241.814,  0, 0])  # Enthalpy of formation @298 K  [kJ/mol]
DHreact_f = np.sum(np.multiply(nu_c, dH_formation_i_f), axis=1).transpose()  # Enthalpy of reaction              [kJ/mol]
# Heat Balance parameters 
lambda_t = 29.58 # W/m/K
Lf = 6.1          #Flame lenght [m]
sigma = 5.670367*1e-8 
a = 16      # lenght [m]
b = 16      # widht [m]
Rf = 1       # radius of anular jacket surrouding the SMR tube [m]
F_gas_NG = 0.13     # [mol/s]
F_gas_Air = 7.17    # [mol/s]
F_purge_gas = 1.18  # [mol/s]

# Auxiliary calculations
DHydr = 2*a*b/(a+b)
Aint_f = np.pi*(Rf+0.92296)**2          # area di entrata miscela fornace [m2]
# Additional Mixing calculations 
M_gas_Purge = 104.8/3600                                # [kg/s]
M_gas_NG = F_gas_NG*MW[0]/1000                          # [kg/s]
M_gas_Air = F_gas_Air*(0.21*MW[5]+0.79*MW[6])
M_in_furnace = M_gas_Air+ M_gas_NG+ M_gas_Purge         #[kg/s]

# Furnace Side Components [CH4, CO, CO2,  H2, H2O, O2, N2]
win_F = np.array([0.02128141, 0, 0.13696721,0.00493927,0, 0.19491593, 0.64189618 ])
Fi_IN = M_in_furnace*win_F/MW                           # [Kmol/s]
Tin_f = 344.3 + 273.25                                  # K 
Pin_f = 1.2                                             # [bar]
################################################################################

# SOLVER FIRST REACTOR
zspan = np.array([0, Length])
N = 100  # Discretization
z = np.linspace(0, Length, N)
y0_R1 = np.concatenate([win_R1, [Tin_R1], [Pin_R1], win_F, [Tin_f] ])

sol = solve_ivp(TubularandFurnaceReactor, zspan, y0_R1, t_eval=z,
                args=(Epsilon, Dp, m_R1, Aint, MW, nu, R, dTube, Twin, RhoC, DHreact, Tc, Pc, F_R1, e_w, lambda_s,Lf,lambda_t,sigma,DHydr,Pin_f,nu_c,Fi_IN,DHreact_f,Rf,Aint_f, M_in_furnace))

wi_out = np.zeros((5, N))
wi_out = sol.y[0:5]
T_R1 = sol.y[5]
P_R1 = sol.y[6]
wi_f_out = np.zeros((7,N))
wi_f_out = sol.y[7:14]
Tf = sol.y[14]

################################################################################
# REACTOR OUTLET - Tube side
# CH4,          CO,             CO2,            H2,              H2O
Fi_out = np.zeros((n_comp, N))
F_tot_out = np.zeros(N);
yi = np.zeros(N)
Mi_out = m_R1 * wi_out  # Mass flowrate per component [kg/s]
for i in range(0, n_comp):
    Fi_out[i] = Mi_out[i, :] / MW[i]  # Molar flowrate per component [kmol/h]
for j in range(0, N):
    F_tot_out[j] = np.sum(Fi_out[:, j])  # Molar flowrate [kmol/h]
    yi = Fi_out / F_tot_out  # outlet Molar fraction to separator
# MWmix_f1 = np.sum(np.multiply(zi_f1,MW))                                # Mixture molecular weight
# F_f1 = M_R1/MWmix_f1*3600                                                # Outlet Molar flowrate [kmol/h]


################################################################
# POST CALCULATION

################################################################
# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.set_xlabel('Reator Lenght [m]');
ax1.set_ylabel('Tg [K]')
ax1.plot(z, T_R1,z,Tf)

ax2.set_xlabel('Reator Lenght [m]');
ax2.set_ylabel('Molar Fraction')
for i in range(0, n_comp):
    ax2.plot(z, yi[i])
ax2.legend(['CH4', 'C0', 'CO2', 'H2', 'H2O'])

ax3.set_xlabel('Reactor Lenght [m]');
ax3.set_ylabel('P [bar]')
ax3.plot(z, P_R1)

# plt.figure(2)
# plt.xlabel('Reactor Lenght [m]'), plt.ylabel('Mass fraction')
# for i in range(0,n_comp):
#     plt.plot(z,wi_out[i])
# plt.legend(['CH4', 'C0','CO2', 'H2','H2O'])
plt.show()

