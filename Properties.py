import numpy as np 


def Kinetics(T, R, kr_list, Pi, RhoC, Epsilon): 
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
    
    return rj,kr


def furnaceTransfer(Tf,Tc_f,n_comp, MW_f, Pc_f, yi, Cpmix_f , RhoGas, u_f ,D_h): 
    ######################################################################################################
    ################################## Heat Transfer Coefficient Calculation #############################
    ######################################################################################################
    n_comp = 7
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
    k_i     = a + b*Tf + c*Tf**2                                                                      # thermal conductivity of gas [W/m/K]
    A_matrix = np.identity(n_comp)
    thermal_conductivity_array = np.zeros(n_comp)
    
    Gamma = 210*(Tc_f*MW_f**3/Pc_f**4)**(1/6)     # reduced inverse thermal conductivity [W/mK]^-1
    Tr = Tf/Tc_f
    k = (np.exp(0.0464*Tr)-np.exp(-0.2412*Tr))
    
    for i in range(0,n_comp-1):
        for j in range(i+1,n_comp): 
            A_matrix[i,j] = ( 1+ ((Gamma[j]*k[i])/(Gamma[i]*k[j]))**(0.5) * (MW_f[j]/MW_f[i])**(0.25) )**2 / ( 8*(1 + MW_f[i]/MW_f[j])**(0.5) )
            A_matrix[j,i] = (1 + ((Gamma[i] * k[j]) / (Gamma[j] * k[i])) ** (0.5) * (MW_f[i] / MW_f[j]) ** (0.25)) ** 2 / (
                        8 * (1 + MW_f[j] / MW_f[i]) ** (0.5))
            #A_matrix[j,i] = k_i[j]/k_i[i]*MW[i]/MW[j] * A_matrix[i,j]

    for i in range(0,n_comp):
        den = 0
        for j in range(0,n_comp): 
            den += yi[j]*A_matrix[j,i]

        num = yi[i]*k_i[i]
        thermal_conductivity_array[i] = num/den 
    lambda_f   = sum(thermal_conductivity_array)                                                        # Thermal conductivity of the mixture [W/m/K]
    # K_gas = 0.9
    # Wilke Method for low-pressure gas viscosity   
    mu_i    = (A + B*Tf + C*Tf**2)*1e-7                                                               # viscosity of gas [micropoise]
    PHI = np.identity(n_comp)                                                                       # initialization for PHI calculation
    dynamic_viscosity_array = np.zeros(n_comp)

    for i in range(0,n_comp-1):
        for j in range(i+1,n_comp): 
            PHI[i,j] = ( 1+ (mu_i[i]/mu_i[j])**(0.5) * (MW_f[j]/MW_f[i])**(0.25) )**2 / ( 8*(1 + MW_f[i]/MW_f[j])**(0.5) )
            PHI[j,i] = mu_i[j]/mu_i[i]*MW_f[i]/MW_f[j] * PHI[i,j]

    for i in range(0,n_comp):
        den = 0
        for j in range(0,n_comp): 
            den += yi[j]*PHI[j,i]

        num = yi[i]*mu_i[i]
        dynamic_viscosity_array[i] = num/den 
    
    DynVis  = sum(dynamic_viscosity_array)                                                  # Dynamic viscosity [kg/m/s]

    Pr_f = Cpmix_f*DynVis/lambda_f                                                            # Prandtl number
    Re_f = RhoGas * u_f * D_h / DynVis                                                        # Reynolds number []

    #h_t = K_gas/Dp*(2.58*Re**(1/3)*Pr**(1/3)+0.094*Re**(0.8)*Pr**(0.4))                     # Convective coefficient tube side [W/m2/K]
    h_f = lambda_f/D_h*(0.023*(Re_f)**0.8*Pr_f**0.3)                                                     # Convenctive coefficient furnace side [W/m2/K] (Dittus-Boeleter's equation)
    #h_f = 1000 # Convective coefficient furnace side fixed [W/m2/K]
    
    return h_f 

def Auxiliary_Calc_Furnace(x_f,f_furnace,MW,m_gas,F_R1,Aint,Epsilon,R,T_F):  
    ########## Aux. Calculations for the Tube 
    ni_f = f_furnace/3600*x_f                                       # Molar flowrate per component [kg/s]
    mi_f = np.multiply(x_f,MW)                                      # Mass flowrate per component [kmol/s]
    ntot = np.sum(ni_f)                                             # Molar flowrate per tube [kmol/s]
    omega_f = mi_f / np.sum(mi_f)
    Ppa = 1.01325 * 1E5                                             # Pressure [Pa]
    MWmix_f = np.sum(x_f*MW)                                        # Mixture
    # Estimation of physical properties with ideal mixing rules
    RhoGas_f = (Ppa*MWmix_f) / (R*T_F)  / 1000                                        # Gas mass density [kg/m3]
    VolFlow_R1 = m_gas / RhoGas_f                                                     # Volumetric flow per tube [m3/s]
    u_f = (F_R1*1000) * R * T_F / (Aint*Ppa)                                        # Superficial Gas velocity if the tube was empy (Coke at al. 2007)
    u_f = VolFlow_R1 / (Aint)                                               # Gas velocity in the tube [m/s]
    return u_f,omega_f,MWmix_f,RhoGas_f

def HeatTransfer(T,Tc,n_comp, MW, Pc, yi, Cpmix, RhoGas,dTube, Dp, Epsilon, e_w, u, dTube_out, lambda_s,D_h): 
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
    Re = RhoGas * u * Dp / DynVis                                                        # Reynolds number []

    #h_t = K_gas/Dp*(2.58*Re**(1/3)*Pr**(1/3)+0.094*Re**(0.8)*Pr**(0.4))                     # Convective coefficient tube side [W/m2/K]

    
    # Overall transfer coefficient in packed beds, Dixon 1996 
    eps = 0.9198/((dTube/Dp)**2) + 0.3414
    aw = (1-1.5*(dTube/Dp)**(-1.5))*(lambda_gas/Dp)*(Re**0.59)*Pr**(1/3)                    # wall thermal transfer coefficient [W/m2/K]
    ars = 0.8171*e_w/(2-e_w)*(T/1000)**3                                                    # [W/m2/K]
    aru = (0.8171*(T/1000)**3) / (1+(eps/(1-eps))*(1-e_w)/e_w)
    lamba_er_o = Epsilon*(lambda_gas+0.95*aru*Dp)+0.95*(1-Epsilon)/(2/(3*lambda_s)+1/(10*lambda_gas+ars*Dp))
    lambda_er = lamba_er_o+0.11*lambda_gas*Re*Pr**(1/3)/(1+46*(Dp/dTube_out)**2)            # effective radial conductivity [W/m/K]
    Bi = aw*dTube_out/2/lambda_er
    
    h_t = aw
    U = 1 / ( 1/aw + dTube_out/6/lambda_er)*((Bi+3)/(Bi+4))                                 # J/m2/s/K = W/m2/K
 
    return U,lambda_gas,DynVis,h_t

def Diffusivity(R,T,P,yi, n_comp, MW, MWmix, e_s, tau): 
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
    den = 0
    for i in range(0,n_comp):
        for j in range(0,n_comp):
            den += np.sum(yi[j] / Dij[i,j])
        Dmi [i] = (1-yi[i])/ den               # Molecular diffusion coefficient for component with Wilke's Equation
        Dki [i] = 9700 * pore_radius * (T / MW[i]) # Knudsen diffusion [cm2/s]

    pore_diameter = 130 * 1e-8   # pore diameter in cm, average from Xu-Froment II
    Dki = pore_diameter/3*(8*R*T/(MWmix/1e3)/np.pi)**0.5            # Knudsen diffusion [cm2/s]
    Deff = 1 / ( (e_s/tau)* (1/Dmi + 1/Dki))                        # Effective diffusion [cm2/s]
    return Deff

def EffectivenessF(p_h,c_h,n_h,s_h,Dp,lambda_gas,Deff,kr): 
    ##### Effectiveness factors calculation from CFD models, based on the work of Alberton, 2009 #########
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
    return Eta 