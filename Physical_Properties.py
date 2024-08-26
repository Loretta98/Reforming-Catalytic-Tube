import numpy as np

def calculate_DH_reaction(T, MW, omega, DHreact, nu):

    # Shomate Equation coefficients for CH4, CO, CO2, H2, H2O
    c1 = np.array([-0.7030298, 25.56759,  24.99735, 33.066178 , 30.09 ])
    c2 = np.array([108.4773,   6.096130,  55.18696, -11.363417, 6.832514])/1000
    c3 = np.array([-42.52157,  4.054656, -33.69137, 11.432816 ,  6.793435])/1e6
    c4 = np.array([5.862788, -2.671301,   7.948387, -2.772874 ,  -2.534480])/1e9
    c5 = np.array([0.678565,  0.131021,  -0.136638, -0.158558 , 0.082139])*1e6

    # Calculate molar specific heat per component [J/molK]
    Cp_mol = c1 + c2*T + c3*T**2 + c4*T**3 + c5/T**2

    # Calculate mass specific heat per component [J/kgK]
    Cp = Cp_mol/MW*1000

    # Calculate mass specific heat of the mixture [J/kgK]
    Cpmix = np.sum(Cp * omega)

    # Calculate the enthalpy of the reaction at the gas temperature [J/mol]
    DH_reaction = DHreact*1000 + np.sum(nu * (
        c1 * (T - 298) + 
        c2 * (T**2 - 298**2) / 2 + 
        c3 * (T**3 - 298**3) / 3 + 
        c4 * (T**4 - 298**4) / 4 - 
        c5 * (1/T - 1/298)), 1)

    # Convert to J/kmol
    DH_reaction = DH_reaction * 1000

    return DH_reaction,Cpmix


def calculate_heat_transfer_coefficients(T,MW,n_comp,Tc,Pc,yi,Cpmix,RhoGas,u,Dp,dTube,e_w,Epsilon,lambda_s,dTube_out): 

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
    
    h_t = aw
    return U,h_t,h_f,lambda_gas,DynVis

def calculate_diffusivity(T,P,n_comp,yi,MWmix,MW,e_s,tau,R):

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
        den = 0
        for j in range(0,n_comp):
            den += np.sum(yi[j] / Dij[i,j])
        Dmi [i] = (1-yi[i])/ den               # Molecular diffusion coefficient for component with Wilke's Equation
        Dki [i] = 9700 * pore_radius * (T / MW[i]) # Knudsen diffusion [cm2/s]

    pore_diameter = 130 * 1e-8   # pore diameter in cm, average from Xu-Froment II
    Dki = pore_diameter/3*(8*R*T/(MWmix/1e3)/np.pi)**0.5            # Knudsen diffusion [cm2/s]
    Deff = 1 / ( (e_s/tau)* (1/Dmi + 1/Dki))                        # Effective diffusion [cm2/s]
    
    return Deff 

def calculate_effectiveness_factor(kr,Dp,c_h,n_h,s_h,lambda_gas,Deff,p_h):
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

    return Eta
