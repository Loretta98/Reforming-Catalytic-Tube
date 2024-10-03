# This is a sample Python script.
import numpy 
#  QUESTO CODICE CONTIENE MODELLO COMPLETO DI 1 REATTORE DI SMR 
#  CINETICA Xu-Froment 

import numpy as np
from scipy.integrate import solve_ivp, quad, solve_bvp
from scipy.optimize import fsolve
import sympy as sp
from Input import*
from Properties import*


def TubularReactor(z,y,Epsilon,Dp,m_gas,Aint,MW,nu,R,dTube,Twin,RhoC,DHreact,Tc,Pc,F_R1,e_w, lambda_s,e_s,tau,p_h,c_h,n_h,s_h):

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

    U,lambda_gas,DynVis = HeatTransfer(T,Tc,n_comp, MW, Pc, yi, Cpmix, RhoGas,dTube, Dp, Epsilon, e_w, u, dTube_out, lambda_s)
    rj,kr = Kinetics(T,R, kr_list, Pi, RhoC, Epsilon)
    Deff = Diffusivity(R,T,P,yi, n_comp, MW, MWmix, e_s, tau)
    Eta = EffectivenessF(p_h,c_h,n_h,s_h,Dp,lambda_gas,Deff,kr)

    Deff_CH4 = Deff[0]*1E-4             # Effective diffusion [m2/s]
    Deff_list.append(Deff_CH4)
    Eta_list.append(Eta)

#####################################################################
# Equations
    Reactor1 = Aint / (m_gas*3600) * MW[0] * np.sum(np.multiply(nu[:, 0], np.multiply(Eta, rj)))
    Reactor2 = Aint / (m_gas*3600) * MW[1] * np.sum(np.multiply(nu[:, 1], np.multiply(Eta, rj)))
    Reactor3 = Aint / (m_gas*3600) * MW[2] * np.sum(np.multiply(nu[:, 2], np.multiply(Eta, rj)))
    Reactor4 = Aint / (m_gas*3600) * MW[3] * np.sum(np.multiply(nu[:, 3], np.multiply(Eta, rj)))
    Reactor5 = Aint / (m_gas*3600) * MW[4] * np.sum(np.multiply(nu[:, 4], np.multiply(Eta, rj)))

    term_0 = np.sum(np.multiply(DH_reaction, np.multiply(Eta,rj)))
    term_1 = - Aint/ ((m_gas*3600)*Cpmix) * term_0
    term_2 = (np.pi*dTube/(m_gas*Cpmix))*U*(Tw - T)
    Reactor6 =  term_1 + term_2

    Reactor7 = ( (-150 * (((1-Epsilon)**2)/(Epsilon**3)) * DynVis * u/ (Dp**2) - (1.75* ((1-Epsilon)/(Epsilon**3)) * m_gas*u/(Dp*Aint))  ) ) / 1e5
    
    return np.array([Reactor1, Reactor2, Reactor3, Reactor4, Reactor5, Reactor6, Reactor7])


#######################################################################

# SOLVER FIRST REACTOR

zspan = np.array([0,Length])
N = 100                                             # Discretization
z = np.linspace(0,Length,N)
# Tw = 1000.4 + 12.145*z + 0.011*z**2
Tw = 9500+273.15 # K
y0_R1  = np.concatenate([omegain_R1, [Tin_R1], [Pin_R1]])

sol = solve_ivp(TubularReactor, zspan, y0_R1, t_eval=z,
                 args=(Epsilon, Dp, m_R1, Aint, MW, nu, R, dTube, Twin, RhoC, DHreact, Tc, Pc, F_R1, e_w, lambda_s,e_s,tau, p_h,c_h,n_h,s_h))

wi_out = np.zeros( (5,np.size(sol.y[0])) )
wi_out = sol.y[0:5]                              
T_R1 = sol.y[5]
P_R1 = sol.y[6]


################################################################################
# REACTOR OUTLET
# CH4,          CO,             CO2,            H2,              H2O
Fi_out = np.zeros((n_comp,np.size(wi_out[0])))
F_tot_out = np.zeros(np.size(wi_out[0])); yi = np.zeros(np.size(wi_out[0]))
print('m_R1 = ',m_R1*3600*Nt)
Mi_out = m_R1 * wi_out                                     # Mass flowrate per component [kg/s]
print('composizione', np.sum(wi_out[:,-1]))
Mout = np.sum(Mi_out)*3600*Nt
print('Mout = ', Mout)
print('Total Water at the outlet', Mi_out[4,-1]*3600*Nt)
for i in range(0,n_comp):
    Fi_out[i] = Mi_out[i,:]/MW[i]                                     # Molar flowrate per component [kmol/h]
for j in range(0,np.size(wi_out[0])):
    F_tot_out[j] = np.sum(Fi_out[:,j])                                              # Molar flowrate [kmol/h]
yi = Fi_out/F_tot_out                                                           # outlet Molar fraction to separator
# MWmix_f1 = np.sum(np.multiply(zi_f1,MW))                                # Mixture molecular weight
#F_f1 = M_R1/MWmix_f1*3600                                                # Outlet Molar flowrate [kmol/h]
print('Outlet Wet composition', yi[:,-1])
Fi_out_ = np.zeros((n_comp-1,np.size(wi_out[0])))
for i in range(0,n_comp-1):
    Fi_out_[i] = Mi_out[i,:]/MW[i]                                     # Molar flowrate per component [kmol/h]
for j in range(0,np.size(wi_out[0])):
    F_tot_out[j] = np.sum(Fi_out_[:,j])                                              # Molar flowrate [kmol/h]
yi_ = Fi_out_ / F_tot_out
print('Outlet Dry composition', yi_[:,-1])