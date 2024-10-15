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
import matplotlib.pyplot as plt

def solve_tube_wall_temperature(T_gas, Tf, r_in, r_out, h_c, h_f, k_w, sigma, e_w, e_f):
    def diff_eqs(r, Tw):
        dTw_dr = Tw[1]
        d2Tw_dr2 = -Tw[1] / r
        return np.vstack((dTw_dr, d2Tw_dr2))

    # def boundary_conditions(Tw_a, Tw_b):
    #     dTw_dr_in = -1/k_w * (h_c * (Tw_a[0] - T_gas) + e_w * sigma * Tw_a[0]**4)
    #     dTw_dr_out = -1/k_w * (h_f * (Tf - Tw_b[0]) + e_f * sigma * Tf**4 - e_w * sigma * Tw_b[0]**4)
    #     return np.array([Tw_a[1] - dTw_dr_in, Tw_b[1] - dTw_dr_out])

    def boundary_conditions(Tw_a, Tw_b):
        # Inner boundary (at r_in)
        Q_in = h_c * (Tw_a[0] - T_gas) + e_w * sigma * Tw_a[0]**4
        dTw_dr_in = -Q_in / k_w
        #print(f"Inner boundary heat flux: {Q_in}")
        geometrical = 2**3/(2*np.pi*r_in)
        # Outer boundary (at r_out)
        Q_out = h_f * (Tf - Tw_b[0]) + (e_f + 0.58*0.6*geometrical) * sigma * Tf**4 - e_w * sigma * Tw_b[0]**4
        dTw_dr_out = -Q_out / k_w
        #print(f"Outer boundary heat flux: {Q_out}")

        return np.array([Tw_a[1] - dTw_dr_in, Tw_b[1] - dTw_dr_out])

    r = np.linspace(r_in, r_out, 100)
    initial_temperature_guess = T_gas + ((Tf-200) - T_gas) * (r - r_in) / (r_out - r_in)
    #initial_gradient_guess = np.gradient(initial_temperature_guess, r))
    # Initial guess for the temperature gradient
    initial_gradient_guess = np.zeros_like(r)  # Start with zero gradient (can refine if necessary)
    initial_guess = np.vstack((initial_temperature_guess, initial_gradient_guess))

    solution = solve_bvp(diff_eqs, boundary_conditions, r, initial_guess)
    #plt.plot(r,solution.y[0])
    #plt.show()
    if not solution.success:
        raise RuntimeError("Tube wall temperature BVP did not converge")

    return solution.y[0]  # Return the tube wall temperature profile

def TubularReactor(z,y,Epsilon,Dp,m_gas,Aint,MW,nu,R,dTube,Twin,RhoC,DHreact,Tc,Pc,F_R1,e_w, lambda_s,e_s,tau,p_h,c_h,n_h,s_h):

    omega = y[0:5]
    T =     y[5]
    P =     y[6]
    r_in = dTube/2
    r_out = dTube_out/2

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

    #u_f,omega_f,MWmix_f,RhoGas_f = Auxiliary_Calc_Furnace(x_f,f_furnace,MW_f,m_gas,F_R1,A_f,Epsilon,R,Tf)

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

    # Shomate Equation coefficients for CH4, CO, CO2, H2, H2O, O2, N2
    c1 = np.array([-0.7030298, 25.56759,  24.99735, 33.066178 , 30.09, 30.03235, 19.50583 ])
    c2 = np.array([108.4773,   6.096130,  55.18696, -11.363417, 6.832514, 8.72972,19.88705])/1000
    c3 = np.array([-42.52157,  4.054656, -33.69137, 11.432816 ,  6.793435, -3.988133, -8.59853])/1e6
    c4 = np.array([5.862788, -2.671301,   7.948387, -2.772874 ,  -2.534480, 0.788313, 1.369784])/1e9
    c5 = np.array([0.678565,  0.131021,  -0.136638, -0.158558 , 0.082139, -0.74159, 0.527601])*1e6

    # Cp_mol_f = c1+c2*Tf+c3*Tf**2+c4*Tf**3+c5/Tf**2                        # Molar specific heat per component [J/molK]
    # Cp_f = Cp_mol_f/MW_f*1000                                             # Mass specific heat per component [J/kgK]
    # Cpmix_f = np.sum(Cp_f*omega_f)                                        # Mass specific heat [J/kgK]
    # yi_f = np.array([0, 0.06, 0.11, 0, 0.13, 0, 0.7])  # composition of the exhaust gas
    U,lambda_gas,DynVis,h_t = HeatTransfer(T,Tc,n_comp, MW, Pc, yi, Cpmix, RhoGas,dTube, Dp, Epsilon, e_w, u, dTube_out, lambda_s,D_h)
    #h_f = furnaceTransfer(Tf,Tc_f,n_comp, MW_f, Pc_f, yi_f, Cpmix_f , RhoGas_f, u_f ,D_h)
    rj,kr = Kinetics(T,R, kr_list, Pi, RhoC, Epsilon)
    Deff = Diffusivity(R,T,P,yi, n_comp, MW, MWmix, e_s, tau)
    Eta = EffectivenessF(p_h,c_h,n_h,s_h,Dp,lambda_gas,Deff,kr)

    Deff_CH4 = Deff[0]*1E-4             # Effective diffusion [m2/s]
    Deff_list.append(Deff_CH4)
    Eta_list.append(Eta)

    h_f = h_t
    # Solve the tube wall temperature profile
    Tw_profile = solve_tube_wall_temperature(T, Tf, r_in, r_out, h_t, h_f, k_w, sigma, e_w, e_f)

    # Use the tube wall temperature from the profile in the model equations
    Twin = Tw_profile[0]  # Taking an average temperature for simplicity, adjust as needed
    Twout = Tw_profile[-1]
    Tw_list.append(Twout)

#####################################################################
# Equations
    Reactor1 = Aint / (m_gas*3600) * MW[0] * np.sum(np.multiply(nu[:, 0], np.multiply(Eta, rj)))
    Reactor2 = Aint / (m_gas*3600) * MW[1] * np.sum(np.multiply(nu[:, 1], np.multiply(Eta, rj)))
    Reactor3 = Aint / (m_gas*3600) * MW[2] * np.sum(np.multiply(nu[:, 2], np.multiply(Eta, rj)))
    Reactor4 = Aint / (m_gas*3600) * MW[3] * np.sum(np.multiply(nu[:, 3], np.multiply(Eta, rj)))
    Reactor5 = Aint / (m_gas*3600) * MW[4] * np.sum(np.multiply(nu[:, 4], np.multiply(Eta, rj)))

    term_0 = np.sum(np.multiply(DH_reaction, np.multiply(Eta,rj)))
    term_1 = - Aint/ ((m_gas*3600)*Cpmix) * term_0
    term_2 = (np.pi*dTube/(m_gas*Cpmix))*U*(Twin - T)
    Reactor6 =  term_1 + term_2

    Reactor7 = ( (-150 * (((1-Epsilon)**2)/(Epsilon**3)) * DynVis * u/ (Dp**2) - (1.75* ((1-Epsilon)/(Epsilon**3)) * m_gas*u/(Dp*Aint))  ) ) / 1e5

    return np.array([Reactor1, Reactor2, Reactor3, Reactor4, Reactor5, Reactor6, Reactor7])


#######################################################################

# SOLVER FIRST REACTOR

zspan = np.array([0,Length])
N = 13                                             # Discretization
z = np.linspace(0,Length,N)
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