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
from Input_Data import * 


# Function to generate Chebyshev points
def chebyshev_points(n):
    return np.cos(np.pi * (2 * np.arange(1, n+1) - 1) / (2 * n))

# Function to generate the differentiation matrix for the collocation points
def differentiation_matrix(n, z_points):
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = (-1)**(i+j) / (z_points[i] - z_points[j])
        D[i, i] = -np.sum(D[i, :])
    return D

#####################################################################################################################################
################################################# Tubular reactor resolution ########################################################
#####################################################################################################################################

# Pseudo_homogeneus model for a 1D reformer unit, resolved with an orthogonal collation method 

def ReformerUnit(z,y,D,Epsilon,Dp,Aint,MW,nu,R,dTube,Twin,RhoC,DHreact,Tc,Pc,F_R1,e_w, lambda_s,e_s,tau,p_h,c_h,n_h,s_h,s_w,kw,mesh_size,m_gas,x_f,f_furnace):

    n = len(D)
    xi = y[0:5*n]
    Tw = y[5*n:5*(n+1)]  # Wall temperature
    Tf = y[5*(n+1):5*(n+2)]  # Fluid temperature
    T = y[5*(n+2):5*(n+3)]  # Reactor temperature
    P = y[5*(n+3):5*(n+4)]  # Reactor pressure

    xi_single = np.ones(n_comp)
    for i in range(0,n_comp):
        xi_single[i] = xi[i*(n)]

    f_gas = m_gas / (np.sum(np.multiply(xi_single, MW[:-2])))  # [kmol/s]

    u,omega,MWmix,RhoGas,Pi = Auxiliary_Calc_Tube(xi_single,f_gas,MW,P,m_gas,F_R1,Aint,Epsilon,R,T)
    u_f,omega_f,MWmix_f,RhoGas_f = Auxiliary_Calc_Furnace(x_f,f_furnace,MW,m_gas,F_R1,Aint,Epsilon,R,Tf)

    # Estimation of the necessary parameters
    #T_single = T[0]; P_single = P[0]; Tw_single = Tw[0], Tf_single = Tf[0]
    # Extract single scalar values for temperature and pressure
    T_single = T  # Reactor temperature is a scalar
    P_single = P  # Reactor pressure is a scalar
    Tw_single = Tw  # Wall temperature is a scalar
    Tf_single = Tf  # Fluid temperature is a scalar

    DH_reaction,Cpmix,Cpmix_f = calculate_DH_reaction(T_single, MW, omega, DHreact, nu, omega_f) #J/kmol
    Uint, h_t, lambda_gas, DynVis, lambda_ax, Re, Uext, Ur,lambda_f = calculate_heat_transfer_coefficients(T_single,MW,n_comp,Tc,Pc,xi_single,Cpmix,Cpmix_f,RhoGas,RhoGas_f,u,Dp,dTube,e_w,Epsilon,lambda_s,dTube_out,u_f,D_h,x_f,Tf_single)# J/m2/s/K = W/m2/K
    Deff,Dax = calculate_diffusivity(T_single,P_single,n_comp,xi_single,MWmix,MW,e_s,tau,R,Epsilon,u,Dp,Re)
    rj, kr = calculate_kinetics(T_single, R, Pi,RhoC,Epsilon)
    Eta = calculate_effectiveness_factor(kr,Dp,c_h,n_h,s_h,lambda_gas,Deff,p_h)

    h_env = 0.1                                                                 # Convective coefficient external environment [W/m2/K]
    Thick = 0.01                                                                # Tube Thickness [m]

    Deff_CH4 = Deff[0]*1E-4                                                     # Effective diffusion [m2/s]
    Deff_list.append(Deff_CH4)
    Eta_list.append(Eta)
    h_t_list.append(h_t)
    U_list.append(Uint)
    ############################################### Tube Wall temperature distribution #################################################################################

    # Create a radial mesh and initial guess for temperature distribution
    r_in = dTube/2                                                                  # tube internal radius [m]
    T_f = 850+273.15            # Furnace Temperature
    #T_f = Tw


#################################################################################################################################################
################################################################## Equations#####################################################################
    Dk = np.ones(n_comp)
    for i in range(0,n_comp): 
        Dk[i] = Dax[i]*1e4*Epsilon/u # [m]

    # Mass Balance terms 
    dxdz = D @ xi               # first derivative of molar composition 
    d2xdz2 = D @ dxdz           # second derivative of molar composition 

    x_1 = -dxdz[0] + Aint / (f_gas*3600) * np.sum(np.multiply(nu[:, 0], np.multiply(Eta, rj))) + Dk[0]*d2xdz2[0]  # dxi/dx [molar based]
    x_2 = -dxdz[1] + Aint / (f_gas*3600) * np.sum(np.multiply(nu[:, 1], np.multiply(Eta, rj))) + Dk[1]*d2xdz2[1]
    x_3 = -dxdz[2] + Aint / (f_gas*3600) * np.sum(np.multiply(nu[:, 2], np.multiply(Eta, rj))) + Dk[2]*d2xdz2[2]
    x_4 = -dxdz[3] + Aint / (f_gas*3600) * np.sum(np.multiply(nu[:, 3], np.multiply(Eta, rj))) + Dk[3]*d2xdz2[3]
    x_5 = -dxdz[4] + Aint / (f_gas*3600) * np.sum(np.multiply(nu[:, 4], np.multiply(Eta, rj))) + Dk[4]*d2xdz2[4]

    # Energy Balance terms 
    dTdz =  D @ T 
    dTwdz = D @ Tw
    #dTfdz = D @ Tf

    d2Tdz2 =    D @ dTdz 
    d2Twdz2 =   D @ dTwdz
    #d2Tfdz =    D @ dTfdz

    Temp_w = -d2Twdz2 + 1/(kw*s_w)*( Uint*(Tw-T) + Uext*(Tw-Tf) + Ur*(Tw**4 - Tf**4))
    
    # Furnace is isothermal as a first guess
    #Temp_f = - dTfdz + 1/u_f * (1/RhoGas_f/Cpmix_f)* ( lambda_f*d2Tfdz - Nt*np.pi()*dTube_out/A_f*(Uext*(Tf-Tw) + Ur*(Tf**4-Tw**4)) + Q)
    Temp_f = 0 
    
    term_0 = np.sum(np.multiply(DH_reaction, np.multiply(Eta,rj)))*1e6 # kmol/m3/h * kJ/mol = J/h/m3 1e6 
    term_1 = - Aint/ ((m_gas*3600)*Cpmix) * term_0
    term_2 = (np.pi*dTube/(m_gas*Cpmix))*Uint*(Tw - T)
    term_3 =  Epsilon*lambda_ax
    
    Temp =  -dTdz + 1/RhoGas/Cpmix/u* (d2Tdz2*term_3) + term_2 + term_1

    # Momentum Balance
    dPdz = ( (-150 * (((1-Epsilon)**2)/(Epsilon**3)) * DynVis * u/ (Dp**2) - (1.75* ((1-Epsilon)/(Epsilon**3)) * m_gas*u/(Dp*Aint))  ) ) / 1e5
    Press = -dPdz
    
    dydz = np.array([x_1.flatten(), x_2.flatten(), x_3.flatten(), x_4.flatten(), x_5.flatten(), Temp_w, Temp_f, Temp, Press])
    
    return dydz


######################################################################################################################################################

# SOLVER FIRST REACTOR
zspan = np.array([0,Length])
N = 100                                             # Discretization
z = np.linspace(0,Length,N)

n_collocation_points = 5
# Guess points
yin = np.ones((5,n_collocation_points))
for i in range(0,5): 
    yin[i,:] = x_in_R1[i]
    
Tin_w = np.ones(n_collocation_points)*Tin_w
Tin_R1 = np.ones(n_collocation_points)*Tin_R1
Tin_f = np.ones(n_collocation_points)*Tin_f
Pin_R1 = np.ones(n_collocation_points)*Pin_R1

# Reshape the temperature and pressure arrays to 2D
Tin_w = Tin_w.reshape(1, n_collocation_points)
Tin_f = Tin_f.reshape(1, n_collocation_points)
Tin_R1 = Tin_R1.reshape(1, n_collocation_points)
Pin_R1 = Pin_R1.reshape(1, n_collocation_points)

y0_R1 = np.concatenate([yin, Tin_w, Tin_f, Tin_R1, Pin_R1], axis=0).ravel()
#y0_R1  = np.concatenate([yin, [Tin_w], [Tin_f], [Tin_R1], [Pin_R1]])
z_points = chebyshev_points(n_collocation_points)
D = differentiation_matrix(n_collocation_points, z_points)

sol = solve_ivp(ReformerUnit, zspan, y0_R1, t_eval=z,
                 args=(D, Epsilon, Dp, Aint, MW, nu, R, dTube, Twin, RhoC, DHreact, Tc, Pc, F_R1, e_w, lambda_s,e_s,tau, p_h,c_h,n_h,s_h,s_w,kw,mesh_size,m_R1,x_f,f_furnace))

yi_out = np.zeros( (5,np.size(sol.y[0])) )
yi_out = sol.y[0:5] 
T_W1 = sol.y[5]
T_F1 = sol.y[6]                             
T_R1 = sol.y[7]
P_R1 = sol.y[8]

################################################################################
# # REACTOR OUTLET
# # CH4,          CO,             CO2,            H2,              H2O
# Fi_out = np.zeros((n_comp,np.size(wi_out[0])))
# F_tot_out = np.zeros(np.size(wi_out[0])); yi = np.zeros(np.size(wi_out[0]))
# Mi_out = m_R1 * wi_out                                        # Mass flowrate per component [kg/h]
# for i in range(0,n_comp):
#     Fi_out[i] = Mi_out[i,:]/MW[i]                                     # Molar flowrate per component [kmol/h]
# for j in range(0,np.size(wi_out[0])):
#     F_tot_out[j] = np.sum(Fi_out[:,j])                                              # Molar flowrate [kmol/h]
# yi = Fi_out/F_tot_out                                                           # outlet Molar fraction to separator
# # MWmix_f1 = np.sum(np.multiply(zi_f1,MW))                                # Mixture molecular weight
# #F_f1 = M_R1/MWmix_f1*3600                                                # Outlet Molar flowrate [kmol/h]

################################################################
# # POST CALCULATION
# #Tw = np.array(Tw_list)
# z = np.linspace(0,Length,np.size(wi_out[0]))
# z1 = np.linspace(0,Length,np.size(Tw))
# Tf = np.ones(np.size(z1))*(850+273.15)
# #Tw = Twin + 12.145*z + 0.011*z**2
# #Tw = 150*np.log(2*z+1)+Twin

################################################################
# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.set_xlabel('Reator Lenght [m]'); ax1.set_ylabel('T [K]')
ax1.plot(z,T_R1,z,T_W1,z,T_F1)
ax1.legend(['Tg','Tw','Tf'])

ax2.set_xlabel('Reator Lenght [m]'); ax2.set_ylabel('Molar Fraction')
for i in range(0,n_comp):
    ax2.plot(z, yi_out[i])
ax2.legend(['CH4', 'C0','CO2', 'H2','H2O'])

ax3.set_xlabel('Reactor Lenght [m]'); ax3.set_ylabel('P [bar]')
ax3.plot(z,P_R1)



