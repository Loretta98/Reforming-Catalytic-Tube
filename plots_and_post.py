import numpy as np 
import matplotlib.pyplot as plt 
from Input_Data import * 
from Reformer_Tube import wi_out, T_R1, P_R1

################################################################################
# REACTOR OUTLET
# CH4,          CO,             CO2,            H2,              H2O
Fi_out = np.zeros((n_comp,np.size(wi_out[0])))
F_tot_out = np.zeros(np.size(wi_out[0])); yi = np.zeros(np.size(wi_out[0]))
Mi_out = m_R1 * wi_out                                        # Mass flowrate per component [kg/h]
for i in range(0,n_comp):
    Fi_out[i] = Mi_out[i,:]/MW[i]                                     # Molar flowrate per component [kmol/h]
for j in range(0,np.size(wi_out[0])):
    F_tot_out[j] = np.sum(Fi_out[:,j])                                              # Molar flowrate [kmol/h]
yi = Fi_out/F_tot_out                                                           # outlet Molar fraction to separator
# MWmix_f1 = np.sum(np.multiply(zi_f1,MW))                                # Mixture molecular weight
#F_f1 = M_R1/MWmix_f1*3600                                                # Outlet Molar flowrate [kmol/h]

################################################################
# POST CALCULATION
Tw = np.array(Tw_list)
z = np.linspace(0,Length,np.size(wi_out[0]))
z1 = np.linspace(0,Length,np.size(Tw))
Tf = np.ones(np.size(z1))*(850+273.15)
#Tw = Twin + 12.145*z + 0.011*z**2
#Tw = 150*np.log(2*z+1)+Twin

################################################################
# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.set_xlabel('Reator Lenght [m]'); ax1.set_ylabel('T [K]')
ax1.plot(z,T_R1,z1,Tw,z1,Tf)
ax1.legend(['Tg','Tw','Tf'])

ax2.set_xlabel('Reator Lenght [m]'); ax2.set_ylabel('Molar Fraction')
for i in range(0,n_comp):
    ax2.plot(z, yi[i])
ax2.legend(['CH4', 'C0','CO2', 'H2','H2O'])

ax3.set_xlabel('Reactor Lenght [m]'); ax3.set_ylabel('P [bar]')
ax3.plot(z,P_R1)
# plt.figure()
# Eta_list = np.array(Eta_list)
# z1 = np.linspace(0,Length,np.size(Eta_list[:,0]))
# plt.xlabel('Reactor Lenght [m]'), plt.ylabel('diffusion efficiency')
# plt.plot(z1,Eta_list[:,0],label=r'$\eta1$');
# plt.plot(z1,Eta_list[:,1],label=r'$\eta2$');
# plt.plot(z1,Eta_list[:,2],label=r'$\eta3$')
# plt.legend()

# plt.figure()
# Deff_list = np.array(Deff_list)
# z1 = np.linspace(0,Length,np.size(Deff_list))
# plt.xlabel('Reactor Lenght [m]'), plt.ylabel('diffusion coefficient [m2/s]')
# plt.plot(z1,Deff_list,label='CH4')
# plt.legend()

# plt.figure()
# kr_list = np.array(kr_list)
# z1 = np.linspace(0,Length,np.size(kr_list[:,0]))
# plt.xlabel('Reactor Lenght [m]'), plt.ylabel('Rate of equation')
# plt.plot(z1,kr_list[:,0],label='kr1'); plt.plot(z1,kr_list[:,1],label='kr2'); plt.plot(z1,kr_list[:,2],label='kr3')
# plt.legend()

# plt.figure()
# h_t = np.array(h_t_list)
# z1 = np.linspace(0,Length,np.size(h_t_list))
# U = np.array(U_list)
# z2 = np.linspace(0,Length,np.size(U_list))
# plt.xlabel('Reactor Lenght [m]'), plt.ylabel('tube thermal conductivity [W/m/K]')
# plt.plot(z1,h_t,z2,U)
# plt.legend('h_t','U')

# plt.show()