from Reformer_model_Alberton_approach import* 
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt

################################################################
# POST CALCULATION
#Tw = np.ones(np.size(wi_out[0]))*Twin
z = np.linspace(0,Length,np.size(wi_out[0]))
#Tw = Twin + 12.145*z + 0.011*z**2
#Tw = 150*np.log(2*z+1)+Twin
Tw = np.ones(np.size(z))*Twin
################################################################
# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.set_xlabel('Reator Lenght [m]'); ax1.set_ylabel('T [K]')
ax1.legend(['Tg','Tf'])
ax1.plot(z,T_R1,z,Tw)

# ax2.set_xlabel('Reator Lenght [m]'); ax2.set_ylabel('Molar Fraction')
# for i in range(0,n_comp):
#     ax2.plot(z, yi[i])
# ax2.legend(['CH4', 'C0','CO2', 'H2','H2O'])

ax2.set_xlabel('Reator Lenght [m]'); ax2.set_ylabel('Molar Fraction')
for i in range(0,n_comp-1):
    ax2.plot(z, yi_[i])
ax2.legend(['CH4', 'C0','CO2', 'H2'])

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
plt.show()