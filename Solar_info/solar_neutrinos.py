import numpy as np
import matplotlib.pyplot as plt
#import pymultinest 
from scipy.special import gammaln, erfinv
from scipy.integrate import simps

# Constants ----------------------------------------------------------------
pi = np.pi
i = 1j  # sqrt(-1)

# Fermi constant and unit conversion ---------------------------------------
Gf = 1.1663787e-11  # MeV^-2
cm_to_Mev_inv = 5.076142131e10  # cm^-1 to MeV
Mn = 1.6749e-24  # g (mass of neutron)

# Oscillation parameters ---------------------------------------------------

# Mixing angles ------------------------------------------------------------
theta_12 = 33.45 * 2 * pi / 360  # rad
theta_23 = 49.2 * 2 * pi / 360   # rad
theta_13 = 8.57 * 2 * pi / 360   # rad
delta_cp = 3 * pi / 2  # CP violation angle

c12 = np.cos(theta_12)
s12 = np.sin(theta_12)
c13 = np.cos(theta_13)
s13 = np.sin(theta_13)
c23 = np.cos(theta_23)
s23 = np.sin(theta_23)

# PMNS Matrix --------------------------------------------------------------
U_23 = np.array([[1, 0, 0],
                 [0, c23, s23],
                 [0, -s23, c23]])

U_13 = np.array([[c13, 0, s13 * np.exp(-i * delta_cp)],
                 [0, 1, 0],
                 [-s13 * np.exp(i * delta_cp), 0, c13]])

U_12 = np.array([[c12, s12, 0],
                 [-s12, c12, 0],
                 [0, 0, 1]])

U = U_23 @ U_13 @ U_12  # Matrix multiplication
U_conj = np.conj(U)     # Complex conjugate
U_adj = np.transpose(U_conj)  # Adjoint 

# Mass squared difference----------------------------------------------------
delta_msq_21 = 7.5*10**(-5)*(10**(-6))**2; #MeV^2
delta_msq_31 = 2.55*10**(-3)*(10**(-6))**2; #MeV^2

M_sq = [[0,0,0],[0,delta_msq_21,0],[0,0,delta_msq_31]]

# Solar data----------------------------------------------------------------

flux_norm = np.genfromtxt("fluxes_b16.dat",skip_header=9)
fl_pp = np.genfromtxt("flux_pp")
fl_Be71 = np.genfromtxt("flux_Be71")
fl_Be7 = np.genfromtxt("flux_Be7")
fl_O = np.genfromtxt("flux_O")
fl_N = np.genfromtxt("flux_N")
fl_F = np.genfromtxt("flux_F")
fl_B8 = np.genfromtxt("flux_B8")
fl_pep = np.genfromtxt("flux_pep")
fl_hep = np.genfromtxt("flux_hep")

# 0 :pp ; 1:pep ; 2:hep ; 3:7Be ; 4:8B ; 5:13N ; 6:15O ; 7:17F

flux_pp = flux_norm[0,0]*flux_norm[0,4]*fl_pp[:,1]
E_pp = fl_pp[:,0]
flux_Be71 = flux_norm[3,0]*flux_norm[3,4]*fl_Be71[:,1]
E_Be71 = fl_Be71[:,0]
flux_Be7 = flux_norm[3,0]*flux_norm[3,4]*fl_Be7[:,1]
E_Be7 = fl_Be7[:,0]
flux_pep = flux_norm[1,0]*flux_norm[1,4]*fl_pep[:,1]
E_pep = fl_pep[:,0]
flux_N = flux_norm[5,0]*flux_norm[5,4]*fl_N[:,1]
E_N = fl_N[:,0]
flux_O = flux_norm[6,0]*flux_norm[6,4]*fl_O[:,1]
E_O = fl_O[:,0]
flux_B8 = flux_norm[4,0]*flux_norm[4,4]*fl_B8[:,1]
E_B8 = fl_B8[:,0]
flux_F = flux_norm[7,0]*flux_norm[7,4]*fl_F[:,1]
E_F = fl_F[:,0]
flux_hep = flux_norm[2,0]*flux_norm[2,4]*fl_hep[:,1]
E_hep = fl_hep[:,0]

plt.plot(E_pp,flux_pp,label='pp')
plt.plot(E_Be71,flux_Be71,label='Be71')
plt.plot(E_Be7,flux_Be7,label='Be7')
plt.plot(E_pep,flux_pep,label='pep')
plt.plot(E_N,flux_N,label='N')
plt.plot(E_O,flux_O,label='O')
plt.plot(E_F,flux_F,label='F')
plt.plot(E_B8,flux_B8,label='B8')
plt.plot(E_hep,flux_hep,label='hep')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.ylim([1e-1,1e14])
plt.xlim([1e-1,2*1e1])
plt.ylabel('flux cm^(-2)s^(-1)')
plt.xlabel('E (MeV)')
plt.show()

## Density-------------------------------

solar_info = np.genfromtxt("nudistr_b16_gs98.dat",skip_header=22)

R = solar_info[:,0]
den = solar_info[:,2] # Logarithm (to the base 10) of the electron density in units of
#   cm^{-3}/N_A,
frac_pp = solar_info[:,4]
frac_pep = solar_info[:,5]
frac_hep = solar_info[:,6]
frac_Be7 = solar_info[:,7]
frac_B8 = solar_info[:,8]
frac_N = solar_info[:,9]
frac_O = solar_info[:,10]
frac_F = solar_info[:,11]

NA=6.022*(10**23)
n_e = (10**den)*NA

solar_comp = np.genfromtxt("struct_b16_gs98.dat")

Rn = solar_comp[:,1]
H1 = solar_comp[:,6]
He4 = solar_comp[:,7]
Y_n = He4/(2*H1 + He4)
Rho = solar_comp[:,3]

#interpolation------------------------------------------
N_e = np.interp(Rn,R,n_e) #from simulation
Ne = Rho*(1-Y_n)/Mn #using rho and Mn

frac_pp = np.interp(Rn,R,frac_pp)
frac_pep = np.interp(Rn,R,frac_pep)
frac_hep = np.interp(Rn,R,frac_hep)
frac_Be7 = np.interp(Rn,R,frac_Be7)
frac_B8 = np.interp(Rn,R,frac_B8)
frac_N = np.interp(Rn,R,frac_N)
frac_O = np.interp(Rn,R,frac_O)
frac_F = np.interp(Rn,R,frac_F)
