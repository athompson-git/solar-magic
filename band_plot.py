import numpy as np
import matplotlib.pyplot as plt

def plot_shaded_area(x, y1, y2, color='gray', alpha=0.3, label=None):
    plt.fill_between(x, y1, y2, color=color, alpha=alpha, label=label)

def Pee_pp_day(sin2theta12):
    return 1 - 2 * sin2theta12 * (1 - sin2theta12)

def Pee_b8_day(sin2theta12):
    return sin2theta12

def Flux_obs2tot(flux_obs, Pee, int_type):
    if int_type == 'ES':
        return flux_obs / (Pee + 1/6  * (1 - Pee))
    elif int_type == 'CC':
        return flux_obs / Pee
    elif int_type == 'NC':
        return flux_obs* np.ones(len(Pee))   

def B8_tot(flux,err_upper,err_lower, Pee, int_type = 'NC'):
    err_up = np.sqrt(err_upper[0]**2+err_upper[1]**2)
    err_low = np.sqrt(err_lower[0]**2+err_lower[1]**2)
    upper_flux = Flux_obs2tot(flux + err_up, Pee, int_type)
    lower_flux = Flux_obs2tot(flux - err_low, Pee, int_type)
    return upper_flux, lower_flux, (err_up+err_low)/flux*50

def propagate_uncertainty(b_bestfit, c_bestfit, b_err_upper, b_err_lower, c_err, rho):
    a_bestfit = b_bestfit * c_bestfit
    b_stat, b_sys_upper = b_err_upper
    _, b_sys_lower = b_err_lower  # Lower systematic uncertainty
    c_stat, c_sys = c_err
    # Average systematic uncertainty of b (handling asymmetry)
    b_sys_avg = (abs(b_sys_upper) + abs(b_sys_lower)) / 2
    # statistical uncertainty
    sigma_a_stat = np.sqrt((c_bestfit**2) * (b_stat**2) +
        (b_bestfit**2) * (c_stat**2) + 2 * b_bestfit * c_bestfit * rho * b_stat * c_stat)
    # systematic uncertainty 
    sigma_a_sys = np.sqrt((c_bestfit**2) * (b_sys_avg**2) +
        (b_bestfit**2) * (c_sys**2) + 2 * b_bestfit * c_bestfit * rho * b_sys_avg * c_sys)
    a_err_upper = np.array([sigma_a_stat, sigma_a_sys])
    a_err_lower = np.array([sigma_a_stat, sigma_a_sys])  # Assuming symmetric uncertainties 
    return a_bestfit, a_err_upper, a_err_lower


# Flux values and uncertainties

ppflux_ES = 6.1 # Borexno:
b8flux_ES = 2.308#e6 # SK:
b8flux_NC = 5.54 # SNO: 0806.0989
b8flux_CC = 1.67 #SNO: 0806.0989
# Flux ratio of B8/pp

# Best-fit value
sin2theta12_bm = 0.308
Pee_pp_bm, Pee_b8_bm = Pee_pp_day(sin2theta12_bm), Pee_b8_day(sin2theta12_bm)
ppflux_th = Flux_obs2tot(ppflux_ES, Pee_pp_day(sin2theta12_bm), int_type = 'ES')
raio_pp_b8 = 5.5e-5

ppflux_ES_err_upper = np.array([0.5,0.3])
ppflux_ES_err_lower = np.array([0.5,0.5]) 
ppflux_th_err_upper = np.array([0.6/100,0])*ppflux_th
ppflux_th_err_lower = np.array([0.6/100,0])*ppflux_th

b8flux_ES_err_upper = np.array([0.020,0.039])#*1e6
b8flux_ES_err_lower = np.array([0.020,0.040])#*1e6
b8flux_CC_err_upper = np.array([0.05,0.07])#*1e6
b8flux_CC_err_lower = np.array([0.04,0.08])#*1e6
b8flux_NC_err_upper = np.array([0.33,0.36])#*1e6
b8flux_NC_err_lower = np.array([0.31,0.34])#*1e6








# cross sections 
cross_ES = 9.20*10e-45  #@ 10 MeV
cross_CC = 5.92*10e-42  #@ 10 MeV
cross_NC = 1.75*10e-42  #@ 10 MeV

# target number 
target_NC = 1/cross_NC
target_CC = target_NC
target_ES = 10* target_NC

# Pee for B8 and pp
sin2theta12 = np.linspace(0.2, 0.4, 100)
Pee_b8, Pee_pp = Pee_b8_day(sin2theta12), Pee_pp_day(sin2theta12)
Pee_pp_day(0.2)

# reconstructed flux
b8_tot_ppES_upper, b8_tot_ppES_lower, a = B8_tot(ppflux_ES,ppflux_ES_err_upper,ppflux_ES_err_lower, Pee_pp, int_type = 'ES')
b8_tot_b8ES_upper,b8_tot_b8ES_lower, b = B8_tot(b8flux_ES,b8flux_ES_err_upper,b8flux_ES_err_lower, Pee_b8, int_type = 'ES')
b8_tot_b8CC_upper,b8_tot_b8CC_lower, c  = B8_tot(b8flux_CC,b8flux_CC_err_upper,b8flux_CC_err_lower, Pee_b8, int_type = 'CC')
b8_tot_b8NC_upper,b8_tot_b8NC_lower, d  = B8_tot(b8flux_NC,b8flux_NC_err_upper,b8flux_NC_err_lower, Pee_b8, int_type = 'NC')
b8_tot_ppth_upper,b8_tot_ppth_lower,e  = B8_tot(ppflux_th,ppflux_th_err_upper,ppflux_th_err_lower, Pee_pp, int_type = 'NC')




b8flux_NC2 = b8flux_ES/(Pee_b8_bm+1/6*(1-Pee_b8_bm))
b8_tot_ppES = Flux_obs2tot(b8flux_NC2*(Pee_pp_bm+1/6*(1-Pee_pp_bm)), Pee_pp, int_type = 'ES')
b8_tot_b8ES = Flux_obs2tot(b8flux_ES, Pee_b8, int_type = 'ES')
b8_tot_b8CC = Flux_obs2tot(b8flux_NC2*Pee_b8_bm, Pee_b8, int_type = 'CC')
b8_tot_b8NC = Flux_obs2tot(b8flux_NC2, Pee_b8, int_type = 'NC')


raio_pp_b8 = b8flux_NC/ppflux_th


plt.fill_betweenx([np.min(b8_tot_b8CC),np.max(b8_tot_b8CC)],0.308 - 0.011, 0.308 + 0.012, color='grey', alpha=0.1)
#plot_shaded_area(sin2theta12,raio_pp_b8*b8_tot_ppth_upper,raio_pp_b8*b8_tot_ppth_lower, color='orange', alpha=0.2, label='pp theory')
plot_shaded_area(sin2theta12, b8_tot_b8NC_lower, b8_tot_b8NC_upper, color='gold', alpha=0.1)
plot_shaded_area(sin2theta12, b8_tot_b8ES_lower, b8_tot_b8ES_upper, color='royalblue', alpha=0.1)
plot_shaded_area(sin2theta12, raio_pp_b8*b8_tot_ppES_lower, raio_pp_b8*b8_tot_ppES_upper, color='cornflowerblue', alpha=0.1, label='pp ES')
plot_shaded_area(sin2theta12, b8_tot_b8CC_lower, b8_tot_b8CC_upper, color='tomato', alpha=0.1)


plt.plot(sin2theta12,b8_tot_b8NC, color='orange', label = 'B8 NC')
plt.plot(sin2theta12,b8_tot_b8ES, color='royalblue', label = 'B8 ES')
plt.plot(sin2theta12,b8_tot_ppES,':',color='cornflowerblue', label = 'pp ES')
plt.plot(sin2theta12,b8_tot_b8CC, color='tomato', label = 'B8 CC')
plt.axvline(sin2theta12_bm, color = 'grey', label = 'baseline')

plt.xlabel("sin²(θ₁₂)")
plt.ylabel("B8 Flux")
plt.ylim([np.min(b8_tot_b8CC),np.max(b8_tot_b8CC)])
plt.legend()
plt.show()

