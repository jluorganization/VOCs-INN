# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:46:24 2021

@author: Administrator 


"""


import sys
sys.path.insert(0, '../../Utilities/')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pandas
import math 
from math import gamma
from scipy.integrate import odeint
import matplotlib.dates as mdates
import tensorflow as tf
import numpy as np
from numpy import *
# from numpy import matlib as mb
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import datetime
from pyDOE import lhs
# from scipy.special import gamma
start_time = time.time()


from datetime import datetime
now = datetime.now()
# dt_string = now.strftime("%m-%d-%H-%M")
# dt_string = now.strftime("%m-%d")
dt_string =  '06-21'

# Load Data 
data_frame = pandas.read_csv('Data/BC_data.csv')
I1_new_star = data_frame['I1new']  # T x 1 array
I2_new_star = data_frame['I2new']  # T x 1 array
I1_sum_star = data_frame['I1sum']  # T x 1 array
I2_sum_star = data_frame['I2sum']  # T x 1 array
I1_new_star = I1_new_star.to_numpy(dtype=np.float64)
I2_new_star = I2_new_star.to_numpy(dtype=np.float64)
I1_sum_star = I1_sum_star.to_numpy(dtype=np.float64)
I2_sum_star = I2_sum_star.to_numpy(dtype=np.float64)
I1_new_star = I1_new_star.reshape([len(I1_new_star), 1])
I2_new_star = I2_new_star.reshape([len(I2_new_star), 1])
I1_sum_star = I1_sum_star.reshape([len(I1_sum_star), 1])
I2_sum_star = I2_sum_star.reshape([len(I2_sum_star), 1])
t_star = np.arange(len(I1_new_star))
t_star = t_star.reshape([len(t_star), 1])
N = 5.5e6 + 13 + 306


data_frame_ture = pandas.read_csv('Data/BC_pre_data.csv')
I1_new_ture = data_frame_ture['I1new']  # T x 1 array
I2_new_ture = data_frame_ture['I2new']  # T x 1 array
I1_sum_ture = data_frame_ture['I1sum']  # T x 1 array
I2_sum_ture = data_frame_ture['I2sum']  # T x 1 array
I1_new_ture = I1_new_ture.to_numpy(dtype=np.float64)
I2_new_ture = I2_new_ture.to_numpy(dtype=np.float64)
I1_sum_ture = I1_sum_ture.to_numpy(dtype=np.float64)
I2_sum_ture = I2_sum_ture.to_numpy(dtype=np.float64)
I1_new_ture = I1_new_ture.reshape([len(I1_new_ture), 1])
I2_new_ture = I2_new_ture.reshape([len(I2_new_ture), 1])
I1_sum_ture = I1_sum_ture.reshape([len(I1_sum_ture), 1])
I2_sum_ture = I2_sum_ture.reshape([len(I2_sum_ture), 1])





first_date = np.datetime64('2023-03-26')
last_date = np.datetime64('2024-01-07') + np.timedelta64(7, 'D')
first_date_pred = np.datetime64('2024-01-07') #last_date[6:]+'-'+last_date[0:2]+'-'+str(int(last_date[3:5])-1)
last_date_pred = np.datetime64('2024-04-21') + np.timedelta64(7, 'D')

date_total = np.arange(first_date, last_date, np.timedelta64(7, 'D'))[:,None]
data_mean = np.arange(first_date, last_date, np.timedelta64(7, 'D'))[:,None]
data_pred = np.arange(first_date_pred, last_date_pred, np.timedelta64(7, 'D'))[:,None]


sf = 1e-4

# load data
BetaI1_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/BetaI1_pred_mean.txt')
BetaI2_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/BetaI2_pred_mean.txt')
Gamma1_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/Gamma1_pred_mean.txt')
Gamma2_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/Gamma2_pred_mean.txt')

t_mean = np.arange(len(BetaI1_PINN))


S_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/S_pred_mean.txt') 
S_PINN = S_PINN/sf
I1_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/I1_pred_mean.txt')
I1_PINN = I1_PINN/sf
I2_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/I2_pred_mean.txt')
I2_PINN = I2_PINN/sf
R_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/R_pred_mean.txt') 
R_PINN = R_PINN/sf
I1_sum_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/I1_sum_pred_mean.txt')
I1_sum_PINN = I1_sum_PINN/sf
I2_sum_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/I2_sum_pred_mean.txt')
I2_sum_PINN = I2_sum_PINN/sf
I1_new_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/I1_new_pred_mean.txt')
I1_new_PINN = I1_new_PINN/sf
I2_new_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/I2_new_pred_mean.txt')
I2_new_PINN = I2_new_PINN/sf


    
#%%
# #Interpolations    # 插值函数
Beta_interp = scipy.interpolate.interp1d(t_mean.flatten(), BetaI1_PINN.flatten(), fill_value="extrapolate")

#%%
    ######################################################################
    ################ Predicting by sloving forward problem ###############
    ######################################################################  
#%%    
#Initial conditions for ODE system  
S_init = float(S_PINN[-1])
I1_init = float(I1_PINN[-1])
I2_init = float(I2_PINN[-1])
R_init = float(R_PINN[-1])
I1_sum_init = float(I1_sum_PINN[-1])
I2_sum_init = float(I2_sum_PINN[-1])
U_init = [S_init, I1_init, I2_init, R_init, I1_sum_init, I2_sum_init]

#Parameters      
#Parameters
Gamma1=0.1
Gamma2=0.1
t_pred = np.arange(len(t_mean)-1, len(t_mean)+len(data_pred)-1)
t_pred = t_pred.reshape([len(t_pred),1])  



#ODEs  
def ODEs_mean(X, t, xi, Pert):
    S, I1, I2, R, sumI1, sumI2 = X
    dSdt = -(BetaI1_PINN[-1]  * (1+xi*Pert))*S*I1/N - (BetaI2_PINN[-1]  * (1+xi*Pert))*S*I2/N
    dI1dt = (BetaI2_PINN[-1]  * (1+xi*Pert))*S*I1/N  - (Gamma1_PINN[-1])*I1
    dI2dt = (BetaI2_PINN[-1]  * (1+xi*Pert))*S*I2/N - (Gamma2_PINN[-1])*I2
    dRdt = (Gamma1_PINN[-1])*I1 + (Gamma2_PINN[-1])*I2
    dsumI1dt = (BetaI1_PINN[-1]  * (1+xi*Pert))*S*I1/N
    dsumI2dt = (BetaI2_PINN[-1]  * (1+xi*Pert))*S*I2/N
    return [float(dSdt),  float(dI1dt), float(dI2dt),  float(dRdt), float(dsumI1dt), float(dsumI2dt)]




#%%
Pert0 = 0.10 
Sol_ub_d0 = odeint(ODEs_mean, U_init, t_pred.flatten(), args = (1,Pert0))
S_ub_d0 = Sol_ub_d0[:,0]
I1_ub_d0 = Sol_ub_d0[:,1]
I2_ub_d0 = Sol_ub_d0[:,2]
R_ub_d0 = Sol_ub_d0[:,3]
sumI1_ub_d0 = Sol_ub_d0[:,4]
sumI2_ub_d0 = Sol_ub_d0[:,5]
newI1_ub_d0= np.diff(sumI1_ub_d0)
newI2_ub_d0 = np.diff(sumI2_ub_d0)
newI1_ub_d0 = newI1_ub_d0*I1_new_PINN[-1]/newI1_ub_d0[0]
newI2_ub_d0 = newI2_ub_d0*I2_new_PINN[-1]/newI2_ub_d0[0]

Sol_lb_d0 = odeint(ODEs_mean, U_init, t_pred.flatten(), args = (-1,Pert0))
S_lb_d0 = Sol_lb_d0[:,0]
I1_lb_d0 = Sol_lb_d0[:,1]
I2_lb_d0 = Sol_lb_d0[:,2]
R_lb_d0 = Sol_lb_d0[:,3]
sumI1_lb_d0 = Sol_lb_d0[:,4]
sumI2_lb_d0 = Sol_lb_d0[:,5]
newI1_lb_d0= np.diff(sumI1_lb_d0)
newI2_lb_d0 = np.diff(sumI2_lb_d0)
newI1_lb_d0 = newI1_lb_d0*I1_new_PINN[-1]/newI1_lb_d0[0]
newI2_lb_d0 = newI2_lb_d0*I2_new_PINN[-1]/newI2_lb_d0[0]



#%%
Pert1 = 0.20
Sol_ub_d1 = odeint(ODEs_mean, U_init, t_pred.flatten(), args = (1,Pert1))
S_ub_d1 = Sol_ub_d1[:,0]
I1_ub_d1 = Sol_ub_d1[:,1]
I2_ub_d1 = Sol_ub_d1[:,2]
R_ub_d1 = Sol_ub_d1[:,3]
sumI1_ub_d1 = Sol_ub_d1[:,4]
sumI2_ub_d1 = Sol_ub_d1[:,5]
newI1_ub_d1= np.diff(sumI1_ub_d1)
newI2_ub_d1 = np.diff(sumI2_ub_d1)
newI1_ub_d1 = newI1_ub_d1*I1_new_PINN[-1]/newI1_ub_d1[0]
newI2_ub_d1 = newI2_ub_d1*I2_new_PINN[-1]/newI2_ub_d1[0]


Sol_lb_d1 = odeint(ODEs_mean, U_init, t_pred.flatten(), args = (-1,Pert1))
S_lb_d1 = Sol_lb_d1[:,0]
I1_lb_d1 = Sol_lb_d1[:,1]
I2_lb_d1 = Sol_lb_d1[:,2]
R_lb_d1 = Sol_lb_d1[:,3]
sumI1_lb_d1 = Sol_lb_d1[:,4]
sumI2_lb_d1 = Sol_lb_d1[:,5]
newI1_lb_d1= np.diff(sumI1_lb_d1)
newI2_lb_d1 = np.diff(sumI2_lb_d1)
newI1_lb_d1 = newI1_lb_d1*I1_new_PINN[-1]/newI1_lb_d1[0]
newI2_lb_d1 = newI2_lb_d1*I2_new_PINN[-1]/newI2_lb_d1[0]


#%% 
Sol_mean = odeint(ODEs_mean, U_init, t_pred.flatten(), args = (0,0))
S_mean = Sol_mean[:,0]
I1_mean = Sol_mean[:,1]
I2_mean = Sol_mean[:,2]
R_mean = Sol_mean[:,3]
sumI1_mean = Sol_mean[:,4]
sumI2_mean = Sol_mean[:,5]
newI1_mean = np.diff(sumI1_mean)
newI2_mean = np.diff(sumI2_mean)
newI1_mean = newI1_mean*I1_new_PINN[-1]/newI1_mean[0]
newI2_mean = newI2_mean*I2_new_PINN[-1]/newI2_mean[0]


#%%
######################################################################
######################################################################
############################# Save the results ###############################
######################################################################
###################################################################### 
#%% 
#saver
current_directory = os.getcwd()
relative_path = '/Model1/Prediction-Results-'+dt_string+'/'
save_results_to = current_directory + relative_path
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)    

np.savetxt(save_results_to + 'S_mean.txt', S_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'I1_mean.txt', I1_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'I2_mean.txt', I2_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'R_mean.txt', R_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'newI1_mean.txt', newI1_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'newI2_mean.txt', newI2_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'sumI1_mean.txt', sumI1_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'sumI2_mean.txt', sumI2_mean.reshape((-1,1)))


np.savetxt(save_results_to + 'S_ub_d0.txt', S_ub_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'I1_ub_d0.txt', I1_ub_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'I2_ub_d0.txt', I2_ub_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'R_ub_d0.txt', R_ub_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'newI1_ub_d0.txt', newI1_ub_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'newI2_ub_d0.txt', newI2_ub_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'sumI1_ub_d0.txt', sumI1_ub_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'sumI2_ub_d0.txt', sumI2_ub_d0.reshape((-1,1)))



np.savetxt(save_results_to + 'S_lb_d0.txt', S_lb_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'I1_lb_d0.txt', I1_lb_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'I2_lb_d0.txt', I2_lb_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'R_lb_d0.txt', R_lb_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'newI1_lb_d0.txt', newI1_lb_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'newI2_lb_d0.txt', newI2_lb_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'sumI1_lb_d0.txt', sumI1_lb_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'sumI2_lb_d0.txt', sumI2_lb_d0.reshape((-1,1)))



np.savetxt(save_results_to + 'S_ub_d1.txt', S_ub_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'I1_ub_d1.txt', I1_ub_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'I2_ub_d1.txt', I2_ub_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'R_ub_d1.txt', R_ub_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'newI1_ub_d1.txt', newI1_ub_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'newI2_ub_d1.txt', newI2_ub_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'sumI1_ub_d1.txt', sumI1_ub_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'sumI2_ub_d1.txt', sumI2_ub_d1.reshape((-1,1)))

np.savetxt(save_results_to + 'S_lb_d1.txt', S_lb_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'I1_lb_d1.txt', I1_lb_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'I2_lb_d1.txt', I2_lb_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'R_lb_d1.txt', R_lb_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'newI1_lb_d1.txt', newI1_lb_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'newI2_lb_d1.txt', newI2_lb_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'sumI1_lb_d1.txt', sumI1_lb_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'sumI2_lb_d1.txt', sumI2_lb_d1.reshape((-1,1)))



#%%
######################################################################
############################# Plotting ###############################
######################################################################  


plt.rc('font', size=60)
intervals=[1,2,1]

#%%
plt.rc('font', size=40)

#%%
#BetaI1 curve
beta_pred_0 = np.array([BetaI1_PINN[-1] for i in range(data_pred.shape[0])])

fig, ax = plt.subplots()
ax.plot(data_mean, BetaI1_PINN, 'k-', lw=4, label='VOCs-INN-Training')
ax.plot(data_pred.flatten(), beta_pred_0, 'm--', lw=4, label='Prediction-mean')
plt.fill_between(data_pred.flatten(), \
                 beta_pred_0*(1.1), \
                 beta_pred_0*(0.9), \
                 facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')

plt.fill_between(data_pred.flatten(), \
                 beta_pred_0*(1.2), \
                 beta_pred_0*(0.8), \
                 facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')


ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=35, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 40)
plt.rc('font', size=40)
ax.grid(True)
ax.set_ylabel(r'$\beta_{I1}$', fontsize = 80)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to +'BetaI1.pdf', dpi=300)
plt.savefig(save_results_to +'BetaI1.png', dpi=300)



#BetaI2 curve
beta_pred_0 = np.array([BetaI2_PINN[-1] for i in range(data_pred.shape[0])])
fig, ax = plt.subplots()
ax.plot(data_mean, BetaI2_PINN, 'k-', lw=4, label='VOCs-INN-Training')
ax.plot(data_pred.flatten(), beta_pred_0, 'm--', lw=4, label='Prediction-mean')
plt.fill_between(data_pred.flatten(), \
                 beta_pred_0*(1.1), \
                 beta_pred_0*(0.9), \
                 facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')

plt.fill_between(data_pred.flatten(), \
                 beta_pred_0*(1.2), \
                 beta_pred_0*(0.8), \
                 facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')


ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=35, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 40)
plt.rc('font', size=40)
ax.grid(True)
ax.set_ylabel(r'$\beta_{I2}$', fontsize = 80)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to +'BetaI2.pdf', dpi=300)
plt.savefig(save_results_to +'BetaI2.png', dpi=300)

#%%
#New infectious1
for i in [1,2]:
    fig, ax = plt.subplots()
    plt.fill_between(data_pred[:-1].flatten(), \
                      newI1_lb_d0.flatten(), newI1_ub_d0.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')
    plt.fill_between(data_pred[:-1].flatten(), \
                      newI1_lb_d1.flatten(), newI1_ub_d1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')


    if i==1:
        ax.plot(data_pred[:-1], newI1_mean, 'm--', lw=4, label='Prediction-mean')
        ax.plot(date_total, I1_new_star, 'ro', lw=4, markersize=8, label='Data')
        ax.plot(data_pred[1:-1], I1_new_ture[:-1], 'ro', lw=4, markersize=8)
        ax.plot(data_mean[1:], I1_new_PINN, 'k-', lw=4, label='VOCs-INN-Training')

    if i==2:
        ax.plot(data_pred[:-1], newI1_mean, 'm--', lw=7, label='Prediction-mean')
        ax.plot(data_pred[1:-1], I1_new_ture[:-1], 'ro', lw=4, markersize=8, label='Data')


    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    plt.rc('font', size=40)
    ax.grid(True)
    ax.set_ylabel('daily infectious cases ($\mathbf{I}^{new1}$)', fontsize = 40)
    fig.set_size_inches(w=25, h=12.5)
    if i==1:
        plt.savefig(save_results_to + 'new1_cases.pdf', dpi=300)
        plt.savefig(save_results_to + 'new1_cases.png', dpi=300)
    if i==2:
        plt.savefig(save_results_to + 'new1_cases_zoom.pdf', dpi=300)
        plt.savefig(save_results_to + 'new1_cases_zoom.png', dpi=300)


#New infectious2
for i in [1,2]:
    fig, ax = plt.subplots()
    plt.fill_between(data_pred[:-1].flatten(), \
                     newI2_lb_d0.flatten(), newI2_ub_d0.flatten(), \
                     facecolor=(0.1, 0.2, 0.5, 0.3), interpolate=True, label='Prediction-std-(10%)')
    plt.fill_between(data_pred[:-1].flatten(), \
                     newI2_lb_d1.flatten(), newI2_ub_d1.flatten(), \
                     facecolor=(0.1, 0.5, 0.8, 0.3), interpolate=True, label='Prediction-std-(20%)')

    if i == 1:
        ax.plot(data_pred[:-1], newI2_mean, 'm--', lw=4, label='Prediction-mean')
        ax.plot(date_total, I2_new_star, 'ro', lw=4, markersize=8, label='Data')
        ax.plot(data_pred[1:-1], I2_new_ture[:-1], 'ro', lw=4, markersize=8)
        ax.plot(data_mean[1:], I2_new_PINN, 'k-', lw=4, label='VOCs-INN-Training')

    if i == 2:
        ax.plot(data_pred[:-1], newI2_mean, 'm--', lw=7, label='Prediction-mean')
        ax.plot(data_pred[1:-1], I2_new_ture[:-1], 'ro', lw=4, markersize=8, label='Data')



    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    plt.rc('font', size=40)
    ax.grid(True)
    ax.set_ylabel('daily infectious cases ($\mathbf{I}^{new2}$)', fontsize = 40)
    fig.set_size_inches(w=25, h=12.5)
    if i==1:
        plt.savefig(save_results_to + 'new2_cases.pdf', dpi=300)
        plt.savefig(save_results_to + 'new2_cases.png', dpi=300)
    if i==2:
        plt.savefig(save_results_to + 'new2_cases_zoom.pdf', dpi=300)
        plt.savefig(save_results_to + 'new2_cases_zoom.png', dpi=300)


#Cumulative Infectious  1
for i in [1,2]:
    fig, ax = plt.subplots()
    plt.fill_between(data_pred.flatten(), \
                      sumI1_lb_d0.flatten(), sumI1_ub_d0.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')
    plt.fill_between(data_pred.flatten(), \
                      sumI1_lb_d1.flatten(), sumI1_ub_d1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')


    if i==1:
        ax.plot(data_pred, sumI1_mean, 'm--', lw=4, label='Prediction-mean')
        ax.plot(date_total, I1_sum_star, 'ro', lw=4, markersize=8, label='Data')
        ax.plot(data_pred[1:], I1_sum_ture, 'ro', lw=4, markersize=8, label='Data')
        ax.plot(data_mean, I1_sum_PINN, 'k-', lw=4, label='VOCs-INN-Training')

    if i==2:
        ax.plot(data_pred, sumI1_mean, 'm--', lw=7, label='Prediction-mean')
        ax.plot(data_pred[1:], I1_sum_ture, 'ro', lw=4, markersize=8, label='Data')


    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    plt.rc('font', size=40)
    ax.grid(True)
    ax.set_ylabel('($\mathbf{I1}^{cum}$)', fontsize = 40)
    fig.set_size_inches(w=25, h=12.5)
    if i==1:
        plt.savefig(save_results_to + 'sum1_cases.pdf', dpi=300)
        plt.savefig(save_results_to + 'sum1_cases.png', dpi=300)
    if i==2:
        plt.savefig(save_results_to + 'sum1_cases_zoom.pdf', dpi=300)
        plt.savefig(save_results_to + 'sum1_cases_zoom.png', dpi=300)

#Cumulative Infectious  2
for i in [1,2]:
    fig, ax = plt.subplots()
    plt.fill_between(data_pred.flatten(), \
                      sumI2_lb_d0.flatten(), sumI2_ub_d0.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')
    plt.fill_between(data_pred.flatten(), \
                      sumI2_lb_d1.flatten(), sumI2_ub_d1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')


    if i==1:
        ax.plot(data_pred, sumI2_mean, 'm--', lw=4, label='Prediction-mean')
        ax.plot(date_total, I2_sum_star, 'ro', lw=4, markersize=8, label='Data')
        ax.plot(data_pred[1:], I2_sum_ture, 'ro', lw=4, markersize=8, label='Data')
        ax.plot(data_mean, I2_sum_PINN, 'k-', lw=4, label='VOCs-INN-Training')

    if i==2:
        ax.plot(data_pred, sumI2_mean, 'm--', lw=7, label='Prediction-mean')
        ax.plot(data_pred[1:], I2_sum_ture, 'ro', lw=4, markersize=8, label='Data')



    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    plt.rc('font', size=40)
    ax.grid(True)
    ax.set_ylabel('($\mathbf{I2}^{cum}$)', fontsize = 40)
    fig.set_size_inches(w=25, h=12.5)
    if i==1:
        plt.savefig(save_results_to + 'sum2_cases.pdf', dpi=300)
        plt.savefig(save_results_to + 'sum2_cases.png', dpi=300)
    if i==2:
        plt.savefig(save_results_to + 'sum2_cases_zoom.pdf', dpi=300)
        plt.savefig(save_results_to + 'sum2_cases_zoom.png', dpi=300)



#Current Suspectious
for i in [1,2]:
    fig, ax = plt.subplots()
    plt.fill_between(data_pred.flatten(), \
                      S_lb_d0.flatten(), S_ub_d0.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')
    plt.fill_between(data_pred.flatten(), \
                      S_lb_d1.flatten(), S_ub_d1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')


    if i==1:
        ax.plot(data_pred, S_mean, 'm--', lw=4, label='Prediction-mean')
        ax.plot(data_mean, S_PINN, 'k-', lw=4, label='VOCs-INN-Training')

    if i==2:
        ax.plot(data_pred, S_mean, 'm--', lw=7, label='Prediction-mean')



    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    plt.rc('font', size=40)
    ax.grid(True)
    ax.set_ylabel('$\mathbf{S}$', fontsize = 40)
    fig.set_size_inches(w=25, h=12.5)
    if i==1:
        plt.savefig(save_results_to + 'Current_Suspectious.pdf', dpi=300)
        plt.savefig(save_results_to + 'Current_Suspectious.png', dpi=300)
    if i==2:
        plt.savefig(save_results_to + 'Current_Suspectious_zoom.pdf', dpi=300)
        plt.savefig(save_results_to + 'Current_Suspectious_zoom.png', dpi=300)
#

#
# #%%
#Current infectious
for i in [1,2]:
    fig, ax = plt.subplots()
    plt.fill_between(data_pred.flatten(), \
                      I1_lb_d0.flatten(), I1_ub_d0.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')
    plt.fill_between(data_pred.flatten(), \
                      I1_lb_d1.flatten(), I1_ub_d1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')



    if i==1:
        ax.plot(data_pred, I1_mean, 'm--', lw=4, label='Prediction-mean')
        ax.plot(data_mean, I1_PINN, 'k-', lw=4, label='VOCs-INN-Training')

    if i==2:
        ax.plot(data_pred, I1_mean, 'm--', lw=7, label='Prediction-mean')



    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    plt.rc('font', size=40)
    ax.grid(True)
    ax.set_ylabel('$\mathbf{I1}$', fontsize = 40)
    fig.set_size_inches(w=25, h=12.5)
    if i==1:
        plt.savefig(save_results_to + 'Current_Infectious1.pdf', dpi=300)
        plt.savefig(save_results_to + 'Current_Infectious1.png', dpi=300)
    if i==2:
        plt.savefig(save_results_to + 'Current_Infectious1_zoom.pdf', dpi=300)
        plt.savefig(save_results_to + 'Current_Infectious1_zoom.png', dpi=300)


#Current infectious
for i in [1,2]:
    fig, ax = plt.subplots()
    plt.fill_between(data_pred.flatten(), \
                      I2_lb_d0.flatten(), I2_ub_d0.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')
    plt.fill_between(data_pred.flatten(), \
                      I2_lb_d1.flatten(), I2_ub_d1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')



    if i==1:
        ax.plot(data_pred, I2_mean, 'm--', lw=4, label='Prediction-mean')
        ax.plot(data_mean, I2_PINN, 'k-', lw=4, label='VOCs-INN-Training')

    if i==2:
        ax.plot(data_pred, I2_mean, 'm--', lw=7, label='Prediction-mean')



    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    plt.rc('font', size=40)
    ax.grid(True)
    ax.set_ylabel('$\mathbf{I2}$', fontsize = 40)
    fig.set_size_inches(w=25, h=12.5)
    if i==1:
        plt.savefig(save_results_to + 'Current_Infectious2.pdf', dpi=300)
        plt.savefig(save_results_to + 'Current_Infectious2.png', dpi=300)
    if i==2:
        plt.savefig(save_results_to + 'Current_Infectious2_zoom.pdf', dpi=300)
        plt.savefig(save_results_to + 'Current_Infectious2_zoom.png', dpi=300)

#Current removed
for i in [1,2]:
    fig, ax = plt.subplots()
    plt.fill_between(data_pred.flatten(), \
                      R_lb_d0.flatten(), R_ub_d0.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')
    plt.fill_between(data_pred.flatten(), \
                      R_lb_d1.flatten(), R_ub_d1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')
    plt.fill_between(data_pred.flatten(), \
                      R_lb_d0.flatten(), R_ub_d0.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)
    plt.fill_between(data_pred.flatten(), \
                      R_lb_d1.flatten(), R_ub_d1.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)


    if i==1:
        ax.plot(data_pred, R_mean, 'm--', lw=4, label='Prediction-mean')
        ax.plot(data_mean, R_PINN, 'k-', lw=4, label='VOCs-INN-Training')

    if i==2:
        ax.plot(data_pred, R_mean, 'm--', lw=7, label='Prediction-mean')



    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    plt.rc('font', size=40)
    ax.grid(True)
    ax.set_ylabel('$\mathbf{R}$', fontsize = 40)
    fig.set_size_inches(w=25, h=12.5)
    if i==1:
        plt.savefig(save_results_to + 'Current_Recovered.pdf', dpi=300)
        plt.savefig(save_results_to + 'Current_Recovered.png', dpi=300)
    if i==2:
        plt.savefig(save_results_to + 'Current_Recovered_zoom.pdf', dpi=300)
        plt.savefig(save_results_to + 'Current_Recovered_zoom.png', dpi=300)


