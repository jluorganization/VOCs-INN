# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:17:46 2021

@author: Administrator 

"""


import sys
sys.path.insert(0, '../../Utilities/')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import math 
from math import gamma
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
import pandas

#%%
###Load data
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


I1_new_train = I1_new_star
I2_new_train = I2_new_star
I1_sum_train = I1_sum_star
I2_sum_train = I2_sum_star



#read results  

BetaI1_pred_total = []
BetaI2_pred_total = []
Gamma1_pred_total = []
Gamma2_pred_total = []

###
S_pred_total = []
I1_pred_total = []
I2_pred_total = []
R_pred_total = []
I1_new_pred_total = []
I2_new_pred_total = []
I1_sum_pred_total = []
I2_sum_pred_total = []

### Ode
S_ode_total = []
I1_ode_total = []
I2_ode_total = []
R_ode_total = []
I1_new_ode_total = []
I2_new_ode_total = []
I1_sum_ode_total = []
I2_sum_ode_total = []


from datetime import datetime
now = datetime.now()
dt_string = '06-21'


Model = '1'
relative_path_results0 = '/Model'+Model

 
for j in np.arange(1,11,1):

    casenumber ='set' +str(j)
    current_directory = os.getcwd()
    relative_path_results = relative_path_results0 + '/Train-Results-'+dt_string+'-'+casenumber+'/'
    read_results_to = current_directory + relative_path_results
    # par
    BetaI1_pred = np.loadtxt(read_results_to + 'BetaI1.txt')
    BetaI2_pred = np.loadtxt(read_results_to + 'BetaI2.txt')
    Gamma1_pred = np.loadtxt(read_results_to + 'Gamma1.txt')
    Gamma2_pred = np.loadtxt(read_results_to + 'Gamma2.txt')
    BetaI1_pred_total.append(BetaI1_pred)
    BetaI2_pred_total.append(BetaI2_pred)
    Gamma1_pred_total.append(Gamma1_pred)
    Gamma2_pred_total.append(Gamma2_pred)

    # VOCs-INN
    S_pred = np.loadtxt(read_results_to + 'S.txt')
    I1_pred = np.loadtxt(read_results_to + 'I1.txt')
    I2_pred = np.loadtxt(read_results_to + 'I2.txt')
    R_pred = np.loadtxt(read_results_to + 'R.txt')
    I1_new_pred = np.loadtxt(read_results_to + 'I1_new.txt')
    I2_new_pred = np.loadtxt(read_results_to + 'I2_new.txt')
    I1_sum_pred = np.loadtxt(read_results_to + 'I1_sum.txt')
    I2_sum_pred = np.loadtxt(read_results_to + 'I2_sum.txt')

    S_pred_total.append(S_pred)
    I1_pred_total.append(I1_pred)
    I2_pred_total.append(I2_pred)
    R_pred_total.append(R_pred)
    I1_new_pred_total.append(I1_new_pred)
    I2_new_pred_total.append(I2_new_pred)
    I1_sum_pred_total.append(I1_sum_pred)
    I2_sum_pred_total.append(I2_sum_pred)

    ## ode
    S_ode = np.loadtxt(read_results_to + 'S_ode.txt')
    I1_ode = np.loadtxt(read_results_to + 'I1_ode.txt')
    I2_ode = np.loadtxt(read_results_to + 'I2_ode.txt')
    R_ode = np.loadtxt(read_results_to + 'R_ode.txt')
    I1_new_ode = np.loadtxt(read_results_to + 'I1_new_ode.txt')
    I2_new_ode = np.loadtxt(read_results_to + 'I2_new_ode.txt')
    I1_sum_ode = np.loadtxt(read_results_to + 'I1_sum_ode.txt')
    I2_sum_ode = np.loadtxt(read_results_to + 'I2_sum_ode.txt')


    S_ode_total.append(S_ode)
    I1_ode_total.append(I1_ode)
    I2_ode_total.append(I2_ode)
    R_ode_total.append(R_ode)
    I1_new_ode_total.append(I1_new_ode)
    I2_new_ode_total.append(I2_new_ode)
    I1_sum_ode_total.append(I1_sum_ode)
    I2_sum_ode_total.append(I2_sum_ode)

    
#%%
#Average + std

## par
BetaI1_pred_mean = np.mean(BetaI1_pred_total, axis=0)
BetaI2_pred_mean = np.mean(BetaI2_pred_total, axis=0)
Gamma1_pred_mean = np.mean(Gamma1_pred_total, axis=0)
Gamma2_pred_mean = np.mean(Gamma2_pred_total, axis=0)
BetaI1_pred_std = np.std(BetaI1_pred_total, axis=0)
BetaI2_pred_std = np.std(BetaI2_pred_total, axis=0)
Gamma1_pred_std = np.std(Gamma1_pred_total, axis=0)
Gamma2_pred_std = np.std(Gamma2_pred_total, axis=0)

### VOCs-INN
S_pred_mean = np.mean(S_pred_total, axis=0)
I1_pred_mean = np.mean(I1_pred_total, axis=0)
I2_pred_mean = np.mean(I2_pred_total, axis=0)
R_pred_mean = np.mean(R_pred_total, axis=0) 
I1_new_pred_mean = np.mean(I1_new_pred_total, axis=0)
I2_new_pred_mean = np.mean(I2_new_pred_total, axis=0)
I1_sum_pred_mean = np.mean(I1_sum_pred_total, axis=0)
I2_sum_pred_mean = np.mean(I2_sum_pred_total, axis=0)

S_pred_std = np.std(S_pred_total, axis=0)
I1_pred_std = np.std(I1_pred_total, axis=0)
I2_pred_std = np.std(I2_pred_total, axis=0)
R_pred_std = np.std(R_pred_total, axis=0)
I1_new_pred_std = np.std(I1_new_pred_total, axis=0)
I2_new_pred_std = np.std(I2_new_pred_total, axis=0)
I1_sum_pred_std = np.std(I1_sum_pred_total, axis=0)
I2_sum_pred_std = np.std(I2_sum_pred_total, axis=0)


### ode
S_ode_mean = np.mean(S_ode_total, axis=0)
I1_ode_mean = np.mean(I1_ode_total, axis=0)
I2_ode_mean = np.mean(I2_ode_total, axis=0)
R_ode_mean = np.mean(R_ode_total, axis=0)
I1_new_ode_mean = np.mean(I1_new_ode_total, axis=0)
I2_new_ode_mean = np.mean(I2_new_ode_total, axis=0)
I1_sum_ode_mean = np.mean(I1_sum_ode_total, axis=0)
I2_sum_ode_mean = np.mean(I2_sum_ode_total, axis=0)
S_ode_std = np.std(S_ode_total, axis=0)
I1_ode_std = np.std(I1_ode_total, axis=0)
I2_ode_std = np.std(I2_ode_total, axis=0)
R_ode_std = np.std(R_ode_total, axis=0)
I1_new_ode_std = np.std(I1_new_ode_total, axis=0)
I2_new_ode_std = np.std(I2_new_ode_total, axis=0)
I1_sum_ode_std = np.std(I1_sum_ode_total, axis=0)
I2_sum_ode_std = np.std(I2_sum_ode_total, axis=0)


#%%
#save results  
current_directory = os.getcwd()
relative_path_results = relative_path_results0 + '/Train-Results-'+dt_string+'-Average/'
save_results_to = current_directory + relative_path_results 
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

## par
np.savetxt(save_results_to + 'BetaI1_pred_mean.txt', BetaI1_pred_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'BetaI2_pred_mean.txt', BetaI2_pred_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'Gamma1_pred_mean.txt', Gamma1_pred_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'Gamma2_pred_mean.txt', Gamma2_pred_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'BetaI1_pred_std.txt', BetaI1_pred_std.reshape((-1,1)))
np.savetxt(save_results_to + 'BetaI2_pred_std.txt', BetaI2_pred_std.reshape((-1,1)))
np.savetxt(save_results_to + 'Gamma1_pred_std.txt', Gamma1_pred_std.reshape((-1,1)))
np.savetxt(save_results_to + 'Gamma2_pred_std.txt', Gamma2_pred_std.reshape((-1,1)))

## VOCs-INN
np.savetxt(save_results_to + 'S_pred_mean.txt', S_pred_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'I1_pred_mean.txt', I1_pred_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'I2_pred_mean.txt', I2_pred_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'R_pred_mean.txt', R_pred_mean.reshape((-1,1)))  
np.savetxt(save_results_to + 'I1_new_pred_mean.txt', I1_new_pred_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'I2_new_pred_mean.txt', I2_new_pred_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'I1_sum_pred_mean.txt', I1_sum_pred_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'I2_sum_pred_mean.txt', I2_sum_pred_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'S_pred_std.txt', S_pred_std.reshape((-1,1)))
np.savetxt(save_results_to + 'I1_pred_std.txt', I1_pred_std.reshape((-1,1)))
np.savetxt(save_results_to + 'I2_pred_std.txt', I2_pred_std.reshape((-1,1)))
np.savetxt(save_results_to + 'R_pred_std.txt', R_pred_std.reshape((-1,1)))  
np.savetxt(save_results_to + 'I1_new_pred_std.txt', I1_new_pred_std.reshape((-1,1)))
np.savetxt(save_results_to + 'I2_new_pred_std.txt', I2_new_pred_std.reshape((-1,1)))
np.savetxt(save_results_to + 'I1_sum_pred_std.txt', I1_sum_pred_std.reshape((-1,1)))
np.savetxt(save_results_to + 'I2_sum_pred_std.txt', I2_sum_pred_std.reshape((-1,1)))


## ode
np.savetxt(save_results_to + 'S_ode_mean.txt', S_ode_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'I1_ode_mean.txt', I1_ode_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'I2_ode_mean.txt', I2_ode_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'R_ode_mean.txt', R_ode_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'I1_new_ode_mean.txt', I1_new_ode_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'I2_new_ode_mean.txt', I2_new_ode_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'I1_sum_ode_mean.txt', I1_sum_ode_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'I2_sum_ode_mean.txt', I2_sum_ode_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'S_ode_std.txt', S_ode_std.reshape((-1,1)))
np.savetxt(save_results_to + 'I1_ode_std.txt', I1_ode_std.reshape((-1,1)))
np.savetxt(save_results_to + 'I2_ode_std.txt', I2_ode_std.reshape((-1,1)))
np.savetxt(save_results_to + 'R_ode_std.txt', R_ode_std.reshape((-1,1)))
np.savetxt(save_results_to + 'I1_new_ode_std.txt', I1_new_ode_std.reshape((-1,1)))
np.savetxt(save_results_to + 'I2_new_ode_std.txt', I2_new_ode_std.reshape((-1,1)))
np.savetxt(save_results_to + 'I1_sum_ode_std.txt', I1_sum_ode_std.reshape((-1,1)))
np.savetxt(save_results_to + 'I2_sum_ode_std.txt', I2_sum_ode_std.reshape((-1,1)))




####
first_date = np.datetime64('2023-03-26')
last_date = np.datetime64('2024-01-07') + np.timedelta64(7, 'D')
date_total = np.arange(first_date, last_date, np.timedelta64(7, 'D'))






sf = 1e-4
plt.rc('font', size=30)

# VOCs-INN
#Current Suspectious
fig, ax = plt.subplots()
ax.plot(date_total, S_pred_mean/sf, 'k-', lw=5, label='VOCs-INN')
plt.fill_between(date_total, \
                  (S_pred_mean+S_pred_std)/sf, \
                  (S_pred_mean-S_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(6,6))
plt.rc('font', size=30)
ax.set_ylabel('$S$', fontsize = 80)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Current_Suspectious_VOCs-INN.pdf', dpi=300)
plt.savefig(save_results_to + 'Current_Suspectious_VOCs-INN.png', dpi=300)

#Current Infectious 1
fig, ax = plt.subplots()
ax.plot(date_total, I1_pred_mean/sf, 'k-', lw=5, label='VOCs-INN')
plt.fill_between(date_total, \
                  (I1_pred_mean+I1_pred_std)/sf, \
                  (I1_pred_mean-I1_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
plt.rc('font', size=30)
ax.set_ylabel('$I_{1}$', fontsize = 80)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Current_Infectious1_VOCs-INN.pdf', dpi=300)
plt.savefig(save_results_to + 'Current_Infectious1_VOCs-INN.png', dpi=300)


#Current Infectious 2
fig, ax = plt.subplots()
ax.plot(date_total, I2_pred_mean/sf, 'k-', lw=5, label='VOCs-INN')
plt.fill_between(date_total, \
                  (I2_pred_mean+I2_pred_std)/sf, \
                  (I2_pred_mean-I2_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
plt.rc('font', size=30)
ax.set_ylabel('$I_{2}$', fontsize = 80)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Current_Infectious2_VOCs-INN.pdf', dpi=300)
plt.savefig(save_results_to + 'Current_Infectious2_VOCs-INN.png', dpi=300)




#Current Removed
fig, ax = plt.subplots()
ax.plot(date_total, R_pred_mean/sf, 'k-', lw=5, label='VOCs-INN')
plt.fill_between(date_total, \
                  (R_pred_mean+R_pred_std)/sf, \
                  (R_pred_mean-R_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(6,6))
plt.rc('font', size=30)
ax.set_ylabel('$R$', fontsize = 80)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Current_Removed_VOCs-INN.pdf', dpi=300)
plt.savefig(save_results_to + 'Current_Removed_VOCs-INN.png', dpi=300)

#%%
#New infectious 1
fig, ax = plt.subplots()
ax.plot(date_total[1:], I1_new_pred_mean/sf, 'k-', lw=5, label = 'VOCs-INN')
ax.plot(date_total, I1_new_star, 'ro', markersize=8, label='Turth')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='plain', scilimits=(3,3))
plt.rc('font', size=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('$I^{new}_{1}$', fontsize = 80)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'New_Infectious1_VOCs-INN.pdf', dpi=300)
plt.savefig(save_results_to + 'New_Infectious1_VOCs-INN.png', dpi=300)

#%%
#New infectious 2
fig, ax = plt.subplots()
ax.plot(date_total[1:], I2_new_pred_mean/sf, 'k-', lw=5, label = 'VOCs-INN')
ax.plot(date_total, I2_new_star, 'ro', markersize=8, label='Turth')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='plain', scilimits=(3,3))
plt.rc('font', size=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('$I^{new}_{2}$', fontsize = 80)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'New_Infectious2_VOCs-INN.pdf', dpi=300)
plt.savefig(save_results_to + 'New_Infectious2_VOCs-INN.png', dpi=300)



#%%
#Cumulative Infectious  1
fig, ax = plt.subplots()
ax.plot(date_total, I1_sum_star, 'ro', markersize=8, label='Turth')
ax.plot(date_total, I1_sum_pred_mean/sf, 'k-', lw=5, label = 'PINN')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='plain', scilimits=(3,3))
plt.rc('font', size=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('$I^{c}_{1}$', fontsize = 80)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Cumulative_Infectious1_PINN.pdf', dpi=300)
plt.savefig(save_results_to + 'Cumulative_Infectious1_PINN.png', dpi=300)


#%%
#Cumulative Infectious  2
fig, ax = plt.subplots()
ax.plot(date_total, I2_sum_star, 'ro', markersize=8, label='Turth')
ax.plot(date_total, I2_sum_pred_mean/sf, 'k-', lw=5, label = 'VOCs-INN')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='plain', scilimits=(3,3))
plt.rc('font', size=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('$I^{c}_{2}$', fontsize = 80)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Cumulative_Infectious2_PINN.pdf', dpi=300)
plt.savefig(save_results_to + 'Cumulative_Infectious2_PINN.png', dpi=300)

#%%
#BetaI1
fig, ax = plt.subplots()
ax.plot(date_total, BetaI1_pred_mean, 'k-', lw=5, label = 'VOCs-INN')
plt.fill_between(date_total, \
                  BetaI1_pred_mean+BetaI1_pred_std, \
                  BetaI1_pred_mean-BetaI1_pred_std, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=18, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
plt.rc('font', size=30)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel(r'$\beta_{1}$', fontsize = 80)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'BetaI1_VOCs-INN.pdf', dpi=300)
plt.savefig(save_results_to + 'BetaI1_VOCs-INN.png', dpi=300)


#%%
#BetaI2
fig, ax = plt.subplots()
ax.plot(date_total, BetaI2_pred_mean, 'k-', lw=5, label = 'VOCs-INN')
plt.fill_between(date_total, \
                  BetaI2_pred_mean+BetaI2_pred_std, \
                  BetaI2_pred_mean-BetaI2_pred_std, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=18, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
plt.rc('font', size=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel(r'$\beta_{2}$', fontsize = 80)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'BetaI2_VOCs-INN.pdf', dpi=300)
plt.savefig(save_results_to + 'BetaI2_VOCs-INN.png', dpi=300)


#%%
#Gamma1
fig, ax = plt.subplots()
ax.plot(date_total, Gamma1_pred_mean, 'k-', lw=5, label = 'VOCs-INN')
plt.fill_between(date_total, \
                  Gamma1_pred_mean+Gamma1_pred_std, \
                  Gamma1_pred_mean-Gamma1_pred_std, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=18, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
plt.rc('font', size=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel(r'$\gamma_{1}$', fontsize = 80)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Gamma1_VOCs-INN.pdf', dpi=300)
plt.savefig(save_results_to + 'Gamma1_VOCs-INN.png', dpi=300)


#%%
#Gamma2
fig, ax = plt.subplots()
ax.plot(date_total, Gamma1_pred_mean, 'k-', lw=5, label = 'VOCs-INN')
plt.fill_between(date_total, \
                  Gamma2_pred_mean+Gamma2_pred_std, \
                  Gamma2_pred_mean-Gamma2_pred_std, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=18, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
plt.rc('font', size=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel(r'$\gamma_{2}$', fontsize = 80)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Gamma2_VOCs-INN.pdf', dpi=300)
plt.savefig(save_results_to + 'Gamma2_VOCs-INN.png', dpi=300)


#  ode
#Current Suspectious
fig, ax = plt.subplots()
ax.plot(date_total, S_ode_mean, 'k-', lw=5, label='Odeslover')
plt.fill_between(date_total, \
                  (S_ode_mean+S_ode_std), \
                  (S_ode_mean-S_ode_std), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(6,6))
plt.rc('font', size=30)
ax.set_ylabel('$S$', fontsize = 80)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Current_Suspectious_ode.pdf', dpi=300)
plt.savefig(save_results_to + 'Current_Suspectious_ode.png', dpi=300)

#Current Infectious 1
fig, ax = plt.subplots()
ax.plot(date_total, I1_ode_mean, 'k-', lw=5, label='Odeslover')
plt.fill_between(date_total, \
                  (I1_ode_mean+I1_ode_std), \
                  (I1_ode_mean-I1_ode_std), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
plt.rc('font', size=30)
ax.set_ylabel('$I_{1}$', fontsize = 80)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Current_Infectious1_ode.pdf', dpi=300)
plt.savefig(save_results_to + 'Current_Infectious1_ode.png', dpi=300)

#%%
#Current Infectious 2
fig, ax = plt.subplots()
ax.plot(date_total, I2_ode_mean, 'k-', lw=5, label='Odeslover')
plt.fill_between(date_total, \
                  (I2_ode_mean+I2_ode_std), \
                  (I2_ode_mean-I2_ode_std), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
plt.rc('font', size=30)
ax.set_ylabel('$I_{2}$', fontsize = 80)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Current_Infectious2_ode.pdf', dpi=300)
plt.savefig(save_results_to + 'Current_Infectious2_ode.png', dpi=300)



#%%
#Current Removed
fig, ax = plt.subplots()
ax.plot(date_total, R_ode_mean, 'k-', lw=5, label='Odeslover')
plt.fill_between(date_total, \
                  (R_ode_mean+R_ode_std), \
                  (R_ode_mean-R_ode_std), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(6,6))
plt.rc('font', size=30)
ax.set_ylabel('$R$', fontsize = 80)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Current_Removed_ode.pdf', dpi=300)
plt.savefig(save_results_to + 'Current_Removed_ode.png', dpi=300)

#%%
#New infectious 1
fig, ax = plt.subplots()
ax.plot(date_total[1:], I1_new_ode_mean, 'k-', lw=5, label = 'Odeslover')
ax.plot(date_total, I1_new_star, 'ro', markersize=8, label='Turth')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='plain', scilimits=(3,3))
plt.rc('font', size=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('$I^{new}_{1}$', fontsize = 80)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'New_Infectious1_ode.pdf', dpi=300)
plt.savefig(save_results_to + 'New_Infectious1_ode.png', dpi=300)

#%%
#New infectious 2
fig, ax = plt.subplots()
ax.plot(date_total[1:], I2_new_ode_mean, 'k-', lw=5, label = 'Odeslover')
ax.plot(date_total, I2_new_star, 'ro', markersize=8, label='Turth')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='plain', scilimits=(3,3))
plt.rc('font', size=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('$I^{new}_{2}$', fontsize = 80)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'New_Infectious2_ode.pdf', dpi=300)
plt.savefig(save_results_to + 'New_Infectious2_ode.png', dpi=300)



#%%
#Cumulative Infectious  1
fig, ax = plt.subplots()
ax.plot(date_total, I1_sum_star, 'ro', markersize=8, label='Turth')
ax.plot(date_total, I1_sum_ode_mean, 'k-', lw=5, label = 'Odeslover')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='plain', scilimits=(3,3))
plt.rc('font', size=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('$I^{c}_{1}$', fontsize = 80)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Cumulative_Infectious1_ode.pdf', dpi=300)
plt.savefig(save_results_to + 'Cumulative_Infectious1_ode.png', dpi=300)


#%%
#Cumulative Infectious  2
fig, ax = plt.subplots()
ax.plot(date_total, I2_sum_star, 'ro', markersize=8, label='Turth')
ax.plot(date_total, I2_sum_ode_mean, 'k-', lw=5, label = 'Odeslover')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='plain', scilimits=(3,3))
plt.rc('font', size=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('$I^{c}_{2}$', fontsize = 80)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Cumulative_Infectious2_ode.pdf', dpi=300)
plt.savefig(save_results_to + 'Cumulative_Infectious2_ode.png', dpi=300)



