# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:26:15 2019

@author: othmane.mounjid
"""

### Load paths 
import os 
import sys
Path_parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,Path_parent_directory + "\\Plotting")

####### Import liraries 
import Estim_mean_no_link_saga_2
import numpy as np 
import matplotlib.pyplot as plt

###############################################################################
###############################################################################
########################################## End functions ######################
###############################################################################
###############################################################################

    
#### Compute the result
######### Initialize the parameters 
gamma = 0.1 ## learning rate
s_value = 0 ## price init value
T_max = 1
nb_iter = 100
Time_step = (T_max)/nb_iter
alpha = 0.1
mu = 1
sigma2 = 5 # 5 # 1
pdic = {  'Time_step':Time_step,
          'alpha':alpha,
          'mu':mu,
          'T_max':T_max,
          'nb_iter':nb_iter,
          'sigma2':sigma2}

### Initialization rl algorithm
h_0_init = 10*np.ones((pdic['nb_iter'])) # 100*np.random.rand(pdic['nb_iter'])#
nb_episode = 100
freq_print = 5 
h_0theo = pdic['alpha']*pdic['mu']*pdic['Time_step']*np.ones(nb_iter) ## theoretical value
Error = lambda x : Estim_mean_no_link_saga_2.error_1(h_0theo,x)

#### Alg 2 : SAGA second version **
_exp_mean = 1 ## decrease of the kernel :  when 1 it is the empirical average
n_max = 2 ## nb past values to store
h_0 = np.array(h_0_init)
pctg_0 = 0.01
h_06,h_0_past6,nb_past6,error_h06 = Estim_mean_no_link_saga_2.Loop_super4(nb_episode,_exp_mean,pdic,inner_loop_func=Estim_mean_no_link_saga_2.Loop4_2,n_max=n_max,gamma= gamma,s_value=s_value,Error = Error,freq_print=freq_print,h_0=h_0,pctg_0 =pctg_0)
print(h_06.mean())


## Plot improvement v1
start = 0
option_save = ""
path_Image = "Image"; ImageName = "\\improvement_drift_estim"
df = [ [error_h06[start:,0],np.log(error_h06[start:,1])]]
labels = ["SAGA"]
mark = ['o']
fig = plt.figure(figsize=(8,5))
Estim_mean_no_link_saga_2.Plot_plot(df,labels, xlabel ="Number of iteration", ylabel ="Log L2 - error",
               option=option_save, path =path_Image, ImageName=ImageName, Nset_tick_x = False, mark = mark)
