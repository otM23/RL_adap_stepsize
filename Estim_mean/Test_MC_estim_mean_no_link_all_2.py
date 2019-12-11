# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:43:36 2019

@author: othmane.mounjid
"""


### Load paths 
import os 
import sys
Path_parent_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,Path_parent_directory + "\\Plotting")

####### Import liraries 
import Estim_mean_no_link_bench_2
import Estim_mean_no_link_1_n_2
import Estim_mean_no_link_pass_2
import Estim_mean_no_link_saga_2
import numpy as np 
import matplotlib.pyplot as plt

###############################################################################
################ MC simu estimation of the average value ######################
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
freq_print = 4 
h_0theo = pdic['alpha']*pdic['mu']*pdic['Time_step']*np.ones(nb_iter) ## theoretical value
Error = lambda x : Estim_mean_no_link_bench_2.error_1(h_0theo,x)



NbSimu = 1000
nb_episode = 70
freq_print = 4 
Val = np.zeros(4)
Error_val = np.zeros(4)
Values = np.zeros((NbSimu,4))
if nb_episode % freq_print == 0:
    size_values_all = int(nb_episode//freq_print)    
else:
    size_values_all = int(nb_episode//freq_print)+1
Values_all = np.zeros((NbSimu,4,size_values_all))
print_option = False
pctg_0 = 0.01
for n in range(NbSimu):#  n = 0
    
    h_0 = np.array(h_0_init) # 100*np.random.rand(pdic['nb_iter'])#
    h_01,h_01_past,error_h01 = Estim_mean_no_link_bench_2.Loop_super1(nb_episode,pdic,gamma= gamma,s_value=s_value,Error=Error,freq_print=freq_print,h_0=h_0,print_option =print_option,pctg_0 = pctg_0)

    h_0 = np.array(h_0_init) # 100*np.random.rand(pdic['nb_iter'])#
    h_01_bis,h_01_bis_past,error_h01_bis = Estim_mean_no_link_1_n_2.Loop_super1_bis(nb_episode,pdic,gamma= gamma,s_value=s_value,Error=Error,freq_print=freq_print,h_0=h_0,print_option =print_option,pctg_0 = pctg_0)

    #### Alg 2 : SAGA second version **
    _exp_mean = 1 ## decrease of the kernel :  when 1 it is the empirical average
    n_max = 2 ## nb past values to store
    h_0 = np.array(h_0_init)
    h_06,h_0_past6,nb_past6,error_h06 = Estim_mean_no_link_saga_2.Loop_super4(nb_episode,_exp_mean,pdic,inner_loop_func= Estim_mean_no_link_saga_2.Loop4_2,n_max=n_max,gamma= gamma,s_value=s_value,Error = Error,freq_print=freq_print,h_0=h_0,print_option =print_option,pctg_0 = pctg_0)


    #### Alg 8 : v_2 : increase gamma same sign **
    _exp_mean = 1 ## decrease of the kernel :  when 1 it is the empirical average
    n_max = 9 # 7 ## nb past values to store
    h_0 = np.array(h_0_init)
    alpha_0 = 1
    alpha_max = 3
    alpha_init = 1
    h_012,h_0_past12,nb_past12,error_h012,gamma_012 = Estim_mean_no_link_pass_2.Loop_super5(nb_episode,_exp_mean,pdic,inner_loop_func= Estim_mean_no_link_pass_2.Loop10,n_max=n_max,gamma= gamma,s_value=s_value,Error = Error,freq_print=freq_print,h_0=h_0,alpha_0 = alpha_0, alpha_max = alpha_max,print_option =print_option,pctg_0 = pctg_0,alpha_init = alpha_init)
    
    Values[n,0] = h_01.mean()
    Values[n,1] = h_01_bis.mean()
    Values[n,2] = h_06.mean()
    Values[n,3] = h_012.mean()
    
    Val[0] += h_01.mean()
    Val[1] += h_01_bis.mean()
    Val[2] += h_06.mean()
    Val[3] += h_012.mean()
    
    Error_val[0] += (h_01.mean()*h_01.mean())
    Error_val[1] += (h_01_bis.mean()*h_01_bis.mean())
    Error_val[2] += (h_06.mean()*h_06.mean())
    Error_val[3] += (h_012.mean()*h_012.mean())
    
    ## Not the best way to do it at all : 
    Values_all[n,0] = error_h01[:,1]
    Values_all[n,1] = error_h01_bis[:,1]
    Values_all[n,2] = error_h06[:,1]
    Values_all[n,3] = error_h012[:,1]
    
    if (n % 50) == 0:
        print(" n is :"  + str(n))
        
Mean = Val/(NbSimu)
n = (NbSimu-1) if (NbSimu>1) else NbSimu
Var = ((Error_val)/n)-(NbSimu/n)*Mean*Mean
print(Mean)
print(Var)

###############################################################################
###############################################################################
###################### This is the second main plot ############################
###################### There one other one below ############################
###############################################################################
###############################################################################


## Plot improvement v0
start = 0
end = 15
option_save = "save"
path_Image = "Image"; ImageName = "\\improvement_drift_estim_v2_" + str(NbSimu) + "_simu_f_l_2"
df = [ [error_h01[start:end,0],np.log(Values_all[:,0,start:end].mean(axis=0))],
       [error_h01_bis[start:end,0],np.log(Values_all[:,1,start:end].mean(axis=0))],
       [error_h06[start:end,0],np.log(Values_all[:,2,start:end].mean(axis=0))],
       [error_h012[start:end,0],np.log(Values_all[:,3,start:end].mean(axis=0))]]
labels = ["step_cste", "1/n", "SAGA", "PASS"]
mark = ['o', 'o', '*' , 'x', 'v', "o"]
bbox_to_anchor_0 = (0.7,.75)
fig = plt.figure(figsize=(8,5))
Estim_mean_no_link_bench_2.Plot_plot(df,labels, xlabel ="Number of iterations", ylabel ="Log L2 - error",
               option=option_save, path =path_Image, ImageName=ImageName, Nset_tick_x = False, mark = mark, bbox_to_anchor_0 = bbox_to_anchor_0)
