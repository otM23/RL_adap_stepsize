# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:43:36 2019

@author: othmane.mounjid
"""


### Load paths 
import os 
import sys
Path_parent_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,Path_parent_directory + "\\Plotting")
sys.path.insert(0,Path_parent_directory + "\\Auxiliary")


####### Import liraries 
import Optimal_exe_cj_reduced_f_no_link_bench as opti_exe_bench
import Optimal_exe_cj_reduced_f_no_link_1_n as opti_exe_1_n
import Optimal_exe_cj_reduced_f_no_link_pass as opti_exe_pass
import Optimal_exe_cj_reduced_f_no_link_saga as opti_exe_saga
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import plotting as pltg
import Theo_sol_exec_cj as Thsolexecj


###############################################################################
################ MC simu estimation of the average value ######################
###############################################################################
#### Compute the result
######### Initialize the parameters 
A = 0.25
Q_max = 2
Q_min = -Q_max
T_max = 1
size_q = 80 ## always take it an even number : see size_nu
nb_iter = 100
Step_q = (2*Q_max)/size_q
Time_step = (T_max)/nb_iter
kappa = 0.1
phi = 1
alpha = 0.1
mu = 1
Step_x = 0.01
Step_s = 0.01
size_x = 10
size_s = 10
size_nu = size_q
x_min = -10
s_min = -10
sigma2 = 1 # 0.02 # 0.02
pdic = {  'A':A,
          'Q_max': Q_max,
          'Q_min': Q_min,
          'Step_q':Step_q,
          'Time_step':Time_step,
          'kappa':kappa,
          'phi':phi,
          'alpha':alpha,
          'mu':mu,
          'T_max':T_max,
          'Step_x':Step_x,
          'Step_s':Step_s,
          'Step_q':Step_q,
          'x_min':x_min,
          's_min':s_min,
          'q_min':Q_min ,
          'size_x':size_x,
          'size_s':size_s,
          'size_q':size_q,
          'size_nu':size_nu,
          'nb_iter':nb_iter,
          'sigma2':sigma2}
option_save = "" # "save"



######### Test (RL) methods
########### parameters initialization

########### Compute Theoretical values
v_theo = np.zeros(((pdic['nb_iter']+1),pdic['size_q']))
h_2 = Thsolexecj.Compute_h_2(pdic)
h_1 = Thsolexecj.Compute_h_1(100, h_2, pdic)
h_0 = Thsolexecj.Compute_h_0(100, h_1, pdic)
########### Compute Theoretical v_values 
q_values = np.arange(-pdic['Q_max'],pdic['Q_max'],pdic['Step_q'])
for i in range(nb_iter+1): # i = 0
    v_theo[i,:] = h_0[i] + h_1[i]*q_values - 0.5 * h_2[i] * q_values * q_values
Error = lambda x : opti_exe_1_n.error_1(v_theo,x)


########### global parameters
NbSimu = 1
gamma = 0.05 # 1/nb_episode #0.01#1/nb_episode
v_0_init = 1*np.ones(((pdic['nb_iter']+1),pdic['size_q']))
nb_episode = 10000 # 5000 # 90000  #1200 # 2000 ## ## upper loop iterations
freq_print = 600 ## print frequency
Error_val = np.zeros((pdic['nb_iter']+1,pdic['size_q'],4))
Values = np.zeros((pdic['nb_iter']+1,pdic['size_q'],4))
if nb_episode % freq_print == 0:
    size_values_all = int(nb_episode//freq_print)    
else:
    size_values_all = int(nb_episode//freq_print)+1
Values_all = np.zeros((NbSimu,4,size_values_all))
print_option = False
write_option = False
pctg_0 = 0.001
for n in range(NbSimu):#  n = 0

    ########### Benchmark 
    v_0 = np.array(v_0_init) # 100*np.random.rand(pdic['nb_iter'])#
    v_0_past = None
    inner_loop_func = opti_exe_bench.Loop_within_episode_1
    v_01,error_v01,gamma_01 = opti_exe_bench.Loop_all_episode_1(nb_episode,pdic, inner_loop_func= inner_loop_func, gamma= gamma,Error=Error,freq_print=freq_print,v_0=v_0, v_0_past = v_0_past, pctg_0 = pctg_0)

    ########### Benchmark : gamma vary 
    v_0 = np.array(v_0_init) # 100*np.random.rand(pdic['nb_iter'])#
    v_0_past = None
    v_01_bis,error_v01_bis = opti_exe_1_n.Loop_all_episode_1_bis(nb_episode,pdic, gamma= gamma,Error=Error,freq_print=freq_print,v_0=v_0, v_0_past = v_0_past, pctg_0 = pctg_0)
  
    ########### Alg 2 : SAGA version test   2 : of the proximal gradient algorithm       
    v_0 = np.array(v_0_init) # 100*np.random.rand(pdic['nb_iter'])#
    v_0_past = None
    nb_past = None
    inner_loop_func = opti_exe_saga.Loop_within_episode_5_2
    n_max =  1
    _exp_mean = 1
    v_07,error_v07,gamma_07 = opti_exe_saga.Loop_all_episode_2(nb_episode,pdic,  inner_loop_func= inner_loop_func, gamma= gamma,Error=Error,freq_print=freq_print,v_0 = v_0, v_0_past = v_0_past, nb_past = nb_past, n_max =  n_max, _exp_mean = _exp_mean, pctg_0 = pctg_0)

    ########### Alg 5 :  increase gamma same sign 2 : adaptative version
    v_0 = np.array(v_0_init) # 100*np.random.rand(pdic['nb_iter'])#
    v_0_past = None
    nb_past = None
    inner_loop_func = opti_exe_pass.Loop_within_episode_7
    alpha_0 = 1
    alpha_max = 3
    r = 2/3
    v_010, error_v010, v_010_past, nb_past_010, gamma_010 = opti_exe_pass.Loop_all_episode_4(nb_episode,pdic,  inner_loop_func= inner_loop_func, gamma= gamma,Error=Error,freq_print=freq_print, v_0 = v_0,  alpha_0 = alpha_0, alpha_max = alpha_max, v_0_past = v_0_past, nb_past = nb_past, pctg_0 = pctg_0, r= r)

    Values[:,:,0] += v_01
    Values[:,:,1] += v_01_bis
    Values[:,:,2] += v_07
    Values[:,:,3] += v_010
    
    Error_val[:,:,0] += (v_01*v_01)
    Error_val[:,:,1] += (v_01_bis*v_01_bis)#  debug: print("value is : "+str(value)); print("Error value is : " +str(Error))
    Error_val[:,:,2] += (v_07*v_07)
    Error_val[:,:,3] += (v_010*v_010)

    ## Not the best way to do it at all : 
    Values_all[n,0] = error_v01[:,1]
    Values_all[n,1] = error_v01_bis[:,1]
    Values_all[n,2] = error_v07[:,1]
    Values_all[n,3] = error_v010[:,1]
    
    if (n % 5) == 0:
        print(" n is :"  + str(n))
        
Mean = Values/(NbSimu)
n = (NbSimu-1) if (NbSimu>1) else NbSimu
Var = ((Error_val)/n)-(NbSimu/n)*Mean*Mean


###############################################################################
###############################################################################
###################### This is the fourth main plot ############################
###################### There is no other one below ############################
###############################################################################
###############################################################################

### Save the values
option_save = ""
if option_save == "save":
    pathfile = Path_parent_directory + "\\Data"; fileName_MC_opti_placement = "\\improvement_optimal_placement_f_f_MC"
    np.save(pathfile + fileName_MC_opti_placement, Values_all)
option_load = "never"
if option_load == "load__carefully":
    Values_all_all = np.load(pathfile + fileName_MC_opti_placement + '.npy')
    

## Plot improvement v2
start = 0
option_save = ""
path_Image = Path_parent_directory + "\\Image"; ImageName = "\\improvement_optimal_placement_f_f_l" + str(NbSimu) + "_simu_f"
df = [ [error_v01[start:,0],np.log(Values_all[:,0,start:].mean(axis=0))],
       [error_v01_bis[start:,0],np.log(Values_all[:,1,start:].mean(axis=0))],
       [error_v07[start:,0],np.log(Values_all[:,2,start:].mean(axis=0))],
       [error_v010[start:,0],np.log(Values_all[:,3,start:].mean(axis=0))]]
labels = ["ste_cste", "1/n", "SAGA","PASS"]
mark = ['o', 'o', '*','<']
bbox_to_anchor_0 = (0.7,.75)
fig = plt.figure(figsize=(8,5))
pltg.Plot_plot(df,labels, xlabel ="Number of iterations", ylabel ="Log L2 - error",
               option=option_save, path =path_Image, ImageName=ImageName, Nset_tick_x = False, mark = mark, bbox_to_anchor_0 = bbox_to_anchor_0)
