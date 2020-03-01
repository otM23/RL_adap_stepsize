# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 20:09:12 2020

@author: othmane.mounjid
"""


### Load paths 
import os 
import sys
Current_directory = os.path.dirname(os.path.abspath(__file__))
Path_parent_directory = os.path.dirname(Current_directory)
sys.path.insert(0,Current_directory)
sys.path.insert(0,Path_parent_directory + "\\Plotting")

####### Import liraries 
import Optimal_placing_reduced_no_link_bench_2 as op_place_bench
import Optimal_placing_reduced_no_link_1_n_2 as op_place_1_n
import Optimal_placing_reduced_no_link_pass_2 as op_place_pass
import Optimal_placing_reduced_no_link_saga_2 as op_place_saga
import numpy as np 

###############################################################################
################ Main function  ###############################################
###############################################################################

def Compare_methods_op(Intens_val_bis, gamma , NbSimu = 1, nb_iter = 100, nb_episode = 500, freq_print = 40, qb = 2, bb_pos = 1, write_option = True, seed=5):
    ######### Theoretical solution :  numerical scheme
    tol = 0.1
    size_q = Intens_val_bis.shape[0]
    nb_iter_scheme = 400
    reward_exec_1 = lambda qsame,bb_pos: op_place_pass.reward_exec(qsame, bb_pos, gain = 6, cost_out = -0.6, cost_stay = -0.2)
    df_bis = op_place_pass.Sol_num_scheme(nb_iter_scheme,size_q,Intens_val_bis,tol = tol,reward_exec_1 = reward_exec_1)
    
    
    ##### Initialization of the parameters
    np.random.seed(seed)
    qb_0 = qb# 4
    bb_pos_0 = bb_pos
    Lob_state_0 = [qb_0,bb_pos_0] 
    
     ## learning rate
    h_0_init = 5*np.ones((size_q,size_q+1))
    h_0_Stay_init = np.ones((size_q,size_q+1))
    h_0_Mkt_init = np.ones((size_q,size_q+1))
    h_0 = np.array(h_0_init)
    h_0_Stay = np.array(h_0_Stay_init)
    h_0_Mkt = np.array(h_0_Mkt_init)
    h_0_theo = op_place_pass.Read_h_0_theo(df_bis["Value_opt"].values,size_q,reward_exec_1)
    Error = lambda x : op_place_pass.error_1(np.nan_to_num(h_0_theo),np.nan_to_num(x))

    ########### global parameters
    Error_val = np.zeros((size_q,size_q+1,4))
    Values = np.zeros((size_q,size_q+1,4))
    if nb_episode % freq_print == 0:
        size_values_all = int(nb_episode//freq_print)    
    else:
        size_values_all = int(nb_episode//freq_print)+1
    Values_all = np.zeros((NbSimu,4,size_values_all+1))
    write_option = False
    eta = 1 ## discount factor
    r = 0.5 
    pctg_0 = 0.05
    
    for n in range(NbSimu):#  n = 0
    
        ########### Benchmark 
        h_0 = np.array(h_0_init)
        h_0_Stay = np.array(h_0_Stay_init)
        h_0_Mkt = np.array(h_0_Mkt_init)
        h_01, h_01_past, error_h01, h_01_Stay, h_01_Mkt = op_place_bench.Lob_simu_super_Loop1(nb_episode,Intens_val_bis,nb_iter, gamma= gamma,Lob_state_0=Lob_state_0,freq_print=freq_print,size_q = size_q,Error=Error, write_option = write_option, h_0 = h_0, reward_exec = reward_exec_1, eta = eta, h_0_Stay = h_0_Stay, h_0_Mkt = h_0_Mkt, pctg_0 = pctg_0)
    
        ########### Benchmark : gamma vary 
        h_0 = np.array(h_0_init)
        h_0_Stay = np.array(h_0_Stay_init)
        h_0_Mkt = np.array(h_0_Mkt_init)
        h_01_bis, h_01_bis_past, error_h01_bis, h_01_Stay_bis, h_01_Mkt_bis = op_place_1_n.Lob_simu_super_Loop1_bis(nb_episode,Intens_val_bis,nb_iter, gamma= gamma,Lob_state_0=Lob_state_0,freq_print=freq_print,size_q = size_q,Error=Error, write_option = write_option, h_0 = h_0, reward_exec = reward_exec_1, eta = eta, h_0_Stay = h_0_Stay, h_0_Mkt = h_0_Mkt, pctg_0 = pctg_0)
    
        ########### Alg 2 : SAGA version test   2 : of the proximal gradient algorithm 
        h_0 = np.array(h_0_init)
        h_0_Stay = np.array(h_0_Stay_init)
        h_0_Mkt = np.array(h_0_Mkt_init)
        inner_loop_func = op_place_saga.Lob_simu_inner_loop7
        _exp_mean = 1 ## decrease of the kernel :  when 1 it is the empirical average
        n_max = 1 ## nb past values to store
        h_07,error_h07,nb_past_07,h_07_Stay,h_07_Mkt = op_place_saga.Lob_simu_super_Loop3(nb_episode,Intens_val_bis,nb_iter,_exp_mean,n_max, inner_loop_func = inner_loop_func, gamma= gamma,Lob_state_0=Lob_state_0,freq_print=freq_print,size_q = size_q,Error=Error, write_option = write_option, h_0 = h_0, reward_exec = reward_exec_1, eta = eta, h_0_Stay = h_0_Stay, h_0_Mkt = h_0_Mkt, pctg_0 = pctg_0)
        
        ########### Alg 2 : increase gamma same sign v2 : of the proximal gradient algorithm : 
        h_0 = np.array(h_0_init)
        h_0_Stay = np.array(h_0_Stay_init)
        h_0_Mkt = np.array(h_0_Mkt_init)
        inner_loop_func = op_place_pass.Lob_simu_inner_loop11
        alpha_0 = 1
        alpha_max = 4
        r = 2/3
        pctg_0 = 0.05
        h_011,error_h011,nb_past_011,h_011_Stay,h_011_Mkt = op_place_pass.Lob_simu_super_Loop4(nb_episode,Intens_val_bis,nb_iter, inner_loop_func = inner_loop_func, gamma= gamma,Lob_state_0=Lob_state_0,freq_print=freq_print,size_q = size_q,Error=Error, write_option = write_option, h_0 = h_0, reward_exec = reward_exec_1, eta = eta, h_0_Stay = h_0_Stay, h_0_Mkt = h_0_Mkt, alpha_0 = alpha_0, alpha_max = alpha_max, r = r, pctg_0 = pctg_0)
    
        Values[:,:,0] += h_01
        Values[:,:,1] += h_01_bis
        Values[:,:,2] += h_07
        Values[:,:,3] += h_011
        
        Error_val[:,:,0] += (h_01*h_01)
        Error_val[:,:,1] += (h_01_bis*h_01_bis)#  debug: print("value is : "+str(value)); print("Error value is : " +str(Error))
        Error_val[:,:,2] += (h_07*h_07)
        Error_val[:,:,3] += (h_011*h_011)
    
        ## Not the best way to do it at all : 
        Values_all[n,0] = error_h01[:,1]
        Values_all[n,1] = error_h01_bis[:,1]
        Values_all[n,2] = error_h07[:,1]
        Values_all[n,3] = error_h011[:,1]
        
        if (n % 5) == 0:
            print(" n is :"  + str(n))
        
    Mean = Values/(NbSimu)
    n = (NbSimu-1) if (NbSimu>1) else NbSimu
    Var = ((Error_val)/n)-(NbSimu/n)*Mean*Mean
    return {'index': error_h01[:,0], '1/n' : np.log(Values_all[:,1,:].mean(axis=0)) , 'step_cste' : np.log(Values_all[:,0,:].mean(axis=0)), 'SAGA' : np.log(Values_all[:,2,:].mean(axis=0))\
            , 'PASS' : np.log(Values_all[:,3,:].mean(axis=0)), 'mean' : Mean, 'var' : Var}