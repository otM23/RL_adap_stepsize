# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:24:01 2020

@author: othmane.mounjid
"""

### Load paths 
import os 
import sys
Current_directory = os.path.dirname(os.path.abspath(__file__))
Path_parent_directory = os.path.dirname(Current_directory)
sys.path.insert(0,Current_directory)
sys.path.insert(0,Path_parent_directory + "\\Plotting")


### Import libraries
import numpy as np
import rLAlgorithms.constant as Estim_mean_bench
import rLAlgorithms.oneOverN as Estim_mean_1_n
import rLAlgorithms.saga as Estim_mean_saga
import rLAlgorithms.pass_ as Estim_mean_pass
import plotting as pltg
import matplotlib.pyplot as plt


#### Compare different methods
def Compare_methods(gamma, NbSimu = 100, nb_episode = 70, T_max = 1, nb_iter = 100, alpha = 0.1, mu = 1, sigma2 = 5, freq_print = 4):
    ### Initialization of parameters    
    s_value = 0 ## price init value
    Time_step = (T_max)/nb_iter
    pdic = {  'Time_step':Time_step, 'alpha':alpha, 'mu':mu, 'T_max':T_max, 'nb_iter':nb_iter, 'sigma2':sigma2}
    ### Initialization rl algorithm
    h_0_init = 1*np.ones((pdic['nb_iter'])) # 100*np.random.rand(pdic['nb_iter'])#
    h_0theo = pdic['alpha']*pdic['mu']*pdic['Time_step']*np.ones(nb_iter) ## theoretical value
    Error = lambda x : np.linalg.norm(x -h_0theo)
    print_option = False ## subsidiary results
    pctg_0 = 0.01 ## stabilization percenteage of the average past error
    ### Initialisation outputs
    Val = np.zeros(4)
    Error_val = np.zeros(4)
    Values = np.zeros((NbSimu,4))
    size_values_all = int(nb_episode//freq_print)
    if nb_episode % freq_print != 0:
        size_values_all += 1   
    Values_all = np.zeros((NbSimu,4,size_values_all))
    
    ## main routine
    for n in range(NbSimu):
        ## step-cste 
        h_0 = np.array(h_0_init) # 100*np.random.rand(pdic['nb_iter'])#
        h_01,h_01_past,error_h01 = Estim_mean_bench.Loop_super1(nb_episode,pdic,gamma= gamma,s_value=s_value,Error=Error,freq_print=freq_print,h_0=h_0,print_option =print_option,pctg_0 = pctg_0)

        ## 1/n
        h_0 = np.array(h_0_init) # 100*np.random.rand(pdic['nb_iter'])#
        h_01_bis,h_01_bis_past,error_h01_bis = Estim_mean_1_n.Loop_super1_bis(nb_episode,pdic,gamma= gamma,s_value=s_value,Error=Error,freq_print=freq_print,h_0=h_0,print_option =print_option,pctg_0 = pctg_0)

        ## SAGA 
        _exp_mean = 1 ## decrease of the kernel :  when 1 it is the empirical average
        n_max = 2 ## nb past values to store
        h_0 = np.array(h_0_init)
        h_06,h_0_past6,nb_past6,error_h06 = Estim_mean_saga.Loop_super4(nb_episode,_exp_mean,pdic,inner_loop_func= Estim_mean_saga.Loop4_2,n_max=n_max,gamma= gamma,s_value=s_value,Error = Error,freq_print=freq_print,h_0=h_0,print_option =print_option,pctg_0 = pctg_0)

        ## PASS
        _exp_mean = 1 ## decrease of the kernel :  when 1 it is the empirical average
        n_max = 9 # 7 ## nb past values to store
        h_0 = np.array(h_0_init)
        alpha_0 = 1
        alpha_max = 3
        alpha_init = 1
        h_012,h_0_past12,nb_past12,error_h012,gamma_012 = Estim_mean_pass.Loop_super5(nb_episode,_exp_mean,pdic,inner_loop_func= Estim_mean_pass.Loop10,n_max=n_max,gamma= gamma,s_value=s_value,Error = Error,freq_print=freq_print,h_0=h_0,alpha_0 = alpha_0, alpha_max = alpha_max,print_option =print_option,pctg_0 = pctg_0,alpha_init = alpha_init)

        ## save values
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
    start = 0
    end = 15
    return {'index': error_h01[start:end,0], '1/n' : np.log(Values_all[:,1,start:end].mean(axis=0)) , 'step_cste' : np.log(Values_all[:,0,start:end].mean(axis=0)), 'SAGA' : np.log(Values_all[:,2,start:end].mean(axis=0))\
            , 'PASS' : np.log(Values_all[:,3,start:end].mean(axis=0)), 'mean' : Mean, 'var' : Var}

   
if __name__ == "__main__":
    ##### test 
    gamma_values = np.arange(0.1,0,-0.01) ## learning rate
    NbSimu = 100
    nb_episode = 70
    
    
    for gamma in gamma_values:
        res = Compare_methods(gamma, NbSimu = NbSimu, nb_episode = nb_episode)
        index = res['index']
        error1 = res['1/n']
        error2 = res['step_cste']
        error3 = res['SAGA']
        error4 = res['PASS']
        
        ### save values 
        res_glob = np.concatenate([index,error1,error2,error3,error4]).reshape((-1,5))
        np.save('Results\\res_gamma'+str(gamma.round(3)),res_glob)
        
        ### plot values    
        option_save = ""; path_Image = ""; ImageName = ""
        df = [ [index,error1],
              [index,error2],
              [index,error3],
              [index,error4]]   
        labels = ["1/n", "step_cste", "SAGA", "PASS"]
        mark = ['o', '*' , 'x', 'v']
        bbox_to_anchor_0 = (0.7,.75)
        plt.figure(figsize=(8,5))
        pltg.Plot_plot(df,labels, xlabel ="Number of iterations", ylabel ="Log L2 - error",
                      option=option_save, path =path_Image, ImageName=ImageName, Nset_tick_x = False, mark = mark, bbox_to_anchor_0 = bbox_to_anchor_0)
        plt.show()
