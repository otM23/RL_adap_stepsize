# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:46:32 2019

@author: othmane.mounjid
"""

### Load paths 
import os 
import sys
Path_parent_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,Path_parent_directory + "/Plotting")
sys.path.insert(0,Path_parent_directory + "/Auxiliary")
print(Path_parent_directory)


####### Import liraries 
import Optimal_exe_cj_reduced_f_no_link_bench as opti_exe_bench
import Optimal_exe_cj_reduced_f_no_link_1_n as opti_exe_1_n
import Optimal_exe_cj_reduced_f_no_link_pass as opti_exe_pass
import Optimal_exe_cj_reduced_f_no_link_saga as opti_exe_saga
import numpy as np 
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
#import plotting as pltg
import Theo_sol_exec_cj as Thsolexecj
#import pdb

#pdb.set_trace()

###############################################################################
################ MC simu estimation of the average value ######################
###############################################################################

######### Initialize the parameters 
A = 0.25
Q_max = float(2)
Q_min = -Q_max
T_max = float(1)
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
pdic = {  'A': A,
          'Q_max': Q_max,
          'Q_min': Q_min,
          'Step_q': Step_q,
          'Time_step': Time_step,
          'kappa': kappa,
          'phi': phi,
          'alpha': alpha,
          'mu': mu,
          'T_max': T_max,
          'Step_x': Step_x,
          'Step_s': Step_s,
          'Step_q': Step_q,
          'x_min': x_min,
          's_min': s_min,
          'q_min': Q_min ,
          'size_x': size_x,
          'size_s': size_s,
          'size_q': size_q,
          'size_nu': size_nu,
          'nb_iter': nb_iter,
          'sigma2': sigma2
          }
option_save = "" # "save"

########### Compute Theoretical values
v_theo = np.zeros(((pdic['nb_iter']+1),pdic['size_q']))
h_2 = Thsolexecj.Compute_h_2(pdic)
h_1 = Thsolexecj.Compute_h_1(100, h_2, pdic)
h_0 = Thsolexecj.Compute_h_0(100, h_1, pdic)
########### Compute Theoretical v_values 
q_values = np.arange(-pdic['Q_max'],pdic['Q_max'],pdic['Step_q'])
for i in range(nb_iter+1): # i = 0
    v_theo[i,:] = h_0[i] + h_1[i]*q_values - 0.5 * h_2[i] * q_values * q_values

def Error(x):
    return opti_exe_1_n.error_1(v_theo,x)


########### Remaining parameters
#NbSimu = 4
gamma = 0.05 # 1/nb_episode # 0.01 # 1/nb_episode
v_0_init = 1*np.ones(((pdic['nb_iter']+1),pdic['size_q']))
nb_episode = 120000 ## 10000 ## 5000 ## 9000 ## 1200 ## 2000 ## upper loop iterations
freq_print = 3000 ## print frequency ##
pctg_0 = 0.001
path_folder = Path_parent_directory + "/Result"
kwargs = {'v_0_init': v_0_init,
          'nb_episode' : nb_episode,
          'pdic': pdic,
          'gamma':gamma,
          'pctg_0': pctg_0,
          'freq_print': freq_print,
          'path_folder': path_folder 
          }


#### Multi processing simple loop
def Simple_loop(start,end,shared_list_res, Values_list, Error_list, Error, kwargs):
    pdic = kwargs['pdic']
    Error_val = np.zeros((4,pdic['nb_iter']+1,pdic['size_q']))
    Values = np.zeros((4,pdic['nb_iter']+1,pdic['size_q']))
    for n in range(start,end):#  n = 0  

        print(' n is ' + str(n) + ' start ')          
        ########### Benchmark 
        v_0 = np.array(kwargs['v_0_init']) # 100*np.random.rand(pdic['nb_iter'])#
        v_0_past = None
        inner_loop_func = opti_exe_bench.Loop_within_episode_1
        v_01,error_v01,gamma_01 = opti_exe_bench.Loop_all_episode_1(kwargs['nb_episode'],pdic, inner_loop_func= inner_loop_func, gamma= kwargs['gamma'],Error = Error ,freq_print= kwargs['freq_print'], v_0 = v_0, v_0_past = v_0_past, pctg_0 = kwargs['pctg_0'])
    
        ########### Benchmark : gamma vary 
        v_0 = np.array(kwargs['v_0_init']) # 100*np.random.rand(pdic['nb_iter'])#
        v_0_past = None
        v_01_bis,error_v01_bis = opti_exe_1_n.Loop_all_episode_1_bis(kwargs['nb_episode'],pdic, gamma= kwargs['gamma'] ,Error = Error, freq_print= kwargs['freq_print'],v_0=v_0, v_0_past = v_0_past, pctg_0 = kwargs['pctg_0'])
      
        ########### Alg 2 : SAGA version test   2 : of the proximal gradient algorithm       
        v_0 = np.array(kwargs['v_0_init']) # 100*np.random.rand(pdic['nb_iter'])#
        v_0_past = None
        nb_past = None
        inner_loop_func = opti_exe_saga.Loop_within_episode_5_2
        n_max =  1
        _exp_mean = 1
        v_07,error_v07,gamma_07 = opti_exe_saga.Loop_all_episode_2(kwargs['nb_episode'],pdic, inner_loop_func= inner_loop_func, gamma= kwargs['gamma'] ,Error= Error, freq_print= kwargs['freq_print'],v_0 = v_0, v_0_past = v_0_past, nb_past = nb_past, n_max =  n_max, _exp_mean = _exp_mean, pctg_0 = kwargs['pctg_0'])
    
        ########### Alg 5 :  increase gamma same sign 2 : adaptative version
        v_0 = np.array(v_0_init) # 100*np.random.rand(pdic['nb_iter'])#
        v_0_past = None
        nb_past = None
        inner_loop_func = opti_exe_pass.Loop_within_episode_7
        alpha_0 = 1
        alpha_max = 3
        r = 2/3
        v_010, error_v010, v_010_past, nb_past_010, gamma_010 = opti_exe_pass.Loop_all_episode_4(kwargs['nb_episode'], pdic,  inner_loop_func= inner_loop_func, gamma= kwargs['gamma'] ,Error= Error, freq_print= kwargs['freq_print'], v_0 = v_0,  alpha_0 = alpha_0, alpha_max = alpha_max, v_0_past = v_0_past, nb_past = nb_past, pctg_0 = kwargs['pctg_0'], r= r)
    

        ## Keep mean :: Not the best way to do it at all :
        Values[0] += v_01
        Values[1] += v_01_bis
        Values[2] += v_07
        Values[3] += v_010
 
        ## Keep variance ::    
        Error_val[0] += (v_01*v_01)
        Error_val[1] += (v_01_bis*v_01_bis)#  debug: print("value is : "+str(value)); print("Error value is : " +str(Error))
        Error_val[2] += (v_07*v_07)
        Error_val[3] += (v_010*v_010)

        ## Keep error :: Not the best way to do it at all :
        list_ = []
        list_.append(error_v01[:,1])
        list_.append(error_v01_bis[:,1])
        list_.append(error_v07[:,1])
        list_.append(error_v010[:,1])
        shared_list_res.append(list_)    
        
        ### save values
        path_bench = kwargs['path_folder'] + "/Bench"
        np.save(path_bench + "/res_" + str(n) , v_01)
        path_bench = kwargs['path_folder'] + "/1_n"
        np.save(path_bench + "/res_" + str(n) , v_01_bis)
        path_bench = kwargs['path_folder'] + "/saga"
        np.save(path_bench + "/res_" + str(n) , v_07)
        path_bench = kwargs['path_folder'] + "/pass"
        np.save(path_bench + "/res_" + str(n) , v_010)

        path_bench = kwargs['path_folder'] + "/Bench"
        np.save(path_bench + "/err_" + str(n) , error_v01)
        path_bench = kwargs['path_folder'] + "/1_n"
        np.save(path_bench + "/err_" + str(n) , error_v01_bis)
        path_bench = kwargs['path_folder'] + "/saga"
        np.save(path_bench + "/err_" + str(n) , error_v07)
        path_bench = kwargs['path_folder'] + "/pass"
        np.save(path_bench + "/err_" + str(n) , error_v010)

        print(' n is ' + str(n) + ' end ')
          
    Values_list.append(Values)
    Error_list.append(Error_val)
    
    
def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj())

def Load_np_mp_files(path_bench):
    Value_list = list()
    file_list = os.listdir(path_bench)
    for elt in file_list:
        Value_list.append(np.load(path_bench + '\\'+ elt))
    return np.array(Value_list)



#### Testing multiprocessing simple loop
if __name__ == '__main__':

    manager = multiprocessing.Manager()
    jobs = []
    Nb_simu = 1000  # 
    nb_jobs =  multiprocessing.cpu_count()
    step_job = (Nb_simu // nb_jobs)
    shared_list_res = manager.list()
    Values_list = manager.list()
    Error_list = manager.list()
    List_glob = manager.list()
    
    #pdb.set_trace()
    #start = 0
    #end = 1
    #Simple_loop(start,end,shared_list_res, Values_list, Error_list, Error, kwargs)
    
    # Step 1: Init multiprocessing.Pool()
    pool = multiprocessing.Pool(nb_jobs)
    
    # Step 2: `pool`     
    results = pool.starmap(Simple_loop, [(i*step_job,(i+1)*step_job,shared_list_res, Values_list, Error_list, Error, kwargs) for i in range(nb_jobs)])

    # Step 3: Don't forget to close
    pool.close() 
    
    #################################################################################
    #################################################################################
    ######################## This is the fourth main plot ###########################
    ######################## There is no other one below  ###########################
    #################################################################################
    #################################################################################
    
    ### Plot improvement v2
    #Values_all = np.array(shared_list_res)
    #size_values_all = int(nb_episode//freq_print) 
    #index = freq_print * np.arange(1,size_values_all+1)
    #if nb_episode % freq_print != 0:
        #val_index = np.array([nb_episode])
        #index = np.concatenate((index,val_index))  
    
    #start = 0    
    #option_save = ""
    #path_Image = Path_parent_directory + "\\Image"; ImageName = "\\improvement_optimal_placement_f_f_l" + str(NbSimu) + "_simu_f"
    #df = [ [index,np.log(Values_all[:,0,start:].mean(axis=0))],
           #[index,np.log(Values_all[:,1,start:].mean(axis=0))],
           #[index,np.log(Values_all[:,2,start:].mean(axis=0))],
           #[index,np.log(Values_all[:,3,start:].mean(axis=0))]]
    #labels = ["ste_cste", "1/n", "SAGA","PASS"]
    #mark = ['o', 'o', '*','<']
    #bbox_to_anchor_0 = (0.7,.75)
    #fig = plt.figure(figsize=(8,5))
    #pltg.Plot_plot(df,labels, xlabel ="Number of iterations", ylabel ="Log L2 - error",
                   #option=option_save, path =path_Image, ImageName=ImageName, Nset_tick_x = False, mark = mark, bbox_to_anchor_0 = bbox_to_anchor_0)
