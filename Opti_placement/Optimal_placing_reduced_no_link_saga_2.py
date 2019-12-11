# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 23:26:06 2019

@author: othmane.mounjid
"""

### Load paths 
import os 
import sys
Path_parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,Path_parent_directory + "\\Plotting")

####### Import liraries 
import numpy as np 
import pandas as pd

####### RL approach
#### Step 1 : estimation of the error: auxiliary function
def error_1(bench,x):
    return np.linalg.norm(x -bench)

def prox_1(x,_eta = 0): ## projection function
  if x>=0:
      return x - _eta
  else:
      return x + _eta
  
###############################################################################
###############################################################################
########################################## Start RL functions #################
###############################################################################
###############################################################################

#### Reward after execution
def reward_exec(qsame, bb_pos, gain = 2, cost_out = -1, cost_stay = -0.5):
    if bb_pos ==  0: ## win if execution
        return gain
    elif bb_pos ==  -1: ## cost of a market order
        return cost_out
    else : ## cost of waiting
        return cost_stay

#### Function that encodes the new order book state after a market order
def State_after_market(Lob_state_0,reward_0,reward_exec=reward_exec):
    Lob_state_res = list(Lob_state_0)
    reward_res = 0 # float(reward_0)
    h_0_market = 0
    if (Lob_state_res[1] >= 1): 
        ### Regeneration 
        Lob_state_res[1] = -1
        Lob_state_res[0] =  max(Lob_state_res[0] - 1,0)
        reward_res = reward_exec(Lob_state_res[0],Lob_state_res[1])
    return [Lob_state_res,reward_res,h_0_market]

#### Inner loops  
###### Alg 2 inner loop : SAGA version test 2 : of the proximal gradient algorithm : 
def Lob_simu_inner_loop7(h_0,h_0_Stay,h_0_Mkt,h_0_past,h_0_past_stay,h_0_past_mkt,Lob_state_0,Intens_val,nb_iter,exp_mean,n_max, write_option = True, size_q = 2, gamma = 0.1, Error = None, reward_exec = None, nb_past = None, eta = 1, prox_fun = prox_1):
    cols_interest = ['Limit', 'Cancel', 'Market']
    Lob_state_before = list(Lob_state_0) #
    Lob_state = list(Lob_state_0) #
    if (nb_past is None) :
        nb_past = np.zeros((size_q,size_q+1)) # For bb_pos : index 0 is market order, index 1 is execution 

    if write_option:
        df_res = pd.DataFrame(np.zeros((nb_iter+1,3)), columns = ['BB size','BB pos','Event'])
        df_res.loc[0,:] = Lob_state + [None]
    intensities_values = np.zeros(3)
    h_0_cum = np.zeros((size_q,size_q+1))
    h_0_cum_stay = np.zeros((size_q,size_q+1)) ## compute in the same time the value of staying in the lob
    h_0_cum_mkt = np.zeros((size_q,size_q+1)) ## compute in the same time the value of a market order
    
    ## Main loop
    for n in range(nb_iter):# n = 0
        #### Get market decision
        Lob_state_before = list(Lob_state)
        index_row_b = Lob_state[0]
        intensities_values[:3] = Intens_val.loc[index_row_b, cols_interest]
        if (Lob_state[0] == 1) and (Lob_state[1] == 1): ## constraint cancellation impossible when i am the LAST one remaining in the queue
            intensities_values[1] = 0
        times = np.random.exponential(1/intensities_values)
        index_min = times.argmin()
        
        
        #### Apply market decision
        if index_min == 0: ## Buy Limit order
            Lob_state[0] = min(Lob_state[0] + 1,size_q-1)
            reward = reward_exec(Lob_state[0], Lob_state[1])
        elif index_min == 1: ## Buy Cancel order
            if (Lob_state[0] <= 1) and (Lob_state[1] <= 0) : # index_min = 1
                ### Regeneration 
                Lob_state[0] = max(Lob_state[0] - 1,0)
                reward = 0
            elif (Lob_state[0] > 1) and (Lob_state[1] <= Lob_state[0] - 1)  :
                Lob_state[0] -= 1
                reward = reward_exec(Lob_state[0], Lob_state[1])
            elif (Lob_state[0] > 1) and (Lob_state[1] == Lob_state[0]) :
                Lob_state[0] -= 1
                Lob_state[1] -= 1
                reward = reward_exec(Lob_state[0], Lob_state[1])
            else:
                raise Exception(" Error cancellation impossible")
        elif index_min == 2: ## Sell Market order
            if (Lob_state[1] == 1):
                Lob_state[0] = max(Lob_state[0] - 1,0)
                Lob_state[1] = 0
                reward = reward_exec(Lob_state[0], Lob_state[1])
            elif (Lob_state[1] > 1):
                Lob_state[0] -= 1   
                Lob_state[1] -= 1 
                reward = reward_exec(Lob_state[0], Lob_state[1])
            elif (Lob_state[1] <= 0):
                Lob_state[0] = max(Lob_state[0] - 1,0)
                reward = 0
            else:
                raise Exception(" Error Sell Market order impossible")
                    
        ### the value of stay decision : h_0 stay 
        if (Lob_state[1] <= 0):
            h_0_stay = 0
        else:
            h_0_stay = h_0[Lob_state[0],Lob_state[1]+1] # TRANSLATE BECAUSE OF -1 is an execution by a market order
        
        ### the value of market decision : h_0 market
        Lob_state_mkt,reward_mkt,h_0_market = State_after_market(Lob_state_before,reward,reward_exec=reward_exec)
        
        ### Update the values  h_0, h_0_cum, h_0_past and nb_past
        h_0_cum_stay[Lob_state_before[0],Lob_state_before[1]+1] = (reward + eta*h_0_stay - h_0_Stay[Lob_state_before[0],Lob_state_before[1]+1])
        h_0_cum_mkt[Lob_state_before[0],Lob_state_before[1]+1] = (reward_mkt + eta*h_0_market - h_0_Mkt[Lob_state_before[0],Lob_state_before[1]+1])
        h_0_cum[Lob_state_before[0],Lob_state_before[1]+1] = (max(reward_mkt + eta*h_0_market, reward + eta*h_0_stay) - h_0[Lob_state_before[0],Lob_state_before[1]+1])

        ## update h_0, 
        j = nb_past[Lob_state_before[0],Lob_state_before[1]+1] # np.random.randint(self._n_max)  
        nb_values = exp_mean[:].sum() #exp_mean.sum() #j + 1
        if j == 0:
            ### update h_0
            h_0[Lob_state_before[0],Lob_state_before[1]+1] += gamma*(h_0_cum[Lob_state_before[0],Lob_state_before[1]+1])
            h_0_Stay[Lob_state_before[0],Lob_state_before[1]+1] += gamma*(h_0_cum_stay[Lob_state_before[0],Lob_state_before[1]+1])
            h_0_Mkt[Lob_state_before[0],Lob_state_before[1]+1] += gamma*(h_0_cum_mkt[Lob_state_before[0],Lob_state_before[1]+1])
            ### update past
            h_0_past[Lob_state_before[0],Lob_state_before[1]+1,nb_past[Lob_state_before[0],Lob_state_before[1]+1]] = h_0_cum[Lob_state_before[0],Lob_state_before[1]+1] 
            h_0_past_stay[Lob_state_before[0],Lob_state_before[1]+1,nb_past[Lob_state_before[0],Lob_state_before[1]+1]] = h_0_cum_stay[Lob_state_before[0],Lob_state_before[1]+1] 
            h_0_past_mkt[Lob_state_before[0],Lob_state_before[1]+1,nb_past[Lob_state_before[0],Lob_state_before[1]+1]] = h_0_cum_mkt[Lob_state_before[0],Lob_state_before[1]+1] 
        elif j < n_max-1:
            r = np.random.randint(0,j)
            nb_values = exp_mean[:j].sum()#exp_mean.sum() #j + 1
            ### update h_0
            h_0[Lob_state_before[0],Lob_state_before[1]+1] += gamma*(h_0_cum[Lob_state_before[0],Lob_state_before[1]+1] - h_0_past[Lob_state_before[0],Lob_state_before[1]+1,r] + (((h_0_past[Lob_state_before[0],Lob_state_before[1]+1,:j]*exp_mean[:j]).sum())/(nb_values)))
            h_0_Stay[Lob_state_before[0],Lob_state_before[1]+1] += gamma*(h_0_cum_stay[Lob_state_before[0],Lob_state_before[1]+1] - h_0_past_stay[Lob_state_before[0],Lob_state_before[1]+1,r] + (((h_0_past_stay[Lob_state_before[0],Lob_state_before[1]+1,:j]*exp_mean[:j]).sum())/(nb_values)))
            h_0_Mkt[Lob_state_before[0],Lob_state_before[1]+1] += gamma*(h_0_cum_mkt[Lob_state_before[0],Lob_state_before[1]+1] - h_0_past_mkt[Lob_state_before[0],Lob_state_before[1]+1,r] + (((h_0_past_mkt[Lob_state_before[0],Lob_state_before[1]+1,:j]*exp_mean[:j]).sum())/(nb_values)))
            ### update past
            h_0_past[Lob_state_before[0],Lob_state_before[1]+1,nb_past[Lob_state_before[0],Lob_state_before[1]+1]] = h_0_cum[Lob_state_before[0],Lob_state_before[1]+1] 
            h_0_past_stay[Lob_state_before[0],Lob_state_before[1]+1,nb_past[Lob_state_before[0],Lob_state_before[1]+1]] = h_0_cum_stay[Lob_state_before[0],Lob_state_before[1]+1] 
            h_0_past_mkt[Lob_state_before[0],Lob_state_before[1]+1,nb_past[Lob_state_before[0],Lob_state_before[1]+1]] = h_0_cum_mkt[Lob_state_before[0],Lob_state_before[1]+1] 
        else:
            r = np.random.randint(0,n_max)
            nb_values = exp_mean[:].sum()#exp_mean.sum() #j + 1
            ### update h_0
            h_0[Lob_state_before[0],Lob_state_before[1]+1] += gamma*(h_0_cum[Lob_state_before[0],Lob_state_before[1]+1] - h_0_past[Lob_state_before[0],Lob_state_before[1]+1,r] + (((h_0_past[Lob_state_before[0],Lob_state_before[1]+1,:]*exp_mean[:]).sum())/(nb_values)))
            h_0_Stay[Lob_state_before[0],Lob_state_before[1]+1] += gamma*(h_0_cum_stay[Lob_state_before[0],Lob_state_before[1]+1] - h_0_past_stay[Lob_state_before[0],Lob_state_before[1]+1,r] + (((h_0_past_stay[Lob_state_before[0],Lob_state_before[1]+1,:]*exp_mean[:]).sum())/(nb_values)))
            h_0_Mkt[Lob_state_before[0],Lob_state_before[1]+1] += gamma*(h_0_cum_mkt[Lob_state_before[0],Lob_state_before[1]+1] - h_0_past_mkt[Lob_state_before[0],Lob_state_before[1]+1,r] + (((h_0_past_mkt[Lob_state_before[0],Lob_state_before[1]+1,:]*exp_mean[:]).sum())/(nb_values)))
            ### update past        
            h_0_past[Lob_state_before[0],Lob_state_before[1]+1,r] = h_0_cum[Lob_state_before[0],Lob_state_before[1]+1] 
            h_0_past_stay[Lob_state_before[0],Lob_state_before[1]+1,r] = h_0_cum_stay[Lob_state_before[0],Lob_state_before[1]+1] 
            h_0_past_mkt[Lob_state_before[0],Lob_state_before[1]+1,r] = h_0_cum_mkt[Lob_state_before[0],Lob_state_before[1]+1] 
            

        nb_past[Lob_state_before[0],Lob_state_before[1]+1] = min(nb_past[Lob_state_before[0],Lob_state_before[1]+1]+1,n_max-1)
        

        ### Write result
        if write_option:
            df_res.loc[n+1,:] = Lob_state + [index_min]
        
        if Lob_state_before[1]<= 0:
            raise Exception(" Error Lob_state_before[1]<= 0")

        ## When out of the domain reinject randomly :  normally it never happens it is just for safety
        if (Lob_state[1] <= 0):
            qsame = np.random.randint(1,size_q)
            bb_pos = np.random.randint(1,qsame+1)
            Lob_state_before = [qsame,bb_pos]
            Lob_state = list(Lob_state_before)
                
    error_val = Error(h_0)        
    if write_option:
        return {'lob':Lob_state, 'History':df_res ,'Result':[h_0,h_0_past,error_val,nb_past,h_0_Stay,h_0_Mkt]}
    else:
        return {'lob':Lob_state, 'History':pd.DataFrame() ,'Result':[h_0,h_0_past,error_val,nb_past,h_0_Stay,h_0_Mkt]}

#### Super loops
###### Alg 1 super loop : average value over the past observations : this function takes the inner loop function as an argument
def Lob_simu_super_Loop3(nb_episode,Intens_val,nb_iter,_exp_mean,n_max,inner_loop_func = Lob_simu_inner_loop7, gamma= 0.2,Lob_state_0=None,freq_print=100,size_q = 2,Error=None, write_option = False, h_0 = None, reward_exec = None, nb_past = None, eta = 1, h_0_Stay = None, h_0_Mkt = None, prox_fun =  prox_1, print_option =  True, pctg_0 = 0.1):
    size_mean = int(nb_episode//freq_print)
    if (h_0 is None) :
        h_0 = np.ones((size_q,size_q+1)) # For bb_pos : index 0 is market order, index 1 is execution 
    if (h_0_Stay is None) :
        h_0_Stay = np.ones((size_q,size_q+1)) # For bb_pos : index 0 is market order, index 1 is execution 
    if (h_0_Mkt is None) :
        h_0_Mkt = np.ones((size_q,size_q+1)) # For bb_pos : index 0 is market order, index 1 is execution 
    h_0_past = np.zeros((size_q,size_q+1,2))
    h_0_past_stay = np.zeros((size_q,size_q+1,n_max))
    h_0_past_mkt = np.zeros((size_q,size_q+1,n_max))
    nb_past = np.zeros((size_q,size_q+1),dtype = int)
    exp_mean = _exp_mean**(np.arange(n_max,0,-1))
    
    ## FINAL CONSTRAINT
    for qsame in range(size_q): # qsame is the size
        h_0[qsame,qsame+2:] = np.nan
        h_0[qsame,0] = reward_exec(qsame, -1)# market
        h_0[qsame,1] = reward_exec(qsame, 0)# execution
        h_0_Stay[qsame,qsame+2:] = np.nan
        h_0_Mkt[qsame,qsame+2:] = np.nan
        
        
    ## Errors
    error_within = np.zeros(freq_print) ## 
    error_within_estim = np.zeros(freq_print)
    count_within = 0
    count_reward = 0
#    mean_reward = np.zeros((size_mean,2))
#    mean_error_estim = np.zeros((size_mean,2))
#    
#    ## initialisation pass
#    gamma_0 = float(gamma)

    mean_reward = np.zeros((size_mean+1,2))
    mean_error_estim = np.zeros((size_mean+1,2))
    
    ## initialisation pass
    gamma_0 = float(gamma)
    ## initialisation of the error
    mean_reward[count_reward] = Error(h_0)
    count_reward += 1
    
    
    ### Parameters ; upper level 
    h_0_before = np.array(h_0)
    h_0_Stay_before = np.array(h_0_Stay)
    h_0_Mkt_before = np.array(h_0_Mkt) 
    h_0_past_before = np.array(h_0_past)
    count_period =0 
    
    for ep in range(nb_episode):
        h_0,h_0_past,error_val,nb_past,h_0_Stay,h_0_Mkt = inner_loop_func(h_0,h_0_Stay,h_0_Mkt,h_0_past,h_0_past_stay,h_0_past_mkt,Lob_state_0,Intens_val,nb_iter,exp_mean,n_max, write_option = write_option, size_q = size_q, gamma = gamma, Error = Error, reward_exec = reward_exec, nb_past = nb_past, eta = eta, prox_fun = prox_fun)['Result']
        error_within[count_within] = error_val
        error_within_estim[count_within] = np.linalg.norm(h_0_past)
        count_within += 1
        if (((ep %(freq_print))==freq_print-1) and (ep> 0)):
            if print_option:
                print(" frequency is : " + str(ep))
            mean_reward[count_reward] = (count_reward*freq_print,error_within.mean())
            mean_error_estim[count_reward] = (count_reward*freq_print,error_within_estim.mean())
            ## update gamma :
            index_count_before = max(count_reward-1,1)#0
            pctg_last = ((mean_error_estim[index_count_before,1] - mean_error_estim[count_reward,1])/mean_error_estim[index_count_before,1])
            if (pctg_last <= pctg_0) and (count_reward >= 1) and (count_period >= 3):
                h_0 = np.array(h_0_before)
                h_0_Stay = np.array(h_0_Stay_before)
                h_0_Mkt = np.array(h_0_Mkt_before) 
                h_0_past = np.array(h_0_past_before)                
                
                gamma_0 = max(gamma_0/2,0.01) # count_gamma_0 = 1
                count_period = 0
            else : 
                h_0_before = np.array(h_0)
                h_0_Stay_before = np.array(h_0_Stay)
                h_0_Mkt_before = np.array(h_0_Mkt)
                h_0_past_before = np.array(h_0_past)                
            error_within[:] = 0
            count_within = 0
            count_reward += 1
            count_period += 1
    if (count_within==0):
        return  [h_0,mean_reward,nb_past,h_0_Stay,h_0_Mkt]
    else:
        val = np.array((count_reward*freq_print,error_within[:count_within].mean())).reshape((-1,2))
        return [h_0,np.concatenate((mean_reward,val)),nb_past,h_0_Stay,h_0_Mkt]      

###############################################################################
###############################################################################
########################################## Theoretical solution functions #####
###############################################################################
###############################################################################

#### Compute the theoretical solution

#### Takes the infinitesimal generator and return the probability transition matrix and the average reward
def P_prob(Tilde_Q,z,Qmax0):
    size = Qmax0*Qmax0
    P = np.zeros((size,size))
    ## Zero diagonal terms
    diagonal_indexes = np.arange(size)
    idx_true = Tilde_Q[diagonal_indexes,diagonal_indexes] == 0
    idx_zero = diagonal_indexes[idx_true]
    P[idx_zero,idx_zero] = 1
    
    ## Non zero diagonal terms
    idx_nzero = diagonal_indexes[~idx_true] 
    P[idx_nzero] = (-Tilde_Q[idx_nzero]/Tilde_Q[idx_nzero,idx_nzero][:,None])
    P[idx_nzero,idx_nzero] = 0
    z[idx_nzero] = -z[idx_nzero]/Tilde_Q[idx_nzero,idx_nzero]

    return [P,z]

#### Build the infinitesimal generator  and reward for stay decision
def Build_Q_stay(Intens_val, size_q,reward_exec):
    tilde_size_q = size_q - 1
    size_Q_tilde = (tilde_size_q)*(tilde_size_q)
    Q_tilde = np.zeros((size_Q_tilde,size_Q_tilde))
    z_1 = np.zeros(size_Q_tilde)
    
    ## Build Q transition matrix 
    for qsame in range(tilde_size_q) : # QSame Loop // qsame =0
        for bb_pos in range(qsame+1) : # QOpp Loop // bb_pos = 0
            CumIntens = 0.
            ## Cancellation order bid side : 
            if (qsame > 0) and (bb_pos == qsame) : ## the limit is not totally consumed  // No regeneration
                 CumIntens +=  Intens_val.loc[qsame+1,'Cancel'] 
                 Q_tilde[qsame*tilde_size_q+bb_pos][(qsame-1)*tilde_size_q+bb_pos-1] += Intens_val.loc[qsame+1,'Cancel'] 
                 z_1[qsame*tilde_size_q+bb_pos] += Intens_val.loc[qsame+1,'Cancel']*reward_exec((qsame+1),bb_pos+1)
            elif (qsame > 0) and (bb_pos < qsame):
                 CumIntens +=  Intens_val.loc[qsame+1 ,'Cancel']
                 Q_tilde[qsame*tilde_size_q+bb_pos][(qsame-1)*tilde_size_q+bb_pos] += Intens_val.loc[qsame+1,'Cancel']
                 z_1[qsame*tilde_size_q+bb_pos] += Intens_val.loc[qsame+1,'Cancel']*reward_exec((qsame+1),bb_pos+1)

            ## Market order bid side : 
            if (bb_pos == 0) : ## the limit is not totally consumed  // No regeneration
                 CumIntens +=  Intens_val.loc[qsame+1 ,'Market'] 
                 z_1[qsame*tilde_size_q+bb_pos] += Intens_val.loc[qsame+1,'Market']*reward_exec((qsame+1),0)
            elif (bb_pos > 0):
                 CumIntens +=  Intens_val.loc[qsame+1 ,'Market']
                 Q_tilde[qsame*tilde_size_q+bb_pos][(qsame-1)*tilde_size_q+bb_pos-1] += Intens_val.loc[qsame+1,'Market'] 
                 z_1[qsame*tilde_size_q+bb_pos] += Intens_val.loc[qsame+1,'Market']*reward_exec((qsame+1),bb_pos)

                                   
            ## Insertion order bid side :
            if (qsame < size_q -2) : ## when qsame = Qmax -1  no more order can be added to the bid limit
                 CumIntens +=  Intens_val.loc[qsame+1,'Limit'] # IntensVal['lambdaIns'][qsame*Qmax0+qopp]  
                 Q_tilde[qsame*tilde_size_q+bb_pos][(qsame+1)*tilde_size_q+bb_pos] += Intens_val.loc[qsame+1,'Limit']            
                 z_1[qsame*tilde_size_q+bb_pos] += Intens_val.loc[qsame+1,'Limit']*reward_exec((qsame+1),bb_pos+1)
            
            
            ## Nothing happen 
            Q_tilde[qsame*tilde_size_q+bb_pos][qsame*tilde_size_q+bb_pos] += (- CumIntens ) 
    
    Q_tilde,z_1 = P_prob(Q_tilde,np.array(z_1),tilde_size_q) 
    return [Q_tilde,z_1]

#### Build the infinitesimal generator  and reward for market decision
def Build_Q_market(Intens_val, size_q,reward_exec):
    tilde_size_q = size_q - 1
    size_Q_tilde = (tilde_size_q)*(tilde_size_q)
    z_1 = np.zeros(size_Q_tilde)
    
    ## Build Q transition matrix 
    for qsame in range(tilde_size_q) : # QSame Loop // qsame =0
        z_1[qsame*tilde_size_q : qsame*tilde_size_q + (qsame+1)] = reward_exec(qsame+1, -1)
       
    return z_1

#### Computation of theo solution
def Sol_num_scheme(nb_iter,size_q,Intens_val,tol = 0.1,reward_exec_1=None):
    ## Build Q stay 
    Q_stay,z_stay = Build_Q_stay(Intens_val, size_q,reward_exec_1)
    ## Build z_market
    z_mkt = Build_Q_market(Intens_val, size_q,reward_exec_1)
    
    tilde_size_q = size_q - 1
    u_1 = -5*np.ones(tilde_size_q*tilde_size_q)
    u_next_1 = -5*np.ones(tilde_size_q*tilde_size_q)
    for qsame in range(tilde_size_q): # qsame is the size
        u_1[qsame*tilde_size_q+qsame+1:(qsame+1)*tilde_size_q] = 0
        u_next_1[qsame*tilde_size_q+qsame+1:(qsame+1)*tilde_size_q] = 0
    index_next_1 = np.zeros(tilde_size_q*tilde_size_q)
    n = 0
    error = 2*tol+1
    while (n <= nb_iter) and (error > tol):
        ### save old value ## debug
        u_1 = np.array(u_next_1)
        ### stay value 
        stay_u = Q_stay.dot(u_next_1) + z_stay
        ### market value
        market_u = z_mkt
        ### compute optimal value
        u_next_1 = np.maximum(stay_u, market_u)
        ### find optimal decision
        index_next_1 =  np.argmax([stay_u,market_u], axis = 0)
        ### Update error
        error = np.sqrt(np.nan_to_num(np.abs(u_1 - u_next_1)).max())
        n += 1

    ### Presentation of the results
    stay_u = Q_stay.dot(u_next_1) + z_stay 
    market_u = z_mkt
    for qsame in range(tilde_size_q): # qsame is the size
        stay_u[qsame*tilde_size_q+qsame+1:(qsame+1)*tilde_size_q] = np.nan
        market_u[qsame*tilde_size_q+qsame+1:(qsame+1)*tilde_size_q] = np.nan
        index_next_1[qsame*tilde_size_q+qsame+1:(qsame+1)*tilde_size_q] = -1
        u_next_1[qsame*tilde_size_q+qsame+1:(qsame+1)*tilde_size_q] = np.nan
    nb_col = 6
    nb_row = tilde_size_q*tilde_size_q
    df = pd.DataFrame(np.zeros((nb_row,nb_col)), columns = ['BB size','BB pos','Limit','Market','Value_opt','Decision'])
    df['BB size'] = np.repeat(np.arange(1,tilde_size_q+1),tilde_size_q)
    df['BB pos'] = np.tile(np.arange(1,tilde_size_q+1),tilde_size_q) 
    df['Limit'] = stay_u
    df['Market'] = market_u
    df['Value_opt'] = u_next_1
    df['Decision'] = index_next_1
    return df

#### Resize the result
def Read_h_0_theo(h,size_q,reward_exec_1):
    h_0_theo = np.ones((size_q,size_q+1)) # For bb_pos : index 0 is market order, index 1 is execution 
    h_0_theo[1:,2:] = np.nan_to_num(h.reshape((size_q-1,size_q-1)))
    ## FINAL CONSTRAINT
    for qsame in range(size_q): # qsame is the size
        h_0_theo[qsame,qsame+2:] = np.nan
        h_0_theo[qsame,0] = reward_exec_1(qsame, -1)# market
        h_0_theo[qsame,1] = reward_exec_1(qsame, 0)# execution
    return h_0_theo
     