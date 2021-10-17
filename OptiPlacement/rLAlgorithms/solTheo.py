# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:08:08 2021

@author: othma
"""

####### Import liraries 
import numpy as np 
import pandas as pd

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
     