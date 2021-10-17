# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 14:39:24 2021

@author: othma
"""

import numpy as np

## l2 norm between vectors
def error_1(bench,x):
    return np.linalg.norm(x -bench)

## projection function
def prox_1(x,_eta = 0): 
  if x>=0:
      return x - _eta
  else:
      return x + _eta
  
def prox_id(x):
  return x

def prox_2(x,_eta):
  return x/(1 + 2*_eta)


def choose_elt_rnd(size_values,val_min,step, param = {}):
    index = np.random.randint(0,size_values)
    return val_min + index * step