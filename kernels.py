#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 07:44:57 2022

@author: vkiit
"""

import numpy as np
import torch

from pykeops.torch import Genred #, LazyTensor, generic_argmin

from skimage.transform import resize

import matplotlib.colors as mcolors
import matplotlib.pylab as plt

import time
import random
#from scipy import stats



## ADDITIONAL OT FUNCTIONS 

class kernel:
  def __init__(self,dim,**kwargs):
    if not 'cost' in kwargs:
      kwargs['cost'] = 'SqDist(x,y)'
    if not 'dtype' in kwargs:
      kwargs['dtype'] = "float64"
    self.cost = kwargs['cost']
    self.dtype = kwargs['dtype']
    self.dim = dim

  def diameter(self, *args):
    return Genred(self.cost, [ "x = Vi("+str(self.dim)+")", "y = Vj("+str(self.dim)+")"], reduction_op = "Max", dtype = self.dtype,  axis = 0)(*args).max()
  
  def marginal(self,*args, axis = 0, flag = True):
    if flag:
        variables = [ "x = Vi("+str(self.dim)+")", "y = Vj("+str(self.dim)+")", "reg_inv = Pm(1)", "log_a = Vi(1)", "log_b = Vj(1)", "f = Vi(1)", "g = Vj(1)"]
        formula  = "(-" + self.cost + "+f+g)*reg_inv + log_a + log_b" 
        return Genred(formula, variables, reduction_op = "LogSumExp", dtype = self.dtype,  axis=axis)(*args)
    else: 
        variables = [ "x = Vi("+str(self.dim)+")", "y = Vj("+str(self.dim)+")", "reg = Pm(1)", "u = Vi(1)", "v = Vj(1)"]
        formula  = "u*Exp(-" + self.cost + "/reg)*v" 
        return Genred(formula, variables, reduction_op = "Sum", dtype = self.dtype,  axis=axis)(*args)    

  def prod(self,*args, axis = 0):
    if axis==1: 
      variables = [ "x = Vi("+str(self.dim)+")", "y = Vj("+str(self.dim)+")", "reg = Pm(1)", "v = Vj(1)"]
      formula  = "Exp(-" + self.cost + "/reg)*v" 
      return Genred(formula, variables, reduction_op = "Sum", dtype = self.dtype,  axis=axis)(*args)
    else:
      variables = [ "x = Vi("+str(self.dim)+")", "y = Vj("+str(self.dim)+")", "reg = Pm(1)", "u = Vi(1)"]
      formula  = "u*Exp(-" + self.cost + "/reg)" 
      return Genred(formula, variables, reduction_op = "Sum", dtype = self.dtype,  axis=axis)(*args)

  def prod_stab(self,*args, axis = 0):
    if axis==1: 
      variables = [ "x = Vi("+str(self.dim)+")", "y = Vj("+str(self.dim)+")", "reg = Pm(1)", "g = Vj(1)", "w = Vj(1)"]
      formula  = "(-" + self.cost + "+g)/reg " 
      return (Genred(formula, variables, reduction_op = "Max_SumShiftExpWeight", dtype = self.dtype,  axis=axis, formula2 = "w")(*args))
    else:
      variables = [ "x = Vi("+str(self.dim)+")", "y = Vj("+str(self.dim)+")", "reg = Pm(1)", "f = Vi(1)", "w = Vi(1)"]
      formula  = "(-" + self.cost + "+f)/reg " 
      return (Genred(formula, variables, reduction_op = "Max_SumShiftExpWeight", dtype = self.dtype,  axis=axis, formula2 = "w")(*args))  
  
  def softmin(self, *args, axis = 0, weights = None):
    if axis==1: 
      if weights is None:
          variables = [ "x = Vi("+str(self.dim)+")", "y = Vj("+str(self.dim)+")", "reg_inv = Pm(1)", "g = Vj(1)"]
          formula  = "-" + self.cost + "*reg_inv + g"
          return -(Genred(formula, variables, reduction_op = "LogSumExp", dtype = self.dtype,  axis=axis)(*args))/args[2]
      else:
          variables = [ "x = Vi("+str(self.dim)+")", "y = Vj("+str(self.dim)+")", "reg_inv = Pm(1)", "g = Vj(1)", "r = Vi(1)", "w = Vj(1)" ]
          formula  = "-" + self.cost + "*reg_inv + g + r"
          formula2 = "w"
          return -(Genred(formula, variables, reduction_op = "LogSumExp", dtype = self.dtype,  axis=axis, formula2=formula2)(*args,weights))/args[2]
    else:
      if weights is None:
          variables = [ "x = Vi("+str(self.dim)+")", "y = Vj("+str(self.dim)+")", "reg_inv = Pm(1)", "f = Vi(1)"]
          formula  = "-" + self.cost + "*reg_inv + f"
          return -(Genred(formula, variables, reduction_op = "LogSumExp", dtype = self.dtype,  axis=axis)(*args))/args[2]
      else:
          variables = [ "x = Vi("+str(self.dim)+")", "y = Vj("+str(self.dim)+")", "reg_inv = Pm(1)", "f = Vi(1)",  "c = Vj(1)", "w = Vi(1)"]
          formula  = "-" + self.cost + "*reg_inv + f + c"
          formula2 = "w"
          return -(Genred(formula, variables, reduction_op = "LogSumExp", dtype = self.dtype,  axis=axis, formula2=formula2)(*args,weights))/args[2]

  def OTdist(self,*args): 
    variables = [ "x = Vi("+str(self.dim)+")", "y = Vj("+str(self.dim)+")", "reg = Pm(1)", "u = Vi(1)", "v = Vj(1)"]
    formula  = self.cost+"*u*Exp(-" + self.cost + "/reg)*v"
    return Genred(formula, variables, reduction_op = "Sum", dtype = self.dtype,  axis=0)(*args).sum()

  def OTdist_log(self,*args): 
    variables = [ "x = Vi("+str(self.dim)+")", "y = Vj("+str(self.dim)+")", "reg_inv = Pm(1)", "f = Vi(1)", "g = Vj(1)"]
    formula  = self.cost+"*Exp(-" + self.cost + "*reg_inv+f+g)"
    return Genred(formula, variables, reduction_op = "Sum", dtype = self.dtype,  axis=0)(*args).sum()
  

  def KLdist(self,*args): 
    variables = [ "x = Vi("+str(self.dim)+")", "y = Vj("+str(self.dim)+")", "reg = Pm(1)", "a = Vi(1)", "b = Vj(1)", "u = Vi(1)", "v = Vj(1)"]
    formula  = "a*u*Exp(-" + self.cost + "/reg)*b*v*(Log(u)+Log(v))- a*u*Exp(-" + self.cost + "/reg)*b*v + a*Exp(-" + self.cost + "/reg)*b"
    return Genred(formula, variables, reduction_op = "Sum", dtype = self.dtype,  axis=0)(*args).sum()  

  def ROTdist(self,*args): 
    variables = [ "x = Vi("+str(self.dim)+")", "y = Vj("+str(self.dim)+")", "reg = Pm(1)", "u = Vi(1)", "v = Vj(1)"]
    formula  = self.cost+"*u*Exp(-" + self.cost + "/reg)*v*reg*(Log(u)+Log(v))" 
    return Genred(formula, variables, reduction_op = "Sum", dtype = self.dtype,  axis=0)(*args).sum()


def cost_tensor(X, cost = 'SqDist(x,y)', gpu = True, dtype = torch.float64):   
    m = len(X)
    dev = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    if not torch.is_tensor(X[0]):
        X = list(torch.tensor(X[i], device = dev, dtype = dtype) for i in range(m))
    elif X[0].device.type == 'cpu' and dev.type == 'cuda':
        X = list(X[i].to(device = dev, dtype = dtype) for i in range(m))
    n = list(X[i].shape[0] for i in range(m))
    coord = torch.cartesian_prod( *(torch.arange(0,n[i], dtype = torch.uint8, device = dev) for i in range(m)))
    C = torch.zeros(n, dtype=torch.float64, device = dev)
    for k in range(len(coord)):
        coord_ = coord[k].tolist()
        if cost == 'Coulomb':
            c = torch.tensor(0,dtype = dtype, device = dev)
            for i in range(m):
                for j in range(i):
                    dX  = torch.norm(X[i][coord_[i],:]-X[j][coord_[j],:])
                    c += 1/dX if dX > 0. else float('inf')
        elif cost == 'RepulsiveHarmonic':
            c = torch.tensor(0,dtype = dtype, device = dev)
            for i in range(m):
                for j in range(m):
                    c-= ((X[i][coord_[i],:]-X[j][coord_[j],:])**2).sum()
        elif cost == 'Barycentric':
            Xbar = torch.zeros_like(X[0][0,:], dtype = dtype, device = dev)
            for i in range(m):
                Xbar += X[i][coord_[i]]
            Xbar /= m
            c = torch.tensor(0, dtype = dtype, device = dev)
            for i in range(m):
                c += ((X[i][coord_[i]]-Xbar)**2).sum()
            c /= m
        elif cost == 'SqDist(x,y)':
            c = torch.tensor(0,dtype = dtype, device = dev)
            for i in range(m):
                for j in range(i):
                    dX  = ((X[i][coord_[i],:]-X[j][coord_[j],:])**2).sum()
                    c += dX
        C[tuple(coord_)] += c
    
    return C