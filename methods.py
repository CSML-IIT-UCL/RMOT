#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 09:18:02 2021

@author: vkiit
"""

import numpy as np
import torch
import time

from kernels import kernel
from utils import distance, kshape, lse__


######## MAIN ALGORITHMS
     
def sinkhorn_dual(a, x, b, y, reg, K, metric = "kl", numItermax=10000, stopThr=1e-6, verbose=False,
               log=False, dist_log = False, gpu = True, dtype = "float64", q = 0.25):
    r"""

    ----------
  
    """
    dev = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    torchtype = torch.float32 if dtype == "float32" else torch.float64
    
    m = x.shape[0]
    n = y.shape[0]

    a = torch.as_tensor(a, dtype=torchtype, device = dev).reshape(m,1)
    log_a = a.log()
    x = torch.as_tensor(x, dtype=torchtype, device = dev)
    b = torch.as_tensor(b, dtype=torchtype, device = dev).reshape(n,1)
    log_b = b.log()
    y = torch.as_tensor(y, dtype=torchtype, device = dev)
    reg = torch.tensor([reg], dtype=torchtype, device=dev)
    reg_inv = 1/ reg
    
    diam_ = torch.tensor([K.diameter(x,y)], dtype=torchtype, device=dev)
    
    if q is not None and q<1 and q>0:
        numInnerItermax = int(np.log(reg/diam_)/np.log(q))
    else:
        q = None
        numInnerItermax = 1

    
    get_reg_inv = lambda n: (1/diam_) * (1/q)**n if q is not None and n < np.log((diam_*reg_inv).item() / 1.5) / np.log(1/q) else 1/reg # exponential decreasing
    
    reg_inv_ = get_reg_inv(0)
    
    f = torch.zeros_like(a, dtype=torchtype, device = dev)
    g = torch.zeros_like(b, dtype=torchtype, device = dev)
    
    
    f_ = K.softmin(x,y,reg_inv_,log_b+g*reg_inv_, axis = 1)
    d_log_r = reg_inv_*(f-f_)
    g_ = K.softmin(x,y,reg_inv_,log_a+f*reg_inv_, axis = 0)
    d_log_c =reg_inv_*(g-g_)
        
    viol_r = distance(a,-d_log_r, metric = 'kl', log = True, reduce=False)
    viol_c = distance(b,-d_log_c, metric = 'kl', log = True, reduce=False)    
    if metric == "kl":
        stopThr_val =  viol_r.sum() + viol_c.sum()
    else:
        stopThr_val =  distance(a,-d_log_r, metric = metric, log = True) + distance(b,-d_log_c, metric = metric, log = True)

    if log:     
        res = [stopThr_val.cpu()]
        log = dict()
        if dist_log:
            rot_dist = [K.ROTdist(x,y,reg,a*(f*reg_inv).exp(),b*(g*reg_inv).exp()).item()]
            ot_dist = [K.OTdist(x,y,reg,a*(f*reg_inv).exp(), b*(g*reg_inv).exp() ).item()]            
            log['ROTdist'] = rot_dist
            log['OTdist'] = ot_dist
        log['res'] = res
        start_time = time.perf_counter()
        if dev=='cuda':
            torch.cuda.synchronize()
        time_ = [start_time - start_time]
        log['time'] = time_ 
                
    it = 0
    for it in range(numItermax):
        if numInnerItermax>0:
            if 1/get_reg_inv(it) -reg > reg**2:
                reg_inv_ = get_reg_inv(it)
            else:
                reg_inv_ = reg_inv

        for iit in range(numInnerItermax+1):    
            
            f = K.softmin(x,y,reg_inv_,g*reg_inv_+ log_b, axis = 1)            
            g_ = K.softmin(x,y,reg_inv_,f*reg_inv_+ log_a, axis = 0)
                        
            d_log_c = reg_inv_*(g-g_)
            g = g_
            
            viol_c = distance(b,-d_log_c, metric = 'kl', log = True, reduce=False)
                       
            if metric == "kl":
                stopThr_val =  viol_c.sum()
            else:
                stopThr_val =  distance(b,-d_log_c, metric = metric, log = True)
                                    
                       
            if stopThr_val <= stopThr or np.isnan(stopThr_val):
                break          
        
        if numInnerItermax>1 and reg_inv_<reg_inv:    
            reg_inv_ = get_reg_inv(it+1)
            log_r = - K.softmin(x,y,reg_inv_,g*reg_inv_+ log_b, axis = 1)*reg_inv_ + f*reg_inv_ + log_a
            log_c = - K.softmin(x,y,reg_inv_,f*reg_inv_+ log_a, axis = 0)*reg_inv_ + g*reg_inv_ + log_b
            stopThr_val = distance(a,log_r-log_a, metric = metric, log = True) + distance(b,log_c-log_b, metric = metric, log = True)
   
        
        if verbose:
            print('{:5s}|{:12s} in metric {:6s}'.format('It.', 'Residual', metric ) + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(it, stopThr_val ))
        
        if log:
            if dist_log:
                rot_dist.append(K.ROTdist(x,y,reg,a*(f*reg_inv).exp(),b*(g*reg_inv).exp()).item())
                ot_dist.append(K.OTdist(x,y,reg,a*(f*reg_inv).exp(),b*(g*reg_inv).exp()).item())   
            res.append(stopThr_val.cpu())
            
            if dev=='cuda':
                torch.cuda.synchronize()
            time_.append(time.perf_counter() - start_time)   
        
        
        if np.isnan(stopThr_val) or (stopThr_val <= stopThr and reg_inv_-reg_inv <= stopThr):
            break
    
        
    if np.isnan(stopThr_val) or (stopThr_val > stopThr and stopThr>0.) or reg_inv_-reg_inv > stopThr:
            print('Warning: Algorithm Sinkhorn for eta={:1.2e} did not converge, current error is {:1.2e}'.format(reg.item(),stopThr_val))
        
    if log:
        if dist_log:
            log['ROTdist'] = rot_dist
            log['OTdist'] = ot_dist
        log['res'] = np.asarray(res)
        log['time'] = np.asarray(time_)
        return [f,g], log 
    else:
        return [f,g]


def sinkhorn(a, x, b, y, reg, K, metric = "kl", numItermax=10000, stopThr=1e-6, verbose=False,
               log=False, dist_log = False, gpu = True, dtype = "float64"):
    r"""


    References
    ----------
  
    """
    dev = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    torchtype = torch.float32 if dtype == "float32" else torch.float64
    
    m = x.shape[0]
    n = y.shape[0]

    a = torch.as_tensor(a, dtype=torchtype, device = dev).reshape(m,1)
    x = torch.as_tensor(x, dtype=torchtype, device = dev)
    b = torch.as_tensor(b, dtype=torchtype, device = dev).reshape(n,1)
    y = torch.as_tensor(y, dtype=torchtype, device = dev)
    reg = torch.tensor([reg], dtype=torchtype, device=dev)

    u = torch.ones_like(a, dtype=torchtype, device = dev)
    v = torch.ones_like(b, dtype=torchtype, device = dev)

    ku = K.prod(x,y,reg,a, axis = 0)
    kv = K.prod(x,y,reg,b, axis = 1) 
    
    print('initial error on rows: ',(K.marginal(x,y,reg,a,b, axis = 1, flag = False) - a).abs().sum().item())
    print('initial error on cols: ',(K.marginal(x,y,reg,a,b, axis = 0, flag = False) - b).abs().sum().item())

    if metric == "kl":
        viol1, viol2 = (a*(kv-1-kv.log())).squeeze(), (b*(ku-1-ku.log())).squeeze()
    elif ( metric == "Linf" or metric == "L1"):
        viol1, viol2 = ( a*(kv-1) ).abs().squeeze(), ( b*(ku-1) ).abs().squeeze()
    else:
        print('Unsupported metric, using kl!')
        viol1, viol2 = viol1, viol2 = (a*(kv-1-kv.log())).squeeze(), (b*(ku-1-ku.log())).squeeze()

    #stopThr_val = torch.max(torch.abs(r-a).sum(), torch.abs(c-b).sum())
    if metric == "Linf":
        stopThr_val =  torch.max(viol1.sum(),viol2.sum())
    else:
        stopThr_val =  viol1.sum()+ viol2.sum()

    if log:     
        res = [stopThr_val.cpu()]
        log = dict()
        if dist_log:
          rot_dist = [K.ROTdist(x,y,reg,a*u,b*v).item()]
          ot_dist = [K.OTdist(x,y,reg,a*u,b*v).item()]            
          log['ROTdist'] = rot_dist
          log['OTdist'] = ot_dist
        log['res'] = res
        start_time = time.perf_counter()
        if dev=='cuda':
            torch.cuda.synchronize()
        time_ = [start_time - start_time]
        log['time'] = time_ 
        

    kv = K.prod(x,y,reg,b*v, axis = 1) 
    for it in range(numItermax):
        
        
        u = 1 / kv 
        v = 1 / K.prod(x, y, reg, a*u, axis = 0) 
        
        # print('error on rows: ',(K.marginal(x,y,reg,a*u,b*v, axis = 1, flag = False) - a).abs().sum().item())
        # print('error on cols: ',(K.marginal(x,y,reg,a*u,b*v, axis = 0, flag = False) - b).abs().sum().item())
        
        kv = K.prod(x, y, reg, b*v, axis = 1)

        if ( metric == "Linf" or metric == "L1"):
            viol = ( a*(u*kv-1) ).squeeze().abs()
        else:
            viol = (a*(u*kv-1-kv.log()-u.log())).squeeze()
        
        stopThr_val = viol.squeeze().sum()

        if log:
          if dist_log:
            rot_dist.append(K.ROTdist(x,y,reg,a*u,b*v).item())
            ot_dist.append(K.OTdist(x,y,reg,a*u,b*v).item())   
          res.append(stopThr_val.cpu())
          if dev=='cuda':
            torch.cuda.synchronize()
          time_.append(time.perf_counter() - start_time)                       

        if verbose:
          print('{:5s}|{:12s} in metric {:6s}'.format('It.', 'Residual', metric ) + '\n' + '-' * 19)
          print('{:5d}|{:8e}|'.format(it, stopThr_val ))
    
        if stopThr_val <= stopThr:
          break

    if stopThr_val > stopThr and stopThr>0.:
        print('Warning: Algorithm Sinkhorn for eta={:1.2e} did not converge!'.format(reg.item()))



    if log:
      if dist_log:
        log['ROTdist'] = rot_dist
        log['OTdist'] = ot_dist
      log['res'] = np.asarray(res)
      log['time'] = np.asarray(time_)
      return [u*a,v*b],log
    else:
      return [u*a,v*b]

def batchGreenkhorn_dual(a, x, b, y, reg, K, batch = 0.25, metric = "kl", numItermax=1000, stopThr=1e-6, verbose=False,
               log=False, dist_log = False, gpu = True, dtype = "float64", q = 0.25):
    r"""

    References
    ----------
  
    """
    dev = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    torchtype = torch.float32 if dtype == "float32" else torch.float64
    
    m = x.shape[0]
    n = y.shape[0]
    
    if batch==0:
      tau1, tau2 = 1,1
    else:
      tau1, tau2 = int(batch*m), int(batch*n)
    bs1, bs2 = int(np.floor(m/tau1)), int(np.ceil(n/tau2))

    itStep = bs1+bs2 
    
    diam_ = torch.tensor([K.diameter(x,y)], dtype=torchtype, device=dev)
    
    if q is not None and q<1 and q>0:
        numInnerItermax = int(np.log(reg/diam_)/np.log(q))
    else:
        q = None
        numInnerItermax = 1

    numItermax *= itStep
    numInnerItermax *= itStep

    a = torch.as_tensor(a, dtype=torchtype, device = dev).reshape(m,1)
    log_a = a.log()
    x = torch.as_tensor(x, dtype=torchtype, device = dev)
    b = torch.as_tensor(b, dtype=torchtype, device = dev).reshape(n,1)
    log_b = b.log()
    y = torch.as_tensor(y, dtype=torchtype, device = dev)
    reg = torch.tensor([reg], dtype=torchtype, device=dev)
    reg_inv = 1/ reg
    
    
    get_reg_inv = lambda n: (1/diam_) * (1/q)**n if q is not None and n < np.log((diam_*reg_inv).item() / 1.5) / np.log(1/q) else 1/reg # geometric decrease
    
    reg_inv_ = get_reg_inv(0)
    
    
    f = torch.zeros_like(a, dtype=torchtype, device = dev)
    g = torch.zeros_like(b, dtype=torchtype, device = dev)
    
    
    f_ = K.softmin(x,y,reg_inv_,log_b+g*reg_inv_, axis = 1)
    d_log_r = reg_inv_*(f-f_)
    #f = f_    
    g_ = K.softmin(x,y,reg_inv_,log_a+f*reg_inv_, axis = 0)
    d_log_c = reg_inv_*(g-g_)
    #g = g_
    
    log_c = d_log_c + log_b
    log_r = d_log_r + log_a
    
    #print(-log_r + - K.softmin(x,y,reg_inv_,g*reg_inv_+ log_b, axis = 1)*reg_inv_ + f*reg_inv_ + log_a)
    #print(-log_c + - K.softmin(x,y,reg_inv_,f*reg_inv_+ log_a, axis = 0)*reg_inv_ + g*reg_inv_ + log_b)
    
    viol_r = distance(a,d_log_r, metric = 'kl', log = True, reduce=False)
    viol_c = distance(b,d_log_c, metric = 'kl', log = True, reduce=False)    
    if metric == "kl":
        stopThr_val =  viol_r.sum() + viol_c.sum()
    elif metric == "Linf":
        stopThr_val =  torch.max(distance(a,d_log_r, metric = metric, log = True),distance(b,d_log_c, metric = metric, log = True))
    else:
        stopThr_val =  distance(a,d_log_r, metric = metric, log = True) + distance(b,d_log_c, metric = metric, log = True)

    if log:     
        res = [stopThr_val.cpu()]
        log = dict()
        if dist_log:
            rot_dist = [K.ROTdist(x,y,reg,a*(f*reg_inv).exp(),b*(g*reg_inv).exp()).item()]
            ot_dist = [K.OTdist(x,y,reg,a*(f*reg_inv).exp(), b*(g*reg_inv).exp() ).item()]            
            log['ROTdist'] = rot_dist
            log['OTdist'] = ot_dist
        log['res'] = res
        start_time = time.perf_counter()
        if dev=='cuda':
            torch.cuda.synchronize()
        time_ = [start_time - start_time]
        log['time'] = time_ 

                
    for it in range(numItermax):

        for iit in range(numInnerItermax):
            
            viol_r_, i = torch.topk(viol_r,tau1, sorted = False)
            viol_c_, j = torch.topk(viol_c,tau2, sorted = False)
            
            if viol_r_.sum() >= viol_c_.sum():                
                f_ = K.softmin(x[i].contiguous(), y, reg_inv_, g*reg_inv_+ log_b, axis = 1)
                ipo = f_ > f[i]
                ine = f_ < f[i]
                w = ((f_*reg_inv_).exp()-(f[i]*reg_inv_).exp()).abs().log()                 
                if any(ipo):
                    delta_c_p = (g - K.softmin(kshape(x[i][ipo]),y,reg_inv_, kshape(w[ipo]) + kshape(log_a[i][ipo]), axis = 0))*reg_inv_ + log_b
                    log_c = lse__(log_c, delta_c_p)                
                if any(ine):
                    delta_c_n = (g- K.softmin(kshape(x[i][ine]),y,reg_inv_, kshape(w[ine]) + kshape(log_a[i][ine]),  axis = 0))*reg_inv_ + log_b 
                    log_c = lse__(log_c, delta_c_n, flag = True)
                
                f[i] = f_
                log_r[i] = log_a[i]
                viol_r[i] = 0
                viol_c = distance(b,-log_b + log_c, metric = 'kl', log = True, reduce=False)                 
                             
            else:
                g_ = K.softmin(x,y[j].contiguous(), reg_inv_, f*reg_inv_ + log_a, axis = 0)        
                jpo = g_ > g[j]
                jne = g_ < g[j]
                w = ((g_*reg_inv_).exp()-(g[j]*reg_inv_).exp()).abs().log()                 
                if any(jpo):
                    delta_r_p = (f - K.softmin(x,kshape(y[j][jpo]),reg_inv_, kshape(w[jpo]) + kshape(log_b[j][jpo]), axis = 1))*reg_inv_ + log_a
                    log_r = lse__(log_r, delta_r_p)                
                if any(jne):
                    delta_r_n = (f - K.softmin(x,kshape(y[j][jne]),reg_inv_, kshape(w[jne]) + kshape(log_b[j][jne]),  axis = 1))*reg_inv_ + log_a
                    log_r = lse__(log_r, delta_r_n, flag = True)
                
                
                g[j] = g_ 
                log_c[j] = log_b[j]
                viol_c[j] = 0
                
                viol_r = distance(a,-log_a + log_r, metric = 'kl', log = True, reduce=False)                 

                
            if metric == "kl":
                stopThr_val =  viol_c.sum() + viol_r.sum()
            elif metric=="Linf":
                stopThr_val =  torch.max(distance(a,-log_a+log_r, metric = metric, log = True),distance(b,-log_b+log_c, metric = metric, log = True))
            else:
                stopThr_val =  distance(b,-log_b+log_c, metric = metric, log = True) + distance(a,-log_a+log_r, metric = metric, log = True)
                       
            
            if (it+1)%itStep==0 and log:
                if dist_log:
                    rot_dist.append(K.ROTdist(x,y,reg,a*(f*reg_inv).exp(),b*(g*reg_inv).exp()).item())
                    ot_dist.append(K.OTdist(x,y,reg,a*(f*reg_inv).exp(),b*(g*reg_inv).exp()).item())   
                res.append(stopThr_val.cpu())
                
                if dev=='cuda':
                    torch.cuda.synchronize()
                time_.append(time.perf_counter() - start_time)                       

            if verbose:
                print('{:5s}|{:12s} in metric {:6s}'.format('It.', 'Residual', metric ) + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(it, stopThr_val ))
            
            
            if stopThr_val <= stopThr or np.isnan(stopThr_val):
                break
        
        if numInnerItermax>itStep and reg_inv_<reg_inv:    
            reg_inv_ = get_reg_inv(it+1)
            log_r = - K.softmin(x,y,reg_inv_,g*reg_inv_+ log_b, axis = 1)*reg_inv_ + f*reg_inv_ + log_a
            log_c = - K.softmin(x,y,reg_inv_,f*reg_inv_+ log_a, axis = 0)*reg_inv_ + g*reg_inv_ + log_b
            stopThr_val = distance(a,log_r-log_a, metric = metric, log = True) + distance(b,log_c-log_b, metric = metric, log = True)   
          
        
        if np.isnan(stopThr_val) or (stopThr_val <= stopThr and reg_inv_-reg_inv <= stopThr):
            break
        
    if np.isnan(stopThr_val) or (stopThr_val > stopThr and stopThr>0.) or reg_inv_-reg_inv > stopThr:
            print('Warning: Algorithm BatchGreenkhorn({:2.0%}) for eta={:1.2e} did not converge, current error is {:1.2e}'.format(batch, reg.item(),stopThr_val))
    
    
    if log:
        if dist_log:
            log['ROTdist'] = rot_dist
            log['OTdist'] = ot_dist
        log['res'] = np.asarray(res)
        log['time'] = np.asarray(time_)
        return [f,g], log 
    else:
        return [f,g]
    

def batchGreenkhorn(a, x, b, y, reg, K,  batch = 0.25, metric = "kl", stab = False, numItermax=10000, stopThr=1e-6, verbose=False,
                    
               log=False, dist_log = False, gpu = True, dtype = "float64"):
    r"""
    

    References
    ----------
  
    """
    dev = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    torchtype = torch.float32 if dtype == "float32" else torch.float64
    
    m = x.shape[0]
    d = x.shape[1]
    n = y.shape[0]

    a = torch.as_tensor(a, dtype=torchtype, device = dev).reshape(m,1)
    x = torch.as_tensor(x, dtype=torchtype, device = dev)
    b = torch.as_tensor(b, dtype=torchtype, device = dev).reshape(n,1)
    y = torch.as_tensor(y, dtype=torchtype, device = dev)
    reg = torch.tensor([reg], dtype=torchtype, device=dev)

    if batch==0:
      tau1, tau2 = 1,1
    else:
      tau1, tau2 = int(batch*m), int(batch*n)
    bs1, bs2 = int(np.floor(m/tau1)), int(np.ceil(n/tau2))

    itStep = bs1+bs2 

    numItermax *= itStep

    u = torch.ones_like(a, dtype=torchtype, device = dev)
    v = torch.ones_like(b, dtype=torchtype, device = dev)
 
    ku = K.prod(x,y,reg,a, axis = 0)
    kv = K.prod(x,y,reg,b, axis = 1)

    viol1, viol2 = (a*(kv-1-kv.log())).squeeze(), (b*(ku-1-ku.log())).squeeze()

    if (metric == "Linf"):
        stopThr_val =  torch.max((a*(1-kv)).squeeze().abs().sum(), (b*(1-ku)).squeeze().abs().sum())
    elif (metric == "L1"):
        stopThr_val =  (a*(1-kv)).squeeze().abs().sum() + (b*(1-ku)).squeeze().abs().sum()
    else:
        stopThr_val = viol1.sum()+viol2.sum()

    if log:
      res = [stopThr_val.cpu()]
      log = dict()
      log['res'] = res
      if dist_log:
        rot_dist = [K.ROTdist(x,y,reg,a*u,b*v).item()]
        ot_dist = [K.OTdist(x,y,reg,a*u,b*v).item()]            
        log['ROTdist'] = rot_dist
        log['OTdist'] = ot_dist
      start_time = time.perf_counter()
      if dev=='cuda':
          torch.cuda.synchronize()
      time_ = [start_time - start_time]
      log['time'] = time_ 

    if verbose:
        print('{:5s}|{:12s} in metric {:6s}'.format('It.', 'Residual', metric ) + '\n' + '-' * 19)
        print('{:5d}|{:8e}|'.format(0, stopThr_val ))
    
    for it in range(numItermax):
        # greedy choice
        viol1_, i = torch.topk(viol1,tau1, sorted = False)
        viol2_, j = torch.topk(viol2,tau2, sorted = False)    
          
        if viol1_.sum() >= viol2_.sum():
            old_ui = u.clone()[i]
            u[i] = 1 / kv[i]           
            ku += K.prod(x[i,:].contiguous(),y,reg,a[i]*(u[i]-old_ui).contiguous(), axis = 0)
            viol1[i] = 0
            viol2 = (b*(v*ku-1-ku.log()-v.log())).squeeze()              
        else:
            old_vj = v.clone()[j]
            v[j] = 1 / ku[j]
            kv += K.prod(x,y[j,:].contiguous(),reg,b[j]*(v[j]-old_vj).contiguous(), axis = 1)
            viol2[j] = 0
            viol1 = (a*(u*kv-1-kv.log()-u.log())).squeeze()

        if (metric == "Linf"):
            stopThr_val =  torch.max(( a*(u*kv-1) ).squeeze().abs().sum(), ( b*(v*ku-1) ).squeeze().abs().sum())
        elif (metric == "L1"):
            stopThr_val =  ( a*(u*kv-1) ).squeeze().abs().sum() + ( b*(v*ku-1) ).squeeze().abs().sum()
        else:
            stopThr_val = viol1.sum()+viol2.sum()

        if stab and (it % (itStep*int(100*reg)) == 0):
            kv = K.prod(x,y,reg,b*v, axis = 1) 
            ku = K.prod(x,y,reg,a*u, axis = 0)

        if (it+1)%itStep==0 and log:
          if dist_log:
            rot_dist.append(K.ROTdist(x,y,reg,a*u,b*v).item())
            ot_dist.append(K.OTdist(x,y,reg,a*u,b*v).item())   
          res.append(stopThr_val.cpu())
          if dev=='cuda':
            torch.cuda.synchronize()
          time_.append(time.perf_counter() - start_time)                        

        if verbose and (it+1) % (itStep) == 0:
            print('{:5s}|{:12s} in metric {:6s}'.format('It.', 'Residual', metric ) + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(it, stopThr_val ))
    
        if stopThr_val <= stopThr:
            break

    if stopThr_val > stopThr and stopThr>0.:
        print('Warning: Algorithm BatchGreenkhorn({:2.0%}) for eta={:1.2e} did not converge'.format(batch, reg.item()))
    
    if log:
      if dist_log:
        rot_dist.append(K.ROTdist(x,y,reg,a*u,b*v).item())
        ot_dist.append(K.OTdist(x,y,reg,a*u,b*v).item()) 
        log['ROTdist'] = rot_dist
        log['OTdist'] = ot_dist
      res.append(stopThr_val.cpu())
      log['res'] = np.asarray(res)
      if dev=='cuda':
        torch.cuda.synchronize()
      time_.append(time.perf_counter() - start_time)
      log['time'] = np.asarray(time_)
      return [u*a,v*b],log
    else:
      return [u*a,v*b]  
  

def sinkhorn_uniform_dual(x, y, reg, K, metric = "kl", numItermax=100, stopThr=1e-6, verbose=False,
               log=False, dist_log = False, gpu = True, dtype = "float64", q = 0.5):
    r"""

    References
    ----------
  
    """
    dev = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    torchtype = torch.float32 if dtype == "float32" else torch.float64
    
    m = x.shape[0]
    n = y.shape[0]

    a = torch.ones((m,1), dtype=torchtype, device = dev)/m
    log_a = a.log()
    x = torch.as_tensor(x, dtype=torchtype, device = dev)
    b = torch.ones((n,1), dtype=torchtype, device = dev)/n
    log_b = b.log()
    y = torch.as_tensor(y, dtype=torchtype, device = dev)
    reg = torch.tensor([reg], dtype=torchtype, device=dev)
    reg_inv = 1/ reg
    
    diam_ = torch.tensor([K.diameter(x,y)], dtype=torchtype, device=dev)
    
    if q>=1 or q<0:
        q = None
        numOuterIter =  1
    else:
        numOuterIter =  int(np.log((diam_*reg_inv).cpu().item()) / np.log(1/q))+1

    
    get_reg_inv = lambda n: ((1/diam_) * (1/q)**n, int((numItermax-1)*(1- q**n)+1)) if q is not None and n < numOuterIter-1 else (1/reg, numItermax)
    
    (reg_inv_, numItermax_) = get_reg_inv(0)
    
    f = torch.zeros_like(a, dtype=torchtype, device = dev)
    g = torch.zeros_like(b, dtype=torchtype, device = dev)
    
    
    f_ = K.softmin(x,y,reg_inv_,log_b+g*reg_inv_, axis = 1)
    d_log_r = reg_inv_*(f-f_)
    #f = f_    
    g_ = K.softmin(x,y,reg_inv_,log_a+f*reg_inv_, axis = 0)
    d_log_c =reg_inv_*(g-g_)
    #g = g_
    
    
    viol_r = distance(a,-d_log_r, metric = 'kl', log = True, reduce=False)
    viol_c = distance(b,-d_log_c, metric = 'kl', log = True, reduce=False)    
    if metric == "kl":
        stopThr_val =  viol_r.sum() + viol_c.sum()
    elif metric == "Linf":
        stopThr_val =  torch.max(distance(a,d_log_r, metric = metric, log = True),distance(b,d_log_c, metric = metric, log = True))
    else:
        stopThr_val =  distance(a,d_log_r, metric = metric, log = True) + distance(b,d_log_c, metric = metric, log = True)

    if log:     
        res = [stopThr_val.cpu()]
        log = dict()
        if dist_log:
            rot_dist = [((f*a).sum()+(b*g).sum() -reg*(d_log_r+log_a).exp().sum()-1).item()]
            #ot_dist = [K.OTdist(x,y,reg,a*(f*reg_inv).exp(), b*(g*reg_inv).exp() ).item()]            
            log['ROTdist'] = rot_dist
            #log['OTdist'] = ot_dist
        log['res'] = res
        start_time = time.perf_counter()
        if dev=='cuda':
            torch.cuda.synchronize()
        time_ = [start_time - start_time]
        log['time'] = time_ 
        
    it_total = 0            
    for it in range(numOuterIter):
        (reg_inv_, numItermax_) = get_reg_inv(it)
        for iit in range(numItermax_):    
            
            f = K.softmin(x,y,reg_inv_,g*reg_inv_+ log_b, axis = 1)            
            g_ = K.softmin(x,y,reg_inv_,f*reg_inv_+ log_a, axis = 0)
                        
            d_log_c = reg_inv_*(g-g_)
            g = g_
            
            viol_c = distance(b,-d_log_c, metric = 'kl', log = True, reduce=False)
                       
            if metric == "kl":
                stopThr_val =  viol_c.sum()
            elif metric == "Linf":
                stopThr_val =  distance(b,d_log_c, metric = metric, log = True)
            else:
                stopThr_val =  distance(b,d_log_c, metric = metric, log = True)
                        
            if log:
                if dist_log:
                    rot_dist.append( ((f*a).sum()+(b*g).sum() -reg*(d_log_r+log_a).exp().sum()-1).item() )
                    #ot_dist.append(K.OTdist(x,y,reg,a*(f*reg_inv).exp(),b*(g*reg_inv).exp()).item())   
                res.append(stopThr_val.cpu())
                
                if dev=='cuda':
                    torch.cuda.synchronize()
                time_.append(time.perf_counter() - start_time)                       
            
            it_total +=1
            if verbose:
                print('{:5s}|{:12s} in metric {:6s}'.format('It.', 'Residual', metric ) + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(it_total, stopThr_val ))
                
            if (stopThr_val <= stopThr or torch.isnan(stopThr_val)):
                break

        
    if torch.isnan(stopThr_val) or (stopThr_val > stopThr):
            print('Warning: Algorithm Dual Sinkhorn for eta={:1.2e} did not converge, current error is {:1.2e}'.format(reg.item(),stopThr_val))

    
    if log:
        if dist_log:
            log['ROTdist'] = rot_dist
            #log['OTdist'] = ot_dist
        log['res'] = np.asarray(res)
        log['time'] = np.asarray(time_)
        return [f,g], log 
    else:
        return [f,g]

    
def sinkhorn_uniform_log_2(x, y, reg, K, metric = "kl", numItermax=10000, stopThr=1e-6, verbose=False,
               log=False, dist_log = True, gpu = True, dtype = "float64", q = 0.9):
    r"""

    References
    ----------
  
    """
    dev = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    torchtype = torch.float32 if dtype == "float32" else torch.float64
    
    m = x.shape[0]
    n = y.shape[0]
    
    diam_ = torch.tensor([K.diameter(x,y)], dtype=torchtype, device=dev)
    
    numInnerItermax = int(np.log(reg.cpu()/diam_.cpu())/np.log(q))

    a = torch.ones((m,1), dtype=torchtype, device = dev)/m
    log_a = a.log()
    x = torch.as_tensor(x, dtype=torchtype, device = dev)
    b = torch.ones((n,1), dtype=torchtype, device = dev)/n
    log_b = b.log()
    y = torch.as_tensor(y, dtype=torchtype, device = dev)
    reg = torch.tensor([reg], dtype=torchtype, device=dev)
    reg_inv = 1/ reg
    
    
    get_reg_inv = lambda n: (1/diam_) * (1/q)**n if q is not None and n < numInnerItermax-1 else 1/reg # exponential decreasing
    
    reg_inv_ = get_reg_inv(0)
    
    f = torch.zeros_like(a, dtype=torchtype, device = dev)
    g = torch.zeros_like(b, dtype=torchtype, device = dev)
    
    
    f_ = K.softmin(x,y,reg_inv_,log_b+g*reg_inv_, axis = 1)
    d_log_r = reg_inv_*(f-f_)
    #f = f_    
    g_ = K.softmin(x,y,reg_inv_,log_a+f*reg_inv_, axis = 0)
    d_log_c =reg_inv_*(g-g_)
    #g = g_
    
    viol_r = distance(a,-d_log_r, metric = 'kl', log = True, reduce=False)
    viol_c = distance(b,-d_log_c, metric = 'kl', log = True, reduce=False)    
    
    print((d_log_r+log_a).exp().sum(), (d_log_c+log_b).exp().sum())
    if metric == "kl":
        stopThr_val =  viol_r.sum() + viol_c.sum()
    elif metric == "Linf":
        stopThr_val =  torch.max(distance(a,d_log_r, metric = metric, log = True),distance(b,d_log_c, metric = metric, log = True))
    else:
        stopThr_val =  distance(a,d_log_r, metric = metric, log = True) + distance(b,d_log_c, metric = metric, log = True)

    s0 = reg*(d_log_c-log_b).logsumexp(0)    

    if log:     
        res = [stopThr_val.cpu()]
        log = dict()
        if dist_log:
            rot_dist = [0]
            #ot_dist = [K.OTdist_log(x,y,reg_inv,f*reg_inv+log_a, g*reg_inv+log_b ).item()]            
            log['ROTdist'] = rot_dist
            #log['OTdist'] = ot_dist
        log['res'] = res
        start_time = time.perf_counter()
        if dev=='cuda':
            torch.cuda.synchronize()
        time_ = [start_time - start_time]
        log['time'] = time_ 
                
    for it in range(numInnerItermax+1):
        
        reg_inv_ = get_reg_inv(it)
        
        for iit in range(2):    
            
            f = K.softmin(x,y,reg_inv_,g*reg_inv_+ log_b, axis = 1)            
            g_ = K.softmin(x,y,reg_inv_,f*reg_inv_+ log_a, axis = 0)
                        
            d_log_c = reg_inv_*(g-g_)
            g = g_
            
            viol_c = distance(b,-d_log_c, metric = 'kl', log = True, reduce=False)
            
                       
            if metric == "kl":
                stopThr_val =  viol_c.sum()
            elif metric == "Linf":
                stopThr_val =  distance(b,d_log_c, metric = metric, log = True)
            else:
                stopThr_val =  distance(b,d_log_c, metric = metric, log = True)
            
                
            if log and iit==1:
                if dist_log:
                    rot_dist.append( ((f*a).sum()+(b*g).sum() - reg*(d_log_c-log_b).logsumexp(0) + s0).item())
                    #ot_dist.append( K.OTdist_log(x,y,reg_inv,f*reg_inv+log_a, g*reg_inv+log_b ).item() )    
                res.append(stopThr_val.cpu())
                
                if dev=='cuda':
                    torch.cuda.synchronize()
                time_.append(time.perf_counter() - start_time)                       
    
            if verbose:
                print('{:5s}|{:12s} in metric {:6s}'.format('It.', 'Residual', metric ) + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(it, stopThr_val ))
                      
       
    if log:
        if dist_log:
            log['ROTdist'] = rot_dist
            #log['OTdist'] = ot_dist
        log['res'] = np.asarray(res)
        log['time'] = np.asarray(time_)
        return [f,g], log 
    else:
        return [f,g]

    
def sinkhorn_uniform(x, y, reg, K, metric = "kl", numItermax=10000, stopThr=1e-6, verbose=False,
               log=False, dist_log = False, gpu = True, dtype = "float64", numInnerItermax = 0, reg0 = 1e4):
    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix
 
    The algorithm used is based on the paper
  
    The function solves the following optimization problem:
 
    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)
 
        s.t. \gamma 1 = a
 
             \gamma^T 1= b
 
             \gamma\geq 0
    where :
 
    - M is the (dim_a, dim_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (histograms, both sum to 1)
 
 
 
    Parameters
    ----------
    alpha : ndarray, shape (m,d+1) for d=1,2,3
        source distribution M samples from the ground space R^d, alpha[:,0] is the probability wigths 
    beta : ndarray, shape (n,d+1) for d=1,2,3
        target distribution M samples from the ground space R^d, alpha[:,0] is the probability wigths    
    reg : float
        Regularization term >0
    C : string using KeOps formulas, optional
        reprresenting cost function, eg. SqDist(x,y), (x|y), Acos((x|y)), ...
    batch: float
        % of the batch-sizes for 1st and 2nd marginal
    stab : bool, optional
        lod-domain stabilization of the iterates
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    log : bool, optional
        record log if True

 
    Returns
    -------
    gamma : ndarray, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
 
    Examples
    --------
 
    >>> alpha =[[.5, 0],[.5, 1]]
    >>> beta = [[.5, -1],[.5, 1]]
    >>> BatchGreenkhorn(alpha, beta, 1)


    References
    ----------
  
    """
    dev = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    torchtype = torch.float32 if dtype == "float32" else torch.float64
    
    m = x.shape[0]
    n = y.shape[0]

    x = torch.as_tensor(x, dtype=torchtype, device = dev)
    y = torch.as_tensor(y, dtype=torchtype, device = dev)
    reg = torch.tensor([reg], dtype=torchtype, device=dev)
    

    u = torch.ones((m,1), dtype=torchtype, device = dev)
    v = torch.ones((n,1), dtype=torchtype, device = dev)

    ku = K.prod(x,y,reg,u, axis = 0)
    kv = K.prod(x,y,reg,v, axis = 1) 

    if metric == "kl":
        viol1, viol2 = (kv/n-1-(kv/n).log()).squeeze()/m, (ku/m-1-(ku/m).log()).squeeze()/n
    elif ( metric == "Linf" or metric == "L1"):
        viol1, viol2 = (1-kv/n).abs().squeeze()/m, (1-ku/m).abs().squeeze()/n
    else:
        print('Unsupported metric, using kl!')
        viol1, viol2 =  (kv/n-1-(kv/n).log()).squeeze()/m, (ku/m-1-(ku/m).log()).squeeze()/n

    #stopThr_val = torch.max(torch.abs(r-a).sum(), torch.abs(c-b).sum())
    if metric == "Linf":
        stopThr_val =  torch.max(viol1.sum(),viol2.sum())
    else:
        stopThr_val =  viol1.sum()+ viol2.sum()

    if log:     
        res = [stopThr_val.cpu()]
        log = dict()
        if dist_log:
          rot_dist = [K.ROTdist(x,y,reg,u/m,v/n).item()]
          ot_dist = [K.OTdist(x,y,reg,u/m,v/n).item()]            
          log['ROTdist'] = rot_dist
          log['OTdist'] = ot_dist
        log['res'] = res
        start_time = time.perf_counter()
        if dev=='cuda':
            torch.cuda.synchronize()
        time_ = [start_time - start_time]
        log['time'] = time_ 

    kv = K.prod(x,y,reg, v, axis = 1) 
    for it in range(numItermax):
          
        u = n / kv 
        v = m / K.prod(x, y, reg, u, axis = 0) 
        
        kv = K.prod(x, y, reg, v, axis = 1)

        if ( metric == "Linf" or metric == "L1"):
            viol = ( u*kv/n-1 ).squeeze().abs()/m
        else:
            viol = (u*kv/n-1-(kv/n).log()-u.log()).squeeze() / m
        
        stopThr_val = viol.squeeze().sum()

        if log:
          if dist_log:
            rot_dist.append(K.ROTdist(x,y,reg,u/m,v/n).item())
            ot_dist.append(K.OTdist(x,y,reg,u/m,v/n).item())   
          res.append(stopThr_val.cpu())
          if dev=='cuda':
            torch.cuda.synchronize()
          time_.append(time.perf_counter() - start_time)                       

        if verbose:
          print('{:5s}|{:12s} in metric {:6s}'.format('It.', 'Residual', metric ) + '\n' + '-' * 19)
          print('{:5d}|{:8e}|'.format(it, stopThr_val ))
    
        if stopThr_val <= stopThr:
          break

    if stopThr_val > stopThr and stopThr>0.:
        print('Warning: Algorithm Sinkhorn for eta={:1.2e} did not converge!'.format(reg.item()))

    if log:
      if dist_log:
        log['ROTdist'] = rot_dist
        log['OTdist'] = ot_dist
      log['res'] = np.asarray(res)
      log['time'] = np.asarray(time_)
      return [u/m,v/n],log
    else:
      return [u/m,v/n]
  
def batchGreenkhorn_uniform_dual(x, y, reg, K, batch = 0.25, metric = "kl", numItermax=100, stopThr=1e-6, verbose=False,
               log=False, dist_log = False, gpu = True, dtype = "float64", q = 0.5):
    r"""

    References
    ----------
  
    """
    dev = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    torchtype = torch.float32 if dtype == "float32" else torch.float64
    
    m = x.shape[0]
    n = y.shape[0]
    
    if batch==0:
      tau1, tau2 = 1,1
    else:
      tau1, tau2 = int(batch*m), int(batch*n)
    bs1, bs2 = int(np.floor(m/tau1)), int(np.ceil(n/tau2))

    itStep = bs1+bs2 
    
    numItermax *= itStep

    a = torch.ones((m,1), dtype=torchtype, device = dev)/m
    log_a = a.log()
    x = torch.as_tensor(x, dtype=torchtype, device = dev)
    b = torch.ones((n,1), dtype=torchtype, device = dev)/n
    log_b = b.log()
    y = torch.as_tensor(y, dtype=torchtype, device = dev)
    reg = torch.tensor([reg], dtype=torchtype, device=dev)
    reg_inv = 1/ reg
    
    
    diam_ = torch.tensor([K.diameter(x,y)], dtype=torchtype, device=dev)
    
    if q>=1 or q<0:
        q = None
        numOuterIter =  1
    else:
        numOuterIter =  int(np.log((diam_*reg_inv).cpu().item()) / np.log(1/q))+1
    
    
    get_reg_inv = lambda n: ((1/diam_) * (1/q)**n, int((numItermax-1)*(1- q**n)+1)) if q is not None and n < numOuterIter-1 else (1/reg, numItermax)
    
    (reg_inv_, numItermax_) = get_reg_inv(0)
    
    
    f = torch.zeros_like(a, dtype=torchtype, device = dev)
    g = torch.zeros_like(b, dtype=torchtype, device = dev)
    
    
    f_ = K.softmin(x,y,reg_inv_,log_b+g*reg_inv_, axis = 1)
    d_log_r = reg_inv_*(f-f_)
    #f = f_    
    g_ = K.softmin(x,y,reg_inv_,log_a+f*reg_inv_, axis = 0)
    d_log_c = reg_inv_*(g-g_)
    #g = g_
    
    viol_r = distance(a,d_log_r, metric = 'kl', log = True, reduce=False)
    viol_c = distance(b,d_log_c, metric = 'kl', log = True, reduce=False)    
    if metric == "kl":
        stopThr_val =  viol_r.sum() + viol_c.sum()
    elif metric == "Linf":
        stopThr_val =  torch.max(distance(a,d_log_r, metric = metric, log = True),distance(b,d_log_c, metric = metric, log = True))
    else:
        stopThr_val =  distance(a,d_log_r, metric = metric, log = True) + distance(b,d_log_c, metric = metric, log = True)

    if log:     
        res = [stopThr_val.cpu()]
        log = dict()
        if dist_log:
            rot_dist = [ ((f*a).sum()+(b*g).sum() -reg*(d_log_r+log_a).exp().sum()-1).item() ]
            #ot_dist = [K.OTdist_log(x,y,reg_inv,f*reg_inv+log_a, g*reg_inv+log_b ).item()]            
            log['ROTdist'] = rot_dist
            #log['OTdist'] = ot_dist
        log['res'] = res
        start_time = time.perf_counter()
        if dev=='cuda':
            torch.cuda.synchronize()
        time_ = [start_time - start_time]
        log['time'] = time_ 
    
    it_total = 0
    for it in range(numOuterIter):
        
        (reg_inv_, numItermax_) = get_reg_inv(it)

        for iit in range(numItermax):
            
            viol_r_, i = torch.topk(viol_r,tau1, sorted = False)
            viol_c_, j = torch.topk(viol_c,tau2, sorted = False)
            
            if viol_r_.sum() >= viol_c_.sum():                
                f_ = K.softmin(x[i].contiguous(), y, reg_inv_, g*reg_inv_+ log_b, axis = 1)
                
                ipo = f_ > f[i]
                ine = f_ < f[i]
                w = ((f_*reg_inv_).exp()-(f[i]*reg_inv_).exp()).abs().log()   
                if any(ipo):
                    delta_c_p = - K.softmin(kshape(x[i],ipo),y,reg_inv_, kshape(w,ipo) + kshape(log_a[i],ipo), axis = 0)*reg_inv_ #(g - K.softmin(kshape(x[i][ipo]),y,reg_inv_, kshape(w[ipo]) + kshape(log_a[i][ipo]), axis = 0))*reg_inv_ + log_b
                    d_log_c = lse__(d_log_c, delta_c_p)                
                if any(ine):
                    delta_c_n = - K.softmin(kshape(x[i],ine),y,reg_inv_, kshape(w,ine) + kshape(log_a[i],ine),  axis = 0)*reg_inv_ #(g- K.softmin(kshape(x[i][ine]),y,reg_inv_, kshape(w[ine]) + kshape(log_a[i][ine]),  axis = 0))*reg_inv_ + log_b 
                    d_log_c = lse__(d_log_c, delta_c_n, flag = True)
                
                
                f[i] = f_
                d_log_r[i] = -f_*reg_inv_ 
                viol_r[i] = 0
                
                #log_c = - K.softmin(x,y,reg_inv_,f*reg_inv_+ log_a, axis = 0)*reg_inv_ + g*reg_inv_ + log_b
                #print('cols:',(-log_c  - K.softmin(x,y,reg_inv_,f*reg_inv_+ log_a, axis = 0)*reg_inv_ + g*reg_inv_ + log_b).norm(p=1))
                viol_c = distance(b, d_log_c+g*reg_inv_, metric = 'kl', log = True, reduce=False)  
                
                viol_c_sum = viol_c.sum()
                
                if np.isnan(viol_c_sum.cpu()):
                    f = K.softmin(x,y,reg_inv_,g*reg_inv_+ log_b, axis = 1)
                    d_log_r = - f*reg_inv_
                    viol_r = torch.zeros_like(viol_r)
                    g_ = K.softmin(x,y,reg_inv_,f*reg_inv_+ log_a, axis = 0)
                    d_log_c = - g_*reg_inv_ 
                    viol_c = distance(b,d_log_c+g*reg_inv_, metric = metric, log = True, reduce = False)
                             
            else:
                g_ = K.softmin(x,y[j].contiguous(), reg_inv_, f*reg_inv_ + log_a, axis = 0)
                
                jpo = g_ > g[j]
                jne = g_ < g[j]
                w = ((g_*reg_inv_).exp()-(g[j]*reg_inv_).exp()).abs().log()                 
                if any(jpo):
                    delta_r_p = - K.softmin(x,kshape(y[j],jpo),reg_inv_, kshape(w,jpo) + kshape(log_b[j],jpo), axis = 1)*reg_inv_ # (f - K.softmin(x,kshape(y[j][jpo]),reg_inv_, kshape(w[jpo]) + kshape(log_b[j][jpo]), axis = 1))*reg_inv_ + log_a
                    d_log_r = lse__(d_log_r, delta_r_p)                
                if any(jne):
                    delta_r_n = - K.softmin(x,kshape(y[j],jne),reg_inv_, kshape(w,jne) + kshape(log_b[j],jne),  axis = 1)*reg_inv_ #(f- K.softmin(x,kshape(y[j][jne]),reg_inv_, kshape(w[jne]) + kshape(log_b[j][jne]),  axis = 1))*reg_inv_ + log_a
                    d_log_r = lse__(d_log_r, delta_r_n, flag = True)                
                                
                g[j] = g_ 
                d_log_c[j] = -g_*reg_inv_ 
                viol_c[j] = 0
                
                #log_r = - K.softmin(x,y,reg_inv_,g*reg_inv_+ log_b, axis = 1)*reg_inv_ + f*reg_inv_ + log_a
                #print('rows:',(-log_r  - K.softmin(x,y,reg_inv_,g*reg_inv_+ log_b, axis = 1)*reg_inv_ + f*reg_inv_ + log_a).norm(p=1))
                viol_r = distance(a,d_log_r+f*reg_inv_, metric = 'kl', log = True, reduce=False)                 
                
                viol_r_sum = viol_r.sum()
                
                if np.isnan(viol_r_sum.cpu()):
                    g = K.softmin(x,y,reg_inv_,f*reg_inv_+ log_a, axis = 0)
                    d_log_c = - g*reg_inv_
                    viol_c = torch.zeros_like(viol_c)
                    f_ = K.softmin(x,y,reg_inv_,g*reg_inv_+ log_b, axis = 1)
                    d_log_r = - f_*reg_inv_ 
                    viol_r = distance(a,d_log_r+g*reg_inv_, metric = metric, log = True, reduce = False)
                    
                
            if metric == "kl":
                stopThr_val =  viol_c_sum + viol_r_sum
            elif metric=="Linf":
                stopThr_val =  torch.max(distance(a,d_log_r+f*reg_inv_, metric = metric, log = True),distance(b,d_log_c+g*reg_inv_, metric = metric, log = True))
            else:
                stopThr_val =  distance(a,d_log_r+f*reg_inv_, metric = metric, log = True) + distance(b,d_log_c+g*reg_inv_, metric = metric, log = True)
            it_total +=1
            if verbose:
                print('{:5s}|{:12s} in metric {:6s}'.format('It.', 'Residual', metric ) + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(it_total, stopThr_val ))
            
            if (it_total+1)%itStep==0 and log:
                if dist_log:
                    rot_dist.append( ((f*a).sum()+(b*g).sum() -reg*(d_log_r+log_a).exp().sum()-1).item())
                    #ot_dist.append( K.OTdist_log(x,y,reg_inv,f*reg_inv+log_a, g*reg_inv+log_b ).item() )    
                res.append(stopThr_val.cpu())
                
                if dev=='cuda':
                    torch.cuda.synchronize()
                time_.append(time.perf_counter() - start_time)   
            
            if (stopThr_val <= stopThr or torch.isnan(stopThr_val) ):
            #    print(f'          Stopped at {it} -> {iit+1} with current error {stopThr_val:.2e}')
                break
        
        if q is not None and reg_inv_<reg_inv:    
            (reg_inv_,_) = get_reg_inv(it+1)
            f = K.softmin(x,y,reg_inv_,g*reg_inv_+ log_b, axis = 1)
            d_log_r = - f*reg_inv_
            viol_r = torch.zeros_like(viol_r)
            g_ = K.softmin(x,y,reg_inv_,f*reg_inv_+ log_a, axis = 0)
            d_log_c = - g_*reg_inv_ 
            viol_c = distance(b,d_log_c+g*reg_inv_, metric = metric, log = True, reduce = False)
            stopThr_val = viol_c.sum()
        #print(f"Error for eta={1/reg_inv_.item():1.2e} at {it} is {stopThr_val:.2e}")
        #print(' ')     
              
        if torch.isnan(stopThr_val) or (stopThr_val <= stopThr and reg_inv_ == reg_inv):
            break
        
        
    if torch.isnan(stopThr_val) or (stopThr_val > stopThr) or reg_inv_-reg_inv > stopThr:
            print('Warning: Algorithm BatchGreenkhorn({:2.0%}) for eta={:1.2e} did not converge, current error is {:1.2e}'.format(batch, reg.item(),stopThr_val))
    
    #correction = (f.T@g).mean()
    #f -= correction
    #g += correction
    
    #print('error on rows: ',(K.marginal(x,y,reg_inv, log_a, log_b,f,g, axis = 1, flag = True).exp() - a).abs().sum().item())
    #print('error on cols: ',(K.marginal(x,y,reg_inv, log_a, log_b,f,g, axis = 0, flag = True).exp() - b).abs().sum().item())
    
    if log:
        if dist_log:
            log['ROTdist'] = rot_dist
            #log['OTdist'] = ot_dist
        log['res'] = np.asarray(res)
        log['time'] = np.asarray(time_)
        return [f,g], log 
    else:
        return [f,g]


def batchGreenkhorn_uniform(x, y, reg, K,  batch = 0.25, metric = "kl", stab = False, numItermax=10000, stopThr=1e-6, verbose=False,
                    
               log=False, dist_log = False, gpu = True, dtype = "float64"):
    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix
   
    The function solves the following optimization problem:
 
    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)
 
        s.t. \gamma 1 = a
 
             \gamma^T 1= b
 
             \gamma\geq 0
    where :
 
    - M is the (dim_a, dim_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (histograms, both sum to 1)
 
 
 
    Parameters
    ----------
    alpha : ndarray, shape (m,d+1) for d=1,2,3
        source distribution M samples from the ground space R^d, alpha[:,0] is the probability wigths 
    beta : ndarray, shape (n,d+1) for d=1,2,3
        target distribution M samples from the ground space R^d, alpha[:,0] is the probability wigths    
    reg : float
        Regularization term >0
    C : string using KeOps formulas, optional
        reprresenting cost function, eg. SqDist(x,y), (x|y), Acos((x|y)), ...
    batch: float
        % of the batch-sizes for 1st and 2nd marginal
    stab : bool, optional
        lod-domain stabilization of the iterates
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    log : bool, optional
        record log if True

 
    Returns
    -------
    gamma : ndarray, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
 
    Examples
    --------
 
    >>> alpha =[[.5, 0],[.5, 1]]
    >>> beta = [[.5, -1],[.5, 1]]
    >>> BatchGreenkhorn(alpha, beta, 1)


    References
    ----------
  
    """
    dev = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    torchtype = torch.float32 if dtype == "float32" else torch.float64
    
    m = x.shape[0]
    n = y.shape[0]

    x = torch.as_tensor(x, dtype=torchtype, device = dev)
    y = torch.as_tensor(y, dtype=torchtype, device = dev)
    reg = torch.tensor([reg], dtype=torchtype, device=dev)

    if batch==0:
      tau1, tau2 = 1,1
    else:
      tau1, tau2 = int(batch*m), int(batch*n)
    bs1, bs2 = int(np.ceil(m/tau1)), int(np.ceil(n/tau2))

    itStep = bs1+bs2 

    numItermax *= itStep

    u = torch.ones((m,1), dtype=torchtype, device = dev)
    v = torch.ones((n,1), dtype=torchtype, device = dev)
 
    ku = K.prod(x,y,reg,u, axis = 0)
    kv = K.prod(x,y,reg,v, axis = 1)

    viol1, viol2 = (kv/n-1-(kv/n).log()).squeeze()/m, (ku/m-1-(ku/m).log()).squeeze()/n
    
    if (metric == "Linf"):
        stopThr_val =  torch.max( (1-kv/n).abs().squeeze().sum()/m, (1-ku/m).abs().squeeze().sum()/n )
    elif (metric == "L1"):
        stopThr_val =  (1-kv/n).abs().squeeze().sum()/m + (1-ku/m).abs().squeeze().sum()/n
    else:
        stopThr_val = viol1.sum()+viol2.sum()

    if log:
      res = [stopThr_val.cpu()]
      log = dict()
      log['res'] = res
      if dist_log:
        rot_dist = [K.ROTdist(x,y,reg,u/m,v/n).item()]
        ot_dist = [K.OTdist(x,y,reg,u/m,v/n).item()]            
        log['ROTdist'] = rot_dist
        log['OTdist'] = ot_dist
      start_time = time.perf_counter()
      if dev=='cuda':
          torch.cuda.synchronize()
      time_ = [start_time - start_time]
      log['time'] = time_ 

    if verbose:
        print('{:5s}|{:12s} in metric {:6s}'.format('It.', 'Residual', metric ) + '\n' + '-' * 19)
        print('{:5d}|{:8e}|'.format(0, stopThr_val ))
    
    for it in range(numItermax):
        # greedy choice
        viol1_, i = torch.topk(viol1,tau1, sorted = False)
        viol2_, j = torch.topk(viol2,tau2, sorted = False)    
          
        if viol1_.sum() >= viol2_.sum():
            old_ui = u.clone()[i]
            u[i] = n / kv[i]           
            ku += K.prod(x[i,:].contiguous(),y,reg,(u[i]-old_ui).contiguous(), axis = 0)
            viol1[i] = 0
            viol2 = (v*ku/m-1-(ku/m).log()-v.log()).squeeze()/n              
        else:
            old_vj = v.clone()[j]
            v[j] = m / ku[j]
            kv += K.prod(x,y[j,:].contiguous(),reg,(v[j]-old_vj).contiguous(), axis = 1)
            viol2[j] = 0
            viol1 = (u*kv/n-1-(kv/n).log()-u.log()).squeeze()/m

        if (metric == "Linf"):
            stopThr_val =  torch.max( (1-u*kv/n).abs().squeeze().sum()/m, (1-v*ku/m).abs().squeeze().sum()/n )
        elif (metric == "L1"):
            stopThr_val =  (1-u*kv/n).abs().squeeze().sum()/m + (1-v*ku/m).abs().squeeze().sum()/n
        else:
            stopThr_val = viol1.sum()+viol2.sum()

        if stab and (it % (itStep*int(100*reg)) == 0):
            kv = K.prod(x,y,reg,v, axis = 1) 
            ku = K.prod(x,y,reg,u, axis = 0)

        if (it+1)%itStep==0 and log:
          if dist_log:
            rot_dist.append(K.ROTdist(x,y,reg,u/m,v/n).item())
            ot_dist.append(K.OTdist(x,y,reg,u/m,v/n).item())   
          res.append(stopThr_val.cpu())
          if dev=='cuda':
            torch.cuda.synchronize()
          time_.append(time.perf_counter() - start_time)                        

        if verbose and (it+1) % (itStep) == 0:
            print('{:5s}|{:12s} in metric {:6s}'.format('It.', 'Residual', metric ) + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(it, stopThr_val ))
    
        if stopThr_val <= stopThr:
            break

    if stopThr_val > stopThr and stopThr>0.:
        print('Warning: Algorithm BatchGreenkhorn({:2.0%}) for eta={:1.2e} did not converge'.format(batch, reg.item()))
    
    if log:
      if dist_log:
        rot_dist.append(K.ROTdist(x,y,reg,u/m,v/n).item())
        ot_dist.append(K.OTdist(x,y,reg,u/m,v/n).item()) 
        log['ROTdist'] = rot_dist
        log['OTdist'] = ot_dist
      res.append(stopThr_val.cpu())
      log['res'] = np.asarray(res)
      if dev=='cuda':
        torch.cuda.synchronize()
      time_.append(time.perf_counter() - start_time)
      log['time'] = np.asarray(time_)
      return [u/m,v/n],log
    else:
      return [u/m,v/n]    
    
def multiBatchGreenkhorn(A, M, reg = 1, tau = 0, metric = "kl", numItermax=10000, stopThr=1e-6, verbose=False,
               log=False, gpu = True, to_numpy = False, numInnerItermax = 0, reg0 = 1e4):
    r"""
    Solve the entropic regularization multi-marginal optimal transport problem and return the optimal scalings v of the MOT tensor
 
    The algorithm used is based on the paper:
 
    The function solves the following optimization problem:
 
    .. math::
        \pi = arg\min_\pi <\pi,M>_F + reg\cdot\Omega(\pi)
 
        s.t. R_i(\pi) = a_i, i = 1,2,...,m
 
    where :
    - R_i is i-th marginal of tensor \gamma
    - M is the (dim_a_1,..., dim_a_m) metric cost tensor
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\pi)=\sum_{i_1,...,i_m} \pi_{i_1,...,i_m}(\log(\pi_{i_1,...,i_m})-1)`
    - a_i's are histograms (all sum to 1) for i=1,2,...,m
 
 
 
    Parameters
    ----------
    A : list of length m of ndarrays a_i of the shape (dim_a_i,) i=1,..., m
        samples weights in the source domain
    M : tensor, shape (dim_a_1,..., dim_a_m)
        loss matrix
    reg : float
        Regularization term >0
    tau : float in [0,1] that for tau > 0 batch_k = ceiling(tau * dim_a_k), while if tau = 0 batch = 1
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    log : bool, optional
        record log if True
 
    Returns
    -------
    gamma : tensor, shape (dim_a_1,..., dim_a_m)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
 
    Examples
    --------
 
    >>> import tensorflow as tf
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> BatchGreenkhorn([a,b], M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])
 
    """
    dev = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

    M = torch.as_tensor(M, dtype=torch.float64, device = dev)
    reg = torch.as_tensor(reg, dtype=torch.float64, device = dev)

    m = len(M.shape)

    # strings and shapes for Einstein summation 
    ltrs = 'abcdefghijklmnopqrstuvwxyz'
    ltrs = ltrs[0:m]
    scaling_str = ltrs
    sum_str = ltrs[0:m-1]
    shape = list(slice(None, None, None) for _ in range(m))    
    for i in range(m):
        scaling_str += ','+ ltrs[i]
        if i<m-1:
            sum_str += ','+ ltrs[i]
    scaling_str += '->'
    sum_str += '->'+ltrs[0:m-1]

    

    if len(A) == 0:
        for k in range(m):
            A[k] = torch.ones((M.shape[k],), dtype=torch.float64, device = dev) / M.shape[k]
    
    for i in range(m):
        A[i] = torch.as_tensor(A[i], dtype=torch.float64, device = dev)
    dim_A = list( A[i].shape[0] for i in range(m) )        
    V = list( torch.ones(dim_A[i], dtype=torch.float64, device = dev) for i in range(m))
    V0 = list( A[i] for i in range(m))

    # batch sizes 
    if tau == 0:
        batch = list(1 for _ in range(m))
        itStep = np.sum(dim_A) 
    else:
        batch = list(int(dim_A[i]*tau)  for i in range(m))
        itStep = np.sum(list(int(np.ceil(dim_A[i]/batch[i]))  for i in range(m)))
        
    numItermax *= itStep

    # Kernel formation K = np.exp(-M/reg)
    K = torch.empty_like(M, device = dev)
    torch.divide(M, -reg, out=K)
    K = torch.where(K>-np.inf,torch.exp(K), 0.)
    #torch.exp(K, out=K)
    K = torch.einsum(scaling_str+ltrs,K,*(V0[t] for t in range(m)))

    # initial marginals, violations KL(A[k],R[k]) and l_1 distance to transport polytope
    R = []
    viol = []
    dist = torch.zeros((m), dtype=torch.float64, device = dev)
    for i in range(m):
        R.append( torch.einsum(scaling_str+ltrs[i],K,*(V[t] for t in range(m)) )) 
        viol.append( A[i]*torch.log(A[i]/R[i])-A[i]+R[i] ) 
        dist[i] = torch.abs(A[i]-R[i]).sum()
 
    if metric == 'Linf': 
        stopThr_val = torch.max(dist) 
    elif metric=='L1':
        stopThr_val = torch.sum(dist) 
    else:
        stopThr_val = np.asarray(list(viol[i].sum() for i in range(m))).sum()
 
    if log:
        res = [np.float64(stopThr_val)]
        log = dict()
        log['res'] = res
        start_time = time.perf_counter()
        if dev=='cuda':
            torch.cuda.synchronize()
        time_ = [start_time - start_time]
        log['time'] = time_ 

 
    for it in range(numItermax):

        # greedy choice
        #k = m
        #viol_, L = torch.topk(viol[m],batch[m], sorted = False)
        viol_max = -1. #viol_.sum()
        for i in range(m):
          viol_, ind_ = torch.topk(viol[i],batch[i], sorted = False)
          viol_sum = viol_.sum()
          if viol_sum >= viol_max:
              viol_max = viol_sum
              k = i
              L = ind_
 
        old_V = torch.clone(V[k][L])
        V[k][L] *= A[k][L] / R[k][L]
        R[k][L] = A[k][L]
        viol[k][L] = torch.zeros((batch[k]), dtype=torch.float64, device = dev)
        dist[k] = torch.abs(A[k]-R[k]).sum()

        shape[k] = L
        Kp = torch.einsum(scaling_str + ltrs[0:k]+ltrs[k+1:m],K[shape], *(V[t] for t in range(k)),V[k][L]- old_V,*(V[t] for t in range(k+1,m)) )
        shape[k] = slice(None,None,None)

        for i in range(k):
            dR = torch.einsum(ltrs[0:m-1]+'->'+ltrs[i],Kp); 
            R[i] += dR
            viol[i] = A[i]*torch.log(A[i]/R[i])-A[i]+R[i] 
            dist[i]=torch.abs(A[i]-R[i]).sum() 

        for i in range(k+1,m):
            dR = torch.einsum(ltrs[0:m-1]+'->'+ltrs[i-1],Kp); 
            R[i] += dR
            viol[i] = A[i]*torch.log(A[i]/R[i])-A[i]+R[i] 
            dist[i]=torch.abs(A[i]-R[i]).sum() 

        if metric == 'Linf': 
            stopThr_val = torch.max(dist) 
        elif metric=='L1':
            stopThr_val = torch.sum(dist) 
        else:
            stopThr_val = np.asarray(list(viol[i].sum() for i in range(m))).sum()

        if (it+1)%itStep==0 and log:
            res.append(np.float64(stopThr_val))
            if dev=='cuda':
              torch.cuda.synchronize()
            time_.append(time.perf_counter() - start_time)
                       

        if verbose:
                if it % (itStep*1) == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(it, stopThr_val ))
        
        if stopThr_val <= stopThr:
            break
        
    else:
        print('Warning: Algorithm BatchGreenkhorn({:2.0%}) for eta={:1.2e} did not converge'.format(tau, reg.item()))
 
    if log:
        if ~((it+1)%itStep==0):
          res.append(np.float64(stopThr_val))        
          if dev=='cuda':
            torch.cuda.synchronize()
          time_.append(time.perf_counter() - start_time)
        log['res'] = np.asarray(res)
        log['time'] = np.asarray(time_)    

    if to_numpy:
        if log:
            #return torch.einsum(scaling_str+ltrs,K,*(V[t] for t in range(m))).cpu().numpy(), log
            return list(V[t].cpu().numpy() for t in range(m)), log
        else:
            return torch.einsum(scaling_str+ltrs,K,*(V[t] for t in range(m))).cpu().numpy()
    else:
        if log:
            #return torch.einsum(scaling_str+ltrs,K,*(V[t] for t in range(m))), log
            return list(V[t] for t in range(m)), log
        else:
            return torch.einsum(scaling_str+ltrs,K,*(V[t] for t in range(m)))
    
def multiSinkhorn(A, M, reg = 1, metric = 'kl', numItermax=10000, stopThr=1e-6, verbose=False, log=False, gpu = True, to_numpy = False, numInnerItermax = 0, reg0 = 1e4):
    r"""
    Solve the entropic regularization multi-marginal optimal transport problem and return the optimal scalings v of the MOT tensor
 
    The algorithm used is based on the paper:
 
    The function solves the following optimization problem:
 
    .. math::
        \pi = arg\min_\pi <\pi,M>_F + reg\cdot\Omega(\pi)
 
        s.t. R_i(\pi) = a_i, i = 1,2,...,m
 
    where :
    - R_i is i-th marginal of tensor \gamma
    - M is the (dim_a_1,..., dim_a_m) metric cost tensor
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\pi)=\sum_{i_1,...,i_m} \pi_{i_1,...,i_m}(\log(\pi_{i_1,...,i_m})-1)`
    - a_i's are histograms (all sum to 1) for i=1,2,...,m
 
 
 
    Parameters
    ----------
    A : list of length m of ndarrays a_i of the shape (dim_a_i,) i=1,..., m
        samples weights in the source domain
    M : tensor, shape (dim_a_1,..., dim_a_m)
        loss matrix
    reg : float
        Regularization term >0
    tau : float in [0,1] that for tau > 0 batch_k = ceiling(tau * dim_a_k), while if tau = 0 batch = 1
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    log : bool, optional
        record log if True
 
    Returns
    -------
    gamma : tensor, shape (dim_a_1,..., dim_a_m)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
 
    Examples
    --------
 
    >>> import tensorflow as tf
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> BatchGreenkhorn([a,b], M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])
 
    """
    dev = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

    M = torch.as_tensor(M, dtype=torch.float64, device = dev)
    reg = torch.as_tensor(reg, dtype=torch.float64, device = dev)
    
    get_reg = lambda n: (reg0 - reg) * np.exp(-n) + reg if not numInnerItermax==0 else reg # exponential decreasing

    m = len(M.shape)
    
    numItermax *= m 

    # strings for Einstein summation 
    ltrs = 'abcdefghijklmnopqrstuvwxyz'
    ltrs = ltrs[0:m]
    scaling_str = ltrs
    sum_str = ltrs[0:m-1]
    for i in range(m):
        scaling_str += ','+ ltrs[i]
        if i<m-1:
            sum_str += ','+ ltrs[i]
    scaling_str += '->'
    sum_str += '->'+ltrs[0:m-1]

    if len(A) == 0:
        A = list( torch.ones((M.shape[k],), dtype=torch.float64, device = dev) / M.shape[k] for k in range(m))
    
    dim_A = []
    V = []
    for k in range(m):
        A[k] = torch.as_tensor(A[k], dtype=torch.float64, device = dev)
        dim_A.append(A[k].shape[0])        
        V.append(torch.ones(dim_A[k], dtype=torch.float64, device = dev))
    V0 = list( A[i] for i in range(m))

    # Kernel formation K = np.exp(-M/reg)
    K = torch.empty_like(M, device = dev)
    torch.divide(M, -reg, out=K)
    K = torch.where(K>-np.inf,torch.exp(K), 0.)
    #torch.exp(K, out=K)
    K = torch.einsum(scaling_str+ltrs,K,*(V0[t] for t in range(m)))

    # initial marginals, violations KL(A[k],R[k]) and l_inf-l_1 distance to transport polytope
    R = []
    viol = []
    dist = torch.zeros((m), dtype=torch.float64, device = dev)
    for i in range(m):
        R.append( torch.einsum(scaling_str+ltrs[i],K,*(V[t] for t in range(m)) )) 
        viol.append( A[i]*torch.log(A[i]/R[i])-A[i]+R[i] ) 
        dist[i] = torch.abs(A[i]-R[i]).sum()
 
    if metric == 'Linf': 
        stopThr_val = torch.max(dist) 
    elif metric=='L1':
        stopThr_val = torch.sum(dist) 
    else:
        stopThr_val = np.asarray(list(viol[i].sum() for i in range(m))).sum()
    
    if log:
        res = [np.float64(stopThr_val)]
        log = dict()
        log['res'] = res
        start_time = time.perf_counter()
        if dev=='cuda':
            torch.cuda.synchronize()
        time_ = [start_time - start_time]
        log['time'] = time_ 
 
    for it in range(numItermax): 

        viol_max = -1.
        # greedy choice
        for i in range(m):
          viol_sum = viol[i].sum() 
          if viol_sum >= viol_max:
              viol_max = viol_sum
              k = i

        V[k] *= A[k] / R[k]
        R[k] = A[k]
        viol[k] = torch.zeros((dim_A[i]), dtype=torch.float64, device = dev)
        dist[k] = torch.zeros((1), dtype=torch.float64, device = dev)

        Kp = torch.einsum(scaling_str +ltrs[0:k]+ltrs[k+1:m], K, *(V[t] for t in range(m) ))

        for i in range(k): 
            R[i] = torch.einsum(ltrs[0:m-1]+'->'+ltrs[i],Kp)
            viol[i] = A[i]*torch.log(A[i]/R[i])-A[i]+R[i] 
            dist[i] = torch.abs(A[i]-R[i]).sum()

        for i in range(k+1,m):
            R[i] = torch.einsum(ltrs[0:m-1]+'->'+ltrs[i-1],Kp)
            viol[i] = A[i]*torch.log(A[i]/R[i])-A[i]+R[i] 
            dist[i]=torch.abs(A[i]-R[i]).sum()
        
        if metric == 'Linf': 
            stopThr_val = torch.max(dist) 
        elif metric=='L1':
            stopThr_val = torch.sum(dist) 
        else:
            stopThr_val = np.asarray(list(viol[i].sum() for i in range(m))).sum()

        if (it+1)%m==0 and log:
            res.append(np.float64(stopThr_val))
            if dev=='cuda':
              torch.cuda.synchronize()
            time_.append(time.perf_counter() - start_time)                        
  
        if verbose:
            print(
                '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(it, stopThr_val ))
        
        if stopThr_val <= stopThr:
            break
    else:
        print('Warning: Algorithm MultiSinkhorn for eta={:1.2e} did not converge!'.format(reg.item()))
 
    if log:
        if ~((it+1)%m==0):
            res.append(np.float64(stopThr_val))  
            if dev=='cuda':
              torch.cuda.synchronize()
            time_.append(time.perf_counter() - start_time)
        log['res'] = np.asarray(res)
        log['time'] = np.asarray(time_)

    if to_numpy:
        if log:
            #return torch.einsum(scaling_str+ltrs,K,*(V[t] for t in range(m))).cpu().numpy(), log
            return list(V[t].cpu().numpy() for t in range(m)), log
        else:
            return torch.einsum(scaling_str+ltrs,K,*(V[t] for t in range(m))).cpu().numpy()
    else:
        if log:
            #return torch.einsum(scaling_str+ltrs,K,*(V[t] for t in range(m))), log
            return list(V[t] for t in range(m)), log
        else:
            return torch.einsum(scaling_str+ltrs,K,*(V[t] for t in range(m)))

def multiCyclicSinkhorn(A, M, reg = 1, metric = 'kl', numItermax=10000, stopThr=1e-6, verbose=False, log=False, gpu = True, to_numpy = False, numInnerItermax = 0, reg0 = 1e4):
    r"""
    Solve the entropic regularization multi-marginal optimal transport problem and return the optimal scalings v of the MOT tensor
 
    The algorithm used is based on the paper:
 
    The function solves the following optimization problem:
 
    .. math::
        \pi = arg\min_\pi <\pi,M>_F + reg\cdot\Omega(\pi)
 
        s.t. R_i(\pi) = a_i, i = 1,2,...,m
 
    where :
    - R_i is i-th marginal of tensor \gamma
    - M is the (dim_a_1,..., dim_a_m) metric cost tensor
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\pi)=\sum_{i_1,...,i_m} \pi_{i_1,...,i_m}(\log(\pi_{i_1,...,i_m})-1)`
    - a_i's are histograms (all sum to 1) for i=1,2,...,m
 
 
 
    Parameters
    ----------
    A : list of length m of ndarrays a_i of the shape (dim_a_i,) i=1,..., m
        samples weights in the source domain
    M : tensor, shape (dim_a_1,..., dim_a_m)
        loss matrix
    reg : float
        Regularization term >0
    tau : float in [0,1] that for tau > 0 batch_k = ceiling(tau * dim_a_k), while if tau = 0 batch = 1
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    log : bool, optional
        record log if True
 
    Returns
    -------
    gamma : tensor, shape (dim_a_1,..., dim_a_m)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
 
    Examples
    --------
 
    >>> import tensorflow as tf
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> BatchGreenkhorn([a,b], M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])
 
    """
    dev = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

    M = torch.as_tensor(M, dtype=torch.float64, device = dev)
    reg = torch.as_tensor(reg, dtype=torch.float64, device = dev)
    
    get_reg = lambda n: (reg0 - reg) * np.exp(-n) + reg if not numInnerItermax==0 else reg # exponential decreasing

    m = len(M.shape)
    
    numItermax *= m

    # strings for Einstein summation 
    ltrs = 'abcdefghijklmnopqrstuvwxyz'
    ltrs = ltrs[0:m]
    scaling_str = ltrs
    sum_str = ltrs[0:m-1]
    for i in range(m):
        scaling_str += ','+ ltrs[i]
        if i<m-1:
            sum_str += ','+ ltrs[i]
    scaling_str += '->'
    sum_str += '->'+ltrs[0:m-1]

    if len(A) == 0:
        for k in range(m):
            A[k] = torch.ones((M.shape[k]), dtype=torch.float64, device = dev) / M.shape[k]
    
    dim_A = []
    V = []
    for k in range(m):
        A[k] = torch.as_tensor(A[k], dtype=torch.float64, device = dev)
        dim_A.append(A[k].shape[0])        
        V.append(torch.ones(dim_A[k], dtype=torch.float64, device = dev))
    V0 = list( A[i] for i in range(m))

    # Kernel formation K = np.exp(-M/reg)
    K = torch.empty_like(M, device = dev)
    torch.divide(M, -reg, out=K)
    K = torch.where(K>-np.inf,torch.exp(K), 0.)
    #torch.exp(K, out=K)
    K = torch.einsum(scaling_str+ltrs,K,*(V0[t] for t in range(m)))

    # initial marginals, violations KL(A[k],R[k]) and l_inf-1 distance to transport polytope
    R = []
    viol = []
    dist = torch.zeros((m), dtype=torch.float64, device = dev)
    for i in range(m):
        R.append( torch.einsum(scaling_str+ltrs[i],K,*(V[t] for t in range(m)) )) 
        viol.append( A[i]*torch.log(A[i]/R[i])-A[i]+R[i] ) 
        dist[i] = torch.abs(A[i]-R[i]).sum()
 
    if metric == 'Linf': 
        stopThr_val = torch.max(dist) 
    elif metric=='L1':
        stopThr_val = torch.sum(dist) 
    else:
        stopThr_val = np.asarray(list(viol[i].sum() for i in range(m))).sum()   
 
    if log:
        res = [np.float64(stopThr_val)]
        log = dict()
        log['res'] = res
        start_time = time.perf_counter()
        if dev=='cuda':
            torch.cuda.synchronize()
        time_ = [start_time - start_time]
        log['time'] = time_ 
        
 
    for it in range(numItermax):

        # cyclic choice
        k = (it % m) 

        V[k] *= A[k] / R[k]
        R[k] = A[k]
        viol[k] = torch.zeros((dim_A[i]), dtype=torch.float64, device = dev)
        dist[k] = torch.zeros((1), dtype=torch.float64, device = dev)

        Kp = torch.einsum(scaling_str +ltrs[0:k]+ltrs[k+1:m], K, *(V[t] for t in range(m) ))

        for i in range(k): 
            R[i] = torch.einsum(ltrs[0:m-1]+'->'+ltrs[i],Kp)
            viol[i] = A[i]*torch.log(A[i]/R[i])-A[i]+R[i] 
            dist[i] = torch.abs(A[i]-R[i]).sum()

        for i in range(k+1,m):
            R[i] = torch.einsum(ltrs[0:m-1]+'->'+ltrs[i-1],Kp)
            viol[i] = A[i]*torch.log(A[i]/R[i])-A[i]+R[i] 
            dist[i]=torch.abs(A[i]-R[i]).sum()
        
        if metric == 'Linf': 
            stopThr_val = torch.max(dist) 
        elif metric=='L1':
            stopThr_val = torch.sum(dist) 
        else:
            stopThr_val = np.asarray(list(viol[i].sum() for i in range(m))).sum()

        if (it+1)%m==0 and log:
            res.append(np.float64(stopThr_val))
            if dev=='cuda':
              torch.cuda.synchronize()
            time_.append(time.perf_counter() - start_time)                          
  
        if verbose:
            print(
                '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(it, stopThr_val ))
        
        if stopThr_val <= stopThr:
            break
    else:
        print('Warning: Algorithm Cyclic Sinkhorn for eta={:1.2e} did not converge!'.format(reg.item()))
 
    if log:
        if ~((it+1)%m==0):
            res.append(np.float64(stopThr_val))  
            if dev=='cuda':
              torch.cuda.synchronize()
            time_.append(time.perf_counter() - start_time)
        log['res'] = np.asarray(res)
        log['time'] = np.asarray(time_)

    if to_numpy:
        if log:
            #return torch.einsum(scaling_str+ltrs,K,*(V[t] for t in range(m))).cpu().numpy(), log
            return list(V[t].cpu().numpy() for t in range(m)), log
        else:
            return torch.einsum(scaling_str+ltrs,K,*(V[t] for t in range(m))).cpu().numpy()
    else:
        if log:
            #return torch.einsum(scaling_str+ltrs,K,*(V[t] for t in range(m))), log
            return list(V[t] for t in range(m)), log
        else:
            return torch.einsum(scaling_str+ltrs,K,*(V[t] for t in range(m)))

