#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 07:52:31 2022

@author: vkiit
"""

import numpy as np
import torch

import matplotlib.colors as mcolors
import matplotlib.pylab as plt

import time
#import random

from kernels import kernel, cost_tensor


def benchmark(dataset, dual = False):
    if dual:
        from methods import sinkhorn_uniform_dual as sinkhorn_uniform
        from methods import sinkhorn_dual as sinkhorn
        from methods import batchGreenkhorn_uniform_dual as batchGreenkhorn_uniform
        from methods import batchGreenkhorn_dual as batchGreenkhorn
    else:
        from methods import sinkhorn, sinkhorn_uniform 
        from methods import batchGreenkhorn, batchGreenkhorn_uniform
                
    params = dataset['params']
    data = dataset['data']

    print("-"*60 + "\n"+"Running the experiment on "+ ("GPU..." if (params['gpu'] and torch.cuda.is_available()) else "CPU...") + "\n"+"-"*60)

    d = params['dimension']
    K = kernel(d, cost = params['cost'], gpu = params['gpu'])    
    
    # initialize kernel computations
    x_ = torch.rand((10,d), dtype = torch.float64, device = torch.device('cuda' if torch.cuda.is_available() and params['gpu'] else 'cpu'))
    one_ = torch.ones((10,1), dtype = torch.float64, device = torch.device('cuda' if torch.cuda.is_available() and params['gpu'] else 'cpu'))
    reg_ = torch.tensor([1.], dtype = torch.float64, device = torch.device('cuda' if torch.cuda.is_available() and params['gpu'] else 'cpu'))
    _ = K.diameter(x_,x_)
    _ = K.prod(x_,x_,reg_,one_, axis = 0)
    _ = K.prod(x_,x_,reg_,one_, axis = 1)

    times = np.zeros([len(params['sizes']),len(params['regularizations']), len(params['batches']),params['trials'], 100*params['maxIter']+2], dtype=np.float64)
    residuals = np.zeros([len(params['sizes']),len(params['regularizations']),len(params['batches']), params['trials'], 100*params['maxIter']+2], dtype=np.float64)
    times_total = np.zeros([len(params['sizes']),len(params['regularizations']), len(params['batches']),params['trials']], dtype=np.float64)
    for N_,N in enumerate(params['sizes']):
      
      print("\n"+"Historgram size n = {}".format(N) + "\n"+"-"*40)
      
      if dataset['type']=='same support':
        x = data['support'][N_]
        corr_ = K.diameter(x,x)

      for t in range(params['trials']):
        if dataset['type']=='point clouds':
          source_supp = data['marginals'][N_][t][0].contiguous()
          target_supp = data['marginals'][N_][t][1].contiguous()
          corr_ = K.diameter(source_supp,target_supp)
        elif dataset['type']=='same support':
          source_supp, target_supp = x, x
          source = data['marginals'][N_][t][0].contiguous()
          target = data['marginals'][N_][t][1].contiguous()
        else:  
          source_supp = data['marginals'][N_][t][0][:,:-1].contiguous()
          target_supp = data['marginals'][N_][t][1][:,:-1].contiguous()
          source = data['marginals'][N_][t][0][:,-1].contiguous()
          target = data['marginals'][N_][t][1][:,-1].contiguous()
          corr_ = K.diameter(source_supp,target_supp)

        for tau_,tau in enumerate(params['batches']):
          for eta_,eta in enumerate(params['regularizations']):
            if tau == 1:
              if dataset['type']=='point clouds':
                _, Log_ = sinkhorn_uniform(source_supp, target_supp, eta*corr_, K, metric = params['metric'], stopThr=params['tolerance'], numItermax= params['maxIter'], log = True, gpu = params['gpu'])
              else:
                _, Log_ = sinkhorn(source, source_supp, target, target_supp, eta*corr_, K, metric = params['metric'], stopThr=params['tolerance'], numItermax= params['maxIter'], log = True, gpu = params['gpu'])
              times[N_,eta_,tau_,t][0:len(Log_['time'])] = Log_['time']
              residuals[N_,eta_,tau_,t][0:len(Log_['res'])] = Log_['res']
              times_total[N_,eta_,tau_,t] = Log_['time'][-1]
            else:
              if dataset['type']=='point clouds':
                _, Log_ = batchGreenkhorn_uniform(source_supp, target_supp, eta*corr_, K, batch = tau, metric = params['metric'], stopThr=params['tolerance'], numItermax= params['maxIter'], log = True, gpu = params['gpu'])  
              else:          
                _, Log_= batchGreenkhorn(source, source_supp, target, target_supp, eta*corr_, K, batch = tau, metric = params['metric'], stopThr=params['tolerance'], numItermax= params['maxIter'], log = True, gpu = params['gpu'])            
              times[N_,eta_,tau_,t][0:len(Log_['time'])] = Log_['time']
              residuals[N_,eta_,tau_,t][0:len(Log_['res'])] = Log_['res']
              times_total[N_,eta_,tau_,t] = Log_['time'][-1]
    results = dict()
    results['residuals'] = residuals
    results['times'] = times
    results['total times'] = times_total
    return results

def benchmark_multi(dataset):

    from methods import multiCyclicSinkhorn, multiSinkhorn, multiBatchGreenkhorn
    
    params = dataset['params']
    data = dataset['data']

    flag = (params['gpu'] and torch.cuda.is_available())
    print("-"*60 + "\n"+"Running the experiment on "+ ("GPU..." if flag else "CPU...") + "\n"+"-"*60)

    m = params['m']
    

    times = np.zeros([len(params['sizes']),len(params['regularizations']), len(params['batches']),params['trials'], params['maxIter']+2], dtype=np.float64)
    residuals = np.zeros([len(params['sizes']),len(params['regularizations']),len(params['batches']), params['trials'], params['maxIter']+2], dtype=np.float64)
    times_total = np.zeros([len(params['sizes']),len(params['regularizations']), len(params['batches']),params['trials']], dtype=np.float64)
    for N_,N in enumerate(params['sizes']):
      
      print("\n"+"Historgram size n = {}".format(N) + "\n"+"-"*40)
      
      if dataset['type']=='same support':
        if flag:
            torch.cuda.synchronize()
        print('Computing cost tensor...')
        start_time = time.perf_counter()
        C = cost_tensor(data['support'][N_], cost = params['cost'], gpu = params['gpu'])
        if flag:
            torch.cuda.synchronize()
        print('Cost tensor construction: ',time.perf_counter()-start_time)
        corr_ = C.max()#K.diameter(x,x)

      for t in range(params['trials']):
        if dataset['type']=='point clouds':
          X = list(data['marginals'][N_][t][i] for i in range(m))
          if flag:
            torch.cuda.synchronize()
          print('Computing cost tensor...')
          start_time = time.perf_counter()  
          C = cost_tensor(X, cost = params['cost'], gpu = params['gpu'])
          if flag:
            torch.cuda.synchronize()
          print('Cost tensor construction: ',time.perf_counter()-start_time)
          corr_ = C.max()
          A = list(torch.ones(data['marginals'][N_][t][i].shape[0])/ data['marginals'][N_][t][i].shape[0] for i in range(m))  
        elif dataset['type']=='same support':
          A = list(data['marginals'][N_][t][i] for i in range(m))
        else:  
          X = list(data['marginals'][N_][t][i][:,:-1] for i in range(m))
          if flag:
            torch.cuda.synchronize()
          print('Computing cost tensor...')
          start_time = time.perf_counter()  
          C = cost_tensor(X, cost = params['cost'], gpu = params['gpu'])
          if flag:
            torch.cuda.synchronize()
          print('Cost tensor construction: ',time.perf_counter()-start_time)
          A = list(data['marginals'][N_][t][i][:,-1] for i in range(m))
          corr_ = C.max()

        for tau_,tau in enumerate(params['batches']):
          for eta_,eta in enumerate(params['regularizations']):
            if tau ==0:
              _ , Log_ = multiCyclicSinkhorn(A, C, eta*corr_, metric = params['metric'], stopThr=params['tolerance'], numItermax= params['maxIter'], log = True, gpu = params['gpu'])    
              times[N_,eta_,tau_,t][0:len(Log_['time'])] = Log_['time']
              residuals[N_,eta_,tau_,t][0:len(Log_['res'])] = Log_['res']
              times_total[N_,eta_,tau_,t] = Log_['time'][-1]          
            elif tau == 1:
              _ , Log_ = multiSinkhorn(A, C, eta*corr_, metric = params['metric'], stopThr=params['tolerance'], numItermax= params['maxIter'], log = True, gpu = params['gpu'])
              times[N_,eta_,tau_,t][0:len(Log_['time'])] = Log_['time']
              residuals[N_,eta_,tau_,t][0:len(Log_['res'])] = Log_['res']
              times_total[N_,eta_,tau_,t] = Log_['time'][-1]
            else:
              _, Log_ = multiBatchGreenkhorn(A, C, eta*corr_, tau = tau, metric = params['metric'], stopThr=params['tolerance'], numItermax= params['maxIter'], log = True, gpu = params['gpu'])  
              times[N_,eta_,tau_,t][0:len(Log_['time'])] = Log_['time']
              residuals[N_,eta_,tau_,t][0:len(Log_['res'])] = Log_['res']
              times_total[N_,eta_,tau_,t] = Log_['time'][-1]
    results = dict()
    results['residuals'] = residuals
    results['times'] = times
    results['total times'] = times_total
    return results    

def benchmark_multi_2(dataset):

    from methods import multiCyclicSinkhorn, multiSinkhorn, multiBatchGreenkhorn
    
    params = dataset['params']
    data = dataset['data']

    flag = (params['gpu'] and torch.cuda.is_available())
    print("-"*60 + "\n"+"Running the experiment on "+ ("GPU..." if flag else "CPU...") + "\n"+"-"*60) 

    times = np.zeros([len(params['m']),len(params['regularizations']), len(params['batches']),params['trials'], params['maxIter']+2], dtype=np.float64)
    residuals = np.zeros([len(params['m']),len(params['regularizations']),len(params['batches']), params['trials'], params['maxIter']+2], dtype=np.float64)
    times_total = np.zeros([len(params['m']),len(params['regularizations']), len(params['batches']),params['trials']], dtype=np.float64)
    for N_,N in enumerate(params['m']):
      
      print("\n"+"Number of marginals m = {}".format(N) + "\n"+"-"*40)
      
      if dataset['type']=='same support':
        if flag:
            torch.cuda.synchronize()
        print('Computing cost tensor...')
        start_time = time.perf_counter()
        C = cost_tensor(data['support'][N_], cost = params['cost'], gpu = params['gpu'])
        if flag:
            torch.cuda.synchronize()
        print('Cost tensor construction: ',time.perf_counter()-start_time)
        corr_ = C.max()#K.diameter(x,x)

      for t in range(params['trials']):
        if dataset['type']=='point clouds':
          X = list(data['marginals'][N_][t][i] for i in range(N))
          if flag:
            torch.cuda.synchronize()
          print('Computing cost tensor...')
          start_time = time.perf_counter()  
          C = cost_tensor(X, cost = params['cost'], gpu = params['gpu'])
          if flag:
            torch.cuda.synchronize()
          print('Cost tensor construction: ',time.perf_counter()-start_time)
          corr_ = C.max()
          A = list(torch.ones(data['marginals'][N_][t][i].shape[0])/ data['marginals'][N_][t][i].shape[0] for i in range(N))  
        elif dataset['type']=='same support':
          A = list(data['marginals'][N_][t][i] for i in range(N))
        else:  
          X = list(data['marginals'][N_][t][i][:,:-1] for i in range(N))
          if flag:
            torch.cuda.synchronize()
          print('Computing cost tensor...')
          start_time = time.perf_counter()  
          C = cost_tensor(X, cost = params['cost'], gpu = params['gpu'])
          if flag:
            torch.cuda.synchronize()
          print('Cost tensor construction: ',time.perf_counter()-start_time)
          A = list(data['marginals'][N_][t][i][:,-1] for i in range(N))
          corr_ = C.max()

        for tau_,tau in enumerate(params['batches']):
          for eta_,eta in enumerate(params['regularizations']):
            if tau ==0:
              _ , Log_ = multiCyclicSinkhorn(A, C, eta*corr_, metric = params['metric'], stopThr=params['tolerance'], numItermax= params['maxIter'], log = True, gpu = params['gpu'])    
              times[N_,eta_,tau_,t][0:len(Log_['time'])] = Log_['time']
              residuals[N_,eta_,tau_,t][0:len(Log_['res'])] = Log_['res']
              times_total[N_,eta_,tau_,t] = Log_['time'][-1]
            elif tau == 1:
              _ , Log_ = multiSinkhorn(A, C, eta*corr_, metric = params['metric'], stopThr=params['tolerance'], numItermax= params['maxIter'], log = True, gpu = params['gpu'])
              times[N_,eta_,tau_,t][0:len(Log_['time'])] = Log_['time']
              residuals[N_,eta_,tau_,t][0:len(Log_['res'])] = Log_['res']
              times_total[N_,eta_,tau_,t] = Log_['time'][-1]
            else:
              _, Log_ = multiBatchGreenkhorn(A, C, eta*corr_, tau = tau, metric = params['metric'], stopThr=params['tolerance'], numItermax= params['maxIter'], log = True, gpu = params['gpu'])  
              times[N_,eta_,tau_,t][0:len(Log_['time'])] = Log_['time']
              residuals[N_,eta_,tau_,t][0:len(Log_['res'])] = Log_['res']
              times_total[N_,eta_,tau_,t] = Log_['time'][-1]
    results = dict()
    results['residuals'] = residuals
    results['times'] = times
    results['total times'] = times_total
    return results 

    
def my_plot(dataset, sizes = 'all', etas = 'all' , batches = 'all', trials = 'all', plots = ['res vs iter', 'res vs time', 'comp_res vs iter', 'time vs size', 'time vs reg', 'comp_time vs reg', 'comp_res vs batch', 'comp_time vs batch'], titles = True, leg = True, Tmax = 100, agregation = 'mean'):
    params = dataset['params']
    results = dataset['results']
    plot_size = [12,9]
    font_size = 40
    line_width = 5
    plot_size, font_size, line_width  = [6,4], 10, 2
    
    
    m = params['m']

    if sizes=='all':
      sizes_ = range(len(params['sizes']))
      sizes = params['sizes']
    else:
      sizes_ = [N_ for N_, N in enumerate(params['sizes']) if N in sizes]
    if etas=='all':
      etas_ = range(len(params['regularizations']))
      etas = params['regularizations']
    else:
      etas_ = [eta_ for eta_, eta in enumerate(params['regularizations']) if eta in etas]
    if trials=='all':
      trials = range(params['trials'])
    if batches=='all':
      batches_ = range(len(params['batches']))
      batches = params['batches']
    else:
      batches_ = [tau_ for tau_, tau in enumerate(params['batches']) if tau in batches]
    if not type(Tmax)==list:
        Tmax = min(Tmax,params['maxIter'])+1

    cols = ['k'] + list(list(mcolors.TABLEAU_COLORS)[i] for i in [3,2,0,6,1,4,5,7,8,9]) + ['gray','indigo', 'fuchsia','darkolivegreen','dodgerblue','gold','blueviolet','firebrick','lawngreen','darkturquoise']
    mrks = 'o'*21#'ov^sd+*.-|' 

    #residuals = results['residuals']
    residuals = np.take(results['residuals'], sizes_,0) 
    residuals = np.take(residuals, etas_,1)
    residuals = np.take(residuals, batches_,2)
    residuals = np.take(residuals, trials,3)

    #times = results['times']
    times = np.take(results['times'], sizes_,0) 
    times = np.take(times, etas_,1)
    times = np.take(times, batches_,2)
    times = np.take(times, trials,3)

    #times_total = results['total times']
    times_total = np.take(results['total times'], sizes_,0) 
    times_total = np.take(times_total, etas_,1)
    times_total = np.take(times_total, batches_,2)
    times_total = np.take(times_total, trials,3)

    if 'comp_res vs iter' in plots:
      for N_, N in enumerate(sizes):
        fig = plt.figure(figsize=(plot_size[0]*len(etas), plot_size[1]))
        for eta_, eta in enumerate(etas):
          plt.subplot(1,len(etas),eta_+1)
          plt.tight_layout(w_pad = 1+font_size/10)
          
          for tau_, tau in enumerate(batches):
            if not type(Tmax)==list:
                T = Tmax
                for t in range(len(trials)):
                  T_ = (residuals[N_,eta_,tau_,t,:Tmax]>0).argmin()-1
                  if T_<0: 
                    T_ = Tmax
                  else:
                    T_ += 1
                  T = min(T,T_)
                if tau_ == 0: 
                    T0 = T
                else:
                    T = min(T,T0)
            else:
                T = Tmax[eta_]
            #log_comp = 10*np.log10(residuals[N_,eta_,0,:,:T] / residuals[N_,eta_,tau_,:,:T])  
            #val = np.mean(log_comp, axis=-2)
            #var = np.std(log_comp, axis=-2)
            comp = residuals[N_,eta_,0,:,:T] / residuals[N_,eta_,tau_,:,:T] 
            val = np.mean(1/comp, axis=-2)
            var = np.std(1/comp, axis=-2)
            if tau == 1:
              plt.plot(range(T),val, '-',color=cols[tau_],label='Sinkhorn' if m==2 else 'MultiSinkhorn', linewidth = line_width)
              plt.yscale('log')
              plt.fill_between(range(T), val-var, val+var , color=cols[tau_], alpha=0.2)
            elif tau == 0:
              plt.plot(range(T),val, '-',color=cols[tau_],label='Cyclic Sinkhorn', linewidth = line_width)
              plt.fill_between(range(T), val-var, val+var , color=cols[tau_], alpha=0.2)
            elif tau == -1:
              plt.plot(range(T),val, '-',color=cols[tau_],label='Random Sinkhorn', linewidth = line_width)
              plt.fill_between(range(T), val-var, val+var , color=cols[tau_], alpha=0.2)
            else: 
              plt.plot(range(T),val, '-',color=cols[tau_],label='BatchGreenkhorn({:2.0%})'.format(tau),linewidth = line_width)
              plt.fill_between(range(T), val-var, val+var, color=cols[tau_], alpha=0.2)

          if titles: plt.title('Size n={} and regularization $\eta=${}$||C||_\\infty$'.format(N,eta),fontsize = font_size)
          plt.xlabel('normalized cycles T', fontsize = font_size)
          #plt.ylim(bottom = 1e-3)
          
          if eta_==0:
              if leg: plt.legend( loc = 'lower left',fontsize = font_size)
              plt.ylabel('compettive ratio $\\rho_{ \\tau}$',fontsize = font_size)
          
          plt.rc('xtick', labelsize=font_size)
          plt.rc('ytick', labelsize=font_size)
    
    if 'res vs iter' in plots:
      mean_res = np.mean(residuals, axis=-2)
      std_res = np.std(residuals, axis=-2)
      for N_, N in enumerate(sizes):
        fig = plt.figure(figsize=(plot_size[0]*len(etas), plot_size[1]))
        for eta_, eta in enumerate(etas):
          plt.subplot(1,len(etas),eta_+1)
          plt.tight_layout(w_pad = 1+font_size/10)
          for tau_, tau in enumerate(batches):
            val = mean_res[N_,eta_,tau_][mean_res[N_,eta_,tau_,]>0.]
            var = std_res[N_,eta_,tau_][mean_res[N_,eta_,tau_]>0.]
            T = len(val)
            if tau == 1:
              plt.plot(range(T),val, '-',color=cols[tau_],label='Sinkhorn' if m==2 else 'MultiSinkhorn',linewidth = line_width)
              plt.yscale('log')
              plt.fill_between(range(T), val-var, val+var , color=cols[tau_], alpha=0.2)
            elif tau == 0:
              plt.plot(range(T),val, '-',color=cols[tau_],label='Cyclic Sinkhorn',linewidth = line_width)
              plt.fill_between(range(T), val-var, val+var , color=cols[tau_], alpha=0.2)
            elif tau == -1:
              plt.plot(range(T),val, '-',color=cols[tau_],label='Random Sinkhorn',linewidth = line_width)
              plt.fill_between(range(T), val-var, val+var , color=cols[tau_], alpha=0.2)  
            else: 
              plt.plot(range(T),val, '-',color=cols[tau_],label='BatchGreenkhorn({:2.0%})'.format(tau),linewidth = line_width)
              plt.fill_between(range(T), val-var, val+var, color=cols[tau_], alpha=0.2)

          if titles: plt.title('Size n={} and regularization $\eta=${}$||C||_\\infty$'.format(N,eta),fontsize = font_size)
          plt.xlabel('normalized cycles T', fontsize = font_size)
          plt.ylim(bottom = params['tolerance'])
          
          if eta_==0:
              if leg: plt.legend( loc = 'upper right',fontsize = font_size)
              plt.ylabel('distance to polytope  d',fontsize = font_size)
              
          plt.rc('xtick', labelsize=font_size)
          plt.rc('ytick', labelsize=font_size)
    
    # if 'res vs time' in plots:
    #   t = random.randrange(len(trials)) if len(trials)>1 else 0
    #   for N_, N in enumerate(sizes):
    #     fig = plt.figure(figsize=(plot_size[0]*len(etas), plot_size[1]))
    #     for eta_, eta in enumerate(etas):
    #       plt.subplot(1,len(etas),eta_+1)
    #       plt.tight_layout(w_pad = 1+font_size/10)
    #       for tau_, tau in enumerate(batches):
    #         y_ = residuals[N_,eta_,tau_,t][residuals[N_,eta_,tau_,t]>0.]
    #         x_ = times[N_,eta_,tau_,t][residuals[N_,eta_,tau_,t]>0.]
    #         if tau == 1:
    #           plt.plot(x_, y_, '-',color=cols[tau_],label='Sinkhorn' if m==2 else 'MultiSinkhorn',linewidth = line_width)
    #           plt.yscale('log')
    #         elif tau == 0:
    #           plt.plot(x_, y_, '-',color=cols[tau_],label='Cyclic Sinkhorn',linewidth = line_width)
    #         else: 
    #           plt.plot(x_, y_, '-',color=cols[tau_],label='BatchGreenkhorn({:2.0%})'.format(tau),linewidth = line_width)

    #       if titles: plt.title('Trial no. {} :  n={} and $\eta=${}$||C||_\\infty$'.format(t,N,eta),fontsize = font_size)
    #       plt.xlabel('computation time [s]', fontsize = font_size)
    #       plt.ylim(bottom = params['tolerance'])
          
    #       if eta_==0:
    #           if leg: plt.legend( loc = 'upper right',fontsize = font_size)
    #           plt.ylabel('distance to polytope d',fontsize = font_size)

    #       plt.rc('xtick', labelsize=font_size)
    #       plt.rc('ytick', labelsize=font_size)


    if 'time vs size' in plots:
      mean_times = np.mean(times_total, axis=-1) if agregation=='mean' else np.sum(times_total, axis=-1)
      std_times = np.std(times_total, axis=-1)
      fig = plt.figure(figsize=(plot_size[0]*len(etas), plot_size[1]))
      for eta_, eta in enumerate(etas):
        plt.subplot(1,len(etas),eta_+1)
        plt.tight_layout(w_pad = 1+font_size/10)
        for tau_, tau in enumerate(batches):
          if tau == 1:
            plt.plot(np.array(sizes), mean_times[:,eta_,tau_].squeeze(), '-', marker = mrks[tau_],color=cols[tau_],label='Sinkhorn' if m==2 else 'MultiSinkhorn',linewidth = line_width)
            if agregation=='mean':
              plt.fill_between(np.array(sizes), mean_times[:,eta_,tau_].squeeze() - std_times[:,eta_,tau_].squeeze(), mean_times[:,eta_,tau_].squeeze() + std_times[:,eta_,tau_].squeeze(), color=cols[tau_], alpha=0.2)
          else: 
            if tau == 0:
              plt.plot(np.array(sizes), mean_times[:,eta_,tau_].squeeze(), '-', marker = mrks[tau_], color=cols[tau_],label='Cyclic Sinkhorn',linewidth = line_width)
              if agregation=='mean':
                plt.fill_between(np.array(sizes), mean_times[:,eta_,tau_].squeeze() - std_times[:,eta_,tau_].squeeze(), mean_times[:,eta_,tau_].squeeze() + std_times[:,eta_,tau_].squeeze(), color=cols[tau_], alpha=0.2)
            elif tau == -1:  
              plt.plot(np.array(sizes), mean_times[:,eta_,tau_].squeeze(), '-', marker = mrks[tau_], color=cols[tau_],label='Cyclic Sinkhorn',linewidth = line_width)
              if agregation=='mean':
                plt.fill_between(np.array(sizes), mean_times[:,eta_,tau_].squeeze() - std_times[:,eta_,tau_].squeeze(), mean_times[:,eta_,tau_].squeeze() + std_times[:,eta_,tau_].squeeze(), color=cols[tau_], alpha=0.2)
            else:
              plt.plot(np.array(sizes), mean_times[:,eta_,tau_].squeeze(), '-', marker = mrks[tau_], color=cols[tau_],label='BatchGreenkhorn({:2.0%})'.format(tau),linewidth = line_width)
              if agregation=='mean':
                plt.fill_between(np.array(sizes), mean_times[:,eta_,tau_].squeeze() - std_times[:,eta_,tau_].squeeze(), mean_times[:,eta_,tau_].squeeze() + std_times[:,eta_,tau_].squeeze(), color=cols[tau_], alpha=0.2)
        
        if titles: plt.title('Regularization $\eta=${}'.format(eta) ,fontsize=font_size)
        plt.xlabel('size n',fontsize=font_size)
        if eta_==0:
            if leg: plt.legend( loc = 'upper left',fontsize=font_size)
            plt.ylabel('time [s]',fontsize=font_size)
            
        plt.rc('xtick', labelsize=font_size)
        plt.rc('ytick', labelsize=font_size)


    if 'time vs reg' in plots:
      val = np.mean(times_total, axis=-1) if agregation=='mean' else np.sum(times_total, axis=-1)
      print(np.ceil(100*val/60)/100)
      var = np.std(times_total, axis=-1)
      x = (1/np.array(etas))
      fig = plt.figure(figsize=(plot_size[0]*len(sizes), plot_size[1]))
      for N_, N in enumerate(sizes):
        plt.subplot(1,len(sizes),N_+1)
        plt.tight_layout(w_pad = 1+font_size/10)
        for tau_, tau in enumerate(batches):
          if tau == 1:
            plt.plot(x, val[N_,:,tau_].squeeze(), '-', marker = mrks[tau_],color=cols[tau_],label='Sinkhorn' if m==2 else 'MultiSinkhorn',linewidth = line_width)
            if agregation=='mean':
                plt.fill_between(x, val[N_,:,tau_].squeeze() - var[N_,:,tau_].squeeze(), val[N_,:,tau_].squeeze() + var[N_,:,tau_].squeeze(), color=cols[tau_], alpha=0.2)
          else: 
            if tau == 0:
              plt.plot(x, val[N_,:,tau_].squeeze(), '-', marker = mrks[tau_], color=cols[tau_],label='Cyclic Sinkhorn',linewidth = line_width)
              if agregation=='mean':
                plt.fill_between(x, val[N_,:,tau_].squeeze() - var[N_,:,tau_].squeeze(), val[N_,:,tau_].squeeze() + var[N_,:,tau_].squeeze(), color=cols[tau_], alpha=0.2)
            elif tau == -1:
              plt.plot(x, val[N_,:,tau_].squeeze(), '-', marker = mrks[tau_], color=cols[tau_],label='Random Sinkhorn',linewidth = line_width)
              if agregation=='mean':
                plt.fill_between(x, val[N_,:,tau_].squeeze() - var[N_,:,tau_].squeeze(), val[N_,:,tau_].squeeze() + var[N_,:,tau_].squeeze(), color=cols[tau_], alpha=0.2)
            else:
              plt.plot(x, val[N_,:,tau_].squeeze(), '-', marker = mrks[tau_], color=cols[tau_],label='BatchGreenkhorn({:2.0%})'.format(tau),linewidth = line_width)
              if agregation=='mean':
                plt.fill_between(x, val[N_,:,tau_].squeeze() - var[N_,:,tau_].squeeze(), val[N_,:,tau_].squeeze() + var[N_,:,tau_].squeeze(), color=cols[tau_], alpha=0.2)
        
        if titles: plt.title('Support size n={}'.format(N) ,fontsize=font_size)
        plt.xlabel('$||C||_{\infty}/\eta$',fontsize=font_size)
        if N_==0:
          if leg: plt.legend( loc = 'upper left',fontsize=font_size)
          plt.ylabel('time [s]',fontsize=font_size)
        
        plt.rc('xtick', labelsize=font_size)
        plt.rc('ytick', labelsize=font_size)

    if 'comp_time vs reg' in plots:
    
      if agregation=='mean':
        comp = np.zeros_like(times_total)
        for tau_, tau in enumerate(batches):
          comp[:,:,tau_,:] =  times_total[:,:,0,:] / times_total[:,:,tau_,:] 
        val = np.mean(comp, axis=-1)
        var = np.std(comp, axis=-1)
      else:
        times_total = times_total.sum(-1)
        comp = np.zeros_like(times_total)  
        for tau_, tau in enumerate(batches):
          comp[:,:,tau_] =  times_total[:,:,0] / times_total[:,:,tau_]  
        val = comp

      x = (1/np.array(etas))
      fig = plt.figure(figsize=(plot_size[0]*len(sizes), plot_size[1]))
      for N_, N in enumerate(sizes):
        plt.subplot(1,len(sizes),N_+1)
        plt.tight_layout(w_pad = 1+font_size/10)
        for tau_, tau in enumerate(batches): 
          if tau == 1:
            plt.plot(x, val[N_,:,tau_].squeeze(), '-', marker = mrks[tau_],color=cols[tau_],label='Sinkhorn' if m==2 else 'MultiSinkhorn',linewidth = line_width)
            if agregation=='mean':
                plt.fill_between(x, val[N_,:,tau_].squeeze() - var[N_,:,tau_].squeeze(), val[N_,:,tau_].squeeze() + var[N_,:,tau_].squeeze(), color=cols[tau_], alpha=0.2)
          else: 
            if tau == 0:
              plt.plot(x, val[N_,:,tau_].squeeze(), '-', marker = mrks[tau_], color=cols[tau_],label='Cyclic Sinkhorn',linewidth = line_width)
              if agregation=='mean':
                plt.fill_between(x, val[N_,:,tau_].squeeze() - var[N_,:,tau_].squeeze(), val[N_,:,tau_].squeeze() + var[N_,:,tau_].squeeze(), color=cols[tau_], alpha=0.2)
            elif tau == -1:
              plt.plot(x, val[N_,:,tau_].squeeze(), '-', marker = mrks[tau_], color=cols[tau_],label='Random Sinkhorn',linewidth = line_width)
              if agregation=='mean':
                plt.fill_between(x, val[N_,:,tau_].squeeze() - var[N_,:,tau_].squeeze(), val[N_,:,tau_].squeeze() + var[N_,:,tau_].squeeze(), color=cols[tau_], alpha=0.2)
            else:
              plt.plot(x, val[N_,:,tau_].squeeze(), '-', marker = mrks[tau_], color=cols[tau_],label='BatchGreenkhorn({:2.0%})'.format(tau),linewidth = line_width)
              if agregation=='mean':
                plt.fill_between(x, val[N_,:,tau_].squeeze() - var[N_,:,tau_].squeeze(), val[N_,:,tau_].squeeze() + var[N_,:,tau_].squeeze(), color=cols[tau_], alpha=0.2)
        
        if titles: plt.title('Support size n={}'.format(N) ,fontsize=font_size)
        plt.xlabel('$||C||_{\infty}/\eta$',fontsize=font_size)
        if N_==0:
          if leg: plt.legend( loc = 'upper left',fontsize=font_size)
          plt.ylabel('speedup factor $\\sigma_{\\tau}$',fontsize=font_size)

        plt.rc('xtick', labelsize=font_size)
        plt.rc('ytick', labelsize=font_size)
        #plt.xlim(left = x[0])


    if 'comp_res vs batch' in plots:
      x  = np.asarray(batches)  
      fig = plt.figure(figsize=(plot_size[0]*len(sizes), plot_size[1]))
      for N_, N in enumerate(sizes):
        plt.subplot(1,len(sizes),N_+1)
        plt.tight_layout(w_pad = 1+font_size/10)
        for eta_, eta in enumerate(etas):
          val, var = np.zeros_like(x), np.zeros_like(x)
          if not type(Tmax)==list:
              T = Tmax
              for tau_, tau in enumerate(batches):
                for t in range(len(trials)):
                  T_ = (residuals[N_,eta_,tau_,t,:Tmax]>0).argmin()-1
                  if T_<0: 
                    T_ = Tmax
                  else:
                    T_ += 1
                  T = min(T,T_)
                if tau_ == 0: 
                    T0 = T
                else:
                    T = min(T,T0)                      
          for tau_, tau in enumerate(batches):
            if type(Tmax)==list:
                T = Tmax[eta_]
            #comp = np.amax((residuals[N_,eta_,0,:,:T] / residuals[N_,eta_,tau_,:,:T]), axis = 1)
            comp = (residuals[N_,eta_,0,:,T-2] / residuals[N_,eta_,tau_,:,T-2])
            if len(trials)>1:
              val[tau_] = np.mean(1/comp,axis=-1)  
              var[tau_] = np.std(1/comp, axis=-1)
            else:
              val[tau_] = 1/comp[0]          
          
          #print('N = {}, eta = {}, T={}, val = {}'.format(N, eta, T-2,val))  
          plt.plot(x,val, '-',color=cols[eta_],label= '$\eta=${}$||C||_\\infty, T = {}$'.format(eta, T-2), linewidth = line_width)
          plt.yscale('log')
          if len(trials)>1:
              plt.fill_between(x, val-var, val+var , color=cols[eta_], alpha=0.2)

          if titles: plt.title('Size n={}'.format(N),fontsize = font_size)
          plt.xlabel('relative batch size ', fontsize = font_size)
          #plt.ylim(bottom = -1)
          
        plt.plot(x, x/x, 'k--', linewidth = line_width/2)  
        if leg: plt.legend( loc = 'upper left',fontsize = font_size)
        if N_==0:
          plt.ylabel('compettive ratio $\\rho_{ \\tau}(T)$',fontsize = font_size)
          
        plt.rc('xtick', labelsize=font_size)
        plt.rc('ytick', labelsize=font_size)

    if 'comp_time vs batch' in plots:
        
      if agregation=='mean':
        comp = np.zeros_like(times_total)
        for tau_, tau in enumerate(batches):
          comp[:,:,tau_,:] =  times_total[:,:,0,:] / times_total[:,:,tau_,:] 
        val = np.mean(comp, axis=-1)
        var = np.std(comp, axis=-1)
      else:
        times_total = times_total.sum(-1)
        comp = np.zeros_like(times_total)  
        for tau_, tau in enumerate(batches):
          comp[:,:,tau_] =  times_total[:,:,0] / times_total[:,:,tau_]  
        val = comp
        
      x  = np.asarray(batches)  

      fig = plt.figure(figsize=(plot_size[0]*len(sizes), plot_size[1]))
      for N_, N in enumerate(sizes):
        plt.subplot(1,len(sizes),N_+1)
        plt.tight_layout(w_pad = 1+font_size/10)
        for eta_, eta in enumerate(etas):           
          plt.plot(x,val[N_,eta_,:].squeeze(), '--',color=cols[eta_+1],label= '$\eta=${}$||C||_\\infty$'.format(eta), linewidth = line_width)
          #plt.yscale('log')
          if len(trials)>1:
              plt.fill_between(x, val[N_,eta_,:].squeeze()-var[N_,eta_,:].squeeze(), val[N_,eta_,:].squeeze()+var[N_,eta_,:].squeeze() , color=cols[eta_], alpha=0.2)

          if titles: plt.title('Size n={}'.format(N),fontsize = font_size)
          plt.xlabel('relative batch size ', fontsize = font_size)
          #plt.ylim(bottom = -1)
        plt.plot(x, x/x, 'k-', linewidth = line_width/2)  
        
        if N_==0:
            if leg: plt.legend( loc = 'lower right',fontsize = font_size)
            plt.ylabel('speedup factor $\\sigma_{ \\tau}$',fontsize = font_size)
          
        plt.rc('xtick', labelsize=font_size)
        plt.rc('ytick', labelsize=font_size)        

    if 'res vs time' in plots:
      #t = random.randrange(len(trials)) if len(trials)>1 else 0
      for eta_, eta in enumerate(etas):
        fig = plt.figure(figsize=(plot_size[0]*len(sizes), plot_size[1]))
        for N_, N in enumerate(sizes):
          plt.subplot(1,len(sizes),N_+1)
          plt.tight_layout(w_pad = 1+font_size/10)
          for tau_, tau in enumerate(batches):
            t = 0
            y_ = residuals[N_,eta_,tau_,t][residuals[N_,eta_,tau_,t]>0.]
            x_ = times[N_,eta_,tau_,t][residuals[N_,eta_,tau_,t]>0.]
            if tau == 1:
              plt.plot(x_, y_, '-',color=cols[tau_],label='MultiSinkhorn',linewidth = line_width)
              plt.yscale('log')
            elif tau == 0:
              plt.plot(x_, y_, '-',color=cols[tau_],label='Cyclic Sinkhorn',linewidth = line_width)
              plt.yscale('log')
            elif tau == -1:
              plt.plot(x_, y_, '-',color=cols[tau_],label='Random Sinkhorn',linewidth = line_width)
            else: 
              plt.plot(x_, y_, '-',color=cols[tau_],label='BatchGreenkhorn({:2.0%})'.format(tau),linewidth = line_width)
            for t in range(1,len(trials)):
                y_ = residuals[N_,eta_,tau_,t][residuals[N_,eta_,tau_,t]>0.]
                x_ = times[N_,eta_,tau_,t][residuals[N_,eta_,tau_,t]>0.]
                plt.plot(x_, y_, '-',color=cols[tau_],linewidth = line_width)            
          if titles: plt.title('m={} and $\eta=${}$||C||_\\infty$'.format(N,eta),fontsize = font_size)
          plt.xlabel('computation time [s]', fontsize = font_size)
          plt.ylim(bottom = params['tolerance'])
          
          if N_==0:
              if leg: plt.legend( loc = 'upper right',fontsize = font_size)
              plt.ylabel('distance to polytope d',fontsize = font_size)

          plt.rc('xtick', labelsize=font_size)
          plt.rc('ytick', labelsize=font_size)
          
    if 'ress vs iter' in plots:
      #t = random.randrange(len(trials)) if len(trials)>1 else 0
      for eta_, eta in enumerate(etas):
        fig = plt.figure(figsize=(plot_size[0]*len(sizes), plot_size[1]))
        for N_, N in enumerate(sizes):
          plt.subplot(1,len(sizes),N_+1)
          plt.tight_layout(w_pad = 1+font_size/10)
          for tau_, tau in enumerate(batches):
            t = 0
            y_ = residuals[N_,eta_,tau_,t][residuals[N_,eta_,tau_,t]>0.]
            #print(y_)
            x_ = range(len(y_))
            if tau == 1:
              plt.plot(x_, y_, '-',color=cols[tau_],label='MultiSinkhorn',linewidth = line_width)
              plt.yscale('log')
            elif tau == 0:
              plt.plot(x_, y_, '-',color=cols[tau_],label='Cyclic Sinkhorn',linewidth = line_width)
              plt.yscale('log')
            elif tau == -1:
              plt.plot(x_, y_, '-',color=cols[tau_],label='Random Sinkhorn',linewidth = line_width)
            else: 
              plt.plot(x_, y_, '-',color=cols[tau_],label='BatchGreenkhorn({:2.0%})'.format(tau),linewidth = line_width)
            for t in range(1,len(trials)):
                y_ = residuals[N_,eta_,tau_,t][residuals[N_,eta_,tau_,t]>0.]
                x_ = range(len(y_))
                plt.plot(x_, y_, '-',color=cols[tau_],linewidth = line_width)            
          if titles: plt.title('m={} and $\eta=${}$||C||_\\infty$'.format(N,eta),fontsize = font_size)
          plt.xlabel('normalized cycles T', fontsize = font_size)
          plt.ylim(bottom = params['tolerance'])
          
          if N_==0:
              if leg: plt.legend( loc = 'upper right',fontsize = font_size)
              plt.ylabel('distance to polytope d',fontsize = font_size)

          plt.rc('xtick', labelsize=font_size)
          plt.rc('ytick', labelsize=font_size)
    

def my_plot_2(dataset, sizes = 'all', etas = 'all' , batches = 'all', trials = 'all', plots = ['res vs iter', 'res vs time', 'comp_res vs iter', 'time vs size', 'time vs reg', 'comp_time vs reg', 'comp_res vs batch', 'comp_time vs batch'], titles = True, leg = True, T = [], agregation = 'mean'):
    params = dataset['params']
    results = dataset['results']
    plot_size = [12,9]
    font_size = 40
    line_width = 5
    plot_size, font_size, line_width  = [8,4], 12, 2

    if sizes=='all':
      sizes_ = range(len(params['m']))
      sizes = params['m']
    else:
      sizes_ = [N_ for N_, N in enumerate(params['m']) if N in sizes]
    if etas=='all':
      etas_ = range(len(params['regularizations']))
      etas = params['regularizations']
    else:
      etas_ = [eta_ for eta_, eta in enumerate(params['regularizations']) if eta in etas]
    if trials=='all':
      trials = range(params['trials'])
    if batches=='all':
      batches_ = range(len(params['batches']))
      batches = params['batches']
    else:
      batches_ = [tau_ for tau_, tau in enumerate(params['batches']) if tau in batches]

    if len(T)==0:
        T = np.ones_like(etas)

    #cols = ['k'] + list(list(mcolors.TABLEAU_COLORS)[i] for i in [3,2,0,6,1,4,5,7,8,9]) + ['gray','indigo', 'fuchsia','darkolivegreen','dodgerblue','gold','blueviolet','firebrick','lawngreen','darkturquoise']
    cols = list(list(mcolors.TABLEAU_COLORS)[i] for i in [0,1,3,2,6,1,4,5,7,8,9]) + ['gray','indigo', 'fuchsia','darkolivegreen','dodgerblue','gold','blueviolet','firebrick','lawngreen','darkturquoise']
    mrks = 'o'*21#'ov^sd+*.-|' 

    #residuals = results['residuals']
    residuals = np.take(results['residuals'], sizes_,0) 
    residuals = np.take(residuals, etas_,1)
    residuals = np.take(residuals, batches_,2)
    residuals = np.take(residuals, trials,3)

    #times = results['times']
    times = np.take(results['times'], sizes_,0) 
    times = np.take(times, etas_,1)
    times = np.take(times, batches_,2)
    times = np.take(times, trials,3)

    #times_total = results['total times']
    times_total = np.take(results['total times'], sizes_,0) 
    times_total = np.take(times_total, etas_,1)
    times_total = np.take(times_total, batches_,2)
    times_total = np.take(times_total, trials,3)

    if 'comp_res vs iter' in plots:
        fig = plt.figure(figsize=(plot_size[0]*len(etas), plot_size[1]))
        for eta_, eta in enumerate(etas):
          plt.subplot(1,len(etas),eta_+1)
          plt.tight_layout(w_pad = 1+font_size/10)
        
          plt.plot(range(T[eta_]),np.ones(T[eta_]), '-',color=cols[0], linewidth = line_width)
          plt.yscale('log')
          for N_, N in enumerate(sizes):
            comp = residuals[N_,eta_,0,:,:T[eta_]] / residuals[N_,eta_,1,:,:T[eta_]] 
            val = np.mean(1/comp, axis=-2)
            var = np.std(1/comp, axis=-2)
            plt.plot(range(T[eta_]),val, '-',color=cols[N_+1],label= 'm = {}'.format(N), linewidth = line_width)
            plt.fill_between(range(T[eta_]), val-var, val+var , color=cols[N_+1], alpha=0.2)
        
          if titles: 
              plt.title('Regularization $\eta=${}$||C||_\\infty$'.format(eta),fontsize = font_size)
          plt.xlabel('normalized cycles T', fontsize = font_size)
          plt.ylim(bottom = 1e-4)
              
          if eta_==0:
            if leg: plt.legend( loc = 'lower left',fontsize = font_size)
            plt.ylabel('compettive ratio $\\rho_{ \\tau}$',fontsize = font_size)
        
          plt.rc('xtick', labelsize=font_size)
          plt.rc('ytick', labelsize=font_size)
  
    if 'res vs iter' in plots:
      mean_res = np.mean(residuals, axis=-2)
      std_res = np.std(residuals, axis=-2)
      for N_, N in enumerate(sizes):
        fig = plt.figure(figsize=(plot_size[0]*len(etas), plot_size[1]))
        for eta_, eta in enumerate(etas):
          plt.subplot(1,len(etas),eta_+1)
          plt.tight_layout(w_pad = 1+font_size/10)
          for tau_, tau in enumerate(batches):
            val = mean_res[N_,eta_,tau_][mean_res[N_,eta_,tau_,]>0.]
            var = std_res[N_,eta_,tau_][mean_res[N_,eta_,tau_]>0.]
            T = len(val)
            if tau == 1:
              plt.plot(range(T),val, '-',color=cols[tau_],label='MultiSinkhorn',linewidth = line_width)
              plt.yscale('log')
              plt.fill_between(range(T), val-var, val+var , color=cols[tau_], alpha=0.2)
            elif tau == 0:
              plt.plot(range(T),val, '-',color=cols[tau_],label='Cyclic Sinkhorn',linewidth = line_width)
              plt.yscale('log')
              plt.fill_between(range(T), val-var, val+var , color=cols[tau_], alpha=0.2)
            elif tau == -1:
              plt.plot(range(T),val, '-',color=cols[tau_],label='Random Sinkhorn',linewidth = line_width)
              plt.yscale('log')
              plt.fill_between(range(T), val-var, val+var , color=cols[tau_], alpha=0.2)   
            else: 
              plt.plot(range(T),val, '-',color=cols[tau_],label='BatchGreenkhorn({:2.0%})'.format(tau),linewidth = line_width)
              plt.fill_between(range(T), val-var, val+var, color=cols[tau_], alpha=0.2)

          if titles: plt.title('Size m={} and regularization $\eta=${}$||C||_\\infty$'.format(N,eta),fontsize = font_size)
          plt.xlabel('normalized cycles T', fontsize = font_size)
          plt.ylim(bottom = params['tolerance'])
          
          if eta_==0:
              if leg: plt.legend( loc = 'upper right',fontsize = font_size)
              plt.ylabel('distance to polytope  d',fontsize = font_size)
              
          plt.rc('xtick', labelsize=font_size)
          plt.rc('ytick', labelsize=font_size)
    
    if 'res vs time' in plots:
      #t = random.randrange(len(trials)) if len(trials)>1 else 0
      for eta_, eta in enumerate(etas):
        fig = plt.figure(figsize=(plot_size[0]*len(sizes), plot_size[1]))
        for N_, N in enumerate(sizes):
          plt.subplot(1,len(sizes),N_+1)
          plt.tight_layout(w_pad = 1+font_size/10)
          for tau_, tau in enumerate(batches):
            t = 0
            y_ = residuals[N_,eta_,tau_,t][residuals[N_,eta_,tau_,t]>0.]
            x_ = times[N_,eta_,tau_,t][residuals[N_,eta_,tau_,t]>0.]
            if tau == 1:
              plt.plot(x_, y_, '-',color=cols[tau_],label='MultiSinkhorn',linewidth = line_width)
              plt.yscale('log')
            elif tau == 0:
              plt.plot(x_, y_, '-',color=cols[tau_],label='Cyclic Sinkhorn',linewidth = line_width)
              plt.yscale('log')
            elif tau == -1:
              plt.plot(x_, y_, '-',color=cols[tau_],label='Random Sinkhorn',linewidth = line_width)
            else: 
              plt.plot(x_, y_, '-',color=cols[tau_],label='BatchGreenkhorn({:2.0%})'.format(tau),linewidth = line_width)
            for t in range(1,len(trials)):
                y_ = residuals[N_,eta_,tau_,t][residuals[N_,eta_,tau_,t]>0.]
                x_ = times[N_,eta_,tau_,t][residuals[N_,eta_,tau_,t]>0.]
                plt.plot(x_, y_, '-',color=cols[tau_],linewidth = line_width)            
          if titles: plt.title('m={} and $\eta=${}$||C||_\\infty$'.format(N,eta),fontsize = font_size)
          plt.xlabel('computation time [s]', fontsize = font_size)
          plt.ylim(bottom = params['tolerance'])
          
          if N_==0:
              if leg: plt.legend( loc = 'upper right',fontsize = font_size)
              plt.ylabel('distance to polytope d',fontsize = font_size)

          plt.rc('xtick', labelsize=font_size)
          plt.rc('ytick', labelsize=font_size)
          
    if 'ress vs iter' in plots:
      #t = random.randrange(len(trials)) if len(trials)>1 else 0
      for eta_, eta in enumerate(etas):
        fig = plt.figure(figsize=(plot_size[0]*len(sizes), plot_size[1]))
        for N_, N in enumerate(sizes):
          plt.subplot(1,len(sizes),N_+1)
          plt.tight_layout(w_pad = 1+font_size/10)
          for tau_, tau in enumerate(batches):
            t = 0
            y_ = residuals[N_,eta_,tau_,t][residuals[N_,eta_,tau_,t]>0.]
            #print(y_)
            x_ = range(len(y_))
            if tau == 1:
              plt.plot(x_, y_, '-',color=cols[tau_],label='MultiSinkhorn',linewidth = line_width)
              plt.yscale('log')
            elif tau == 0:
              plt.plot(x_, y_, '-',color=cols[tau_],label='Cyclic Sinkhorn',linewidth = line_width)
              plt.yscale('log')
            elif tau == -1:
              plt.plot(x_, y_, '-',color=cols[tau_],label='Random Sinkhorn',linewidth = line_width)
            else: 
              plt.plot(x_, y_, '-',color=cols[tau_],label='BatchGreenkhorn({:2.0%})'.format(tau),linewidth = line_width)
            for t in range(1,len(trials)):
                y_ = residuals[N_,eta_,tau_,t][residuals[N_,eta_,tau_,t]>0.]
                x_ = range(len(y_))
                plt.plot(x_, y_, '-',color=cols[tau_],linewidth = line_width)            
          if titles: plt.title('m={} and $\eta=${}$||C||_\\infty$'.format(N,eta),fontsize = font_size)
          plt.xlabel('normalized cycles T', fontsize = font_size)
          plt.ylim(bottom = params['tolerance'])
          
          if N_==0:
              if leg: plt.legend( loc = 'upper right',fontsize = font_size)
              plt.ylabel('distance to polytope d',fontsize = font_size)

          plt.rc('xtick', labelsize=font_size)
          plt.rc('ytick', labelsize=font_size)


    if 'time vs size' in plots:
      mean_times = np.mean(times_total, axis=-1) if agregation=='mean' else np.sum(times_total, axis=-1)
      std_times = np.std(times_total, axis=-1)
      fig = plt.figure(figsize=(plot_size[0]*len(etas), plot_size[1]))
      for eta_, eta in enumerate(etas):
        plt.subplot(1,len(etas),eta_+1)
        plt.tight_layout(w_pad = 1+font_size/10)
        for tau_, tau in enumerate(batches):
          if tau == 1:
            plt.plot(np.array(sizes), mean_times[:,eta_,tau_].squeeze(), '-', marker = mrks[tau_],color=cols[tau_],label= 'MultiSinkhorn',linewidth = line_width)
            if agregation=='mean':
              plt.fill_between(np.array(sizes), mean_times[:,eta_,tau_].squeeze() - std_times[:,eta_,tau_].squeeze(), mean_times[:,eta_,tau_].squeeze() + std_times[:,eta_,tau_].squeeze(), color=cols[tau_], alpha=0.2)
          else: 
            if tau == 0:
              plt.plot(np.array(sizes), mean_times[:,eta_,tau_].squeeze(), '-', marker = mrks[tau_], color=cols[tau_],label='Cyclic Sinkhorn',linewidth = line_width)
              if agregation=='mean':
                plt.fill_between(np.array(sizes), mean_times[:,eta_,tau_].squeeze() - std_times[:,eta_,tau_].squeeze(), mean_times[:,eta_,tau_].squeeze() + std_times[:,eta_,tau_].squeeze(), color=cols[tau_], alpha=0.2)
            elif tau == -1:
              plt.plot(np.array(sizes), mean_times[:,eta_,tau_].squeeze(), '-', marker = mrks[tau_], color=cols[tau_],label='Random Sinkhorn',linewidth = line_width)
              if agregation=='mean':
                plt.fill_between(np.array(sizes), mean_times[:,eta_,tau_].squeeze() - std_times[:,eta_,tau_].squeeze(), mean_times[:,eta_,tau_].squeeze() + std_times[:,eta_,tau_].squeeze(), color=cols[tau_], alpha=0.2)
            else:
              plt.plot(np.array(sizes), mean_times[:,eta_,tau_].squeeze(), '-', marker = mrks[tau_], color=cols[tau_],label='BatchGreenkhorn({:2.0%})'.format(tau),linewidth = line_width)
              if agregation=='mean':
                plt.fill_between(np.array(sizes), mean_times[:,eta_,tau_].squeeze() - std_times[:,eta_,tau_].squeeze(), mean_times[:,eta_,tau_].squeeze() + std_times[:,eta_,tau_].squeeze(), color=cols[tau_], alpha=0.2)
        
        if titles: plt.title('Regularization $\eta=${}'.format(eta) ,fontsize=font_size)
        plt.xlabel('number of marginals m',fontsize=font_size)
        if eta_==0:
            if leg: plt.legend( loc = 'upper left',fontsize=font_size)
            plt.ylabel('time [s]',fontsize=font_size)
            
        plt.rc('xtick', labelsize=font_size)
        plt.rc('ytick', labelsize=font_size)


    if 'time vs reg' in plots:
      val = np.mean(times_total, axis=-1) if agregation=='mean' else np.sum(times_total, axis=-1)
      var = np.std(times_total, axis=-1)
      x = (1/np.array(etas))
      fig = plt.figure(figsize=(plot_size[0]*len(sizes), plot_size[1]))
      for N_, N in enumerate(sizes):
        plt.subplot(1,len(sizes),N_+1)
        plt.tight_layout(w_pad = 1+font_size/10)
        for tau_, tau in enumerate(batches):
          if tau == 1:
            plt.plot(x, val[N_,:,tau_].squeeze(), '-', marker = mrks[tau_],color=cols[tau_],label= 'MultiSinkhorn',linewidth = line_width)
            if agregation=='mean':
                plt.fill_between(x, val[N_,:,tau_].squeeze() - var[N_,:,tau_].squeeze(), val[N_,:,tau_].squeeze() + var[N_,:,tau_].squeeze(), color=cols[tau_], alpha=0.2)
          else: 
            if tau == 0:
              plt.plot(x, val[N_,:,tau_].squeeze(), '-', marker = mrks[tau_], color=cols[tau_],label='Cyclic Sinkhorn',linewidth = line_width)
              if agregation=='mean':
                plt.fill_between(x, val[N_,:,tau_].squeeze() - var[N_,:,tau_].squeeze(), val[N_,:,tau_].squeeze() + var[N_,:,tau_].squeeze(), color=cols[tau_], alpha=0.2)
            else:
              plt.plot(x, val[N_,:,tau_].squeeze(), '-', marker = mrks[tau_], color=cols[tau_],label='BatchGreenkhorn({:2.0%})'.format(tau),linewidth = line_width)
              if agregation=='mean':
                plt.fill_between(x, val[N_,:,tau_].squeeze() - var[N_,:,tau_].squeeze(), val[N_,:,tau_].squeeze() + var[N_,:,tau_].squeeze(), color=cols[tau_], alpha=0.2)
        
        if titles: plt.title('Number of marginals m={}'.format(N) ,fontsize=font_size)
        plt.xlabel('$||C||_{\infty}/\eta$',fontsize=font_size)
        if N_==0:
          if leg: plt.legend( loc = 'upper left',fontsize=font_size)
          plt.ylabel('time [s]',fontsize=font_size)
        
        plt.rc('xtick', labelsize=font_size)
        plt.rc('ytick', labelsize=font_size)


    if 'comp_time vs reg' in plots:
        
        comp = np.zeros_like(times_total)  
        comp[:,:,1,:] =  times_total[:,:,0] / times_total[:,:,1,:]  
        val = np.std(comp, axis=-1)
        var = np.std(comp, axis=-1)

        x = (1/np.array(etas))
        fig = plt.figure(figsize=(plot_size[0], plot_size[1]))
        plt.plot(x, x/x, '-',color=cols[0], linewidth = line_width)
      
        for N_, N in enumerate(sizes):
          plt.plot(x, val[N_,:,1].squeeze(), '-',color=cols[N_+1],label= 'm = {}'.format(N), linewidth = line_width)
          plt.fill_between(x, val[N_,:,1].squeeze() - var[N_,:,1].squeeze(), val[N_,:,1].squeeze() + var[N_,:,1].squeeze() , color=cols[N_+1], alpha=0.2)
         
        plt.xlabel('$||C||_{\infty}/\eta$',fontsize=font_size)
        if leg: 
            plt.legend( loc = 'upper left',fontsize=font_size)
        plt.ylabel('speedup factor $\\sigma_{\\tau}$',fontsize=font_size)

        plt.rc('xtick', labelsize=font_size)
        plt.rc('ytick', labelsize=font_size)

def summary(dataset):
    name = dataset['name']
    Ns = dataset['params']['sizes'] 
    etas = dataset['params']['regularizations']
    batches = dataset['params']['batches']
    trials = dataset['params']['trials']
    m = dataset['params']['m']
    residuals = dataset['results']['residuals']
    times = dataset['results']['total times']
    
    file_name = name + '_summary.txt'
    
    with open(file_name, 'w') as fl:
        print('\n',file = fl)
        print('='*60,file = fl) 
        print(' '*5+'Summary for '+name+' experiment '+' '*15,file = fl)
        print('='*60,file = fl) 
        
    for N_,N in enumerate(Ns):
        for eta_,eta in enumerate(etas):
            with open(file_name, 'a') as fl:
                print('\n',file = fl)
                print('   Residuals report for N = '+str(N)+',  eta = '+str(eta)+'||C||_inf',file = fl)
                print('%'*(24+trials*6),file = fl)
            header_str = '| Trial no. '+' '*(22-len('Trial no. '))
            for t in range(trials):
                str_ = '| '+str(t+1)
                header_str += str_+' '*(6-len(str_)) 
            header_str +='|'
            with open(file_name, 'a') as fl:                              
                print(header_str ,file = fl)
                print('|-----------------------'+'|:---:'*trials+'|',file = fl)
                
            for tau_,tau in enumerate(batches):
                if tau == 0:
                    alg_name = 'Cyclic Sinkhorn'
                elif tau==1:
                    alg_name = 'Sinkhorn' if m==2 else 'Greedy MultiSinkhorn'
                else:
                    alg_name = 'BatchGreenkhorn('+str(int(tau*100))+'%)'
                
                line_str = '| '+alg_name +' '*(22-len(alg_name))
                for t in range(trials):
                    count_str = '| '+str((residuals[N_,eta_,tau_, t,:]>0).sum()-1)
                    line_str += count_str+' '*(6-len(count_str))
                line_str +='|'
                with open(file_name, 'a') as fl:
                    print(line_str, file = fl)
        
    for N_,N in enumerate(Ns):
        for eta_,eta in enumerate(etas):
            with open(file_name, 'a') as fl:
                print('\n',file = fl)
                print('   Times report for N = '+str(N)+',  eta = '+str(eta)+'||C||_inf',file = fl)
                print('%'*(24+trials*10),file = fl)
            header_str = '| Trial no. '+' '*(22-len('Trial no. '))
            for t in range(trials):
                str_ = '| '+str(t+1)
                header_str += str_+' '*(10-len(str_)) 
            header_str +='|'
            with open(file_name, 'a') as fl:                              
                print(header_str ,file = fl)
                print('|-----------------------'+'|:-------:'*trials+'|',file = fl)
                
            for tau_,tau in enumerate(batches):
                if tau == 0:
                    alg_name = 'Cyclic Sinkhorn'
                elif tau==1:
                    alg_name = 'Sinkhorn' if m==2 else 'Greedy MultiSinkhorn'
                else:
                    alg_name = 'BatchGreenkhorn('+str(int(tau*100))+'%)'
                
                line_str = '| '+alg_name +' '*(22-len(alg_name))
                for t in range(trials):
                    count_str = '| '+str(int(times[N_,eta_,tau_, t]*100)/100)
                    line_str += count_str+' '*(10-len(count_str))
                line_str +='|'
                with open(file_name, 'a') as fl:
                    print(line_str, file = fl)     