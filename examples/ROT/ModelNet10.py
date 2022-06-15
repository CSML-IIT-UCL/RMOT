#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 11:18:58 2021

@author: vkiit
""" 

from benchmarks import benchmark, my_plot

import torch
import numpy as np

from torch_geometric.transforms import SamplePoints


if ~('ModelNet' in globals()):
    from torch_geometric.datasets import ModelNet
    ModelNetData = ModelNet('ModelNetData/', name='10', train = False)    

dataset = dict()
dataset['name'] = 'ModelNet3D_final2_scale'
dataset['type'] = 'point clouds'

pairs = 3

params = dict()
params['dimension'] = 3
params['m'] = 2
params['cost'] = "SqDist(x,y)"
params['gpu'] = True
params['sizes'] = [10000, 30000, 50000] if (params['gpu'] and torch.cuda.is_available()) else [100, 300, 500]
params['regularizations'] = [0.05, 0.02, 0.01, 0.005]
params['batches'] = [1, 0.5, 0.25, 0.1, 0.01]
params['trials'] = 10
params['metric'] = "Linf"
params['tolerance'] = 1e-6
params['maxIter'] = 10000

dataset['params'] = params

np.random.seed(13)
sources_ = np.random.choice(range(len(ModelNetData)), size=pairs, replace=False)
targets_ = np.random.choice(range(len(ModelNetData)), size=pairs, replace=False)

for p_ in range(pairs):
    
    # generating data
    data = dict()
    data['marginals'] = []
    data['support'] = []
    for n_, n in enumerate(params['sizes']):
        ModelNetData.transform = SamplePoints(num = n)
        
        print('size: ',n)
        data['marginals'].append( list( [ModelNetData[sources_[p_]].pos.to(torch.float64), ModelNetData[targets_[p_]].pos.to(torch.float64) ] for _ in range(params['trials'])))

    dataset['data'] = data

    # computing
    dataset['results'] = benchmark(dataset)

    # saving results, avoiding data to reduce file size
    dataset['data'] = []
    np.save(dataset['name']+str(p_),dataset)

    # plotting results
    my_plot(dataset,plots=['comp_res vs iter','comp_time vs reg'])