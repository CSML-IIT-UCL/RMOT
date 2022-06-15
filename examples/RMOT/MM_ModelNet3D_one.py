#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 11:18:58 2021

@author: vkiit
"""

from benchmarks import benchmark, benchmark_multi 
from utils import my_plot
import torch
import numpy as np


from torch_geometric.transforms import SamplePoints


if ~('ModelNet' in globals()):
    from torch_geometric.datasets import ModelNet
    ModelNetData = ModelNet('ModelNetData/', name='10', train = False)    

dataset = dict()
dataset['name'] = 'MM_ModelNet3D_final_m'
dataset['type'] = 'point clouds'



params = dict()
params['dimension'] = 3
params['m'] = 6
dataset['name'] = 'MM_ModelNet3D_one_m' + str(params['m'])
params['cost'] = "SqDist(x,y)" if params['m']==2 else "SqDist(x,y)"
params['gpu'] = True
params['sizes'] = [8, 16] if (params['gpu'] and torch.cuda.is_available()) else [8, 16, 24]
params['regularizations'] = [0.5 , 0.1, 0.05, 0.01]
#params['regularizations'] = [0.5, 0.1, 0.05, 0.01, 0.005]
#params['batches'] = [1, 0.9 , 0.8 , 0.7 , 0.6 , 0.5 , 0.45, 0.4 , 0.35, 0.3 , 0.25, 0.2 , 0.15, 0.1 , 0.05, 0.01, 0.005, 0.001]
#params['batches'] = 1 / np.array( [1, 2, 4, 5, 10, 20] )
params['batches'] = [0, 1, 0.5, 0.25, 0.125]
params['trials'] = 10
params['metric'] = "Linf"
params['tolerance'] = 1e-6
params['maxIter'] = 5000


## generating data
data = dict()
data['marginals'] = []
data['support'] = []

np.random.seed(13)
marginals_ = np.random.choice(range(len(ModelNetData)), size=params['m'], replace=False)


for n_, n in enumerate(params['sizes']):
    ModelNetData.transform = SamplePoints(num = n)
    
    print('size: ',n)
    data['marginals'].append( list( list(ModelNetData[marginals_[k]].pos.to(torch.float64) for k in range(params['m']) ) for _ in range(params['trials'])))

dataset['params'] = params
dataset['data'] = data

if params['m']>2:
    dataset['results'] = benchmark_multi(dataset)
    dataset['data'] = []
    np.save(dataset['name'],dataset)
else:
    dataset_1, dataset_2 = dataset.copy(), dataset.copy() 
    dataset_1['results'] = benchmark(dataset)
    dataset_2['results'] = benchmark_multi(dataset)

    np.save(dataset['name']+'_keops',dataset_1)
    np.save(dataset['name']+'_torch',dataset_2)


my_plot(dataset)