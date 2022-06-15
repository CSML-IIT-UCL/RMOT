#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 11:18:58 2021

@author: vkiit
"""

from benchmarks import benchmark_multi_2, my_plot_2
import torch
import numpy as np


dataset = dict()
dataset['name'] = 'Diffrent_m5_'
dataset['type'] = 'point clouds'



params = dict()
params['dimension'] = 1
params['m'] = [3,6,9,12]
params['cost'] = "Barycentric"
params['gpu'] = False
params['sizes'] = [5]
params['regularizations'] = [0.1, 0.05]
params['batches'] = [0, 1]
params['trials'] = 10
params['metric'] = "Linf"
params['tolerance'] = 1e-6
params['maxIter'] = 5000


## generating data
data = dict()
data['marginals'] = []
data['support'] = []

np.random.seed(13)


for m_, m in enumerate(params['m']):
    print('marginals: ',m)
    data['marginals'].append( list( list( torch.normal(mean = torch.linspace(-1., 1.,params['sizes'][0]),std = 0.1).reshape(params['sizes'][0],1) for k in range(m) ) for _ in range(params['trials'])))

dataset['params'] = params
dataset['data'] = data

# computing
dataset['results'] = benchmark_multi_2(dataset)

# saving results
dataset['data'] = []
np.save(dataset['name'],dataset)

# ploting the results
my_plot_2(dataset, plots=['comp_res vs iter'])