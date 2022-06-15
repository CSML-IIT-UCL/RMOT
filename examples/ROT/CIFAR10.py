#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 11:18:58 2021

@author: vkiit
"""

from benchmarks import benchmark, summary
from utils import sample
import torch
import numpy as np

if ~('fashion_mnist' in globals()):
    from keras.datasets import cifar10 
    (X_train, Y_train), _ = cifar10.load_data()
    

dataset = dict()
dataset['name'] = 'LabelDist_CIFAR10_keops'
dataset['type'] = 'point clouds'

## setting up the parameters for the benchmark
d = 32 #dimension of the rscaled image dxd
l = 10 #no. of labels

params = dict()
params['dimension'] = 3*(d**2)
params['m'] = 2
params['cost'] = "SqDist(x,y)"
params['gpu'] = True
params['sizes'] = [2500, 5000] if (params['gpu'] and torch.cuda.is_available()) else [250, 500]
params['regularizations'] = [1/5, 1/10, 1/15, 1/20, 1/25]
params['batches'] = [1, 0.5, 0.25, 0.125]
params['trials'] = int(l*(l-1)/2)
params['metric'] = "Linf"
params['tolerance'] = 1e-6
params['maxIter'] = 5000


# generating data
data = dict()
data['marginals'] = []
data['support'] = []

for n_, n in enumerate(params['sizes']):
    sample_supports = sample(X_train, Y_train, n, gpu =  False)
    print('size: ',n)
    marginals = []
    for t1 in range(l):
        for t2 in range(t1):
            marginals.append([sample_supports[t1], sample_supports[t2]])    
    data['marginals'].append(marginals)


dataset['params'] = params
dataset['data'] = data

# computing
dataset['results'] = benchmark(dataset)

# saving results
dataset['data'] = []
np.save(dataset['name'],dataset)

#making the summary
summary(dataset)