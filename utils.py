#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 07:47:21 2022

@author: vkiit
"""

import numpy as np
import torch

from skimage.transform import resize

import matplotlib.colors as mcolors
import matplotlib.pylab as plt


######## UTILS

def KLfun(a,b):
    return (a*(a/b).log()-a+b).sum()

def distance(a,dl, metric = 'kl', log = False, reduce = True):    
    if metric == "kl":
        dist = -a*dl+a*(dl.exp()-1) if log else a*(dl-dl.log()-1)
    elif metric == "Linf" or metric == "L1":
        dist = a*( 1-(-dl).exp()).abs() if log else a*(dl-1).abs()
    else:
        print('Unsupported metric, using kl!')
        dist = -a*dl+a*(dl.exp()-1) if log else a*(dl-dl.log()-1)
    if reduce:
        dist = dist.sum()
    return dist.squeeze()

def softmin_tensor(eta, C, f, greedy_batch=None):
    if greedy_batch is None:
        B = C.shape[0]
        return -eta * (f.view(B, 1, -1) - C / eta).logsumexp(2).view(B, -1)
    else:
        B = C.shape[0]
        return -eta * (f.view(B, 1, -1)[greedy_batch] - C[greedy_batch,:] / eta).logsumexp(2).view(B, -1)

def lse_(a,b,c):
    alpha = torch.hstack((a,b)).logsumexp(1).view(a.shape[0],-1)         
    return alpha+(1-(c-alpha).exp()).log()

def lse__(a,b,flag = False):
    if flag:
        return a + (1-(b-a).exp()).log() 
    else:
        return torch.hstack((a,b)).logsumexp(1).view(a.shape[0],-1)
                
def kshape(x,idx):
    return x[idx.view(-1)].contiguous()


######## PLOTS & SAMPLING 

def make_1D_mix_gauss(n, m, s, dtype = "float64", gpu = True):
    """return a 1D histogram for a gaussian distribution on [0,1] (`n` bins, means `m` and stds `s`)

    Parameters
    ----------
    n : int
        number of bins in the histogram
    m : float
        mean value of the gaussian distribution
    s : float
        standard deviaton of the gaussian distribution

    Returns
    -------
    h : ndarray (`n`,)
        1D histogram for a gaussian distribution
    """
    dev = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    torchtype = torch.float32 if dtype == "float32" else torch.float64

    x = (1+torch.arange(n, dtype=torchtype, device = dev, requires_grad = False).reshape(n,1))/n
    
    h = (-(x - m[0]) ** 2 / (2 * s[0] ** 2)).exp()
    h /= h.sum()
    for i in range(len(m)-1):
        h1 = (-(x - m[i+1]) ** 2 / (2 * s[i+1] ** 2)).exp()
        h += h1/h1.sum()

    return h / len(m), x

def make_2D_mix_gauss(n, mx, sx, my, sy, dtype = "float64", gpu = True):
    """return a 2D histogram for a gaussian distribution on [0,1]x[0,1] (`n` bins, means `(mx,my) ` and stds `diag(sx,sy)`)

    Parameters
    ----------
    n : int
        number of bins in the histogram
    mx,my : float
        mean value of the gaussian distributions
    sx, sy : float
        standard deviatons of the gaussian distributions

    Returns
    -------
    h : ndarray (`n`,)
        2D histogram for a gaussian distribution
    """
    dev = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    torchtype = torch.float32 if dtype == "float32" else torch.float64

    ind = (1+torch.arange(n, dtype=torchtype, device = dev, requires_grad = False))/n
    x = torch.cartesian_prod(ind, ind)
  
    h = (-(x[:,0] - mx[0]) ** 2 / (2 * sx[0] ** 2)  -(x[:,1] - my[0]) ** 2 / (2 * sy[0] ** 2)).exp()
    h /= h.sum()
    for i in range(len(mx)-1):
        h1 = (-(x[:,0] - mx[i+1]) ** 2 / (2 * sx[i+1] ** 2) -(x[:,1] - my[i+1]) ** 2 / (2 * sy[i+1] ** 2) ).exp()
        h += h1/h1.sum()

    return h.reshape(n**2,1) / len(mx), x

def plot_2D_mix_gauss(ax, n, mx, sx, my, sy):
    """
    """
    ind = np.arange(n)/n
    x, y = np.meshgrid(ind, ind)
  
    h = np.exp(-(x - mx[0]) ** 2 / (2 * sx[0] ** 2)  -(y - my[0]) ** 2 / (2 * sy[0] ** 2))
    #h /= h.sum()
    for i in range(len(mx)-1):
        h1 = np.exp(-(x - mx[i+1]) ** 2 / (2 * sx[i+1] ** 2) -(y - my[i+1]) ** 2 / (2 * sy[i+1] ** 2) )
        h += h1#/h1.sum()
    h /= h.sum()#len(mx)

    #c = ax.pcolormesh(x, y, h, cmap='Greys', vmin=h.min(), vmax=h.max())
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    
def sample_and_resize(X, Y, n, d, gpu = False, dtype ='float64'):
    dev = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    torchtype = torch.float32 if dtype == "float32" else torch.float64
    
    labels = np.unique(Y)
    supports = []
    for label_, label in enumerate(labels):
        ind = np.random.choice(np.where(Y == label)[0], size=n, replace=False)
        X_ = np.zeros((n,d**2))
        for j in range(n):
            X_[j,:] = resize(X[ind[j],:], (d,d)).reshape(d**2)        
        supports.append(torch.as_tensor(X_, dtype=torchtype, device = dev))
    return supports

def sample(X, Y, n, gpu = False, dtype ='float64', labs = 'all'):
    dev = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    torchtype = torch.float32 if dtype == "float32" else torch.float64
    
    if labs == 'all':
        labels = np.unique(Y)
    else:
        labels = np.unique(Y)[0:labs]
    
    labels = np.unique(Y)
    supports = []
    for label_, label in enumerate(labels):
        ind = np.random.choice(np.where(Y == label)[0], size=n, replace=False)      
        supports.append(torch.as_tensor(X[ind,:].reshape(n,-1), dtype=torchtype, device = dev))
    return supports
   
                                    
def plot_3D_objects(data, t = 'all', plot_size = [6,6]):
    cols = ['k'] + list(list(mcolors.TABLEAU_COLORS)[i] for i in [3,2,0,6,1,4,5,7,8,9]) + ['gray','indigo', 'fuchsia','darkolivegreen','dodgerblue','gold','blueviolet','firebrick','lawngreen','darkturquoise']
    mrks = '.ov^sd+*-|'
    m = len(data[0])     
    if t == 'all':
        t = len(data)
    print(m,t)        
    fig = plt.figure(figsize=(plot_size[0]*t, plot_size[1]*m))
    #ax.set_box_aspect([1,1,1])
    for k in range(m):
        for j in range(t):
            ax = fig.add_subplot(m, t, k*t+j+1, projection="3d")
            #shift = torch.ones_like(data[1])
            #shift[:,0] *= 0
            #shift[:,1] *= 0
            #shift[:,2] *= 0
            data_ = data[j][k]#+shift*k
            ax.scatter(data_[:, 0], data_[:, 1], data_[:, 2], color = cols[3+k], marker = mrks[0])
            ax.axis('off')