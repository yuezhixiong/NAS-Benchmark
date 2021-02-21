import re, os, argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import entropy

parser = argparse.ArgumentParser()
parser.add_argument('--save', type=str, default='', help='path')
args = parser.parse_args()

def plot_loss_alpha(path):
    N = int(25000/64)
    # N = 1000
    acc_loss = []
    nop_loss = []
    adv_loss = []
    ood_loss = []
    flp_loss = []
    data = np.load(os.path.join(path, 'loss_data.npy'))
    for d in data:
        acc_loss.append(d['acc'])
        if 'nop' in d:
            nop_loss.append(d['nop'])
        else:
            nop_loss.append(np.nan)
        if 'adv' in d:
            adv_loss.append(d['adv'])
        else:
            adv_loss.append(np.nan)
        if 'ood' in d:
            ood_loss.append(d['ood'])
        else:
            ood_loss.append(np.nan)
        if 'flp' in d:
            flp_loss.append(d['flp'])
        else:
            flp_loss.append(np.nan)              
    
    # adv_outer = len(adv_loss)
    acc_loss = np.convolve(acc_loss, np.ones((N,))/N, mode='valid')[::N]
    nop_loss = np.convolve(nop_loss, np.ones((N,))/N, mode='valid')[::N]
    adv_loss = np.convolve(adv_loss, np.ones((N,))/N, mode='valid')[::N]
    ood_loss = np.convolve(ood_loss, np.ones((N,))/N, mode='valid')[::N]
    flp_loss = np.convolve(flp_loss, np.ones((N,))/N, mode='valid')[::N]
    # acc_loss = acc_loss[:N]
    # nop_loss = nop_loss[:N]

    plt.figure(figsize=(20,10))
    plt.suptitle(args.save)
    plt.subplot(221)
    plt.plot(acc_loss, label='acc')
    plt.plot(nop_loss, label='nop')
    plt.plot(adv_loss, label='adv')
    plt.plot(ood_loss, label='ood')
    plt.plot(flp_loss, label='flp')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    plt.subplot(222)
    data = np.load(os.path.join(path, 'sols.npy'))
    sols = []
    for l in data:
        # if len(l)>1:
        sols.append(l)
    sols = np.array(sols)
    _sols = np.zeros([len(sols), sols.shape[1]])
    for i in range(sols.shape[0]):
        n_obj = len(sols[i])
        if n_obj < 3:
            _sols[i, :n_obj] = sols[i]
        else:
            _sols[i, :] = sols[i]
    sols = _sols
    sols_avg = []
    for i in range(sols.shape[1]):
#         sols_avg.append(sols[:,i][::N])
        sols_avg.append(np.convolve(sols[:,i], np.ones((N,))/N, mode='valid')[::N])
    sols_avg = np.array(sols_avg)

    x = range(sols_avg.shape[1])
    n_obj = len(sols_avg)
    plt.bar(x, sols_avg[0], label='acc', alpha=0.5)
    if n_obj >=2:
        plt.bar(x, sols_avg[1], bottom=sols_avg[0], label='adv', alpha=0.5)
    if n_obj >=3:
        plt.bar(x, sols_avg[2], bottom=sols_avg[0]+sols_avg[1], label='nop', alpha=0.5)
    if n_obj >=4:
        plt.bar(x, sols_avg[3], bottom=sols_avg[0]+sols_avg[1]+sols_avg[2], label='ood', alpha=0.5)
    if n_obj >=5:
        plt.bar(x, sols_avg[4], bottom=sols_avg[0]+sols_avg[1]+sols_avg[2]+sols_avg[3], label='flp', alpha=0.5)
    plt.xlabel('epoch')
    plt.ylabel('MGDA weight')
    # plt.title(path)
    plt.legend()
    print(path, sols.sum(0)) # acc, adv, nop
    # plt.savefig(os.path.join(fig_path, 'plot_loss_sol.pdf'), format='pdf')
    
    alphas_reduce = np.load(os.path.join(path, 'alphas_reduce.npy'))[-1]
    alphas_normal = np.load(os.path.join(path, 'alphas_normal.npy'))[-1]
    print(alphas_reduce.min())
    print(alphas_normal.min())
    # plt.figure(figsize=(20,5))
    plt.subplot(223)
    plt.title('alphas_reduce')
    sns.heatmap(alphas_reduce, annot=True, fmt ='.5f', cmap="YlGnBu", cbar=False)
    plt.subplot(224)
    plt.title('alphas_normal')
    sns.heatmap(alphas_normal, annot=True, fmt ='.5f', cmap="YlGnBu", cbar=False)
    plt.savefig(os.path.join(fig_path, args.save+'.pdf'), format='pdf')

def plot_entropy(path, temperature='A'):
    alphas_normal = np.load(os.path.join(path, 'alphas_normal.npy'))
    alphas_reduce = np.load(os.path.join(path, 'alphas_reduce.npy'))
    alphas_normal_entropys = []
    alphas_reduce_entropys = []

    for epoch in alphas_normal:
        alphas_normal_entropys.append(entropy(epoch).mean())

    for epoch in alphas_reduce:
        alphas_reduce_entropys.append(entropy(epoch).mean())
    
    if temperature=='A':
        tau = [1 / 2**(epoch//10) for epoch in range(50)]
    elif temperature=='B':
        tau = [1 if epoch<10 else 0.1 for epoch in range(50)]
    elif temperature=='C':
        tau = [1 / 10**(epoch//10) for epoch in range(50)]
    elif temperature=='D':
        tau = [1 if epoch<10 else 0.00001 for epoch in range(50)]
    elif temperature=='none':
        tau = [1] * 50

    fig, ax_f = plt.subplots()
    ax_c = ax_f.twinx()
    ax_f.plot(alphas_normal_entropys, label='alphas_normal')
    ax_f.plot(alphas_reduce_entropys, label='alphas_reduce')
    ax_c.plot(tau, 'r-', label='tau')

    ax_f.set_xlabel('epoch')
    ax_f.set_ylabel('entropy')
    ax_c.set_ylabel('tau')
    fig.legend()
    fig.savefig(os.path.join(fig_path, 'plot_entropy.pdf'), format='pdf')

local_path = ''
path = os.path.join(local_path, args.save)
fig_path = os.path.join(local_path, args.save)
os.makedirs(fig_path, exist_ok=True)
# plot_entropy(path, temperature='none')
plot_loss_alpha(path)
# plot_alpha(path)