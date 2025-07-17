import matplotlib
from matplotlib.colors import Normalize
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import glob as glob

def color_map(bins=10):
    min_val, max_val = 0.03, 0.99
    orig_cmap = plt.cm.bone_r
    colors = orig_cmap(np.linspace(min_val, max_val, bins))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)
    cmap.set_bad('k')
    return cmap


def corner_plot(fil, bounds, colnames, savedir=True, **kwargs):
    figsize= kwargs.get('figsize', (16,16))
    fs = kwargs.get('fs', 14)
    bins = kwargs.get('bins',6)
    cmap = kwargs.get('cmap', color_map(bins=bins))
    vmin = kwargs.get('vmin', 0)
    vmax = kwargs.get('vmax', 10)
    get_stats = kwargs.get('get_stats')
    # ylim

    params, truths, data, chi2 = parse_data(fil, chi2_max=50)
    M = len(truths)
    nu = len(bounds)

    fig, axs = plt.subplots(figsize=figsize,ncols=M, nrows=M)
    for M1 in np.arange(M):
        for M2 in np.arange(M):
            if M2 > M1:
                ax = axs[M1,M2]
                ax.remove()
    label = colnames 
    rang=bounds 
    datacolnames = data.columns.to_list()
    chi2 = chi2.values

    norm = Normalize(vmin=vmin, vmax=vmax)
    for i in range(M):
        x = data[datacolnames[i]].to_list()
        ax = axs[i,i]
        bin_means, bin_edges, binnumber = stats.binned_statistic(x,chi2, 'min', bins=bins)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2
        ax.bar(bin_centers,bin_means, bin_width, color=cmap(norm(bin_means)), edgecolor='k')
        ax.axvline(truths[i], color='r', linewidth=1)
        if i != M-1:
            ax.set_xticklabels([])
        ax.set_ylabel('$\chi_r^2$', fontsize=fs)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.tick_params(which='both', direction='in', top=True, right=True, left=True, bottom=True, labelsize=fs)
        ax.minorticks_on()
        if '(deg)' in label[i]:
            lab = label[i].split(' (deg)')[0]
            ax.set_title(lab + ' = ' + str(round(truths[i],2)) + '$^\circ$', fontsize=fs)
        elif '(AU)' in label[i]:
            lab = label[i].split(' (AU)')[0]
            ax.set_title(lab + ' = ' + str(round(truths[i],2)) + ' AU', fontsize=fs)
        else:
            lab = label[i]
            ax.set_title(lab + ' = ' + str(round(truths[i],2)), fontsize=fs)
        ax.set_xlim(bin_centers[0]-bin_width/2, bin_centers[-1]+bin_width/2,)
        print(bin_centers[0]-bin_width/2, bin_centers[-1]+bin_width/2,)
        horline = bin_means.min() + np.sqrt(2/nu)
        ax.axhline(horline, color='k', linestyle='--')
        if not np.isfinite(horline):
            horline = 2
        ylim = kwargs.get('ylim',[0.05, horline+0.2] )
        ax.set_ylim(ylim)
        
    for j in range(M):
        for i in range(j):
            x = data[datacolnames[j]].to_list()
            y = data[datacolnames[i]].to_list()
            ax = axs[j,i]
   
            ax.axvline(truths[i], color='r', linewidth=1)
            ax.axhline(truths[j], color='r', linewidth=1)
            ax.plot(truths[i], truths[j], 'sr', markersize=8)

            ret = stats.binned_statistic_2d(y,x,chi2, 'min', bins=bins)
            val = ret.statistic.T
            if get_stats:
                print(val.min(), val.max())
            im = ax.pcolormesh(ret.x_edge, ret.y_edge, ret.statistic.T,
                cmap=cmap, vmax=vmax, vmin=vmin)

            # ax.set_xlim(rang[i])
            # ax.set_ylim(rang[j])
            ax.set_xlim(ret.x_edge[0], ret.x_edge[-1])
            ax.set_ylim(ret.y_edge[0], ret.y_edge[-1])
            if j != M-1:
                ax.set_xticklabels([])
            if i != 0:
                ax.set_yticklabels([])

            if i == 0:
                ax.set_ylabel(label[j], fontsize=fs)
            if j == M-1:
                ax.set_xlabel(label[i], fontsize=fs)
            ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=fs)
            ax.minorticks_on()
            
            xleft, xright = ax.get_xlim()
            ybottom, ytop = ax.get_ylim()

            ax.set_aspect(abs((xright-xleft)/(ybottom-ytop)))
    plt.subplots_adjust(wspace=0.08, hspace=0.08) 
    if savedir:
        plt.savefig(savedir, dpi=150)
        
    
def parse_data(fil, chi2_max=4):
    all_params = pd.read_csv(fil)
    params =list(all_params.columns)
    params

    truths = all_params.iloc[all_params['chi2'].argmin()]
    truths = truths.drop('chi2').to_list()

    data = all_params.drop('chi2', axis=1)
    chi2 = all_params['chi2']
    if chi2_max is not None:
        data = data.loc[chi2 < chi2_max]
        chi2 = chi2.loc[chi2 < chi2_max]
    return params, truths, data, chi2

def print_acceptable_range_chi2(fil, nu):
    params, truths, data, chi2 = parse_data(fil, chi2_max=50)

    chi2_min = chi2.min()
    horline = chi2.min() + np.sqrt(2/nu)
   
    c = data.loc[chi2 < horline]
    
    print('Best fit params:')
    print(data.loc[chi2==np.min(chi2)].iloc[0])
    
    print('\n chi2', chi2_min )
    print('\n chi2 line: ', round(horline,3), '\n')
    for colname in list(c.columns):
        print(f"{colname}: {round(c[colname].min(),2)}-{round(c[colname].max(),2)}")

    print('\n # DE rounds: ', len(data))
    return chi2_min