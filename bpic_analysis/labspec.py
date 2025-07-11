import numpy as np
import pandas as pd

from astropy.io import fits
from astropy import units as u

from scipy import interpolate
from scipy.optimize import curve_fit

from scipy import optimize
from bpic_analysis.broaden import *
from bpic_analysis.optical_depth import *

def smooth_labspec_special(x, y, X_wn=None):
    if X_wn is None:
        dataCOclump = np.loadtxt('COclump_CO2opticaldepth.txt')
        X_wn = dataCOclump[:,0]
    
    # x = wavenumber
    # y = absorbance
    # X_wn = wavenumber of NIRSpec data
    
    # convert to wavelength
    x_wavelength = (1 / (x/u.cm)).to(u.um).value
    X_wn_wavelength = (1 / (X_wn/u.cm)).to(u.um).value
    
    a = fits.getdata('jwst_nirspec_prism_disp.fits')
    
    # get all model values that fall in nirspec
    IDX = np.where((x_wavelength >= np.nanmin(a['WAVELENGTH'])) &
                    (x_wavelength <= np.nanmax(a['WAVELENGTH'])))
    x_wavelength_data, y_data = x_wavelength[IDX], y[IDX]
    
    # get all nirspec values that fall in model
    IDX2 = np.where((a['WAVELENGTH'] >= np.nanmin(X_wn_wavelength)) & 
                    (a['WAVELENGTH'] <= np.nanmax(X_wn_wavelength)))    
    
    a_wavelength, a_R = a['WAVELENGTH'][IDX2], a['R'][IDX2]

    # get model on same grid as nirspec
    f = interpolate.interp1d(x_wavelength_data, y_data)
    y_data_nirpsecgrid = f(a_wavelength)
    
    # broaded to nirspec
    mody = broaden(a_wavelength, y_data_nirpsecgrid, a_R)
    
    a_wavenumber = (1 / (a_wavelength*u.um)).to(1/u.cm).value
    
    return a_wavenumber, mody

def smooth_labspec(x, y, wv_center=4.26):
    '''BAD -- USE SPECIAL version'''
    a = fits.getdata('jwst_nirspec_prism_disp.fits')
 
    IDX = np.nanargmin(abs(a['WAVELENGTH']-wv_center), axis=0)
    R_nirspecI = a['R'][IDX]
    print(R_nirspecI)

    mody = broaden(x, y, R_nirspecI)
    return x, y, mody

def f(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B

def ice_wn_bounds(icefeature):
    if icefeature == 'H2O':
        wn_min, wn_max = 3000, 3600
    elif icefeature == '12CO2':
        wn_min, wn_max = 2320, 2400
    else:
        wn_min, wn_max = 2240, 2310
    return wn_min, wn_max

def plot_labspec(ax, norm_df, icefeature, labspec_ices_path, labspec_colors, labspec_leglabels, data_leglabel, top_label=True,
                  labspec_residual=False, xlabel = 'wavelength (um)', ylabel='normalized Fdisk/Fstar', 
                  ylabelerr='normalized Fdisk/Fstar err', **kwargs):
    
    wn_min, wn_max = ice_wn_bounds(icefeature)
    linewidth = kwargs.get('linewidth', 1)
    fontsize = kwargs.get('fontsize', 12)
    xlim = kwargs.get('xlim', [wn_max, wn_min])
    ylim = kwargs.get('ylim', [-0.025, 1.5])
    legend = kwargs.get('legend')
    legend_fontsize = kwargs.get('legend_fontsize', fontsize)

    X_wl = norm_df[xlabel].values
    X_wn = 1/(X_wl*u.um.to(u.cm))
    norm_absorbance, norm_absorbancerr = measure_absorbance(norm_df, icefeature, normalize=True, yerr=True, plot=False, 
                                                            xlabel = xlabel, ylabel=ylabel, ylabelerr=ylabelerr)

    ax.plot(X_wn, norm_absorbance, color='k', linewidth=linewidth+1, label=data_leglabel, ds='steps-post')
    # ax.errorbar(X_wn, norm_absorbance, yerr=abs(norm_absorbancerr),fmt='none',
    #              color='k', alpha=0.1, capsize=2, lw=1)
    
    for i, ice in enumerate(labspec_ices_path):
        dat = np.loadtxt(ice, comments='#')
        x = dat[:,0]
        y = dat[:,1]

        if icefeature == '13CO2':
            # remove continuum
            idx = np.where(((x>2250) & (x<2260)) | ((x>2290)&(x<2296)))
            popt, pcov = curve_fit(f, x[idx], y[idx] ) 
            cy = f(x,*popt)
            y = y - cy
            idx0 = np.where((x>2240)&(x<2265) | ((x>2300)&(x<2310)))
            y[idx0] = 0
            idx_sm = np.where((X_wn>2240) & (X_wn<2310))
            xsm, ysm = smooth_labspec_special(x, y, X_wn=X_wn[idx_sm])
        elif icefeature == 'H2O':
            idx_sm = np.where((X_wn>2600) & (X_wn<4000))
            xsm, ysm = smooth_labspec_special(x, y, X_wn=X_wn[idx_sm])
        else:
            xsm, ysm = smooth_labspec_special(x, y)

        ff = interpolate.interp1d(xsm, ysm, kind=3)
        newx = np.linspace(np.min(xsm), np.max(xsm), 500)
        newy = ff(newx)

        ff = interpolate.interp1d(X_wn, norm_absorbance, kind=3)
        newy_real = ff(newx)

        ind = np.where((newx>wn_min) & (newx<wn_max))
        IDXmod = np.max(newy[ind])

        ax.plot(newx, newy/IDXmod,  linestyle='-', color=labspec_colors[i], linewidth=linewidth, alpha=1,label=labspec_leglabels[i])
        if labspec_residual:
            ind = np.where((x>wn_min) & (x<wn_max))
            IDXmod = np.max(y[ind])
            ax.plot(x, y/IDXmod,  linestyle='--', color=labspec_colors[i], linewidth=0.5, alpha=0.5)

    bplot.plot_axes(ax, 'Normalized optical depth (τ)', xlabel='Wavenumber (cm$^{-1}$)', fontsize=fontsize, xlim = xlim, ylim=ylim, legend=legend, legend_fontsize=legend_fontsize, top_label=top_label)
    return ax
              

def one_gaussian(x, height, center, width):
    return height*np.exp(-(x - center)**2/(2*width**2)) 

def two_gaussians(x, h1, c1, w1, h2, c2, w2):
    return three_gaussians(x, h1, c1, w1, h2, c2, w2, 0,0,1)

def three_gaussians(x, h1, c1, w1, h2, c2, w2, h3, c3, w3):
    return (one_gaussian(x, h1, c1, w1) +
        one_gaussian(x, h2, c2, w2) +
        one_gaussian(x, h3, c3, w3))

def plot_ice_multigaussian(ax, norm_df, icefeature, guess, wn_min, wn_max, num_gauss=2, get_stats=True,
                           xlabel = 'wavelength (um)', ylabel='normalized Fdisk/Fstar', 
                  ylabelerr='normalized Fdisk/Fstar err', top_label=True, **kwargs):
    wn_min, wn_max = ice_wn_bounds(icefeature)

    data_color = kwargs.get('data_color', 'r')
    gaussian_color = kwargs.get('gaussian_color', 'k')
    fontsize = kwargs.get('fontsize', 12)
    xlim = kwargs.get('xlim', [wn_max, wn_min])
    ylim = kwargs.get('ylim', [0.03, -0.025])

    X_wl = norm_df[xlabel].values
    X_wn = 1/((X_wl)*u.um.to(u.cm))
    continuum, popt, optical_depth, optical_depth_err = measure_optical_depth(norm_df, icefeature, yerr=True, plot=False, xlabel=xlabel, ylabel=ylabel, ylabelerr = ylabelerr)

    ax.plot(X_wn, optical_depth, color=data_color, linewidth=4)
    IDX = np.where((X_wn < wn_min) & (X_wn>wn_max)) 
    X, Y = X_wn[IDX][::-1], optical_depth[IDX][::-1]
    Y[Y<=0]=0
    
    xx = np.linspace(wn_max, wn_min, 100)

    if num_gauss == 1:
         func = one_gaussian
    elif num_gauss == 2:
        func = two_gaussians
    else:
        func = three_gaussians

    errfunc = lambda p, x, y: (func(x, *p) - y)**2
    optim, pcov, infodict, errmsg, success = optimize.leastsq(errfunc, guess[:], args=(X, Y), full_output=1)
    if (len(Y) > len(guess)) and pcov is not None:
        s_sq = (errfunc(optim, X, Y)**2).sum()/(len(Y)-len(guess))
        pcov = pcov * s_sq
    else:
        pcov = np.inf

    error = [] 
    for i in range(len(optim)):
        try:
            error.append(np.absolute(pcov[i][i])**0.5)
        except:
            error.append( 0.00 )
    perr_leastsq = np.array(error) 

    ax.plot(xx, func(xx, *optim), color=gaussian_color, linewidth=2)
    linestyles = ['--', '-.', ':']
    j = 0
    for i in np.arange(len(optim))[::3]:
        ax.plot(xx, one_gaussian(xx, optim[i], optim[i+1], optim[i+2]), color=gaussian_color, linewidth=1, linestyle=linestyles[j])
        j+=1

    axT = plot_top_wavelength_axis(ax, top_label=top_label, fontsize=fontsize)
    plot_axes(ax, 'reverse optical depth', xlabel='wavenumber (cm$^{-1}$)', fontsize=fontsize, xlim = xlim, ylim=ylim, legend=legend, legend_fontsize=legend_fontsize)

    if get_stats:
        centers = [optim[i] for i in np.arange(1, len(optim))[::3]]
        centererr = [perr_leastsq[i] for i in np.arange(1, len(perr_leastsq))[::3]]
        stds = [optim[i] for i in np.arange(2, len(optim)+1)[::3]]
        stderrs = [perr_leastsq[i] for i in np.arange(2, len(perr_leastsq))[::3]]
        fwhm = 2.355*np.array(stds )
        d = {'center':centers, 'center err':centererr, 'std':stds, 'std err':stderrs, 'fwhm':fwhm}
        df = pd.DataFrame(d, [f'gaussian {i+1}' for i in np.arange(num)])
        return df
    