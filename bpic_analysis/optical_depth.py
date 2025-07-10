import numpy as np
import pandas as pd

from astropy import units as u
from astropy import constants as const
from astropy import modeling

import matplotlib.pyplot as plt

from lmfit.models import GaussianModel
from lmfit.models import SkewedGaussianModel

def measure_absorbance(norm_df, icefeature, normalize=True, yerr=True, plot=False, xlabel = 'wavelength (um)', ylabel='normalized Fdisk/Fstar', ylabelerr='normalized Fdisk/Fstar err', **kwargs):
    x = norm_df['wavelength (um)']
    _, _, optical_depth, optical_depth_err = measure_optical_depth(norm_df, icefeature, yerr=yerr, plot=plot, xlabel=xlabel, ylabel=ylabel, ylabelerr=ylabelerr, **kwargs)
    X_wn = 1/(x*u.um.to(u.cm))

    absorbance = optical_depth/np.log(10)
    
    if yerr:
        absorbanceerr = optical_depth_err / np.log(10)
    else:
        absorbanceerr = None

    if normalize:
        if icefeature == '12CO2':
            wn_min, wn_max = 2320, 2400
        elif icefeature == '13CO2':
            wn_min, wn_max = 2240, 2310

        ind = np.where((X_wn>wn_min) & (X_wn<wn_max))
        norm_absorbance = absorbance / np.nanmax(absorbance[ind])
        if yerr:
            norm_absorbancerr = norm_absorbance * np.sqrt((absorbanceerr/absorbance)**2. + (absorbanceerr.values[ind][np.argmax(absorbance.values[ind])]/np.nanmax(absorbance.values[ind]))**2.)
            norm_absorbancerr =  norm_absorbancerr.values
        else:
            norm_absorbancerr = None
        return norm_absorbance, norm_absorbancerr
    else:
        absorbance, absorbanceerr
    

def measure_optical_depth(norm_df, icefeature, yerr=True, plot=False, xlabel = 'wavelength (um)', ylabel='normalized Fdisk/Fstar', ylabelerr='normalized Fdisk/Fstar err', **kwargs):
    ylims = kwargs.get('ylims', [0,1])
    if icefeature == 'H2O':
        continuum = norm_df.loc[((norm_df[xlabel]> 2.0) & 
                                        (norm_df[xlabel]< 2.6)) | 
                                        ((norm_df[xlabel]> 3.4) & 
                                        (norm_df[xlabel]< 3.6))]
        xmin, xmax = 2, 4
    elif icefeature == '12CO2':
        continuum = norm_df.loc[((norm_df[xlabel]> 4) & 
                                        (norm_df[xlabel]< 4.18)) | 
                                        ((norm_df[xlabel]> 4.32) & 
                                        (norm_df[xlabel]< 4.34)) |
                                        ((norm_df[xlabel]> 4.445) & 
                                        (norm_df[xlabel]< 4.5))]
        xmin, xmax = 4., 4.5

    elif icefeature == '13CO2':
        continuum = norm_df.loc[((norm_df['wavelength (um)']> 4.3) & 
                                        (norm_df['wavelength (um)']< 4.37)) | 
                                        ((norm_df['wavelength (um)']> 4.405) & 
                                        (norm_df['wavelength (um)']< 4.5)) ]
        xmin, xmax = 4.3, 4.5
    z = np.polyfit(continuum['wavelength (um)'], continuum[ylabel],4)
    p = np.poly1d(z)
    optical_depth = (-np.log(norm_df[ylabel]/p(norm_df[xlabel]))).values
    if plot:
        plt.figure()
        plt.plot(norm_df[xlabel], norm_df[ylabel], 'k')
        plt.plot(norm_df[xlabel], p(norm_df[xlabel]), 'r')
        plt.xlabel('wavelength')
        plt.ylabel('normalized Fdisk/Fstar')
        plt.xlim(xmin, xmax)
        plt.ylim(ylims)
        plt.show()

    if yerr:
        optical_deptherr = (norm_df[ylabelerr]/(norm_df[ylabel])).values
    else:
        optical_deptherr = None
    return p(norm_df[xlabel]), z, optical_depth, optical_deptherr

def ice_gaussian_plot(ax, norm_df, icefeature, opdepth=True, skewedgaussian=True, xlabel = 'wavelength (um)', ylabel='normalized Fdisk/Fstar', ylabelerr='normalized Fdisk/Fstar err', mask_bump=False, top_label=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(4,4))
    label = kwargs.get('label', 'data')
    data_color = kwargs.get('data_color', 'k')
    gaussian_color = kwargs.get('gaussian_color', 'r')
    fs = kwargs.get('fs', 12)
    ylim = kwargs.get('ylim', [0.3, -0.1])


    X_wv = norm_df['wavelength (um)'].values
    X_wn = 1/(norm_df['wavelength (um)'].values*u.um.to(u.cm))
    data = norm_df[ylabel].values
    data_err = norm_df[ylabelerr].values

    if icefeature == 'H2O':
        wn_min, wn_max = 2800, 4000
        guassian_wn_min, gaussian_wn_max = 3000, 3750
        left_wn_min, left_wn_max =3000,  3511
        right_wn_min, right_wn_max =3511 , 3750
        A = 2.0e-16 #Gerakines1995
        # A = 1.5e-16 #Bouilloud2015
        # A = 2.2e-16 # Bouilloud2015 from corrected Gerakines1995
        erridx = -5
    elif icefeature == '12CO2':       
        wn_min, wn_max = 2308, 2440
        guassian_wn_min, gaussian_wn_max = 2300, 2440
        left_wn_min, left_wn_max =2300,  2340
        right_wn_min, right_wn_max =2400,  2440
        A = 7.6e-17 #Gerakines1995
        # A = 7.6e-17 #Bouilloud2015
        # A = 1.1e-16 #Bouilloud2015 from corrected Gerakines1995 
        erridx = -2
    elif icefeature == '13CO2':
        wn_min, wn_max = 2240, 2308
        guassian_wn_min, gaussian_wn_max = 2267, 2288
        left_wn_min, left_wn_max = 2250,2275
        right_wn_min, right_wn_max = 2275, 2285
        A = 7.8e-17  #Gerakines1995
        # A = 6.8e-17  #Bouilloud2015
        # A = 1.15e-16 #Brunken2024 from corrected Gerakines1995 from Bouilloud2015
        erridx = -2
    else:
        raise ValueError('icefeature should be H2O, 12CO2, or 13CO2')
    
    continuum, popt, optical_depth, optical_depth_err = measure_optical_depth(norm_df, icefeature, yerr=True, plot=False, xlabel=xlabel, ylabel=ylabel, ylabelerr = ylabelerr)
    IDX = np.where((X_wn > wn_min) & (X_wn < wn_max))

    if opdepth:
        fin_data = optical_depth
        fin_data_err = optical_depth_err
    else:
        fin_data = data/continuum
        fin_data_err = abs(data_err/continuum)

    ax.plot(X_wn[IDX], fin_data[IDX], color=data_color,label=label)
    ax.errorbar(X_wn[IDX][::erridx], fin_data[IDX][::erridx], yerr=fin_data_err[IDX][::erridx], fmt='none', color=data_color, alpha=0.5)

    if skewedgaussian:
        model = SkewedGaussianModel()
        IDX_gauss = np.where((X_wn > guassian_wn_min) & (X_wn<gaussian_wn_max)) 
        y = optical_depth[IDX_gauss]
        yerr = optical_depth_err[IDX_gauss]
        params = model.guess(((np.nan_to_num(y))), x=X_wn[IDX_gauss])  
        result = model.fit(((np.nan_to_num(y))), params, x=X_wn[IDX_gauss], weights=1./yerr, scale_covar=True)
        result_err = result.eval_uncertainty()
        
        yeval = model.eval(result.params, x=X_wn[IDX_gauss])
        chi2 = np.nansum ( (y - yeval)**2. / yerr**2.) / (len(y)-4)
        print('chi2' ,chi2) 

        xx = np.linspace(wn_min, wn_max,100)
        yeval = model.eval(result.params, x=xx)
        ax.plot(xx, yeval, color=gaussian_color, zorder=100, linestyle='--')
        # print(result.fit_report())

        a = result.params['gamma'].value
        ae = result.params['gamma'].stderr
        c = result.params['center'].value
        ce = result.params['center'].stderr
        w = result.params['sigma'].value
        we = result.params['sigma'].stderr

        delta = a / np.sqrt(1 + a**2.)
        m = np.sqrt(2/np.pi) * delta - (1-(np.pi/4)) * ((np.sqrt(2/np.pi)*delta)**3/(1-(2/np.pi)*delta**2.)) - (np.sign(a) / 2 ) * np.exp(-(2*np.pi)/ abs(a))
        wm = w * m
        mode = c + wm

        de = ( 1/ (1+a**2)**(3/2)) * ae
        Ae = np.sqrt(2/np.pi) * de
        Be = de * abs( ((np.pi-4) * (3*np.pi*delta**2. - 2*delta**4.)) / (np.sqrt(2*np.pi) * (np.pi - 2*delta**2.)**2.) )
        Ce = ((np.pi / a**2.) * np.exp(-2*np.pi / abs(a))) * ae
        me = Ae -  Be - Ce
        wme = wm * np.sqrt((we / w)**2. + (me / m)**2.)
        mode_err = np.sqrt(ce**2. + wme**2.)
        height = np.nanmax(yeval)

        # FIT EACH SIDE
        fitter = modeling.fitting.LMLSQFitter(calc_uncertainties=True)
        xxL = np.linspace(mode, left_wn_min,100)
        IDXL = np.where((X_wn > left_wn_min) & (X_wn<mode)) 
        modelL = modeling.models.Gaussian1D(amplitude=height, mean = mode, stddev=5,
                                            fixed={'mean':True, 'amplitude':True}) 

        xxR = np.linspace(right_wn_max, mode,100)
        IDXR = np.where((X_wn >= mode) & (X_wn<right_wn_max)) 
        modelR = modeling.models.Gaussian1D(amplitude=height, mean = mode, stddev=15, 
                                            fixed={'mean':True, 'amplitude':True}) 

        if opdepth:
            fitted_modelL = fitter(modelL, X_wn[IDXL][::-1], fin_data[IDXL][::-1])
            # ax.plot(xxL, fitted_modelL(xxL), color='b',linestyle='--', zorder=101) 

            fitted_modelR = fitter(modelR, X_wn[IDXR][::-1], fin_data[IDXR][::-1])
            # ax.plot(xxR, fitted_modelR(xxR), color='g', linestyle='-.', zorder=101) 
        else:
            fitted_modelL = fitter(modelL, X_wn[IDXL][::-1], 1-(fin_data[IDXL][::-1]))
            # ax.plot(xxL, 1-fitted_modelL(xxL), color=gaussian_color,linestyle='--')

            fitted_modelR = fitter(modelR, X_wn[IDXR][::-1], 1-(fin_data[IDXR][::-1]))
            # ax.plot(xxR, 1-fitted_modelR(xxR), color=gaussian_color, linestyle='-.')  

        fwhm_blue = 2.355*fitted_modelL.stddev.value
        fwhm_blueerr = fitted_modelL.stddev.std
        fwhm_red = 2.355*fitted_modelR.stddev.value
        fwhm_rederr = fitted_modelR.stddev.std 

        # height2 = 0.3989423 * result.params['amplitude'] / ((fitted_modelR.stddev.value/2 + fitted_modelL.stddev.value/2))
        
        height_err = height * np.sqrt( (result.params['amplitude'].stderr / result.params['amplitude'])**2. + (fitted_modelL.stddev.std**2. + fitted_modelR.stddev.std **2.)/(fitted_modelL.stddev.value+fitted_modelR.stddev.value)**2. )


    else:
        model = GaussianModel()
        IDX_gauss = np.where((X_wn > guassian_wn_min) & (X_wn < gaussian_wn_max))
        
        if mask_bump:
            fin_data[fin_data>0.035]=0.03

        if opdepth:
            y = fin_data[IDX_gauss]
            yerr = fin_data_err[IDX_gauss]
        else:
            y = 1-(fin_data[IDX_gauss][::-1])
            yerr = fin_data_err[IDX_gauss][::-1]

        params = model.guess(np.nan_to_num(y), x=X_wn[IDX_gauss])    # depending on the data you need to
        result = model.fit(np.nan_to_num(y), params, x=X_wn[IDX_gauss],  weights=1./yerr, scale_covar=False)
        result_err = result.eval_uncertainty()
        yeval = model.eval(result.params, x=X_wn[IDX_gauss])
        chi2 = np.nansum ( (y - yeval)**2. / yerr**2.) / (len(y)-5)
        print('chi2' , chi2) 
        # print(result.fit_report())
        xx = np.linspace(wn_min, wn_max,100)
        yeval = model.eval(result.params, x=xx)
        ax.plot(xx, yeval, color=gaussian_color, zorder=100, linestyle='--')
        height = result.params['height'].value
        height_err = result.params['height'].stderr
        mode = result.params['center'].value
        mode_err = result.params['center'].stderr
        fwhm = 2.355*result.params['sigma'].value
        fwhm_err = 2.355*result.params['sigma'].stderr
    ax.axvline(mode, color='gray', linestyle=':', lw=0.5)
    
    ax.tick_params(which='both', top=False)
    axT = ax.secondary_xaxis('top', functions=(forward, inverse))
    axT.tick_params(labelsize=fs)
    if top_label:
        axT.set_xlabel('wavelength (μm)', fontsize=fs)
        
    else:
        axT.set_xticklabels([])
    ax.set_xlabel('wavenumber (cm$^{-1}$)', fontsize=fs)
    ax.set_xlim(wn_max, wn_min)
    ax.set_ylim(ylim)
    if kwargs.get('legend'):
        ax.legend(loc ='lower left', fontsize=fs-6)  
    if opdepth:
        ax.set_ylabel('reverse optical depth')
    else:
        ax.set_ylabel('$F/F_\mathrm{cont.}$')

    peak_position_wn = mode
    peak_position_err_wn = mode_err
    x = 1/peak_position_wn
    xerr = x * (peak_position_err_wn/peak_position_wn)
    peak_position = (x*u.cm).to(u.um).value
    peak_position_err = (xerr*u.cm).to(u.um).value

    peak_height =height
    peak_height_err = height_err
    peak_snr = np.nanmax(fin_data[IDX]) / np.nanmax(fin_data_err[IDX][np.argmax(fin_data[IDX])])

    integrated_depth_mod = np.trapz(-result.best_fit, x=X_wn[IDX_gauss])
    if opdepth: 
        integrated_depth_data = np.trapz(-optical_depth[IDX_gauss], x = X_wn[IDX_gauss]) 
        column_density = integrated_depth_mod / A 
        column_density_up = np.trapz(-(result.best_fit - result_err), x =X_wn[IDX_gauss])/ A
        column_density_err = -column_density_up + column_density
    else:
        column_density = np.nan
        column_density_err = np.nan
        integrated_depth_data = np.nansum(1-fin_data[IDX_gauss][::-1])
    integrated_noise_data = np.sqrt(np.nansum(fin_data_err[IDX_gauss]**2.))

    d = {}
    d['peak position wn'] = peak_position_wn
    d['peak position wn err'] = peak_position_err_wn
    d['peak position wl'] = peak_position
    d['peak position wl err'] = peak_position_err
    if skewedgaussian: 
        d['hwhm L'] = fwhm_blue / 2.
        d['hwhm L err'] = fwhm_blueerr / 2.
        d['hwhm R'] = fwhm_red / 2.
        d['hwhm R err'] = fwhm_rederr / 2.
    else:
        d['fwhm'] = fwhm 
        d['fwhm err'] = fwhm_err
    d['peak height'] = peak_height
    d['peak height err'] = peak_height_err
    d['peak SNR'] = peak_snr
    d['integrated depth model'] = integrated_depth_mod
    d['integrated depth data'] = integrated_depth_data
    d['integrated noise data'] = integrated_noise_data
    d['integrated SNR'] = integrated_depth_mod / integrated_noise_data
    if opdepth:
        d['column density 10^16'] = column_density / 1e16
        d['column density err 10^16'] = column_density_err / 1e16
    if icefeature == '13CO2':
        T = (2280-peak_position_wn)/0.03
        num = (np.sqrt(0.06**2. + peak_position_err_wn**2.) / (2280.16 - peak_position_wn))**2.
        den = (0.0006/0.03)**2.
        Terr = T*np.sqrt(num + den)
        d['Temperature'] = T 
        d['Temperature err'] = Terr
    df = pd.DataFrame.from_dict(d, orient='index', columns=['Value'])
    return df