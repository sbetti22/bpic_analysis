# updated 1-13-25 with correct arcsecond separation and center star positions
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import fits
import pysynphot as S
from photutils.aperture import CircularAperture, CircularAnnulus, EllipticalAperture
from photutils.aperture import aperture_photometry

from scipy import interpolate
from astropy.stats import sigma_clip
from scipy import interpolate
from astropy.modeling.models import BlackBody
from photutils.aperture import CircularAperture, ApertureStats
import pandas as pd


def smooth_photosphere(file=None, add_thermal_excess=True):
    if file is None:
        x = np.linspace(0.6025, 5.3025,941)

        a = fits.getdata('/Users/sbetti/Documents/Science/DebrisDisks/betaPic/spectral_extraction_fitting_scripts/extraction/jwst_nirspec_prism_disp.fits')
        f = interpolate.interp1d(a['WAVELENGTH'], a['R'])
        R_nirspecI = f(x)

        stellar_model = fits.open('/Users/sbetti/Documents/Science/DebrisDisks/betaPic/spectral_extraction_fitting_scripts/extraction/bpic_photosphere_Model.fits')
        sm_x = stellar_model[1].data['wave_micron']
        sm_y = stellar_model[1].data['flux_Jy']

        f = interpolate.interp1d(sm_x, sm_y)
        new_y = f(x)
        photosphere_y = broaden(x, new_y, R_nirspecI)
        if add_thermal_excess:
            bb = BlackBody(temperature=500*u.K)
            flux = bb(x) *u.sr
            flux = flux.to(u.erg/(u.Hz*u.s*u.cm**2.)).value
            sp = S.ArraySpectrum(x, flux, waveunits='um', fluxunits='Fnu', name='MySource')
            sp.convert('Jy')
            sp.convert('um')

            # 8.2292e-2 = flux density at 4um from Kadin's Fig 2

            f = interpolate.interp1d(sp.wave, sp.flux)
            fourmicron = f(4)
            BB_flux = sp.flux * (8.2292e-2/fourmicron) # normalize to Fig 2 from Kadin's paper
            photosphere_y = 1.5*BB_flux + photosphere_y
    else:
        a = np.loadtxt(file) 
        x, photosphere_y = a[:,0], a[:,1]
    return x, photosphere_y


def radial_profile_NEWNOISE(data, inMod, residual, err, positions, direction='major', binsize=5, surbright=False, plot=True, ap_shape='circle'):
    d = {}
    derr = {}
    tp = {}

    dres = []
    for wv in np.arange(data.shape[0]):
        DATA = np.nan_to_num(data[wv,:,:])
        INMOD = np.nan_to_num(inMod[wv,:,:])
        RES = np.nan_to_num(residual[wv,:,:])
        ERR = np.nan_to_num(err[wv,:,:])
        if ap_shape=='oval':
            aperture = EllipticalAperture(positions, binsize[0], binsize[1], theta= 3)
        else:
            aperture = CircularAperture(positions, r=binsize)
        
        if surbright:
            # area in arcseconds
            area = aperture.area * (0.1**2. )
            # flux in Jy/arcsecond^2
            DATA /= area
            RES /=  area
            INMOD /=  area
        
        # flux in Jy
        aperture_stats_RES = ApertureStats(RES, aperture)
        aperture_stats_DATA = ApertureStats(DATA, aperture)
        phot_table_DATA = aperture_photometry(DATA, aperture)['aperture_sum']
        aperr = CircularAperture((7,20), r=10)
        phot_table_RES =  aperture_photometry(DATA, aperture, error=3*ERR)['aperture_sum_err'] # aperture_photometry(RES, aperture)['aperture_sum'] 
        
        outMOD_sc = sigma_clip((DATA-RES), sigma_upper=1000, maxiters=20, masked=False, copy=True, axis=(0,1))
        phot_table_outMOD = aperture_photometry(outMOD_sc, aperture)['aperture_sum']
        phot_table_inMOD = aperture_photometry(INMOD, aperture)['aperture_sum'] 

        TP = phot_table_outMOD.value/phot_table_inMOD.value
        d[wv] = phot_table_DATA.value
        tp[wv] = TP.value
        derr[wv] = phot_table_RES#.value

    dict_pos = {}
    dict_err_pos = {}
    tp_fin = {}
    tp_fin_sm = {}
    tp_fin_std = {}
    if direction == 'major':
        pos = 1
        ceny = 66 #68.5
        cenx = 29.5 #30.6
    else:
        x = np.arange(-6,6)
        y = 8*x 
        f = interpolate.interp1d(y,x)
        pos = 0
        cen = 37.5
    for j in np.arange(len(positions)):
        if direction == 'major':
            arc_pos = round(np.sqrt((positions[j][0]-cenx)**2. +(positions[j][1]-ceny)**2.)* 0.1,2)
        else:
            halfsize = np.asarray(data[200,:,:].shape) / 2 * 0.1
            P = (positions[j][1]*0.1) - halfsize[0]
            positions_j = (positions[j][pos]*0.1) - halfsize[1]
            arc_pos = round(f(P)-positions_j,2)
        if arc_pos in list(dict_pos.keys()):
            arc_pos = -arc_pos
        y = [list(d.values())[i][j] for i in np.arange(data.shape[0])]
        tpval = np.array([list(tp.values())[i][j] for i in np.arange(data.shape[0])])
        tpval[np.isinf(tpval)] = np.nan
        avg_tpval0 = np.nanmean(tpval)
        avg_tpval = np.ones_like(tpval) * avg_tpval0
        tp_fin_std[arc_pos] = np.ones_like(tpval) * np.nanstd(tpval)

        tp_fin[arc_pos] = tpval
        tp_fin_sm[arc_pos] = avg_tpval

        dict_pos[arc_pos] = y / avg_tpval
        yerr = [list(derr.values())[i][j] for i in np.arange(residual.shape[0])]
        dict_err_pos[arc_pos] = yerr
    if plot:
        plt.figure(figsize=(2,4))
        halfsize = np.asarray(data[200,:,:].shape) / 2 * 0.1
        extent = [-halfsize[1], halfsize[1], -halfsize[0], halfsize[0]]
        plt.imshow(data[200,:,:], vmin=0, vmax=1.5e-4, origin='lower', cmap='magma', extent=extent, aspect='auto')
        aperture.positions = aperture.positions*0.1 - [halfsize[1], halfsize[0]] 
        if ap_shape == 'oval':
            aperture.a = aperture.a * 0.1
            aperture.b = aperture.b * 0.1
        else:
            aperture.r = aperture.r * 0.1
        aperture.plot(color='cyan', zorder=100)
        plt.xlabel('arcsecond')
        plt.ylabel('arcsecond')
    #     plt.savefig('apertures.pdf', dpi=150)
        plt.show()
    df = pd.DataFrame(tp_fin)
    df2 = pd.DataFrame(tp_fin_sm)
    df3 = pd.DataFrame(tp_fin_std)
    df.to_csv('tp_ovalNE_ovalSW.csv')
    df2.to_csv('tp_ovalNE_ovalSW_avg.csv')
    df3.to_csv('tp_ovalNE_ovalSW_std.csv')
    return dict_pos, dict_err_pos

def contrast_diskflux(disk_flux, stellar_flux):
    contrast_flux = (disk_flux / stellar_flux)
    return contrast_flux 

def normalize_spectra(flux, flux_err, photosphere_y, save_spectra=True, plot=False, **kwargs):
    CRVAL3 = 0.6025000237859786
    CDELT3 = 0.004999999888241291
    x = np.arange(CRVAL3, CRVAL3 + 941*CDELT3, CDELT3)
    
    # contrast disk flux/star flux
    contrast_disk = contrast_diskflux(flux, photosphere_y)
    # normalize at 2.5 um 
    IDX = np.nanargmin(abs(x-2.5), axis=0)
    y = np.array(contrast_disk/contrast_disk[IDX])
    yerr = np.array(flux_err/photosphere_y/contrast_disk[IDX]) 

    arr = {'wavelength (um)':x, 'Flux (Jy)':flux, 'Flux err (Jy)':flux_err, 
            'normalized Fdisk/Fstar':y, 'normalized Fdisk/Fstar err':yerr}
    df = pd.DataFrame(arr)

    if save_spectra:
        '''kwargs = savedir, name'''
        save_spectra_fits(df, **kwargs)
    
    if plot:
        fig, ax = plt.subplots()
        plot_spectra_contrast(ax, x, y, yerr, **kwargs)
        plt.show()
    return df

def save_spectra_fits(df, **kwargs):
    savedir = kwargs.get('savedir','')
    name = kwargs.get('name', 'spectra')

    fits_arr = np.c_[df['wavelength (um)'].values, df['Flux (Jy)'].values, df['Flux err (Jy)'].values]
    print('saving to: ', savedir + f'betaPic_notnorm_TPcorr_{name}.fits')
    fits.writeto(savedir + f'betaPic_notnorm_TPcorr_{name}.fits', fits_arr, overwrite=True)

    fits_arr = np.c_[df['wavelength (um)'].values, df['normalized Fdisk/Fstar'].values, df['normalized Fdisk/Fstar err'].values]
    print('saving to: ', savedir + f'betaPic_diskspectrum_TPcorr_{name}.fits')
    fits.writeto(savedir + f'betaPic_diskspectrum_TPcorr_{name}.fits', fits_arr, overwrite=True)


def plot_dict_spectra(ax, d, derr=None, **kwargs):
    colors = kwargs.get('colors', 'k'*len(d))
    CRVAL3 = 0.6025000237859786
    CDELT3 = 0.004999999888241291
    x = np.arange(CRVAL3, CRVAL3 + 941*CDELT3, CDELT3)
    # x = np.linspace(0.6025, 5.3025,941)
    for i, (key, item) in enumerate(d.items()):
        item = np.array(item)
        
        ax.plot(x, item, linewidth=2, color=colors[i], label=key)
        if derr is not None:
            err = np.array(derr[key])
            ax.fill_between(x, item-err, item+err, color=colors[i], alpha=0.2)

    ax.set_yscale('log')
    ax.set_xlim(0.6, 5.2)
    
    ax.set_xlabel('wavelength (μm)', fontsize=14)
    ylabel = kwargs.get('ylabel', 'Flux (Jy)')
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=14, direction='in', which='major', top=True, right=True, length=7, width=1.5)
    ax.tick_params(labelsize=14, direction='in', which='minor', top=True, right=True, length=5, width=1.5)
    ax.minorticks_on()
    
    leg_title = kwargs.get('legend_title', None)
    ax.legend(loc='upper right', fontsize=8, title=leg_title)
    

def plot_spectra_contrast(ax, x, y, yerr, **kwargs):
    color = kwargs.get('color', 'k')
    title = kwargs.get('title', '')

    ax.plot(x, y, color)
    ax.fill_between(x, y-yerr, y+yerr, color=color, alpha=0.3)
    ax.set_title(title)
    ax.axhline(1, color='k', zorder=0)
    ax.tick_params(labelsize=14, direction='in', which='major', top=True, right=True, length=7, width=1.5)
    ax.tick_params(labelsize=14, direction='in', which='minor', top=True, right=True, length=5, width=1.5)
    ax.minorticks_on()
    ax.set_xlabel('wavelength (μm)', fontsize=14)
    ylabel = kwargs.get('ylabel', 'normalized $F_{disk}/F_{star}$')
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlim(0.6, 5)
    ylim = kwargs.get('ylim', [0, 1.6])
    ax.set_ylim(ylim)
    ax.axvline(4.25)
    ax.axvline(2.7)
    ax.axvline(4.07)
    if 'savefig' in kwargs:
        plt.savefig(kwargs['savefig'], dpi=150, transparent=True)


def plot_dict_spectra_contrast(ax, d, photosphere_y, derr=None, **kwargs):
    colors = kwargs.get('colors', 'k'*len(d))
    direction = kwargs.get('direction', '')
    legend_title = kwargs.get('legend_title', 'spectra')
    CRVAL3 = 0.6025000237859786
    CDELT3 = 0.004999999888241291
    x = np.arange(CRVAL3, CRVAL3 + 941*CDELT3, CDELT3)
    # x = np.linspace(0.6025, 5.3025,941)
    for i, key in enumerate(d.keys()):
        # contrast disk flux/star flux
        contrast_disk = contrast_diskflux(d[key], photosphere_y)
        # normalize at 2.5 um 
        ax.plot(x, contrast_disk, linewidth=2, color=colors[i], label=f'{key}" {direction}')
        y = np.array(contrast_disk)
        if derr is not None:
            err = np.array(derr[key]/photosphere_y)
            ax.fill_between(x, y-err, y+err, color=colors[i], alpha=0.2)
    ax.legend(title=legend_title)
#     ax.axhline(1, color='k', zorder=0)
    ax.tick_params(labelsize=14, direction='in', which='major', top=True, right=True, length=7, width=1.5)
    ax.tick_params(labelsize=14, direction='in', which='minor', top=True, right=True, length=5, width=1.5)
    ax.minorticks_on()
    ax.set_xlabel('wavelength (μm)', fontsize=14)
    ylabel = kwargs.get('ylabel', '$F_{disk}/F_{star}$')
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlim(0.6, 5)
    ax.set_ylim(bottom=0)
    
def plot_dict_spectra_norm(ax, d, photosphere_y, derr=None, **kwargs):
    colors = kwargs.get('colors', 'k'*len(d))
    direction = kwargs.get('direction', '')
    normwave = kwargs.get('normwave', 2.5)
    legend_title = kwargs.get('legend_title', 'spectra')
    # x = np.linspace(0.6025, 5.3025,941)
    CRVAL3 = 0.6025000237859786
    CDELT3 = 0.004999999888241291
    x = np.arange(CRVAL3, CRVAL3 + 941*CDELT3, CDELT3)
    IDX = np.where((x >= normwave) & (x < normwave+0.01))[0][0]
    print(IDX, x[IDX])
    offset = kwargs.get('offset')
    for i, key in enumerate(d.keys()):
        # contrast disk flux/star flux
        contrast_disk = contrast_diskflux(d[key], photosphere_y)
        # normalize at 2.5 um 
        if offset:
            o = i/4
        else:
            o= 0
        ax.plot(x, (contrast_disk/contrast_disk[IDX])+o, linewidth=2, color=colors[i], label=f'{key}" {direction}')
        y = np.array(contrast_disk/contrast_disk[IDX])
        if derr is not None:
            err = np.array(derr[key]/photosphere_y/contrast_disk[IDX])
            ax.fill_between(x, (y-err)+o, (y+err)+o, color=colors[i], alpha=0.2)
    ax.legend(title=legend_title)
    ax.axhline(1, color='k', zorder=0)
    ax.tick_params(labelsize=14, direction='in', which='major', top=True, right=True, length=7, width=1.5)
    ax.tick_params(labelsize=14, direction='in', which='minor', top=True, right=True, length=5, width=1.5)
    ax.minorticks_on()
    ax.set_xlabel('wavelength (μm)', fontsize=14)
    if offset:
        ylabel = kwargs.get('ylabel', 'normalized $F_{disk}/F_{star}$ + offset')
        ax.set_ylabel(ylabel, fontsize=14)
    else:
        ylabel = kwargs.get('ylabel', 'normalized $F_{disk}/F_{star}$')
        ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlim(0.6, 5)
    ax.set_ylim(bottom=0)
