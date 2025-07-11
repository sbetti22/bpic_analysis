import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture, CircularAnnulus, EllipticalAperture
import pyregion
import glob as glob
import matplotlib.patheffects as pe

def plot_spectra(ax, df, offset=0, color='k', zorder=10,
                 xlabel='wavelength (um)', ylabel='normalized Fdisk/Fstar', ylabelerr='normalized Fdisk/Fstar err', leglabel=None, **kwargs):
    linewidth = kwargs.get('linewidth', 1)
    linestyle = kwargs.get('linestyle', '-')
    
    ax.plot(df[xlabel], df[ylabel]+offset, lw=linewidth, color = color, label=leglabel , zorder=zorder, linestyle=linestyle)
    ax.fill_between(df[xlabel], df[ylabel]+offset-df[ylabelerr], df[ylabel]+offset+df[ylabelerr], color = color, alpha=0.2, zorder=zorder-0.5)

def plot_spectra_model(ax, df, offset=0, color='k', zorder=11, label=None, component1=False, component2=False,
                       xlabel='wavelength (um)', ylabel='normalized Fdisk',
                         **kwargs):
    linewidth = kwargs.get('linewidth', 1)
    linestyle = kwargs.get('linestyle', '--')
    label_component1 = kwargs.get('label_component1', 'component 1')
    label_component2 = kwargs.get('label_component2', 'component 2')
    ax.plot(df[xlabel], df[ylabel] + offset, color=color, linestyle=linestyle, linewidth=linewidth, label=label, zorder=zorder)

    if component1:
        label_component1 = kwargs.get('label_component1', None)
        ax.plot(df[xlabel], df[f'{ylabel} comp1'], color='k', linestyle='--', linewidth=0.95, label=label_component1, zorder=zorder)
    if component2:
        label_component2 = kwargs.get('label_component2', None)
        ax.plot(df[xlabel], df[f'{ylabel} comp2'], color='k', linestyle='--', linewidth=0.95, label=label_component2, zorder=zorder)


def plot_spectra_with_model(ax, df_data, df_model, data_color, df_co2model=None, comodel=False, offset=0, xlabel='wavelength (um)', ylabel='normalized Fdisk/Fstar', ylabelerr='normalized Fdisk/Fstar err', modxlabel='wavelength (um)', modylabel='normalized Fdisk', mod2ylabel='normalized Fdisk comp2', **kwargs):

    linestyle = kwargs.get('linestyle', '-')
    linewidth = kwargs.get('linewidth', 1)
    
    if comodel:
        xx = df_data[xlabel].values 
        yy = df_data[ylabel].values / 1e-14
        yerr = df_data[ylabelerr].values / 1e-14
        if df_model is not None:
            Xmod = df_model[modxlabel]
            Ymod = df_model[modylabel]/1e-14
    else:
        xx = df_data[xlabel].values[105:]
        yy = df_data[ylabel].values[105:]
        yerr = df_data[ylabelerr].values[105:]
        if df_model is not None:
            Xmod = df_model[modxlabel]
            Ymod = df_model[modylabel]


    if df_co2model is not None:
        if df_model is not None:
            Ymod2 = df_model[mod2ylabel]
            Xmod_CO2 = df_co2model[modxlabel]
            Ymod_CO2 = df_co2model[modylabel]
            ax.plot(xx, yy+offset, lw=linewidth+0.5, color = data_color,  zorder=1)
            ax.fill_between(xx, (yy-yerr)+offset, (yy+yerr)+offset, color = data_color, alpha=0.2)
            IDX = np.where((xx>=np.min(Xmod_CO2)) & (xx<=np.max(Xmod_CO2)))
            ymodel2 = Ymod2.values[IDX]
            ax.plot(Xmod_CO2, (Ymod_CO2)+ymodel2+offset, lw=linewidth, zorder=1, color='k', linestyle=linestyle)
                
    else:
        ax.plot(xx, yy+offset, lw=linewidth+0.5, color = data_color, zorder=1)
        ax.fill_between(xx, (yy-yerr)+offset, (yy+yerr)+offset, color = data_color, alpha=0.2)
        if df_model is not None:
            ax.plot(Xmod, Ymod+offset, lw=linewidth, zorder=1, color='k', linestyle=linestyle)



def plot_nircam_filters(ax, offset=0.25):
    filters = glob.glob('../JWST_NIRCam_transmissions/*.txt')
    xpos = [1.845, 2.10, 2.5, 2.99, 3.37, 3.65, 4.42]
    filter_cols = {"F182M":'#7FD7FF', 'F210M':'#80F3F8', 'F250M':'#A5FED4', 'F300M':'#D2FFA7', 'F335M':'#F3FF85', 
                'F444W':'#FF937F'}
    for j, i in enumerate(np.sort(filters)):
        name = i.split('/')[-1].split('.')[0]
        if name != 'F360M':
            a = pd.read_csv(i,sep=' ')
            Fx = a['Microns']
            Fy = a['Throughput']
            color='k'
            ax.plot(Fx, (Fy/2)+offset, color=color, linewidth=1)

            ax.fill_between(Fx, offset, (Fy/2)+offset, color=filter_cols[name])
            text_bkg = filter_cols[name]
            ax.text(xpos[j], offset+0.05, name, ha='center', fontsize=5, 
                    path_effects=[pe.withStroke(linewidth=2, foreground=text_bkg)])
            

def plot_nircam_photometry(ax, df, offset=0, zorder=12, color='k', marker='D', 
                           xlabel='wavelength (um)', ylabel='normalized Fdisk/Fstar', fluxlabel ='Flux (Jy)',  fluxlabelerr = 'Flux err (Jy)'):
    ax.plot(df[xlabel], df[ylabel]+offset, marker, zorder=zorder, color=color, mec='k')

    yerr = df[ylabel] * np.sqrt((df[fluxlabelerr].values[2]/df[fluxlabel].values[2])**2. + (df[fluxlabelerr].values/df[fluxlabel].values)**2.)

    ax.errorbar(df[xlabel], df[ylabel]+offset, yerr = yerr,
                    fmt='.', zorder=zorder-0.5, color='k', mec='k', capsize=3)
    
def plot_axes(ax, ylabel, xlabel='Wavelength (μm)', top_label=True, fontsize=14, xlim = [0.4, 5.5], ylim=[0.25, 2.25], legend=True, legend_fontsize=None, legend_loc = 'upper right'):
    if legend_fontsize is None:
        legend_fontsize = fontsize
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.tick_params(which='major', direction='in', length=7, top=True, right=True, labelsize=fontsize)
    ax.tick_params(which='minor', direction='in', length=5, top=True, right=True, labelsize=fontsize)
    ax.minorticks_on()

    if legend:
        ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=True)

    if top_label:
        ax.tick_params(which='both', top=False, labelsize=fontsize)
        axT = ax.secondary_xaxis('top', functions=(forward, inverse))
        if top_label:
            axT.set_xlabel('Wavelength (μm)', fontsize=fontsize)
        else:
            axT.set_xticklabels([])
        axT.tick_params(labelsize=fontsize)
        return axT


def forward(x):
    return 10000 / x

def inverse(x):
    return 10000 / x

def plot_image(ax, data, wv_slice='all', direction='vertical', colorbar=True, black_mask=True, **kwargs):
    vmin = kwargs.get('vmin', 0)
    vmax=  kwargs.get('vmax', 200000)
    cmap = kwargs.get('cmap', 'magma')
    cbar_label = kwargs.get('cbar_label', 'Flux')
    cbar_dir = kwargs.get('cbar_dir', 'right')
    fs = kwargs.get('fs', 12)
    zorder= kwargs.get('zorder', 1)
    if len(data.shape) == 3:
        if wv_slice == 'all':
            coadd_data = np.nansum(data, axis=0)
        else:
            coadd_data = data[wv_slice,:,:]

    coadd_data = np.nan_to_num(coadd_data)
    halfsize = np.asarray(coadd_data.shape) / 2 * 0.1
    extent = [-halfsize[1], halfsize[1], -halfsize[0], halfsize[0]]
    coadd_data[20:60, 41:50] = 0
    coadd_data[70:80, 50:60] = 0
    coadd_data[30:40, 0:10] = 0
    coadd_data[100:110, 10:17] = 0
    coadd_data[98::, 52::] = 0

    im1 = ax.imshow(coadd_data.data, vmin=vmin, vmax=vmax, extent=extent,
            origin='lower', cmap=cmap,aspect='auto', zorder=zorder)
    if colorbar:
        pad = 0 if cbar_dir == 'top' else 0.03
        cbar = plt.colorbar(im1, ax=ax, location=cbar_dir,  pad=pad, 
             )
        cbar.ax.tick_params(labelsize=fs-1)
        cbar.set_label(label=cbar_label,size=fs-1)

    if black_mask:
        finmask = np.copy(coadd_data)
        if not np.ma.isMaskedArray(finmask):
            region_name = "/Users/sbetti/Documents/Science/DebrisDisks/betaPic/spectral_extraction_fitting_scripts/extraction/ds9_mask.reg"
            r = pyregion.open(region_name)
            mask = r.get_mask(shape=finmask.shape)
            finmask = np.ma.array(finmask, mask = mask)

        idx = np.where(finmask.mask == True)
        finmask2 = np.zeros_like(finmask)
        finmask2[idx] = -100
        idx2 = np.where(finmask.mask == False)
        finmask2[idx2] = np.nan
        halfsize = np.asarray(finmask.shape) / 2 * 0.1
        extent = [-halfsize[1], halfsize[1], -halfsize[0], halfsize[0]]
        ax.imshow(finmask2, zorder=1, vmin=-10, vmax = 10,cmap='Greys_r', extent=extent, aspect='auto')

        
        circle = plt.Circle((0, 0), 2, color='k', clip_on=False)
        ax.add_patch(circle)

        # Create a Rectangle patch
        # rect = patches.Rectangle((-0.45, 1.55), 1, 2, angle=25,
        #                         linewidth=1, edgecolor='k', 
        #                         facecolor='k')
        # ax.add_patch(rect)
        # rect = patches.Rectangle((0.3, -4), 1, 2.5, angle=27,
        #                         linewidth=1, edgecolor='k', 
        #                         facecolor='k')
        # ax.add_patch(rect)


    ax.plot(0,0, marker='*', color='white', )
    # vertical
    if direction == 'vertical':
        ax.set_xlabel('Arcsecond', fontsize=fs)
        ax.set_ylabel('Arcsecond', fontsize=fs)
        ax.tick_params(which='both', color='white', labelsize=fs)
        ax.minorticks_on()

        ax.plot([-2.4399, -1.8999], [-4.3048+0.37399, -4.3048], color='white', markevery=[0], marker=(3,0,55), markersize=2, linewidth=0.5)
        ax.plot([-1.8999, -1.4999], [-4.3048, -4.3048+0.4626], color='white', markevery=[-1], marker=(3,0,75), markersize=2, linewidth=0.5)
        ax.text(-1.6, -3.7, 'N',  fontsize=6, color='white')
        ax.text(-2.6, -3.75, 'E', fontsize=6, color='white')
        ax.plot([-2.5, -1.5], [-5, -5], color='white', linewidth=0.75)
        ax.text(-2., -5.05, '1"\n19.6 au', color='white', fontsize=6, va='center', ha='center')
        ax.set_ylim(-6.6, 6.2)
        ax.set_xlim(-2.88,2.73)
        ###########

    else:
        ax.set_xlabel('Arcsecond', rotation=0)
        ax.xaxis.set_label_position('top') 
        ax.set_ylabel('Arcsecond', rotation=270, labelpad=15)
        ax.tick_params(axis='x', labelrotation=270, labeltop=True, labelbottom=False, color='white')
        ax.tick_params(axis='y', labelrotation=270, labeltop=True, labelbottom=False, color='white')
        ax.tick_params(which='both', color='white')
        ax.minorticks_on()

        ax.plot([-2.4399, -1.8999], [4.6788, 4.3048], color='white', markevery=[0], marker=(3,0,55), markersize=2, linewidth=0.5)
        ax.plot([-1.8999, -1.4999], [4.3048, 4.7674], color='white', markevery=[-1], marker=(3,0,75), markersize=2, linewidth=0.5)
        ax.text(-1.7, 5, 'N', rotation=270, fontsize=6, color='white')
        ax.text(-2.75, 4.9, 'E', rotation=270, fontsize=6, color='white')
        ax.plot([-2.2, -2.2], [-4, -5], color='white', linewidth=0.75)
        ax.text(-2.25, -4.5, '1"\n19.6 au', color='white', rotation=270, fontsize=6, va='center', ha='center')

def plot_apertures(ax, positions, aperture_shape, binsize, data, color, theta=None):
    if not isinstance(positions, list):
        positions = [positions]
    halfsize = np.asarray(data[200,:,:].shape) / 2 * 0.1
    if aperture_shape == 'oval':
        aperture = EllipticalAperture(positions, binsize[0], binsize[1], theta= theta)
        aperture.positions = aperture.positions*0.1 - [halfsize[1], halfsize[0]] 
        aperture.a = aperture.a * 0.1
        aperture.b = aperture.b * 0.1
    else:
        aperture = CircularAperture(positions, r=binsize)
        aperture.positions = aperture.positions*0.1 - [halfsize[1], halfsize[0]] 
        aperture.r = aperture.r * 0.1

    aperture.plot(color='white', linewidth=3, ax=ax)

    if len(positions) > 1:
        if not isinstance(color, list):
            color=[color]
        for i in np.arange(len(aperture)):
            aperture[i].plot(color=color[i], linewidth=2, ax=ax)
    else:
        aperture.plot(color=color, linewidth=2, ax=ax)
