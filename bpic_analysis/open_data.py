import numpy as np
import pandas as pd
import pyregion
import h5py

from astropy.io import fits
from astropy import units as u
from astropy.io import fits
from synphot import SpectralElement, SourceSpectrum
from synphot.models import Empirical1D
from synphot import Observation

def convert_Jy(data):
    data_u = data * u.MJy/u.steradian
    # Jy/arcsecond^2
    data_Jyas2 = data_u.to(u.Jy/u.arcsecond**2.)
    data_Jyas2 = data_Jyas2.value
    pixelscale = 0.1
    data_Jy = data_Jyas2 * pixelscale**2.
    return data_Jyas2,data_Jy 

def data_mask(unit='Jy'):
    datadir = '/Users/sbetti/Documents/Science/datasets/JWST/betaPic_NIRSpec/final_image/'
    data = fits.getdata(datadir + 'RDI_betaPic_JWST_NIRSpec.fits')

    region_name = "/Users/sbetti/Documents/Science/DebrisDisks/betaPic/spectral_extraction_fitting_scripts/extraction/ds9_mask.reg"
    r = pyregion.open(region_name)
    mask = r.get_mask(shape=data[0,:,:].shape)
    mask3d = np.repeat(mask[np.newaxis, :, :], 941, axis=0)
    data = np.ma.array(data, mask = mask3d)

    data_u = data * u.MJy/u.steradian
    # Jy/arcsecond^2
    data_Jyas2 = data_u.to(u.Jy/u.arcsecond**2.)
    data_Jyas2 = data_Jyas2.value
    pixelscale = 0.1

    if unit=='Jy':
        print('using Jy')
        data_Jy = data_Jyas2 * pixelscale**2.
        data_Jy = np.nan_to_num(data_Jy)
        data_Jy[:, 20:60, 41:50] = 0
        data_Jy[:, 70:80, 50:60] = 0
        data_Jy = np.ma.array(data_Jy, mask = mask3d)
        return data_Jy
    elif unit == 'Jyas2':
        print('using Jyas2')
        data_Jyas2 = np.nan_to_num(data_Jyas2)
        data_Jyas2[:, 20:60, 41:50] = 0
        data_Jyas2[:, 70:80, 50:60] = 0
        data_Jyas2 = np.ma.array(data_Jyas2, mask = mask3d)
        return data_Jyas2
    else:
        print('using MJy/st')
        data_u = data_u.value
        data_u = np.nan_to_num(data_u)
        data_u[:, 20:60, 41:50] = 0
        data_u[:, 70:80, 50:60] = 0
        data_u = np.ma.array(data_u, mask = mask3d)
        return data_u


def obs_effstim_image(data_Jy, mask=True):
    new_nirspec = np.zeros((data_Jy.shape[1], data_Jy.shape[2]))
    dd = np.nansum(data_Jy.data,axis=0)
    for i in np.arange(data_Jy.shape[1]):
        print(i, end='\r')
        for j in np.arange(data_Jy.shape[2]):
            y = data_Jy[100:-5,i,j]
            x = (np.linspace(0.6025, 5.3025,941)[100:-5] * u.um).to(u.Angstrom)
    #         if dd[i,j] < 0.002:
    #             y *= np.nan
            y = y*u.Jy
            
            Fx = x.value
            Fy = np.ones_like(Fx)

            if np.isnan(y.value).all():
                new_nirspec[i,j] = np.nan
            elif all(xxx <= 0 for xxx in y.value):
                new_nirspec[i,j] = np.nan
            else:
                bp = SpectralElement(Empirical1D, points=Fx, lookup_table=Fy, keep_neg=True)
                final_yunits = 'Jy'

                sp = SourceSpectrum(Empirical1D, points=x, lookup_table=np.nan_to_num(y.value, nan=10000)*u.Jy)

                obs = Observation(sp, bp, binset=range(int(np.min(Fx)), int(np.max(Fx)) ) )
                Fli = obs.effstim(u.Jy)
                new_nirspec[i,j] = Fli.value
    new_nirspec2 = np.copy(new_nirspec)
    new_nirspec2 = new_nirspec2*u.Jy
    new_nirspec2 = new_nirspec2.to(u.mJy).value
    new_nirspec2 = new_nirspec2 / (0.1**2.)
    if mask:
        new_nirspec2 = np.nan_to_num(new_nirspec2)
        new_nirspec2[20:60, 41:50] = 0
        new_nirspec2[70:80, 50:60] = 0
        new_nirspec2[20:25, 35:60] = 0

        new_nirspec2[70:90, 50:60] = -10
        new_nirspec2[50:120, 5:25] = -10
        new_nirspec2[40:50, 0:15] = -10
        new_nirspec2[0:80, 40:60] = -10
        new_nirspec2[0:15, 35:60] = -10
        new_nirspec2[60::, 50::] = -10
        new_nirspec2[115::, 15:20] = -10

        region_name = "/Users/sbetti/Documents/Science/DebrisDisks/betaPic/spectral_extraction_fitting_scripts/extraction/ds9_mask.reg"
        r = pyregion.open(region_name)
        mask = r.get_mask(shape=new_nirspec2.shape)
        new_nirspec2 = np.ma.array(new_nirspec2, mask = mask)

    return new_nirspec2

def open_spectrum_data(disk_spectrum_path, norm=True):

    disk_spectrum = fits.getdata(disk_spectrum_path)

    X = disk_spectrum[:,0][:-5]
    DATA = disk_spectrum[:,1][:-5]
    NOISE  = disk_spectrum[:,2][:-5]
    NOISE[887:892] = NOISE[880]
    if norm:
        yname = 'normalized Fdisk/Fstar'
    else:
        yname = 'Fdisk/Fstar'

    d = {'wavelength (um)': X.byteswap().newbyteorder(),
          yname:DATA.byteswap().newbyteorder(),
            f'{yname} err':NOISE.byteswap().newbyteorder()}
    norm_df = pd.DataFrame(d)
    return norm_df


def open_spectrum_model(disk_model_path, components=True):
    if '.csv' in disk_model_path:
        model = pd.read_csv(disk_model_path)
    else:
        model = h5py.File(disk_model_path, "r")
    Xmod = np.array(model['microns'])
    Ymod = np.array(model['model'])
    if components:
        Ymod1 = np.array(model['model_comp1'])
        Ymod2 = np.array(model['model_comp2'])
        model_df = pd.DataFrame({'wavelength (um)':Xmod.byteswap().newbyteorder(), 
                                 'normalized Fdisk':Ymod.byteswap().newbyteorder(),
                                   'normalized Fdisk comp1':Ymod1.byteswap().newbyteorder(), 
                                   'normalized Fdisk comp2':Ymod2.byteswap().newbyteorder()})
    else:
        model_df = pd.DataFrame({'wavelength (um)':Xmod.byteswap().newbyteorder(), 
                         'normalized Fdisk':Ymod.byteswap().newbyteorder()})
    return model_df