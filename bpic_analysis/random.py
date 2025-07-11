import numpy as np
import pandas as pd


def colors(location):
    d = {'ne_oval':'#D81B60', 'sw_oval':'#FFC107' , 'clump':'#BF6D60' , 'clumpoff':'#1E88E5'}
    return d[location]

def photosphere():
    dd = '/Users/sbetti/Documents/Science/DebrisDisks/betaPic/spectral_extraction_fitting_scripts/extraction/'
    file=dd + 'betapic_R100_photosphere_thermalexcess.txt'
    a = np.loadtxt(file) 
    x, photosphere_y = a[:,0], a[:,1]
    return x, photosphere_y
    
def wavelength_array():
    return np.linspace(0.6025, 5.3025,941)[100:-5]

############ MCMC for CO2 center positions
def log_prior(theta):
    h, c, w, log_f = theta
    if 0.5 < h < 1.5 and 2340 < c < 2360 and 6<w<20 and -10.0 < log_f < 2.0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

def gaussian(x, height, center, width):
    return height*np.exp(-(x - center)**2/(2*width**2)) 


def log_likelihood(theta, x, y, yerr):
    h, c, w, log_f = theta
    model = gaussian(x, h, c, w)
    sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))




def plot_ices_temp_fwhm(ax1, ax2, ices, color, label):
    fwhm = []
    peak = []
    temp = []
    for i, ice in enumerate(ices):
        dat = np.loadtxt(ice)
        T = int(ice.split('/')[-1].split('.0K')[0])
        if T > 50:
            x = dat[:,0]
            y = dat[:,1]
            # IDX = np.where((x < 2400) & (x>2300))  
            x, y = x, y
            x, y = smooth_labspec_special(x, y)
            
            IDX = np.where((x < 2400) & (x>2300))  
            
            f = interpolate.interp1d(x[IDX],y[IDX], kind=2)
            newx = np.linspace(np.min(x[IDX]), np.max(x[IDX]), 1000)
            newy = f(newx)
            PEAK = newx[np.argmax(newy)]

            halfy = np.max(newy) / 2
            
            IDX_right = np.where((newx < 2400) & (newx>PEAK)) 
            half_x_right = np.abs(newy[IDX_right] - halfy).argmin()
            halfx_right = newx[IDX_right][half_x_right]

            IDX_left = np.where((newx < PEAK) & (newx>2300))      
            half_x_left = np.abs(newy[IDX_left] - halfy).argmin()
            halfx_left = newx[IDX_left][half_x_left]

            FWHM =  (halfx_right- halfx_left) # 2* #-PEAK
            fwhm.append(FWHM)
            peak.append(PEAK)
            temp.append(T)
      
            ax2.scatter(PEAK, FWHM,s=(i*4)+10, color=color )
            ax1.scatter(T, FWHM,s=(i*4)+10, color=color )

    ax2.plot(peak, fwhm, '-', color=color, label=label)
    ax1.plot(temp, fwhm, '-', color=color, label=label)