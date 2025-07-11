import numpy as np
import pandas as pd
import miepython
import os
from time import time, sleep

from astropy.io import fits
import astropy.units as u
import astropy.constants as const

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy import interpolate

import matplotlib.gridspec as gridspec

from mpmath import findroot

HERE = os.path.dirname(os.path.abspath(__file__))
optical_constant_path = os.path.join(HERE, 'data_files/optical_constants/')
photosphere_path = os.path.join(HERE, 'data_files/photosphere/')

def _nkindicies(material, nz_iter = 100):
    ## H2O ONLY
    if material == 'H2O':
        file = optical_constant_path + 'h2o-a-Hudgins1993.lnk'
    elif material == 'CO2':
        file = optical_constant_path + 'co2-w-Warren1986.lnk'
    elif material == 'olivine':
        file = optical_constant_path + 'ol-mg40-Dorschner1995.lnk'

    wave_inter = np.logspace(np.log10(0.08), np.log10(5), nz_iter)

    binary_data = np.loadtxt(file, usecols=[0,1,2])
    n_array = np.zeros((binary_data[:,0].shape[0]), dtype=complex)
    for z in range(binary_data[:,0].shape[0]):
        n_array[z] = complex(binary_data[z,1], binary_data[z,2])
    fn = interp1d(binary_data[:,0], binary_data[:,1] )
    fk = interp1d(binary_data[:,0], binary_data[:,2] )
    nz = len(wave_inter)
    fnk = np.zeros((nz), dtype=complex)
    for z in range(len(wave_inter)):
        fnk[z] = complex(fn(wave_inter)[z], fk(wave_inter)[z])
    return fn, fk, wave_inter, fnk

def _photosphere():
    # lu = fits.open('../spectral_extraction_fitting_scripts/extraction/betapic_photosphere_Lu.fits')
    # lu_x = lu[1].data['wave_micron'] * u.um
    # lu_y = lu[1].data['flux_Jy'] * u.Jy

    wu = np.loadtxt(photosphere_path + 'betapic_photosphere_Wu2024.txt')
    wavelength, flux = wu[:,0], wu[:,1]
    wavelength = (1/wavelength/1.097e5*1e4)*u.um
    flux = flux/0.45e18

    flux = flux*u.Jy/u.s
    flux = (flux * (wavelength)/const.c).to(u.Jy)
    flux = (flux*(const.c/(wavelength**2.))).to(u.erg/u.s/u.cm**2./u.AA)
    return wavelength, flux


def calc_beta(a, n, k, rho, nz_iter=100):
    phot_wave, phot_flux = _photosphere()
    wave_inter = np.logspace(np.log10(0.08), np.log10(5), nz_iter)

    Qpr = np.zeros((2,nz_iter))
    Qpr[0] = wave_inter
    
    neff_brug = np.zeros((nz_iter), dtype=complex)
    for z in range(nz_iter):
        x = 2*np.pi*a/wave_inter[z]
        neff_brug[z] = complex(n(wave_inter[z]), -1* k(wave_inter[z]))
        refrel = neff_brug[z]
        qext, qsca, qback, gg = miepython.mie(refrel,x)
        qabs = qext-qsca
        Qpr[1,z] = qext - (gg*qsca)
    F = phot_flux.value
    betas = []
    Qpr_w = Qpr[0, :] 
    Qpr_f = Qpr[1, :] 

    f = interpolate.interp1d(phot_wave, F)
    spF = f(Qpr_w)

    num = np.trapz(Qpr_f*spF, x=Qpr_w)
    den = np.trapz(spF, x=Qpr_w)
    avg_Qpr = num/den
            
    G = const.G
    c = const.c  
    Lstar = (8.7 * u.Lsun).to(u.g*u.cm**2./u.s**3)
    Mstar = (1.75 * u.Msun).to(u.g)
    m_h2o = 2.99e-23 * u.g 

    beta = (3 * avg_Qpr * Lstar) / (16*np.pi * G*c*rho*(a*u.um)*Mstar)
    beta = beta.to(u.m**2*u.g/(u.g*u.m**2))
    return beta.value

def _density(material):
    if material == 'H2O':
        rho = (0.92 * u.g/u.cm**3).to(u.g/u.m**3.) 
    elif material == 'CO2':
        rho = (1.56* u.g/u.cm**3).to(u.g/u.m**3.) 
    else: # olivine
        rho = (3.22* u.g/u.cm**3).to(u.g/u.m**3.) 
    return rho

def get_pure_beta(material, nz_iter = 100):
    
    alist = np.logspace(-1, 1, 100)
    fn, fk, wave_inter, fnk = _nkindicies(material, nz_iter=nz_iter)
    rho = _density(material)

    beta = []
    for a in alist:
        print(a, end='\r')
        beta.append(calc_beta(a, fn, fk, rho))
    return alist, beta

def get_mixture_beta(materials, frac_materials, nz_iter = 100):
    if isinstance(frac_materials, float):
        frac_materials = [frac_materials]
    if isinstance(materials, str):
        materials = [materials]
    num_components = len(materials)
    porosity = 1 - np.sum(frac_materials)

    rho_combo = 0 
    fnk_material = {}
    for i, component in enumerate(materials):
        fn, fk, wave_inter, fnk = _nkindicies(component, nz_iter=nz_iter)
        fnk_material[i] = fnk

        rho = _density(component)
        rho_combo += frac_materials[i] * rho

    n_array = np.zeros((num_components+1, nz_iter), dtype=complex)
    n_array[0,:] = np.ones((nz_iter))
    for key, value in fnk_material.items():
        n_array[key+1,:] = value

    vf_array = np.array([porosity])
    for i in frac_materials:
        vf_array = np.append(vf_array, i)
    N = vf_array.shape[0]
    initial_guess = [complex(2, 1)]
    n_combo = []
    k_combo = []
    for z in range(nz_iter):
        n_bg_test =findroot(lambda n_bg: np.sum((vf_array[n]*(n_array[n,z]**2 - (n_bg)**2)/(n_array[n,z]**2 + 2*(n_bg)**2)) for n in np.arange(0, N)), initial_guess , solver='muller')
        n_combo.append(n_bg_test.real) 
        k_combo.append(n_bg_test.imag)         
    fn_combo = interp1d(wave_inter, n_combo)
    fk_combo = interp1d(wave_inter, k_combo)

    beta = []
    alist = np.logspace(-1, 1, 100)
    for a in alist:
        print(a, end='\r')
        beta.append(calc_beta(a, fn_combo, fk_combo, rho_combo))
    return alist, beta

def plot_beta(ax, a, beta, color='k', leglabel='data', linewidth=1, linestyle='-', fontsize = 14, legend_fontsize=14):
    if isinstance(beta[0], float):
        beta = [beta]
    if isinstance(leglabel, str):
        leglabel = [leglabel]
    if isinstance(color, str):
        color = [color]
    if isinstance(linewidth, (int, float)):
        linewidth = [linewidth]
    if isinstance(linestyle, str):
        linestyle = [linestyle]
    for i in np.arange(len(beta)):
        ax.plot(a, beta[i], color=color[i], linewidth=linewidth[i], label=leglabel[i], linestyle=linestyle[i])
        
    ax.axhline(0.5, color='gray', linewidth=1, linestyle='-', alpha=0.2)

    ax.set_xlabel('minimum grain size (μm)', fontsize=fontsize)
    ax.set_ylabel('β', fontsize=fontsize)

    ax.legend(loc='upper right', fontsize=legend_fontsize)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0.3, 1e1)

    ax.fill_between([-0.1, 10], 0.3, 0.5, alpha=0.1,color='none', hatch='///', ec='k', linewidth=0.5)
    ax.text(0.12, 0.44, 'bound', color='gray')
    ax.text(0.12, 0.52, 'unbound', color='gray')

    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(['0.1', '1.0', '10.0'])
    ax.set_yticks([1, 10])
    ax.set_yticklabels(['1.0', '10.0'])

def save_beta(a, beta, beta_colname, savename):
    if isinstance(beta_colname, str):
        beta = [beta]
        beta_colname = [beta_colname]
    
    d = {'grains size':a}
    for i in np.arange(len(beta_colname)):
        d[beta_colname[i]] = beta[i]
    df = pd.DataFrame(d)
    df.to_csv(savename, index=False)
    







           


        