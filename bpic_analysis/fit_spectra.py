
### fitting 2 compositions (can also be 2 populations)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np

import pandas as pd

from time import time, sleep
from pathlib import Path

from scipy.integrate import quad
import math
import diskmap
import glob
import matplotlib.pyplot as plt
import urllib.request
from astropy.io import fits
import matplotlib.gridspec as gridspec


from sympy.solvers import solve, nonlinsolve

from sympy import Symbol, nsolve
from scipy.interpolate import interp1d
from scipy import optimize
from scipy.optimize import fsolve, fmin
import scipy
import miepython
import cmath
import miepython
from scipy.optimize import curve_fit
import time
from datetime import datetime

from scipy.optimize import differential_evolution


def find_wave_indice(input_data, short_wave, long_wave):
    short_wave_indice = np.nanargmin(abs(input_data[:,0]-short_wave), axis=0)
    long_wave_indice = np.nanargmin(abs(input_data[:,0]-long_wave), axis=0)
    return short_wave_indice, long_wave_indice 


def hgg_phase_function(phi,g, rayleigh_pol = False):
    #Inputs:
    # g - the g 
    # phi - the scattering angle in radians
    g = g[0]

    cos_phi = np.cos(phi)
    g2p1 = g**2 + 1
    gg = 2*g
    k = 1./(4*np.pi)*(1-g*g)
    if not rayleigh_pol:
        return k/(g2p1 - (gg*cos_phi))**1.5
    else: #HGG modified by Rayleigh scattering to generate polarized image, added by Bin Ren
        cos2phi = cos_phi**2
        return k/(g2p1 - (gg*cos_phi))**1.5 * (1 - cos2phi)/(1 + cos2phi)
    
# define a function for Bruggeman's equation
def sum_bg(n_bg, vf, n_array):
    N = vf.shape[0]
    a, b = n_bg
    S = sum((vf[n]*(n_array[n]**2 - (a+b*1j)**2)/(n_array[n]**2 + (N-1)*(a+b*1j)**2)) for n in np.arange(0, N))
    return (S.real, S.imag)


# different optical constants files have different wavelength coverage
# do the interpolation to mathch the wavelength sampling.
def inter_n_i(input_n_file, z_wave,):
    binary_data = input_n_file
    nz = len(z_wave)
    refr_index_cube = np.zeros((nz, 3)) 
    refr_index_cube[:,0] = z_wave
    refr_index_cube_n = interp1d(binary_data[:,0], binary_data[:,1] )
    refr_index_cube_k = interp1d(binary_data[:,0], binary_data[:,2] )

    refr_index_comp_i = np.zeros((nz), dtype=complex)
    for z in range(len(z_wave)):
        refr_index_comp_i[z] = complex(refr_index_cube_n(z_wave)[z], refr_index_cube_k(z_wave)[z])
    

    return z_wave, refr_index_comp_i, 

# creating model specturm
def integrand(a, neff, z_wave, f,  p):
    x = 2*np.pi*a/z_wave
    refrel = neff
    qext, qsca, qback, g = miepython.mie(refrel,x)
    return f*qsca*np.pi* a**2 * a**(p)

def chi2(data, data_unc, model, lnlike = False):
    """Calculate the chi-squared value or log-likelihood for given data and model. 
    Note: if data_unc has values <= 0, they will be ignored and replaced by NaN.
    Input:  data: 2D array, observed data.
            data_unc: 2D array, uncertainty/noise map of the observed data.
            lnlike: boolean, if True, then the log-likelihood is returned.
    Output: chi2: float, chi-squared or log-likelihood value."""
    data_unc[np.where(data_unc <= 0)] = np.nan
    chi2 = np.nansum(((data-model)/data_unc)**2)

    # print('reduced chi2: ', chi2/(np.count_nonzero(~np.isnan(data_unc)-3)))

    if lnlike:
        loglikelihood = -0.5*np.log(2*np.pi)*np.count_nonzero(~np.isnan(data_unc)) - 0.5*chi2 - np.nansum(np.log(data_unc))
        # -n/2*log(2pi) - 1/2 * chi2 - sum_i(log sigma_i) 
        return loglikelihood
    return chi2


def output_ice_nk(csv_data, ice_type, T):
    # ice_type == 'Amorphous' or 'Crystalline'
    crop_data = csv_data[csv_data['ice_type'] == ice_type]
    crop_data = crop_data[crop_data['T'] == T]
    crop_data_nk = np.zeros((3,len(crop_data)))
    crop_data_nk[0,:] = crop_data['lamda'] 
    crop_data_nk[1,:] = crop_data['n'] 
    crop_data_nk[2,:] = crop_data['k'] 
    return crop_data_nk




#######################################

optical_constant_path =  '../spectral_extraction_fitting_scripts/fitting/optical_constants/'


###################################################
g = [0.5]
angles = [90]
f =  hgg_phase_function(math.radians(angles[0]), g, rayleigh_pol = False)
###################################################

pyr = np.loadtxt(optical_constant_path + 'pyr-mg70-Dorschner1995.lnk' , usecols=[0,1,2])
fes = np.loadtxt(optical_constant_path + 'fes-Henning1996.lnk' , usecols=[0,1,2])
nh3 = np.loadtxt(optical_constant_path + 'nh3-m-Martonchik1983.txt', usecols=[0,1,2])
olivine = np.loadtxt(optical_constant_path + 'ol-mg40-Dorschner1995.lnk', usecols=[0,1,2]) #'Olivine_modif.dat'
sil = np.loadtxt(optical_constant_path + 'sic-Draine1993.txt', usecols=[0,1,2])
carbon_c = np.loadtxt(optical_constant_path + 'c-gra-Draine2003.lnk', usecols=[0,1,2])
carbon_a = np.loadtxt(optical_constant_path + 'c-z-Zubko1996.lnk', usecols=[0,1,2])
co2_a = np.loadtxt(optical_constant_path + 'co2-a-Gerakines2020.lnk', usecols=[0,1,2])
co2_c = np.loadtxt(optical_constant_path + 'co2-w-Warren1986.lnk', usecols=[0,1,2])

h2o_a15 = np.loadtxt(optical_constant_path + 'water_ice_Amorphous_15_Mastrapa_nk.txt' ,usecols=[0,1,2])
h2o_a25 = np.loadtxt(optical_constant_path + 'water_ice_Amorphous_25_Mastrapa_nk.txt' ,usecols=[0,1,2])
h2o_a40 = np.loadtxt(optical_constant_path + 'water_ice_Amorphous_40_Mastrapa_nk.txt' ,usecols=[0,1,2])
h2o_a50 = np.loadtxt(optical_constant_path + 'water_ice_Amorphous_50_Mastrapa_nk.txt' ,usecols=[0,1,2])
h2o_a60 = np.loadtxt(optical_constant_path + 'water_ice_Amorphous_60_Mastrapa_nk.txt' ,usecols=[0,1,2])
h2o_a80 = np.loadtxt(optical_constant_path + 'water_ice_Amorphous_80_Mastrapa_nk.txt' ,usecols=[0,1,2])
h2o_a100 = np.loadtxt(optical_constant_path + 'water_ice_Amorphous_100_Mastrapa_nk.txt' , usecols=[0,1,2])
h2o_a120 = np.loadtxt(optical_constant_path + 'water_ice_Amorphous_120_Mastrapa_nk.txt', usecols=[0,1,2])

h2o_c20 = np.loadtxt(optical_constant_path + 'water_ice_Crystalline_20_Mastrapa_nk.txt' ,usecols=[0,1,2])
h2o_c40 = np.loadtxt(optical_constant_path + 'water_ice_Crystalline_40_Mastrapa_nk.txt' ,usecols=[0,1,2])

compositions = {
    'H2O-A15':h2o_a15,
    'H2O-A25':h2o_a25,
    'H2O-A40':h2o_a40,
    'H2O-A50':h2o_a50,
    'H2O-A60':h2o_a60,
    'H2O-A80':h2o_a80,
    'H2O-A100':h2o_a100, 
    'H2O-A120':h2o_a120
}

# compositions = {
#     'H2O-C20':h2o_c20,
#     'H2O-C40':h2o_c40}
import mpmath
###################################################
def make_model(x, f_comp1, COMPS, z_wave):
    inital_refracive_indices_guess = [3, 1] # for refractive indices > normally range between 1-3
    scaling_factor, a_min, a_max, a_exp,porosity = x

    num_components = len(COMPS)
    n_comp = {}

    for i, component in enumerate(COMPS):

        w, n = inter_n_i(component, z_wave)
        n_comp[i] = n

    neff_brug = np.zeros((len(z_wave)), dtype=complex)
    I = np.zeros((len(z_wave)))

    fitted_model_spec = np.zeros((len(z_wave),2))
    fitted_model_spec[:,0] = z_wave

    n_array = np.zeros((num_components+1, len(z_wave)), dtype=complex)
    n_array[0,:] = np.ones((len(z_wave)))
    for key, value in n_comp.items():
        n_array[key+1,:] = value

    if f_comp1 is not None:
        f_C = 1- porosity - np.sum(f_comp1)
        vf_array = np.array([porosity])
        for i in f_comp1:
            vf_array = np.append(vf_array, i)
        vf_array = np.append(vf_array, f_C)
    else:
        f_C = 1- porosity 
        vf_array = np.array([porosity, f_C])
    for z in range(len(z_wave)):
        # n_bg_real, n_bg_imag = fsolve(sum_bg, inital_refracive_indices_guess, args=(vf_array, n_array[:,z]))
        # neff_brug[z] = complex(n_bg_real, -1*n_bg_imag)
        n_bg_test =mpmath.findroot(lambda n_bg: sum((vf_array[n]*(n_array[n,z]**2 - (n_bg)**2)/(n_array[n,z]**2 + 2*(n_bg)**2)) for n in np.arange(0, vf_array.shape[0])), inital_refracive_indices_guess , solver='muller')
        n_bg_real = n_bg_test.real 
        n_bg_imag = n_bg_test.imag                    
        neff_brug[z] = complex(n_bg_real, -1*n_bg_imag)
   


        a = np.linspace(a_min, a_max,700)
        trapz_y = integrand(a, neff_brug[z], z_wave[z], f,  a_exp)
        I[z] = np.trapz(trapz_y, a)


    model = I * scaling_factor
    return model