import numpy as np
import pandas as pd

from astropy import units as u
import matplotlib.pyplot as plt

# functions for photodesorption calcualtions
from scipy.integrate import quad
from scipy.interpolate import interp1d

def inter_n_i(input_n_file, wave_inter ,):
    binary_data = np.loadtxt(input_n_file, usecols=[0,1,2])
    n_array = np.zeros((binary_data[:,0].shape[0]), dtype=complex)
    for z in range(binary_data[:,0].shape[0]):
        n_array[z] = complex(binary_data[z,1], binary_data[z,2])


    refr_index_cube_n = interp1d(binary_data[:,0], binary_data[:,1] )
    refr_index_cube_k = interp1d(binary_data[:,0], binary_data[:,2] )
    nz = len(wave_inter)
    refr_index_comp_i = np.zeros((nz), dtype=complex)
    for z in range(len(wave_inter)):
        refr_index_comp_i[z] = complex(refr_index_cube_n(wave_inter)[z], refr_index_cube_k(wave_inter)[z])
    
    return wave_inter, refr_index_comp_i, 

def cal_dust_temperature(R_star, D, T_star):
    # see equation 2 in Worthen+2024
    # https://iopscience.iop.org/article/10.3847/1538-4357/ad2354

    # Td is the temperature 
    # R_star is the stellar radius
    # D is the stellocentric distance of the dust
    # T_star is the effective temperature of the star

    Td = (R_star**2 / (4*D**2))**(1/5) * T_star
    return Td.to(u.Kelvin)

def cal_dust_temperature_blackbody(R_star, D, T_star):
    R_star = R_star * u.solRad
    T_star = T_star * u.Kelvin
    D = D * u.AU
    Td = 0.707*T_star*(R_star/D)**(1/2)
    return Td.to(u.Kelvin)

def cal_R_star(L_star, T_star):
    sigma = 5.67037442e-8  * u.kg * u.s**-3 * u.Kelvin**-4
#     L_star = L_star * unit.solLum
#     T_star = T_star * unit.Kelvin
    R_star = np.sqrt(L_star/ (4*np.pi*sigma*T_star**4 )).to(u.solRad)
    return R_star

def extinction(tau, flux_incident):
    """
    I_abs = I_0 * e^{-tau}
    Return: abopted flux
    """
    return flux_incident*np.exp(-tau)

def convert_apparent_flux_to_incident_flux(flux_app, dist_au, dist_pc):
    """Converting the apparent flux to the incident flux at the distance of the grain
    """
#     pc = 206264.806 # 1 pc = 206264.806 au
#     dist_pc = dist_pc * pc
    

    flux_in = (dist_pc/dist_au.to(u.pc)).value**2 * flux_app
    return flux_in

def cal_number_of_aborbed_photons(Qabs, dist_au, dist_pc):
    
    wu_photosphere = pd.read_csv('stellar_emission_model_Wu2024/wu_photosphere.csv')
    wu_x, wu_y = wu_photosphere['um'].values, wu_photosphere['erg/s/cm3'].values

    f = interpolate.interp1d(wu_x, wu_y)
    
    # ref.: Grigorieva+2007 (https://ui.adsabs.harvard.edu/abs/2007A%26A...475..755G/abstract)
    wave_min = 0.091 # um
    wave_max = 0.24 # um
    h = 6.62607015e-34 * 1e7 # J/Hz ->  erg/Hz
    c = 2.99792458e+8 *1e6 # um/s
    # here the flux is a constant value between wave_min and wave_max
    def f_intergal(wave, f, Qabs, dist_au, dist_pc):
        FF = f(wave)
        FF2 = convert_apparent_flux_to_incident_flux(FF, dist_au, dist_pc)
        Qabs_wave = Qabs(wave)
        # note 1e-4 is the covertion from erg/s/cm3 * um to erg/s/cm3 * cm = erg/s/cm2, which is the unit of Nabs
        return FF2/(h*c/wave) * Qabs_wave * 1e-4

    # the unit of Nabs is counts/s/cm2
    Nabs, _ = quad(f_intergal, wave_min, wave_max, args=(f, Qabs, dist_au, dist_pc) ) 
    return Nabs #erg/s/cm2

def cal_UV_desoprtion_rate( Nabs, Y, frac=1,):
    """
    Calculating the erosion rate due to UV sputtering for icy grains.
    Ref: Grigorieva et al. 2007 (https://ui.adsabs.harvard.edu/abs/2007A%26A...475..755G/abstract)
    frac: the fraction of surface covered by ice; defaut assumes to be 1
    Y: the desoprtion probability, which is between 4e-4 for amorphous ice and 2e-3 for crystalline ice
    Nabs (photon s^-1 cm^-2): the stellar flux at the distance of the dust grain. 
    """    
    m_h2o = 3e-23 * u.g # g
    rho_h2o = 0.92 * u.g/u.cm**3 #g/cm3
    Nabs = Nabs * 1/(u.s * u.cm**2)
    s_dot = - (frac * m_h2o * Y * Nabs) / (4*rho_h2o)

    # converting from cm/s to um/yr
    return s_dot.to(u.um/u.year)


def cal_collisional_timescale(D, M_submm, a_min, a_max, M_star):
    """
    Collisional timescale: ref: Chen+2006 eq(13)
    Units: 
    D <- au
    M_submm <- earth mass
    a_min <- mm
    a_max <- mm
    M_star <- solar mass
    Output t_coll <- yr
    """
    a_min = a_min.to(u.mm)  #* 1e-3  # from um to mm
    a_max = a_max.to(u.mm)  #* 1e-3  # from um to mm
    t_coll = 3000*u.yr * (D/(30*u.au))**(7/2) * (M_submm/(0.1*u.Mearth))**(-1) * (np.sqrt(a_min*a_max)/(1*u.mm)) * ((1*u.Msun)/M_star)**(1/2)
    return t_coll