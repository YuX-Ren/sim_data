#!/usr/bin/env python

'''
reconstruction based simulation by simply convoluting with CTF and adding proper level of noise 

code modified according to 
aitom/aitom/simulation/backprojection_reconstruction.m
aitom/aitom/simulation/reconstruction__eman2.py

import aitom.simulation.reconstruction__simple_convolution as TSRSC
aitom/aitom/simulation/reconstruction__simple_convolution.py

'''
import copy
import numpy as np
import numpy.fft as NF
from io_util import save_png
import ctf as TIOC
from util import cub_img, sphere_mask, grid_displacement_to_center, grid_distance_to_center, grid_distance_sq_to_center
from geometry import fft_mid_co, rotate, random_rotation_angle_zyz


'''
create a simulated subtomogram given density map. Add proper level of noise.


adapted from
~/ln/frequent_structure/code/GenerateSimulationMap.m
backprojection_reconstruction()
~/src/aitom/aitom/simulation/reconstruction__eman2.py
'''
"""
construct a missing wedge mask, see tom_wedge,
angle represents the angle range of MISSING WEDGE region, the larger, the more missing wedge region!!!
tilt_axis is tilt axis
"""

import warnings

def tom_bandpass(v, low, hi, smooth=None):

    vt = NF.fftn(v)
    vt = NF.fftshift(vt)

    mid_co = fft_mid_co(v.shape)
    if smooth is None:
        d = grid_distance_sq_to_center(grid_displacement_to_center(v.shape, mid_co=mid_co))
        vt[d > hi] = 0.0
        vt[d < low] = 0.0
    else:
        m = sphere_mask(v.shape, center=mid_co, radius=hi, smooth_sigma=smooth)
        if low > 0:            m -= sphere_mask(v.shape, center=mid_co, radius=low, smooth_sigma=smooth)
        vt *= m

    vt = NF.ifftshift(vt)
    vt = NF.ifftn(vt)
    vt = np.real(vt)
    return vt

def wedge_mask(size, ang1, ang2=None, tilt_axis=1, sphere_mask_need=True, verbose=False):
    # should define both tilt axis and electron beam (missing wedge) direction
    warnings.warn("The definition of wedge mask is still ambiguous")

    if ang2 is None:
        ang2 = float(np.abs(ang1))
        ang1 = -ang2

    else:
        assert ang1 < 0
        assert ang2 > 0

    if verbose:
        print('image.vol.wedge.util.wedge_mask()', 'ang1', ang1, 'ang2', ang2, 'tilt_axis', tilt_axis, 'sphere_mask',
              sphere_mask_need)

    ang1 = (ang1 / 180.0) * np.pi
    ang2 = (ang2 / 180.0) * np.pi

    g = grid_displacement_to_center(size=size, mid_co=fft_mid_co(siz=size))

    if tilt_axis == 0:
        # y-z plane
        # y axis
        x0 = g[1]
        # z axis
        x1 = g[2]

    elif tilt_axis == 1:
        # x-z plane
        # x axis
        x0 = g[0]
        # z axis
        x1 = g[2]

    elif tilt_axis == 2:
        # x-y plane
        # x axis
        x0 = g[0]
        # y axis
        x1 = g[1]

    m = np.zeros(size, dtype=float)

    m[np.logical_and(x0 >= (np.tan(ang2) * x1), x0 >= (np.tan(ang1) * x1))] = 1.0
    m[np.logical_and(x0 <= (np.tan(ang1) * x1), x0 <= (np.tan(ang2) * x1))] = 1.0

    if sphere_mask_need:
        m *= sphere_mask(m.shape)

    return m


def tilt_mask(size, tilt_ang1, tilt_ang2=None, tilt_axis=1, light_axis=2, sphere_mask_need=True):
    """wedge mask defined using tilt angles light axis is the direction of electrons"""
    assert tilt_axis != light_axis

    if tilt_ang2 is None:
        tilt_ang2 = float(np.abs(tilt_ang1))
        tilt_ang1 = -tilt_ang2

    else:
        assert tilt_ang1 < 0
        assert tilt_ang2 > 0

    tilt_ang1 = (tilt_ang1 / 180.0) * np.pi
    tilt_ang2 = (tilt_ang2 / 180.0) * np.pi

    g = grid_displacement_to_center(size=size, mid_co=fft_mid_co(siz=size))

    plane_axis = set([0, 1, 2])
    plane_axis.difference_update([light_axis, tilt_axis])
    assert len(plane_axis) == 1
    plane_axis = list(plane_axis)[0]

    x_light = g[light_axis]
    x_plane = g[plane_axis]

    m = np.zeros(size, dtype=float)

    m[np.logical_and(x_light <= (np.tan(tilt_ang1) * x_plane), x_light >= (np.tan(tilt_ang2) * x_plane))] = 1.0
    m[np.logical_and(x_light >= (np.tan(tilt_ang1) * x_plane), x_light <= (np.tan(tilt_ang2) * x_plane))] = 1.0

    if sphere_mask_need:
        m *= sphere_mask(m.shape)

    return m

def do_reconstruction(v, op, signal_variance=None, verbose=False):

    mask = wedge_mask(v.shape, op['model']['missing_wedge_angle']) * sphere_mask(v.shape);     assert      np.all(np.isfinite(mask))

    if 'Dz' in op['ctf']:
        ctf = TIOC.create(Dz=op['ctf']['Dz'], size=v.shape, pix_size=op['ctf']['pix_size'], voltage=op['ctf']['voltage'], Cs=op['ctf']['Cs'], sigma=op['ctf']['sigma'] if 'sigma' in op['ctf'] else None)['ctf']
    else:
        # in this case, we do not have CTF defined
        ctf = np.zeros_like(mask) + 1

    if signal_variance is None:
        signal_variance = calc_variance(v_e=v, ctf=ctf, mask=mask, verbose=verbose)['variance_total']
        
    print ('signal_variance', signal_variance)

    corrfac_t = calc_corrfac(ctf=ctf, mask=mask)
    corrfac = corrfac_t['e_var'] / corrfac_t['ec_var']
    if verbose:     print ('corrfac_t', corrfac_t, 'corrfac', corrfac)

    signal_variance *= corrfac
    if verbose:     print ('signal_variance corrfac', signal_variance)

    vb = do_reconstruction_given_sigma(v=v, ctf=ctf, signal_variance=signal_variance, op=op, verbose=verbose)


    vb_t = vb.copy()
    vb_t = NF.fftn(vb_t)
    vb_t = NF.fftshift(vb_t)
    vb_t *= mask
    vb_t = NF.ifftshift(vb_t)
    vb_t = NF.ifftn(vb_t)
    vb_t = np.real(vb_t)
    assert      np.all(np.isfinite(vb_t))

    return vb_t



'''
this is to save computation, given dm which was masked by wedge mask, and  pre-calculated ctf, variance_sigma,
convolute dm with ctf with proper level of noise added
'''
def do_reconstruction_given_sigma(v, ctf, signal_variance, op, verbose=False):

    op = copy.deepcopy(op)

    Ny = 1 / (2.0 * op['ctf']['pix_size'])
    BPThresh = np.floor(((1/4.0)*(v.shape[0]/2.0))/Ny)
    if verbose: print ('BPThresh', BPThresh)
  
    v = tom_bandpass(v, low=0, hi=BPThresh, smooth=2.0)
    assert np.all(np.isfinite(v))


    WeiProj = 0.5
    WeiMTF = 1 - WeiProj


    mid_co = fft_mid_co(v.shape)

    SNR = op['model']['SNR']
    if SNR is None:     SNR = np.inf


    if np.isfinite(SNR):
        noisy = v + np.random.normal(loc=0.0, scale=np.sqrt(WeiProj/SNR*signal_variance), size=v.shape)
    else:
        noisy = v

    noisy = NF.fftn(noisy)
    noisy = NF.fftshift(noisy)
    noisy *= ctf
    noisy = NF.ifftshift(noisy)
    noisy = NF.ifftn(noisy)
    noisy = np.real(noisy)
    assert np.all(np.isfinite(noisy))

    if np.isfinite(SNR):
        mtf_t = np.random.normal(loc=0.0, scale=np.sqrt(WeiMTF/SNR*signal_variance), size=noisy.shape)
        mtf_t = tom_bandpass(mtf_t, low=0.0, hi=1.0, smooth=0.2*noisy.shape[0])
        noisy += mtf_t
        noisy = tom_bandpass(noisy, low=0.0, hi=BPThresh, smooth=2.0)
        assert np.all(np.isfinite(noisy))


    vb = noisy.copy()

    #if ('result_standardize' in op['model']) and op['model']['result_standardize']:   vb = (vb - vb.mean()) / vb.std()

    return vb


def calc_variance(v_e, ctf, mask, verbose=False):
    re = {}

    v = NF.fftn(v_e)
    v = NF.fftshift(v)
    v *= ctf * mask         # here applying wedge mask is very important for estimating correct level of noice to be added
    v = NF.ifftshift(v)
    v = NF.ifftn(v)
    v = np.real(v)

    re['variance_total'] = v.var()

    return re


def calc_corrfac(ctf, mask):
    e = np.random.normal(loc=0.0, scale=np.sqrt(1.0), size=ctf.shape)
    e = NF.fftn(e)
    e = NF.fftshift(e)
    ec = e * ctf * mask       # error convoluted with ctf. Here applying wedge mask is very important for estimating correct level of noice to be added
    ec = NF.ifftshift(ec)
    ec = NF.ifftn(ec)
    ec = np.real(ec)

    e = NF.ifftshift(e)
    e = NF.ifftn(e)
    e = np.real(e)   # error without ctf

    return {'e_var':e.var(), 'ec_var':ec.var()}



# just a test given all parameters fixed
# def simulation_test0():
#     import aitom.io.file as TIF

#     #op = {'model':{'missing_wedge_angle':30, 'SNR':np.nan}, 'ctf':{'pix_size':1.0, 'Dz':-15.0, 'voltage':300, 'Cs':2.2, 'sigma':0.4}}
#     op = {'model':{'missing_wedge_angle':30, 'SNR':0.05}, 'ctf':{'pix_size':1.0, 'Dz':-5.0, 'voltage':300, 'Cs':2.0, 'sigma':0.4}}

#     v = generate_toy_model(dim_siz=64)
    
#     loc_proportion = 0.1
#     loc_max = np.array(v.shape, dtype=float) * loc_proportion
#     angle = random_rotation_angle_zyz()
#     loc_r = (np.random.random(3)-0.5)*loc_max

#     vr = rotate(v, angle=angle, loc_r=loc_r, default_val=0.0)

#     import aitom_core.simulation.reconstruction.reconstruction__simple_convolution as TSRSC
#     vb = TSRSC.do_reconstruction(vr, op, verbose=True)
#     print ('vb', 'mean', vb.mean(), 'std', vb.std(), 'var', vb.var())

#     if True:
#         vb_rep = do_reconstruction(vr, op, verbose=True)

#         # calculate SNR
#         import scipy.stats as SS
#         vb_corr = SS.pearsonr(vb.flatten(), vb_rep.flatten())[0]
#         vb_snr = 2*vb_corr / (1 - vb_corr)
#         print ('SNR', 'parameter', op['model']['SNR'], 'estimated', vb_snr )         # fsc = ssnr / (2.0 + ssnr)


#     #TIF.put_mrc(vb, '/tmp/vb.mrc', overwrite=True)
#     #TIF.put_mrc(v, '/tmp/v.mrc', overwrite=True)


#     save_png(cub_img(vb)['im'], "/tmp/vb.png")
#     save_png(cub_img(v)['im'], "/tmp/v.png")


#     # save Fourier transform magnitude for inspecting wedge regions
#     vb_f = NF.fftn(vb)
#     vb_f = NF.fftshift(vb_f)
#     vb_f = np.abs(vb_f)
#     vb_f = np.log(vb_f)
#     TIF.put_mrc(vb_f, '/tmp/vb-f.mrc', overwrite=True)



if __name__=='__main__':
    #test_reconstruction0()
    #test_projection()
    simulation_test0()



