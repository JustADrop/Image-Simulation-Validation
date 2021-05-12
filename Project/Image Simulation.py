# -*- coding: utf-8 -*-
"""
Created on Wed May 12 12:23:37 2021

@author: Sean
"""
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from prysm.propagation import psf_sample_to_pupil_sample, Wavefront
from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.geometry import circle
from prysm.otf import mtf_from_psf
from prysm.objects import slantededge, siemensstar
from prysm.convolution import conv
from prysm.polynomials import hopkins
from prysm.detector import bindown

output_res = 100
wlen = 0.550
pixel_pitch = 4.5
f = 50
fno = 1.4
lens_D = f/fno
lens_R = lens_D/2

# The above specification determine a Q of:
Q = (wlen)*(fno)/pixel_pitch

# Need Q_forward >= 2 for forward model, so Q_forward = Q*(oversampling) = 2
oversampling = ceil(2/Q)
Q_forward = round(Q*oversampling, 2)

# Intermediate higher res gives psize = pixel_pitch/oversampling
psize = pixel_pitch/oversampling

# PSF_domain_res will be output_res*oversampling
# Pupil domain samples will be (PSF_domain_res)/Q_forward
samples = ceil(output_res*oversampling/Q_forward)

# Find pupil dx from wanted psize
pup_dx = psf_sample_to_pupil_sample(psize, samples, wlen, f)

# Construct pupil grid, convert to polar, construct normalized r for phase
xi, eta = make_xy_grid(samples, dx=pup_dx)
r, theta = cart_to_polar(xi, eta)
norm_r = r/lens_R

# Construct amplitude function of pupil function
amp = circle(lens_R, r)
amp = amp / amp.sum()

# Construct phase mode
aber = hopkins(0, 4, 0, norm_r, theta, 1) # spherical aberration

# Scale phase mode
phase = aber * (wlen/8*1e3)*16

# Construct pupil function from amp and phase functions, propagate to PSF plane, take square modulus.
P = Wavefront.from_amp_and_phase(amp, phase, wlen, pup_dx)
coherent_PSF = P.focus(f, Q=Q_forward)
PSF = coherent_PSF.intensity

# Plot PSF
PSF_radius = 1.22*wlen*fno
PSF.plot2d(xlim=5*PSF_radius, cmap='gray', clim=(0, .1))

# Construct MTF from PSF
MTF = mtf_from_psf(PSF, PSF.dx)

fx, _ = MTF.slices().x
fig, ax = MTF.slices().plot(['x', 'y', 'azavg'], xlim=(0,150), ylim=(0,1))
ax.axvline(1000/(2*pixel_pitch))
ax.set(xlabel='Spatial frequency, cy/mm', ylabel='MTF')

# Constuct sample grid for edge/star test patterns
x,y = make_xy_grid(shape=PSF.data.shape[0], dx=PSF.dx)
rho, t = cart_to_polar(x, y)

# Construct slanted edge and Siemen's star test patterns
# Edge simulation should be done at ~100 output res
# Star simulation should be done at ~512 output res
edge = slantededge(x, y)
star = siemensstar(rho, t, 40, oradius=x.max()*0.8)

# Blur images by convolving with PSF
convedge = conv(edge, PSF.data)
convstar = conv(star, PSF.data)

# Resample to output resolution
edge_image = bindown(convedge, oversampling)
star_image = bindown(convstar, oversampling)

# Plot edge test pattern and blurred image
fig, axes = plt.subplots(ncols=2, figsize=(10,10))
axes[0].imshow(edge, cmap='gray')
axes[1].imshow(edge_image, cmap='gray')

# Plot star test pattern and blurred image
# fig, axes = plt.subplots(ncols=2, figsize=(10,10))
# axes[0].imshow(star, cmap='gray')
# axes[1].imshow(star_image, cmap='gray')
