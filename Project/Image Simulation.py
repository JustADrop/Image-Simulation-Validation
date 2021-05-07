# -*- coding: utf-8 -*-
"""
Created on Fri May  7 12:47:00 2021

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
from prysm.polynomials import zernike_nm
from prysm.detector import bindown

output_res = 100
wlen = 0.550
pixel_pitch = 4.5
f = 50
fno = 4
lens_D = f/fno
lens_R = lens_D/2

# Zernike spherical, coma(+/-), astig(+/-), curv, dist(+/-)
abers = [(4,0),
         (3,1),
         (3,-1),
         (2,2),
         (2,-2),
         (2,0),
         (1,1),
         (1,-1)
    ]

# The above specifications determine a Q of:
Q = (wlen)*(fno)/pixel_pitch

# Need Q_forward >= 2 for forward model, so Q_forward = Q*(oversampling) = 2
oversampling = ceil(2/Q)
Q_forward = round(Q*oversampling, 1)

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

# Construct phase mode
aber = zernike_nm(abers[0][0], abers[0][1], norm_r, theta) # spherical aberration

# Scale phase mode
# TODO: Look into specific lens MTF's and determine suitable/realistic aberration type and magnitude.
phase = aber # Will be aber*magnitude

# Construct pupil function from amp and phase functions, propagate to PSF plane, take intensity.
P = Wavefront.from_amp_and_phase(amp, phase, wlen, pup_dx)
coherent_PSF = P.focus(f, Q=Q_forward)
PSF = coherent_PSF.intensity

# Plot PSF
PSF_radius = 1.22*wlen*fno
PSF.plot2d(xlim=5*PSF_radius, cmap='gray', clim=(0, .1e7))


# Hist of PSF
# hist = plt.hist(np.histogram(PSF.data))

# Construct MTF from PSF
MTF = mtf_from_psf(PSF, PSF.dx)

# Construct normalized P for easier plotting
# norm_P = Wavefront.from_amp_and_phase(amp, phase, wlen, pup_dx)
# norm_P.data = norm_P.data/np.sum(norm_P.data)
# norm_coherent_PSF = norm_P.focus(f, Q=Q_forward)
# norm_PSF = norm_coherent_PSF.intensity
# norm_PSF_radius = 1.22*(wlen)*fno
# norm_PSF.plot2d(xlim=norm_PSF_radius*5, cmap='gray', clim=(0,0.3))

fx, _ = MTF.slices().x
fig, ax = MTF.slices().plot(['x', 'y', 'azavg'], xlim=(0,250), ylim=(0,1))
ax.plot(fx, ls=':', c='k', alpha=0.75, zorder=1)
ax.set(xlabel='Spatial frequency, cy/mm', ylabel='MTF')

# Constuct sample grid for edge/star test patterns
x,y = make_xy_grid(shape=PSF.data.shape[0], dx=PSF.dx)
rho, t = cart_to_polar(x, y)

# Construct slanted edge and Siemen's star test patterns
# Edge simulation should be done at ~100 output res
# Star simulation should be done at ~512 output res
edge = slantededge(x, y)
# star = siemensstar(rho, t, 40, oradius=110*lens_R)

# Blur images by convolving with PSF
convedge = conv(edge, PSF.data)
# convstar = conv(star, PSF.data)

# Resample to output resolution
edge_image = bindown(convedge, oversampling)
# star_image = bindown(convstar, oversampling)

# Plot edge test pattern and blurred image
fig, axes = plt.subplots(ncols=2, figsize=(10,10))
axes[0].imshow(edge, cmap='gray')
axes[1].imshow(edge_image, cmap='gray')

# Plot star test pattern and blurred image
# fig, axes = plt.subplots(ncols=2, figsize=(10,10))
# axes[0].imshow(star, cmap='gray')
# axes[1].imshow(star_image, cmap='gray')
