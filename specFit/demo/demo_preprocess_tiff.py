# -*- coding: utf-8 -*-

desc = """
Generates a sequence of test images and uses them to create the files needed by
specFit: dataCube.npy, visual.npy, exposures.npy, and timestamps.npy.
"""

import os
import argparse
import numpy as np
from PIL import Image

# Image file name prefix.
prefix = 'demo_preprocess_tiff'

parser = argparse.ArgumentParser(description=desc,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--processed_dir',
                    type=str,
                    default='images/processed/demo',
                    help='Directory to place generated .npy files')

parser.add_argument('--raw_dir',
                    type=str,
                    default='images/raw/demo',
                    help='Directory to store generated images')

parser.add_argument('--Nfiles',
                    type=int,
                    default=256,
                    help='Number of images to generate')

parser.add_argument('--Nx',
                    type=int,
                    default=7,
                    help='Number of horizontal pixels')

parser.add_argument('--Ny',
                    type=int,
                    default=7,
                    help='Number of vertical pixels')

parser.add_argument('--spectra_type',
                    type=int,
                    default=1,
                    help='Spectra of test images.')

args = parser.parse_args()

processed_dir = args.processed_dir
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

raw_dir = args.raw_dir
if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)

Nfiles = args.Nfiles
assert Nfiles > 1, "Required: Nfiles > 1"

Nx = args.Nx
assert Nx > 2, "Required: Nx > 2"

Ny = args.Ny
assert Ny > 2, "Required: Ny > 2"

spectra_type = args.spectra_type
assert spectra_type in [1, 2, 3], "Required: spectra_type = 1, 2, or 3"

###########################################################################
# Create test images. Each pixel has the same time dependence.

img_array = np.zeros((Nx, Ny), dtype=np.uint8)
T = float(Nfiles)
for i in range(0, Nfiles):  # Loop over files
    t = float(i)
    tmp = 0
    if spectra_type == 1:
        # Single period
        tmp = np.sin(4*np.pi*t/T)
    if spectra_type == 2:
        # Power law
        f = np.arange(2, 128, 2)
        for j in range(0, len(f)):
            tmp = tmp + (1./float(f[j]))*np.sin(f[j]*2.*np.pi*t/T)
    if spectra_type == 3:
        # Power law + tail
        f = np.arange(2, 128, 2)
        for j in range(0, len(f)):
            A = 1.0
            if f[j] < 32:
                A = (1./float(f[j]))
            tmp = tmp + A*np.sin(f[j]*2.*np.pi*t/T)

    # Set all pixels to have same value
    img_array[:, :] = np.uint8(255.0*(tmp + 1.0)/2.0)
    im = Image.fromarray(img_array)
    # TODO: %03d assumes Nfiles < 1000. Compute padding based on Nfiles.
    im.save(os.path.join(raw_dir, prefix + '-%03d.tiff' % i))
###########################################################################

###########################################################################
# Read images into data cube.
cube = np.zeros((Nx, Ny, Nfiles), dtype=np.uint8)

timestamps = []
exposures = []
for i in range(0, Nfiles):
    # Normally we would read timestamp from Exif metadata in file
    # and convert to an integer.
    timestamps.append(i*2)

    exposures.append(1.0)

    im = Image.open(os.path.join(raw_dir, prefix + '-%03d.tiff' % i))
    cube[:, :, i] = np.asarray(im)
###########################################################################

###########################################################################
# Compute time-averaged image
cube_avg = np.uint8(np.average(cube, axis=2))
###########################################################################

###########################################################################
# Save files needed for specFit processing
np.save(os.path.join(processed_dir, 'dataCube.npy'), cube)
np.save(os.path.join(processed_dir, 'visual.npy'), cube_avg)
np.save(os.path.join(processed_dir, 'timestamps.npy'), timestamps)
np.save(os.path.join(processed_dir, 'exposures.npy'), exposures)
###########################################################################

print('Wrote %d images to %s. Wrote .npy files to %s.' % (Nfiles, raw_dir, processed_dir))
