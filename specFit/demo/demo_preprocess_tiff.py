# -*- coding: utf-8 -*-
"""
For usage, see
  python demo_preprocess_tiff.py --help
"""

import os
import argparse
import numpy as np
from PIL import Image

prefix = 'preProcessDemo'  # Image file name prefix.

parser = argparse.ArgumentParser(description='demo_preprocess_tiff.py')
parser.add_argument('--processed_dir', type=str, default='images/processed/demo')
parser.add_argument('--raw_dir', type=str, default='images/raw/demo')
parser.add_argument('--Nfiles', type=int, default=256)
parser.add_argument('--Nx', type=int, default=7)
parser.add_argument('--Ny', type=int, default=7)
parser.add_argument('--spectra_type', type=int, default=1, help='Spectra of test images.')

args = parser.parse_args()
processed_dir = args.processed_dir
raw_dir = args.raw_dir
Nfiles = args.Nfiles
Nx = args.Nx
Ny = args.Ny
spectra_type = args.spectra_type

assert Nfiles > 1, "Required: Nfiles > 1"
assert Nx > 2, "Required: Nx > 2"
assert Ny > 2, "Required: Ny > 2"

if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

###########################################################################
# Create test images
imarray = np.zeros((Nx, Ny), dtype=np.uint8)
T = float(Nfiles)
for i in range(0, Nfiles):  # Loop over files
    arr = imarray
    t = float(i)
    if spectra_type == 1:
        # Single period
        tmp = np.sin(4*np.pi*t/T)
    elif spectra_type == 2:
        # Power law
        f = np.arange(2, 128, 2)
        tmp = 0
        for j in range(0, len(f)):
            tmp = tmp + (1./float(f[j]))*np.sin(f[j]*2.*np.pi*t/T)
    elif spectra_type == 3:
        # Power law + tail
        f = np.arange(2, 128, 2)
        tmp = 0
        for j in range(0, len(f)):
            if f[j] < 32:
                A = (1./float(f[j]))
            tmp = tmp + A*np.sin(f[j]*2.*np.pi*t/T)
    else:
        assert False, "Unknown spectra type " % spectra_type

    arr[:, :] = np.uint8(255*(tmp + 1.0)/2.0)
    im = Image.fromarray(arr)
    im.save(os.path.join(raw_dir, prefix+'-%03d.tiff' % i))
###########################################################################

###########################################################################
# Read images
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

# Compute time-averaged image
cube_avg = np.uint8(np.average(cube, axis=2))

###########################################################################
# Save files needed for specFit processing
np.save(os.path.join(processed_dir, 'dataCube.npy'), cube)
np.save(os.path.join(processed_dir, 'visual.npy'), cube_avg)
np.save(os.path.join(processed_dir, 'timestamps.npy'), timestamps)
np.save(os.path.join(processed_dir, 'exposures.npy'), exposures)

print('Wrote %d images to %s. Wrote npy files to %s.' % (Nfiles, raw_dir, processed_dir))
