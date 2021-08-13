'''
Created by Victor Delvigne
ISIA Lab, Faculty of Engineering University of Mons, Mons (Belgium)
victor.delvigne@umons.ac.be
Copyright (C) 2021 - UMons
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
'''

import os 
import torch
import math as m
import numpy as np

import torch.optim as optim

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata

def gen_noise_tensor(shape, mean=0, std=0.01):
    return std*torch.randn((shape))+mean

def random_config(emb_range, hid_range, head_range, layer_range):
    np.random.shuffle(emb_range)
    np.random.shuffle(hid_range)
    np.random.shuffle(head_range)
    np.random.shuffle(layer_range)

    e_dim = emb_range[0]
    h_dim = hid_range[0]
    nhead = head_range[0]
    nlayer = layer_range[0]

    return e_dim, h_dim, nhead, nlayer

class EEGDataset(Dataset):
    def __init__(self, label, eeg):
        self.sig = eeg
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        eeg = self.sig[idx]
        label = self.label[idx]

        return (eeg, label)

def band_image(frequency_band, electrodes_location, img_size=32):
    frequency_band += 1e-25
    locs_2d = elec_proj( electrodes_location )
    frequency_band = frequency_band / np.min( frequency_band )
    frequency_band = frequency_band.reshape( (frequency_band.shape[0], -1) )

    images = image_generation( frequency_band, locs_2d, img_size )
    return images

def elec_proj(loc_3d):
    locs_2d = []
    for l in loc_3d:
        locs_2d.append( azim_proj( l ) )
    return np.asarray( locs_2d )

def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.
    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph( pos[0], pos[1], pos[2] )
    return pol2cart( az, m.pi / 2 - elev )

def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x ** 2 + y ** 2
    r = m.sqrt( x2_y2 + z ** 2 )  # r
    elev = m.atan2( z, m.sqrt( x2_y2 ) )  # Elevation
    az = m.atan2( y, x )  # Azimuth
    return r, elev, az

def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * m.cos( theta ), rho * m.sin( theta )

def image_generation(feature_matrix, electrodes_loc, n_gridpoints):
    n_electrodes = electrodes_loc.shape[0]  # number of electrodes
    n_bands = feature_matrix.shape[1] // n_electrodes  # number of frequency bands considered in the feature matrix
    n_samples = feature_matrix.shape[0]  # number of samples to consider in the feature matrix.

    # Checking the dimension of the feature matrix
    if feature_matrix.shape[1] % n_electrodes != 0:
        print( 'The combination feature matrix - electrodes locations is not working.' )
    assert feature_matrix.shape[1] % n_electrodes == 0
    new_feat = []

    # Reshape a novel feature matrix with a list of array with shape [n_samples x n_electrodes] for each frequency band
    for bands in range( n_bands ):
        new_feat.append( feature_matrix[:, bands * n_electrodes: (bands + 1) * n_electrodes] )

    # Creation of a meshgrid data interpolation
    #   Creation of an empty grid
    grid_x, grid_y = np.mgrid[
                     np.min( electrodes_loc[:, 0] ): np.max( electrodes_loc[:, 0] ): n_gridpoints * 1j,  # along x_axis
                     np.min( electrodes_loc[:, 1] ): np.max( electrodes_loc[:, 1] ): n_gridpoints * 1j  # along y_axis
                     ]

    interpolation_img = []
    #   Interpolation
    #       Creation of the empty interpolated feature matrix
    for bands in range( n_bands ):
        interpolation_img.append( np.zeros( [n_samples, n_gridpoints, n_gridpoints] ) )
    #   Interpolation between the points
    # print('Signals interpolations.')
    for sample in tqdm( range( n_samples ) ):
        for bands in range( n_bands ):
            interpolation_img[bands][sample, :, :] = griddata( electrodes_loc, new_feat[bands][sample, :],
                                                               (grid_x, grid_y), method='cubic', fill_value=np.nan )
    #   Normalization - replacing the nan values by interpolation
    for bands in range( n_bands ):
        interpolation_img[bands][~np.isnan( interpolation_img[bands] )] = scale(
            interpolation_img[bands][~np.isnan( interpolation_img[bands] )] )
        interpolation_img[bands] = np.nan_to_num( interpolation_img[bands] )
    return np.swapaxes( np.asarray( interpolation_img ), 0, 1 )  # swap axes to have [samples, colors, W, H]