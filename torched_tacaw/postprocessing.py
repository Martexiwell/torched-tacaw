#!/usr/bin/env python3
import itertools
from typing import Iterable

# from tqdm import tqdm
#
# import logging
# import os
# import time
#
# import pathlib
# import scipy
# import scipy.constants as c
# import torch
#
# import yaml
import zarr
# from zarr.storage import LocalStore

from filelock import Timeout, FileLock

import numpy as np

# from ase.io import Trajectory
# import ase
#
# import pyms
# from pyms.utils import structure_routines

import tools
# from tools import indices_of_cutout_from_array, cutout_from_array

import core as tc
from core import Config


class Detector:
    def __init__(self, config: Config, label:str):
        self.config = config
        self.label = label
        self.mask = np.zeros(
            config['simulation','kspace','ROI_shape']
        )
        self.parameters = {}

    @classmethod
    def from_file(cls, config_file: str):
        return cls(Config.load_from_yaml(config_file))

    def to_set(self, label):
        return DetectorSet(**{label:self})

    def make_ESTEM_image(self):
        self.compute_ESTEM()
        self.save_estem()

    def compute_ESTEM(self):
        file_zarray = self.config['storage','intensities_zarray']
        print(f'oppening zarr: {file_zarray}')
        data_zarray = zarr.load(file_zarray)
        print('checking sizes')
        chunk_size_x, chunk_size_y = self.config['beam','scanning','batch_shape']
        print('and othe r sizes')
        s0, energy_size, nof_scan_x, nof_scan_y, kx_size, ky_size = data_zarray.shape

        print('allocating memory for image')
        self.ESTEM_image = np.empty([energy_size, nof_scan_x, nof_scan_y])

        # go over all the zarr chunks
        print('looping over all batches in zarr')
        for chunk_id_x, chunk_id_y in itertools.product(range(nof_scan_x // chunk_size_x), range(nof_scan_y // chunk_size_y)):
            # print(f'reading chunk [{chunk_id_x}, {chunk_id_y}]')
            chunk = tools.get_nth_cutout_from_array(
                data_zarray,
                [s0, energy_size, chunk_size_x, chunk_size_y, kx_size, ky_size],
                [0,0,chunk_id_x,chunk_id_y,0,0]
            )
            
            # image_chunk = tools.get_nth_cutout_from_array(
            #     self.ESTEM_image,
            #     [energy_size, chunk_size_x, chunk_size_y],
            #     [0,chunk_id_x, chunk_id_y]
            # )
            # print('computing image')
            start_x = chunk_id_x*chunk_size_x
            stop_x = (chunk_id_x+1)*chunk_size_x
            stop_x = stop_x if stop_x < nof_scan_x else nof_scan_x

            start_y = chunk_id_y*chunk_size_y
            stop_y = (chunk_id_y+1)*chunk_size_y
            stop_y = stop_y if stop_y < nof_scan_y else nof_scan_y

            self.ESTEM_image[:,start_x:stop_x,start_y:stop_y] = np.einsum(
                'xy,eklxy->ekl',
                self.mask,
                chunk[0]
            )
            # print(f'chunk [{chunk_id_x}, {chunk_id_y}] done')

    def save_estem(self, zarr_path:str|None = None):
        """dump data to zarr array
        :param zarr_path: optional path to zarr array, if not provided
        """
        # read zarr path
        try:
            zarr_path = self.config['storage','estem_zarray']
        except KeyError:
            zarr_path = zarr_path if zarr_path else self.config['datafolder']+'estem.zarray'
            self.config['storage']['estem_zarray'] = zarr_path

        try:
            self.config['detectors'][self.label] = self.parameters
        except:
            print('adding detectors to config ... ')
            self.config.config['detectors'] = {self.label: self.parameters}

        # update config
        lock = FileLock(self.config['config_file'] + '.lock', timeout=3 * 60)
        with lock:
            self.config.dump_to_yaml(self.config['config_file'])


        # dump to zarr group
        zarr_group = zarr.open_group(zarr_path, mode='a')
        zarray = zarr_group.create_array(
            name=self.label,
            shape=self.ESTEM_image.shape,
            chunks=self.ESTEM_image.shape,
            dtype='f8',
            overwrite=True,
        )
        zarray[:,:,:] = self.ESTEM_image






class CircularDetector(Detector):
    def __init__(
            self,
            config: Config,
            label:str,
            center_mrad:Iterable[float],
            radius_mrad:float,
            **kwargs
    ):
        super().__init__(config, label, **kwargs)
        Kx, Ky = config.crop_arr_qspace_2ROI(config.get_reciprocal_space_grids_mrad())

        center_x, center_y = center_mrad
        mask = (Kx - center_x) ** 2 + (Ky - center_y) ** 2 <= radius_mrad ** 2

        self.mask = mask.astype(int)

        self.parameters = {
            'type': 'circular',
            'center_mrad' : center_mrad,
            'radius_mrad' : radius_mrad,
        }

class AnularDetector(Detector):
    def __init__(
            self,
            config: Config,
            label: str,
            center_mrad: Iterable[float],
            radius_inner_mrad: float,
            radius_outer_mrad: float,
            **kwargs
    ):
        super().__init__(config, label, **kwargs)
        Kx, Ky = config.crop_arr_qspace_2ROI(config.get_reciprocal_space_grids_mrad())

        center_x, center_y = center_mrad
        mask_outer = (Kx - center_x) ** 2 + (Ky - center_y) ** 2 <= radius_outer_mrad ** 2
        mask_inner = (Kx - center_x) ** 2 + (Ky - center_y) ** 2 <= radius_inner_mrad ** 2

        self.mask = mask_inner.astype(int) * mask_outer.astype(int)

        self.parameters = {
            'type': 'circular',
            'center_mrad': center_mrad,
            'radius_inner_mrad': radius_inner_mrad,
            'radius_outer_mrad': radius_outer_mrad,
        }

class DetectorSet:
    def __init__(
            self,
            # device:torch.device = 'cpu',
            *detectors: Detector,
            **label_detector_pairs,
    ):
        # self.device = device
        self.detectors = {}
        for detector in detectors:
            self.detectors[detector.label] = detector
        for label, detector in label_detector_pairs.items():
            self.detectors[label] = detector


    def add(self, label, detector):
        '''add a new detector'''
        self.detectors[label] = detector

    def make_ESTEM_images(self):
        for label, detector in self.detectors.items():
            detector.make_ESTEM_image()