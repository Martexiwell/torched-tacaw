"""
This module is used to compute the STEM images using detectors

by convention units in reciprocal space are mrad

"""
import zarr
import numpy as np
from filelock import FileLock
import dask.array as da

from typing import Iterable

import warnings

# local imports
from . import tools
from . import core as tc
from .core import Config

warnings.warn('THIS MODULE USES DASK LIBRARY AND DOES NOT SEEM TO WORK PROPERLY WITH MEMORY!')

class DetectorSet:
    """Class for defining detectors and computing the detected image.

    Parameters
    ----------
    config : Config or str
        Config object or str as the config file address
    *args : dict
            detector prescriptions provided as dictionaries must
            include at least 'type' key

    Examples
    --------
    DetectorSet('config.yaml',
                dict(type='circular', center=[70,0], radius=20, label='A'),
                dict(type='circular', center=[0,0], radius=20, label='B'),
                dict(type='annular', center=[0,0], radius_inner=80, radius_outer=100, label='ADF'),
                )


    """
    def __init__(self, config: Config | str, *args):

        # config
        if isinstance(config, str):
            self.config = Config.load_from_yaml(config)
        elif isinstance(config, Config):
            self.config = config

        self.parameters = list()
        self.masks = da.empty(
            shape = [0] + list(self.config['simulation','kspace','ROI_shape']),
            dtype = np.float32,
        )

        self.add_detectors(*args)

    def __len__(self) -> int:
        return len(self.parameters)


    def add_detectors(self, *args: Iterable[dict]):
        """Adds detectors to this detector set.

        Parameters
        ----------
        *args : Iterable[dict]
            list of detector prescriptions provided as dictionaries must
            include at least 'type' key and other keys depending on the type
            valid types:
                'circular' see get_mask_circular for details
                'annular' see get_mask_annular for details
                if 'label' not provided it is by default added as 'detector{L}'
                where L is the index of the detector
        """
        valid_types = ['circular', 'annular']

        for arg in args:
            if not isinstance(arg, dict):
                raise Exception(
                    f"Can't work with arg of type {type(arg)}. "
                    "Provide all arguments as dictionaries, e.g.: "
                    "{'label': 'A', 'type':'circular', 'center': [50,50], 'radius': 10 }.")
            if not 'type' in arg.keys():
                raise Exception(f'A type needs to be provided. Valid types are {valid_types}.')
            if arg['type'] == 'circular':
                mask = self.get_mask_circular(**arg)
            elif arg['type'] == 'annular':
                mask = self.get_mask_annular(**arg)
            else:
                raise Exception(f"Mask of type '{type}' is not supported. Valid types are {valid_types}.")

            # if label not provided, generate label
            if not 'label' in arg.keys():
                arg['label'] = f'detector{len(self)}'

            self.parameters.append(arg)
            self.masks = da.concatenate([self.masks, mask[np.newaxis,:,:]], axis=0)


    def get_mask_circular(
            self,
            radius:float,
            center:list[float,float]=None,
            **kwargs, # not used to cope with other not needed parameters in ar
            ) -> np.ndarray:
        """Returns a mask for circular detector as numpy array of ints

        Parameters
        ----------
        center : list or tuple of floats
            center of the detector in mrad, e.g. [0,50]
        radius :
            radius of the detector in mrad, e.g. 10
        **kwargs :
            not used, here for compatibility and ease of use so that
            arbitrary dict can be used as long as it has necessary
            parameters

        Returns
        -------
            detector mask as integers
        """
        Kx, Ky = self.config.crop_arr_qspace_2ROI(
            self.config.get_reciprocal_space_grids_mrad()
        )

        if center is not None:
            center_x, center_y = center
        else:
            center_x, center_y = [0, 0]

        mask = (Kx - center_x) ** 2 + (Ky - center_y) ** 2 <= radius ** 2

        return mask.astype(int)


    def get_mask_annular(
            self,
            radius_inner: float,
            radius_outer: float,
            center: Iterable[float]=None,
            **kwargs,
    ):
        """Returns a mask for annular detector as numpy array of ints

        Parameters
        ----------
        center : list or tuple of floats, default = [0,0]
            center of the detector in mrad, e.g. [0,50]
        radius_inner :
            inner radius of the detector in mrad, e.g. 70
        radius_outer :
            outer radius of the detector in mrad, e.g. 100
        **kwargs :
            not used, here for compatibility and ease of use so that
            arbitrary dict can be used as long as it has necessary
            parameters

        Returns
        -------

        """
        Kx, Ky = self.config.crop_arr_qspace_2ROI(
            self.config.get_reciprocal_space_grids_mrad()
        )

        if center is not None:
            center_x, center_y = center
        else:
            center_x, center_y = [0,0]

        mask_outer = (Kx - center_x) ** 2 + (Ky - center_y) ** 2 <= radius_outer ** 2
        mask_inner = (Kx - center_x) ** 2 + (Ky - center_y) ** 2 >= radius_inner ** 2

        return mask_inner.astype(int) * mask_outer.astype(int)

    def compute(self) -> None:
        """defines how the integration of the signal over the detectors should be done"""
        file_zarray = self.config['storage', 'intensities_zarray']

        data_zarray = da.from_zarr(file_zarray)

        self.estem_images = da.einsum(
            'ixy,...xy->i...',
            self.masks,
            data_zarray,
        )

    def dump_to_zarr(
            self,
            filename:str=None,
            overwrite:bool=False,
            # indices:Iterable[int]=None,
    ) -> None:
        """Dumps eSTEM images to a zarr file.

        Parameters
        ----------
        filename : str, default = datafolder + 'estem.zarrgroup'
            filename for the zarr group in which the images are stored
            labeled by the detectors' respective labels
        overwrite : bool, default = False
            if True, will overwrite the zarr array if it exists

        Future params
        -------------
        indices : Iterable[int], optional
            Indices of detectors which will be dumped.
            if not provided, all results will be dumped
        """

        if filename is None:
            filename = self.config['datafolder'] + 'estem.zarrgroup'

        for image, parameter_set in zip(self.estem_images, self.parameters):
            label = parameter_set['label']

            da.to_zarr(
                image,
                filename,
                component=label,
                overwrite=overwrite,
            )

        # add parameters into attrs of the zarr group
        store = zarr.open(filename)
        store.attrs['detectors'] = self.parameters

        # update config
        try:
            self.config['detectors'][self.label] = self.parameters
        except:
            print('adding detectors to config ... ')
            self.config.config['detectors'] = {self.label: self.parameters}

        lock = FileLock(self.config['config_file'] + '.lock', timeout=3 * 60)
        with lock:
            self.config.dump_to_yaml(self.config['config_file'])


    def work(self, overwrite:bool = False) -> None:
        """The same as detector_set.compute() followed by detector_set.dum_to_zarr(overwrite)

        Parameters
        ----------
        overwrite : bool, default = False
            if True, will overwrite the zarr array if it exists

        """
        self.compute()
        self.dump_to_zarr(overwrite=overwrite)

