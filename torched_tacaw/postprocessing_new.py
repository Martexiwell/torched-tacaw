"""
This module is used to compute the STEM images using detectors

by convention units in reciprocal space are mrad

"""
import zarr
import numpy as np
from filelock import Timeout, FileLock
import dask.array as da

from typing import Iterable


# local imports
from . import tools
from . import core as tc
from .core import Config


class DetectorSet:
    """Class for defining detectors and computing the detected image.

    Parameters
    ----------
    config : Config or str
        Config object or str as the config file address
    *args :

    **kwargs :
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


    def add_detectors(self, *args):
        """

        Parameters
        ----------
        args :

        Returns
        -------

        """
        for arg in args:
            if not isinstance(arg, dict):
                raise Exception(
                    f"Can't work with arg of type {type(arg)}. "
                    "Provide all arguments as dictionaries, e.g.: "
                    "{'label': 'A', 'type':'circular', 'center': [50,50], 'radius': 10 }.")
            if arg['type'] == 'circular':
                mask = self.get_mask_circular(**arg)
            elif arg['type'] == 'annular':
                mask = self.get_mask_annular(**arg)
            else:
                raise Exception(f"mask of type '{type}' is not supported")

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

    def compute(self):
        file_zarray = self.config['storage', 'intensities_zarray']

        data_zarray = da.from_zarr(file_zarray)

        self.estem_images = da.einsum(
            'ixy,...xy->i...',
            self.masks,
            data_zarray,
        )
