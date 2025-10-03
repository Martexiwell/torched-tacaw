"""
This module is used to compute the STEM images using detectors

by convention units in reciprocal space are mrad

"""
import zarr
import numpy as np
from filelock import FileLock
import torch

from typing import Iterable

# import warnings
import logging
import itertools

# local imports
from . import tools
from .core import Config


class DetectorSet:
    """Class for defining detectors and computing the detected image.

    Parameters
    ----------
    config : Config or str
        Config object or str as the config file address
    *args : dict
            detector prescriptions provided as dictionaries must
            include at least 'type' key
    logger : logging.Logger | bool, optional
        if logger object is provided, logging will be performed in it
        if True is provided, default logger object is used as
            logger = logging.getLogger(__name__)

    Examples
    --------
    DetectorSet('config.yaml',
                dict(type='circular', center=[70,0], radius=20, label='A'),
                dict(type='circular', center=[0,0], radius=20, label='B'),
                dict(type='annular', center=[0,0], radius_inner=80, radius_outer=100, label='ADF'),
                )


    """
    def __init__(
            self,
            config: Config | str,
            *args,
            logger: bool | logging.Logger = None,
            device: str | torch.device = None,
    ):
        self.device: str | torch.device | None = device

        # config
        if isinstance(config, str):
            self.config: Config = Config.load_from_yaml(config)
        elif isinstance(config, Config):
            self.config: Config = config

        # logging
        if logger is None:
            def simplelog(*_args, **_kwargs):
                pass
        else:
            if isinstance(logger, logging.Logger):
                pass
            elif logger is True:
                logger = logging.getLogger(__name__)
            else:
                raise Exception(
                    f'logger of type {type(logger)} is not supported. Must be either logger object or bool.')

            def simplelog(*_args, **_kwargs):
                logger.info(*_args, **_kwargs)

        self.simplelog = simplelog

        # main part
        self.parameters = list()
        self.masks = torch.empty(
            [0] + list(self.config['simulation','kspace','ROI_shape']),
            # dtype = 'double',
            device = self.device,
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

            mask = torch.tensor(mask, device=self.device)

            self.parameters.append(arg)
            self.masks = torch.concatenate(
                [self.masks, mask[np.newaxis,:,:]],
                axis=0,
            )


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
        """integrates the intensities over detectors' areas"""

        file_zarray = self.config['storage', 'intensities_zarray']

        self.simplelog(f'opening zarr: {file_zarray} ...')
        data_zarray = zarr.open(file_zarray)

        chunk_size_x, chunk_size_y = self.config['beam', 'scanning', 'batch_shape']
        s0, energy_size, nof_scan_x, nof_scan_y, kx_size, ky_size = data_zarray.shape
        n_detectors = len(self)

        self.simplelog(f"shapes: {chunk_size_x=}, {chunk_size_y=}, {s0=}, {energy_size=}," +
                       f" {nof_scan_x=}, {nof_scan_y=}, {kx_size=}, {ky_size=}, {n_detectors=}"
                       )

        self.simplelog('allocating array for images ...')
        self.estem_images = torch.empty(
            [n_detectors, energy_size, nof_scan_x, nof_scan_y],
            device=self.device
        )

        self.simplelog('looping over all batches in zarr ...')
        for chunk_id_x, chunk_id_y in itertools.product(
                range(nof_scan_x // chunk_size_x),
                range(nof_scan_y // chunk_size_y)
        ):
            chunk = tools.get_nth_cutout_from_array(
                data_zarray,
                [s0, energy_size, chunk_size_x, chunk_size_y, kx_size, ky_size],
                [0, 0, chunk_id_x, chunk_id_y, 0, 0]
            )

            start_x = chunk_id_x * chunk_size_x
            stop_x = (chunk_id_x + 1) * chunk_size_x
            stop_x = stop_x if stop_x < nof_scan_x else nof_scan_x

            start_y = chunk_id_y * chunk_size_y
            stop_y = (chunk_id_y + 1) * chunk_size_y
            stop_y = stop_y if stop_y < nof_scan_y else nof_scan_y

            self.estem_images[:, :, start_x:stop_x, start_y:stop_y] = torch.einsum(
                'ixy,eklxy->iekl',
                self.masks,
                torch.tensor(chunk[0], device=self.device),
            )


        # self.estem_images = da.einsum(
        #     'ixy,...xy->i...',
        #     self.masks,
        #     data_zarray,
        # )

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

        store = zarr.open(filename)


                # add parameters into attrs of the zarr group

        store.attrs['detectors'] = self.parameters

        for image, parameter_set in zip(self.estem_images, self.parameters):
            detector_label = parameter_set['label']

            store[detector_label] = image

            # update config
            try:
                self.config['detectors'][label] = parameter_set
            except:
                self.simplelog('adding detectors to config ... ')
                self.config.config['detectors'] = {label: parameter_set}

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

