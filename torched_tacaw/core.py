#!/usr/bin/env python3
import itertools
import multiprocessing

import logging
import os
import time
import copy
import textwrap
import warnings

import pathlib
import scipy
import scipy.constants as c
import torch

import yaml
import zarr
from zarr.storage import LocalStore

from filelock import Timeout, FileLock

import numpy as np

from ase.io import Trajectory
import ase

import pyms

# from pyms.utils import structure_routines

# internal imports
from . import tools
from .tools import indices_of_cutout_from_array, cutout_from_array
from . import units
from . import coordinates
from . import io




class Config:
    """Config keeps track of the configuration of simulation,
    getting axes, physics and everything else right.

    Parameters
    ----------
    hardload: bool, default=True
        if true config is initialized only be reading the nested dictionary provided
        without interpretation. Otherwise Config is intialized standardly.
    logger: logging.Logger, optional
        if provided, logs progress of config creation
    **kwargs
        other parameters, see bellow for details

    MISC
    name: str, optional
        for user's convenience
    info: str, optional
        for user's convenience
    datafolder: str
        default directory for data storage
    config_file: str, optional
        filename to store config
        by default datafolder+'config.yaml'

    BEAM
    beam_energy_keV: float
        energy of the beam, in keV
    beam_conv_ang_mrad: float
        convergence semiangle of the beam, in mrad
    scanning_mode: {'unitcell', 'box','box_relative', 'parallelogram'}
        sets the scanning mode used to create corresponding scanning grids
        - unitcell (default)
            performs scanning within the unitcell as provided by the strucutre file
        - box
            performs scanning in rectangular box given by additional arguments
            (in cartesian Angstroms):
            - scanning_box_origin: list[float, float]
            - scanning_box_end   OR   scanning_box_size   -  both: list[float, float]
        - box_relative
            same as box but using relative coordinates of the supercell
        - parallelogram
            performs scanning on a grid given by two vectors
    scanning_shape: iterable[int, int]
        (nx, ny) as the shape of scanning grid
        use [1,1] for no scanning (e.g. for planewave calculation)
    scanning_batch_shape: iterable[int, int]
        shape of scanning batch computed in paralel within one calculation
        (!) IF IT DOES NOT DIVIDE scanning_shape, can result in erroneous results!

    (!) Beware of fence-post problem - when setting up scanning area, "the final
        vertex" is not actually used as a scanning point. E.g.
        >   scanning_mode = 'box_relative'
        >   scanning_shape = [5,4]
        >   scanning_box_origin = [0., 0.]
        >   scanning_box_end = [0.5, 0.8]
        results in scanning coordinates of this form
            ┌──┬──┬──┬──┐ .0
            ├──┼──┼──┼──┤ .2
            ├──┼──┼──┼──┤ .4
        y   └──┴──┴──┴──┘ .6
        └─x .0 .1 .2 .3 .4

    SAMPLE
    sample_name: str, optional
        for user's convenience
    sample_supercell_int: list[int,int,int], default=[1,1,1]
        if the whole cell in trajectory is multiple of unitcell, this can be useful
    sample_temperature_K: float
        sample temperature in K, used to calculate prefactor in TACAW
    sample_structure_file: str, optional
        structure file from which dimensions of unitcell are extracted,
        if not provided, last snapshot of trajectory file is used by default.

    TRAJECTORY
    trajectory_file: str
        (ase).traj file where trajectory is stored
    trajectory_timestep_fs: float
        timestep between snapshots in trajectory
    trajectory_chunks_size: int
        number of snapshots in one tacaw-chunk
    trajectory_chunks_skip_init: int
        how many snapshots in the begining of trajectory to skip, useful when dealing
        with trajectory that thermalizes in the begining
    trajectory_chunks_step: int, default=1
        step between consecutive snapshots from trajectory that are actually used
        into tacaw-chunk, deafult is 1 -- everything is used
    trajectory_chunks_nof: int, optional
        max number of tacaw-chunks generated from trajectory, if not provided,
        as many as possible is used
    trajectory_chunks_overlap: float, default=1.
        overlap factor F, if N is len of chunk,
        then next chunk starts at N/F after previous:
        original            ┌─────────────1.5────(3.)─────┐
        ├────────────────────────────┤├────────────1.0────(2.)─────┤├─────────────0.5────────────┤
                  └─────────────3.─────────────┘
                       └─────────────2.─────────────┘
        corresponds to the mean "in how many different chunks does one given
        part of a trajectory end up in"

    SIMULATION - MULTISLICE
    kspace_shape_full: list[float,float]
        full shape of array used for multislice calculation
    kspace_ROI_mode: str, optional, possible {'center', 'minmax_mrad'}
        mode for selection of Region Of Interest (ROI) in kspace for saving
        if not provided, bandwidth-limited shape is used.
        - center
            ROI will be central region of shape
            - kspace_ROI_shape: list[int,int]
        - mimax_mrad
            enables to select arbitrary rectangular selection given by min
            and max coordinates in mrad:
            - kspace_ROI_min: list[float,float]
            - kspace_ROI_max: list[float,float]
    kspace_bandwidth_limiting: list[float,float], default=[2 / 3, 2 / 3]
        bandwidth limiting used to eliminate artefacts from leakage of electron
        wavefunction into it's "reciprocal space copy"
        (due to periodic boundary conditions)
    n_slices:
        n# of slices for multislice
    center_atoms_in_cell: bool, default=False
        if True, centers the atoms within supercell after loading a snapshot

    SIMULATION - TACAW
    frequency_THz_ROI: list[float, float]
        ROI of frequencies that are kept for storage after FFT in tacaw
    window: str | dict
        window settings used for time windowing before FFT in TACAW
        supported:
        - "hann" (default), equivalent to {"type": "hann"}
            hann window, from scipy, sym=True
        - {"type": "tukey", "alpha": <float>}
            tukey window, from scipy, sym=True

    SIMULATION - GENERAL
    device: str | torch.device, default='cpu'
        device on which calculations in torch are performed, cpu by default
    intensities_zarray: str, optional
        enables to change the storage path of tacaw calculation output,
        by default datafolder+'tacaw.zarray'

    MISC
    comments: optional
        for user's convenience, can be whatever, e.g. list of strings for nicely
        formated text or special calculation parameters a dictionary

    """
    yaml_file_structure = """
            CONFIG FILE STRUCTURE =================================
                * things labeled with asterisk are calculated

                datafolder          : str
                config_file         : str
                beam                : dict
                    energy_keV      : float
                    conv_ang_mrad   : float
                    scanning_numbers    : list [nx:int ny:int]
                        -
                        -
                    scanning_batch_shape
                        -
                        -
                sample
                    name            :str
                    supercell_int
                        -
                        -
                        -
                    structure_file :str
                    unitcell
                        - a:float
                        - b:float
                        - c:float
                trajectory
                    file :str
                    chunks
                        size
                        skip_init
                        nof
                        starts
                            -
                            -
                            -
                    timestep_fs
                simulation
                    kspace
                        shape_full
                            -
                            -
                        bandwidth_limiting
                            -
                            -
                        shape_bandwidth_limited
                            -
                            -
                        ROI_mode
                        ROI_shape
                            -
                            -
                    scangrid
                        xaxis_A
                            -
                            -
                            -
                        yaxis_A
                - # min
                            - # max
                        ROI
                            - # min
                            - # max                      -
                            -
                            -
                        nof_points
                    frequency_THz
                        full

                        ROI_indices
                            - # min_id
                            - # max_id
                        ROI_len
                    n_slices
                    device
                computation_batches
                    nof
                    param_list
                        # list of all chunks to be processsed, each line corresponds
                        # to a computation chunk to be preocessed with params to be passed
                        # into the
                        -   trajectory_chunk_id:
                            scanning_batch_coordinates:
                                -
                                -
                    status_list
                        - 2  # done
                        - 1  # in progress
                        - 0  # untouched
                storage
                comments
        """
    def __init__(
            self,
            hardload:bool=False,
            logger:logging.Logger=None,
            **kwargs
    ):
        self.logger = tools.logger_or_null(logger)
        # if logger is None:
        #     self.logger = tools.NullLogger()
        # elif isinstance(logger, logging.Logger):
        #     self.logger = logger
        # else:
        #     raise Exception(f'invalid logger object provided: {logger}')

        if hardload:
            # if this is chosen, config will be just made as a load of the provided dictionary
            # with no preparatory steps
            self.logger.info('hardloading config - no initialization will be performed')

            self.config = kwargs

        else:
            self.logger.info('initializing config')
            # =========== #
            # READ INPUTS #
            # =========== #
            self.logger.info('reading inputs')

            name:str = kwargs.pop("name", 'TACAW')
            info:str = kwargs.pop("info", '')
            datafolder:str = kwargs.pop('datafolder')
            if not datafolder.endswith('/'): # make sure that it is a directory
                datafolder = datafolder + '/'
            config_file = kwargs.pop('config_file', datafolder+'config.yaml')

            scanning_shape = list(kwargs.pop('scanning_shape'))
            beam = {
                "energy_keV": kwargs.pop("beam_energy_keV"),
                "conv_ang_mrad": kwargs.pop("beam_conv_ang_mrad"),
                "scanning":{
                    "mode": kwargs.pop("scanning_mode", 'unitcell'),
                    "shape": scanning_shape,
                    "batch_shape": list(kwargs.pop("scanning_batch_shape",scanning_shape)),
                }
                # "scanning_numbers": kwargs.pop("beam_scanning_numbers"),
                # "scanning_batch_shape": kwargs.pop("beam_scanning_batch_shape"),
            }
            if not (beam["scanning"]["shape"][0] % beam["scanning"]["batch_shape"][0] == 0 and
                    beam["scanning"]["shape"][1] % beam["scanning"]["batch_shape"][1] == 0 ):
                warnings.warn(
                    f"scanning_batch_shape = {beam['scanning']['batch_shape']} does not divide " +
                    f"scanning_shape = {beam['scanning']['shape']} . This can have unpredictable results."
                )

            sample = {
                "name": kwargs.pop("sample_name",'sample'),
                "supercell_int": kwargs.pop("sample_supercell_int", [1,1,1]),
                # "structure_file": kwargs.pop("sample_structure_file"), # TODO: make this optional
                "temperature_K": kwargs.pop("sample_temperature_K"),
                # "unitcell"         : kwargs.pop("sample_unitcell"),
            }
            if "sample_structure_file" in kwargs:
                sample["structure_file"] = kwargs.pop("sample_structure_file")


            trajectory = {
                "file": kwargs.pop("trajectory_file"),
                "timestep_fs": kwargs.pop("trajectory_timestep_fs"),
                "chunks": {
                    "size": kwargs.pop("trajectory_chunks_size"),
                    "skip_init": kwargs.pop("trajectory_chunks_skip_init"),
                    "step": kwargs.pop("trajectory_chunks_step", 1), # currently is not doing anything  =(
                    "nof": kwargs.pop("trajectory_chunks_nof", None),
                    "overlap": kwargs.pop("trajectory_chunks_overlap", 1.),
                }
            }
            simulation = {
                "kspace": {
                    "shape_full": kwargs.pop("kspace_shape_full"),
                    "ROI_mode": kwargs.pop("kspace_ROI_mode", None),
                    "ROI_shape": kwargs.pop("kspace_ROI_shape", None),
                    "bandwidth_limiting": kwargs.pop("kspace_bandwidth_limiting", [2 / 3, 2 / 3]),
                },
                "n_slices": kwargs.pop("n_slices"),
                "frequency_THz": {
                    "ROI": kwargs.pop("frequency_THz_ROI"),
                },
                "window": kwargs.pop("window", "hann"),
                # "subslices" : kwargs.pop("subslices", [1.0]),
                "device": kwargs.pop("device", 'cpu'),
                "center_atoms_in_cell": kwargs.pop("center_atoms_in_cell", False),
            }
            storage = {
                "intensities_zarray": kwargs.pop("intensities_zarray", datafolder+'tacaw.zarray'),
            }
            comments = kwargs.pop("comments", '')


            # ================== #
            # Collect everything #
            # ================== #

            self.config = {
                'name': name,
                'info': info,
                'datafolder': datafolder,
                'config_file': config_file,
                'beam': beam,
                'sample': sample,
                'trajectory': trajectory,
                'simulation': simulation,
                'storage': storage,
                'comments': comments,
            }


            # ===================== #
            # Calculated parameters #
            # ===================== #
            self.logger.info('calculating parameters')

            ## trajectory
            # file type and file_kwargs
            if 'trajectory_file_type' in kwargs:
                trajectory['file_type'] = kwargs['trajectory_file_type']
                trajectory_file_type = trajectory['file_type']
            else:
                trajectory_file_type = None

            if 'trajectory_file_kwargs' in kwargs:
                trajectory['file_kwargs'] = kwargs['trajectory_file_kwargs']
                trajectory_file_kwargs = trajectory['file_kwargs']
            else:
                trajectory_file_kwargs = {}

            self.logger.info('  probing trajectory')
            trajectory_reader = io.TrajctoryReader(trajectory["file"], trajectory_file_type, **trajectory_file_kwargs )
            nof_snapshots_in_traj = len(trajectory_reader)
            trajectory['chunks']['starts'] = \
                [start for start in range(trajectory['chunks']['skip_init'],
                                          nof_snapshots_in_traj - trajectory['chunks']['size']*trajectory['chunks']['step'],
                                          int(trajectory['chunks']['size']*trajectory['chunks']['step'] / trajectory['chunks']['overlap'])
                                          )
                 ]
            if trajectory['chunks']['nof'] is not None:
                trajectory['chunks']['starts'] = trajectory['chunks']['starts'][0:trajectory['chunks']['nof']]
            del nof_snapshots_in_traj

            trajectory['chunks']['nof'] = len(trajectory['chunks']['starts'])

            # effective timestep between used snapshots
            trajectory['timestep_effective_fs'] = trajectory['timestep_fs'] * trajectory['chunks']['step']




            ## sample
            self.logger.info('  probing sample')

            if "structure_file" in sample:
                self.logger.info('    structure file provided')
                atoms = ase.io.read(trajectory["file"])
            else: # if structure file not provided use the first structure from trajectory
                self.logger.info('    structure file NOT provided -> using trajectory file')
                atoms = io.TrajctoryReader(trajectory["file"])[0]
            cell = atoms.get_cell().array
            sample['unitcell'] = [float(cell[i, i]) for i in range(3)]
            del cell, atoms

            # beam
            # ====

            # scanning
            # --------
            # this part prepares the scanning based on which mode is chosen
            self.logger.info('setting up scanning')
            if beam['scanning']['mode'] == 'unitcell':
                beam['scanning']['origin'] = [0,0]
                if "structure_file" in sample:
                    self.logger.info('    structure file provided')
                    atoms = ase.io.read(sample["structure_file"])
                else:
                    self.logger.info('    structure file NOT provided -> using trajectory file')
                    atoms = ase.io.read(trajectory["file"])
                cell = atoms.get_cell().array
                beam['scanning']['basis_vectors'] = [[float(cell[0,0]),float(cell[0,1])],
                                                     [float(cell[1,0]),float(cell[1,1])]]
                del atoms,cell

            elif beam['scanning']['mode'] in [ 'box', 'rectangle' ]:
                beam['scanning']['origin'] = kwargs.pop('scanning_origin')
                box_size = kwargs.pop('scanning_box_size', None)
                box_end = kwargs.pop('scanning_box_end', None)
                if box_size is not None:
                    beam['scanning']['basis_vectors'] = [[box_size[0],0],
                                                         [0,box_size[1]]]
                    beam['scanning']['box_size'] = box_size
                elif box_end is not None:
                    beam['scanning']['basis_vectors'] = [[box_end[0]-beam['scanning']['origin'][0], 0],
                                                         [0, box_end[1]-beam['scanning']['origin'][1]]]
                    beam['scanning']['box_end'] = box_end
                else:
                    raise Exception(f'Either scanning_box_size or scanning_box_end must be defined')
                del  box_size, box_end

            elif beam['scanning']['mode'] == 'box_relative':
                beam['scanning']['origin'] = kwargs.pop('scanning_origin')
                box_size = kwargs.pop('scanning_box_size', None)
                box_end = kwargs.pop('scanning_box_end', None)
                if box_size is not None:
                    beam['scanning']['basis_vectors'] = [[box_size[0],0],
                                                         [0,box_size[1]]]
                    beam['scanning']['box_size'] = box_size
                elif box_end is not None:
                    beam['scanning']['basis_vectors'] = [[box_end[0]-beam['scanning']['origin'][0], 0],
                                                         [0, box_end[1]-beam['scanning']['origin'][1]]]
                    beam['scanning']['box_end'] = box_end
                else:
                    raise Exception(f'Either scanning_box_size or scanning_box_end must be defined')
                del  box_size, box_end
                self.logger.debug(f'{beam=}')
                beam['scanning']['origin'] = [
                    r * a for r, a in zip(
                        beam['scanning']['origin'],
                        sample['unitcell'])
                ]
                beam['scanning']['basis_vectors'] = [
                    (np.array(r) * a).tolist() for r,a in zip(
                        beam['scanning']['basis_vectors'],
                        sample['unitcell']
                    )
                ]


            elif beam['scanning']['mode'] == 'parallelogram':
                beam['scanning']['origin'] = kwargs.pop('scanning_origin')
                beam['scanning']['basis_vectors'] = kwargs.pop('scanning_basis_vectors')

            beam['scanning']['nof_points'] = scanning_shape[0] * scanning_shape[1]
            del scanning_shape

            # currently there is a problem of unknown mechanism with batch shapes
            # which do not divide scanning shape without remainder. The computation
            # runs but at least the last axes are weird
            if beam['scanning']['shape'][0] % beam['scanning']['batch_shape'][0] != 0 or beam['scanning']['shape'][1] % beam['scanning']['batch_shape'][1] != 0:
                warnings.warn(f"scanning batch shape {beam['scanning']['batch_shape']} does not divide"
                              f"the total scanning shape {beam['scanning']['shape']} ! "
                              f"This will probably lead to wrong handling of the edge-batches."
                              )
            ## simulation

            ### kspace
            simulation["kspace"]["shape_bandwidth_limited"] = \
                np.array(
                    (simulation["kspace"]["shape_full"]
                     * np.array(simulation["kspace"]["bandwidth_limiting"])
                     )
                ).astype(int).tolist()

            #### kspace ROI
            if simulation["kspace"]["ROI_mode"] is None:  # No ROI
                simulation["kspace"]["ROI_shape"] = simulation["kspace"]["shape_bandwidth_limited"]

            elif simulation["kspace"]["ROI_mode"] == 'center':
                if simulation["kspace"]["ROI_shape"] is None:
                    simulation["kspace"]["ROI_shape"] = simulation["kspace"]["shape_bandwidth_limited"]
                ROI_shape = simulation["kspace"]["ROI_shape"]

            elif simulation["kspace"]["ROI_mode"] in ['minmax_mrad', 'mrad', 'minmaxmrad']:
                simulation["kspace"]["ROI_min"] = kwargs.pop("kspace_ROI_min")
                simulation["kspace"]["ROI_max"] = kwargs.pop("kspace_ROI_max")

                invax_x_mrad, invax_y_mrad = self.get_reciprocal_space_axes_mrad()
                x_ROI_indices = np.where(
                    (invax_x_mrad >= simulation["kspace"]["ROI_min"][0]) &
                    (invax_x_mrad <= simulation["kspace"]["ROI_max"][0])
                )[0]
                min_index_x = int(x_ROI_indices[0])
                max_index_x = int(x_ROI_indices[-1] + 1)
                del x_ROI_indices

                y_ROI_indices = np.where(
                    (invax_y_mrad >= simulation["kspace"]["ROI_min"][1]) &
                    (invax_y_mrad <= simulation["kspace"]["ROI_max"][1])
                )[0]
                min_index_y = int(y_ROI_indices[0])
                max_index_y = int(y_ROI_indices[-1] + 1)
                del y_ROI_indices

                simulation["kspace"]["ROI_min_indices"] = [min_index_x, min_index_y]
                simulation["kspace"]["ROI_max_indices"] = [max_index_x, max_index_y]
                simulation["kspace"]["ROI_shape"] = [max_index_x - min_index_x, max_index_y - min_index_y]





            ### frequency
            simulation['frequency_THz']['full'] = (np.array([-1, 1]) / 2 / trajectory['timestep_effective_fs'] * 1e3).tolist()

            # frequencies = np.linspace(*simulation['frequency_THz']['full'], trajectory['chunks']['size'])
            frequencies = self.get_frequency_axis_THz()

            indices = np.where(
                (frequencies >= simulation['frequency_THz']['ROI'][0]) & (
                        frequencies <= simulation['frequency_THz']['ROI'][1])
            )[0].tolist()

            simulation['frequency_THz']['ROI_len'] = len(indices)
            simulation['frequency_THz']['ROI_indices'] = [indices[0], indices[-1] + 1]




            ## Prepare computation batches
            ## ---------------------------

            ### scanning:
            # TODO: REFORM
            # scanning_axis_x_A, scanning_axis_y_A = self.scanning_axes_A(
            #     beam['scanning']['shape'],
            #     sample['unitcell']
            # )
            # simulation['scan_grid'] = {
            #     'xaxis_A': scanning_axis_x_A.tolist(),
            #     'yaxis_A': scanning_axis_y_A.tolist(),
            #     'nof_points': len(scanning_axis_x_A) * len(scanning_axis_y_A)
            # }

            ### make batches
            def make_batches():
                param_list = list()
                status_list = list()

                scanning_batches_max_indices = np.ceil(
                    np.array(beam['scanning']['shape']) / np.array(beam['scanning']['batch_shape'])).astype(int)
                iterator = itertools.product(
                    range(trajectory['chunks']['nof']),
                    range(scanning_batches_max_indices[0]),
                    range(scanning_batches_max_indices[1]),
                )
                for params in iterator:
                    param_list.append({
                        "trajectory_chunk_id": params[0],
                        "scanning_batch_coordinates": [params[1], params[2]]
                    })

                    status_list.append(0)

                return param_list, status_list

            ### collect batches

            param_list, status_list = make_batches()

            computation_batches = {
                'nof': len(param_list),
                'param_list': param_list,
                'status_list': status_list
            }

            self.config['computation_batches'] = computation_batches









    def __getitem__(self, keys):
        # if not isinstance(keys, tuple):
        #     raise TypeError("Keys must be provided as a tuple.")

        current = self.config

        if isinstance(keys, tuple):
            for key in keys:
                current = current[key]

        elif isinstance(keys, str):
            if len(keys.split()) > 1:
                return self[keys.split()]
            else:
                current = current[keys]

                # try:
                #     current = current[key]
                # except KeyError:
                #     f"Key '{key}' not found."
        return current

    @classmethod
    def from_params(cls, **kwargs):
        """
        The parameters for this functions should be passed as keyword arguments
        This function creates yaml file that should be formatted as shown below:

        INPUT: =============================================
            name                    : str
            datafolder              : str
            config_file             : str

            beam_energy_keV         : float
            beam_conv_ang_mrad      : float
            beam_scanning_numbers   : list [nx:int ny:int]          # [n_x, n_y] - will generate square grid...
            beam_scanning_batch_shape : list[int,int]                       # nof wavefunctions in batch

            sample_name             : str
            sample_supercell_int    : list[int, int, int]           # The miller-index-like multipliers of unitcell
            sample_structure_file   : str
            # sample_unitcell         : list[float, float, float]   # a,b,c params of orthogonal lattice

            trajectory_file
            trajectory_timestep_fs
            trajectory_chunks_size
            trajectory_chunks_skip_init
            trajectory_chunks_nof
            trajectory_chunks_overlap

            kspace_shape_full
            kspace_bandwidth_limiting
            kspace_ROI_mode
            kspace_ROI_shape        # shape of
            frequency_THz_ROI       # list[float, float]
            subslices               : list[float,]      # optional, ends with 1.0
            n_slices                : int               # number of slices for multislice

            window                  : dict{'name':'tukey','args':...}
            device                  : str = 'cpu' | 'gpu' / 'cuda'
            intensities_zarray
        """

        return cls(**kwargs)

        # first read all params from kwargs

        # name = kwargs.pop("name", 'TACAW')
        #
        # beam = {
        #     "energy_keV": kwargs.pop("beam_energy_keV"),
        #     "conv_ang_mrad": kwargs.pop("beam_conv_ang_mrad"),
        #     "scanning_numbers": kwargs.pop("beam_scanning_numbers"),
        #     "scanning_batch_shape": kwargs.pop("beam_scanning_batch_shape"),
        # }
        # sample = {
        #     "name": kwargs.pop("sample_name"),
        #     "supercell_int": kwargs.pop("sample_supercell_int"),
        #     "structure_file": kwargs.pop("sample_structure_file"),
        #     "temperature_K": kwargs.pop("sample_temperature_K"),
        #     # "unitcell"         : kwargs.pop("sample_unitcell"),
        # }
        # trajectory = {
        #     "file": kwargs.pop("trajectory_file"),
        #     "timestep_fs": kwargs.pop("trajectory_timestep_fs"),
        #     "chunks": {
        #         "size": kwargs.pop("trajectory_chunks_size"),
        #         "skip_init": kwargs.pop("trajectory_chunks_skip_init"),
        #         "nof": kwargs.pop("trajectory_chunks_nof", None),
        #         "overlap": kwargs.pop("trajectory_chunks_overlap", 1),
        #     }
        # }
        # simulation = {
        #     "kspace": {
        #         "shape_full": kwargs.pop("kspace_shape_full"),
        #         "ROI_mode": kwargs.pop("kspace_ROI_mode", None),
        #         "ROI_shape": kwargs.pop("kspace_ROI_shape", None),
        #         "bandwidth_limiting": kwargs.pop("kspace_bandwidth_limiting", [2 / 3, 2 / 3]),
        #     },
        #     "frequency_THz": {
        #         "ROI": kwargs.pop("frequency_THz_ROI"),
        #     },
        #     # "subslices" : kwargs.pop("subslices", [1.0]),
        #     "n_slices": kwargs.pop("n_slices"),
        #     "device": kwargs.pop("device"),
        #     "window": kwargs.pop("window"),
        #     "center_atoms_in_cell": kwargs.pop("center_atoms_in_cell", True),
        # }
        # storage = {
        #     "intensities_zarray": kwargs.pop("intensities_zarray"),
        # }
        # comments = kwargs.pop("comments", '')
        #
        # # Calculated parameters
        #
        # ## beam
        #
        # ## sample
        # atoms = ase.io.read(sample["structure_file"])
        # cell = atoms.get_cell().array
        # sample['unitcell'] = [float(cell[i, i]) for i in range(3)]
        # del cell, atoms
        #
        # ## trajectory
        # nof_snapshots = len(Trajectory(trajectory["file"]))
        # trajectory['chunks']['starts'] = \
        #     [start for start in range(trajectory['chunks']['skip_init'],
        #                               nof_snapshots - trajectory['chunks']['size'],
        #                               int(trajectory['chunks']['size'] / trajectory['chunks']['overlap'])
        #                               )
        #      ]
        # if trajectory['chunks']['nof'] is None:
        #     trajectory['chunks']['nof'] = len(trajectory['chunks']['starts'])
        # else:
        #     trajectory['chunks']['starts'] = trajectory['chunks']['starts'][0:trajectory['chunks']['nof']]
        # del nof_snapshots
        #
        # ## simulation
        #
        # simulation["kspace"]["shape_bandwidth_limited"] = \
        #     np.array(
        #         (simulation["kspace"]["shape_full"]
        #          * np.array(simulation["kspace"]["bandwidth_limiting"])
        #          )
        #     ).astype(int).tolist()
        #
        # # kspace ROI mode
        # if simulation["kspace"]["ROI_mode"] is None:  # No ROI
        #     simulation["kspace"]["ROI_shape"] = simulation["kspace"]["shape_bandwidth_limited"]
        # elif simulation["kspace"]["ROI_mode"] == 'center':
        #     if simulation["kspace"]["ROI_shape"] is None:
        #         simulation["kspace"]["ROI_shape"] = simulation["kspace"]["shape_bandwidth_limited"]
        #     ROI_shape = simulation["kspace"]["ROI_shape"]
        # elif simulation["kspace"]["ROI_mode"] == 'minmax_mrad':
        #     simulation["kspace"]["ROI_min_mrad"] = kwargs.pop("ROI_min_mrad")
        #     simulation["kspace"]["ROI_max_mrad"] = kwargs.pop("ROI_max_mrad")
        #
        #     # invdim_invA = self.get_invsupercell_dims_mrad()
        #
        #     simulation["kspace"]["ROI_min_indices"]
        #     simulation["kspace"]["ROI_max_indices"]
        #
        # simulation['frequency_THz']['full'] = (np.array([-1, 1]) / 2 / trajectory['timestep_fs'] * 1e3).tolist()
        #
        # frequencies = np.linspace(*simulation['frequency_THz']['full'], trajectory['chunks']['size'])
        #
        # indices = np.where(
        #     (frequencies >= simulation['frequency_THz']['ROI'][0]) & (
        #                 frequencies <= simulation['frequency_THz']['ROI'][1])
        # )[0].tolist()
        #
        # simulation['frequency_THz']['ROI_len'] = len(indices)
        # simulation['frequency_THz']['ROI_indices'] = [indices[0], indices[-1] + 1]
        #
        # ## Prepare computation batches
        #
        # ### scanning:
        # scanning_axis_x_A, scanning_axis_y_A = cls.scanning_axes_A(
        #     beam['scanning_numbers'],
        #     sample['unitcell']
        # )
        # simulation['scan_grid'] = {
        #     'xaxis_A': scanning_axis_x_A.tolist(),
        #     'yaxis_A': scanning_axis_y_A.tolist(),
        #     'nof_points': len(scanning_axis_x_A) * len(scanning_axis_y_A)
        # }
        #
        # ### make batches
        # def make_batches():
        #     param_list = list()
        #     status_list = list()
        #
        #     scanning_batches_max_indices = np.ceil( np.array(beam['scanning_numbers']) / np.array(beam["scanning_batch_shape"]) ).astype(int)
        #     iterator = itertools.product(
        #         range(trajectory['chunks']['nof']),
        #         range(scanning_batches_max_indices[0]),
        #         range(scanning_batches_max_indices[1]),
        #     )
        #     for params in iterator:
        #         param_list.append({
        #             "trajectory_chunk_id": params[0],
        #             "scanning_batch_coordinates": [params[1], params[2]]
        #         })
        #
        #         status_list.append(0)
        #
        #     return param_list, status_list
        #
        # ### collect batches
        #
        # param_list, status_list = make_batches()
        #
        # computation_batches = {
        #     'nof': len(param_list),
        #     'param_list': param_list,
        #     'status_list': status_list
        # }
        #
        # # Collect everything
        #
        # config = {
        #     'datafolder': kwargs.pop('datafolder'),
        #     'config_file': kwargs.pop('config_file'),
        #     'beam': beam,
        #     'sample': sample,
        #     'trajectory': trajectory,
        #     'simulation': simulation,
        #     'storage': storage,
        #     'comments': comments,
        #     'computation_batches': computation_batches
        # }

        # return cls(**config)

        # with open(parameters['config_file'], 'w') as f:
        #     yaml.dump(parameters, f)






    def dump_to_yaml(self, file_path=None):
        file_path = file_path or self.config['config_file']
        tools.ensure_valid_path_file(file_path)

        class CustomDumper(yaml.Dumper):
            # Custom yaml Dumper to make short lists flat and nested lists neat
            def represent_list(self, data):
                # Check if the list contains nested lists
                if all(isinstance(i, list) for i in data):
                    return self.represent_sequence('tag:yaml.org,2002:seq', data, default_flow_style=True)
                # Check for short lists
                elif len(data) <= 3:
                    return self.represent_sequence('tag:yaml.org,2002:seq', data, default_flow_style=True)

        with open(file_path, 'w') as file:
            yaml.dump(self.config, file, sort_keys=False, Dumper=CustomDumper)

    @classmethod
    def load_from_yaml(cls, file_path, logger=None):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return cls(**config, hardload=True, logger=logger)



    def create_zarr_array(self):
        """
        Creates the zarr array, if it already exists, it is rewritten!
        """
        config = self.config
        # Define the shape, chunks, and dtype based on your configuration
        shape = (
            1,  # dummy axis
            config['simulation']['frequency_THz']['ROI_len'],
            *config['beam']['scanning']['shape'],
            *config['simulation']['kspace']['ROI_shape']
        )
        chunks = (
            1,  # dummy axis
            config['simulation']['frequency_THz']['ROI_len'],
            *config['beam']['scanning']['batch_shape'],
            *config['simulation']['kspace']['ROI_shape']
        )
        dtype = 'float64'  # or 'complex128' if you need complex numbers

        # Create the Zarr store and array
        zarr_store = LocalStore(config['storage']['intensities_zarray'])
        zarr_array = zarr.zeros(
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            store=zarr_store,
            # synchronizer=zarr.sync.ProcessSynchronizer(zarr_store.path + '.sync'),
            overwrite=True  # Overwrite if the array already exists
        )

        print(f'Zarr array {dtype} created with shape {shape} and chunks {chunks}')


    # DEPRECATED
    # @staticmethod
    # def scanning_axes_A(
    #         scanning_numbers,
    #         unitcell
    # ):
    #     cell_param_x = unitcell[0]
    #     cell_param_y = unitcell[1]
    #     scanning_axes_A = (
    #         np.linspace(0, cell_param_x, scanning_numbers[0]+1)[:-1],
    #         np.linspace(0, cell_param_y, scanning_numbers[1]+1)[:-1]
    #     )
    #     return scanning_axes_A


    def get_scanning_grids_indices(self):
        """wrapper around coordinates.scanning_grids_indices() which uses scanning shape"""
        shape = self['beam','scanning','shape']
        return coordinates.scanning_grids_indices(shape)

    def get_scanning_grids_coordinates_fractional(self):
        """wrapper around coordinates.scanning_grids_coordinates_fractional() which uses scanning shape"""
        shape = self['beam', 'scanning', 'shape']
        return coordinates.scanning_grids_coordinates_fractional(shape)

    def get_scanning_grids_coordinates_cartesian(self, scanning_batch_coordinates:list[int,int] | None = None):
        """wrapper around coordinates.scanning_grids_coordinates_cartesian() using actual config parameters
        Parameters
        ----------
        scanning_batch_coordinates : list[int,int], optional
            if scanning batch coordinates are provided, retruns only cutout from the whole grid

        Returns
        -------
        numpy.ndarray
            (2, nx, ny) array with cartesian coordinates (in Angstrom)
        """
        shape = self['beam', 'scanning', 'shape']
        basis_vectors = self['beam', 'scanning', 'basis_vectors']
        origin = self['beam', 'scanning', 'origin']

        scanning_grids = coordinates.scanning_grids_coordinates_cartesian(shape, basis_vectors, origin)

        if scanning_batch_coordinates is not None:
            scanning_grids = [
                tools.get_nth_cutout_from_array(
                    grid,
                    self.config['beam']['scanning']['batch_shape'],
                    scanning_batch_coordinates
                )
                for grid in scanning_grids
            ]
            scanning_grids = np.array(scanning_grids)

        return scanning_grids

    # DEPRECATED
    # def get_scanning_grids(
    #         self,
    #         scanning_batch_coordinates:list[int,int] | None = None
    # ):
    #     """DEPRECATED: returns an array in shape (2, nx, ny) of meshgrids"""
    #     xaxis = self.config['simulation']['scan_grid']['xaxis_A']
    #     yaxis = self.config['simulation']['scan_grid']['yaxis_A']
    #
    #
    #     scanning_grids = np.meshgrid(xaxis, yaxis,indexing='ij' )
    #
    #
    #     if scanning_batch_coordinates is not None:
    #         scanning_grids = tuple((tools.get_nth_cutout_from_array(
    #             grid,
    #             self.config['beam']['scanning_batch_shape'],
    #             scanning_batch_coordinates
    #         ) for grid in scanning_grids ))
    #
    #
    #     return np.array(scanning_grids)


    # ====== supercell reciprocal dimensions and grids ====== #

    def get_supercell_dims_A(self):
        return np.array(self['sample', 'unitcell']) * np.array(
        self['sample', 'supercell_int'])


    def get_realpixel_dims_A(self):
        supercell_dims_A = self.get_supercell_dims_A()
        return supercell_dims_A[:2] / np.array(self['simulation', 'kspace', 'shape_full'])


    def get_reciprocal_supercell_dims_invA(self):
        'returns dimensions (x,y) of inverse supercell in 1/A'
        return 1 / self.get_realpixel_dims_A()

    def get_reciprocal_supercell_dims_mrad(self):
        return self.get_reciprocal_supercell_dims_invA() / self.get_wavenumber_invA() * 1e3

    def get_reciprocal_space_axes_invA(self) -> tuple[np.ndarray, np.ndarray]:

        d_x, d_y = self.get_realpixel_dims_A()
        x_axis = np.fft.fftshift(np.fft.fftfreq(
            self['simulation', 'kspace', 'shape_full', 0],
            d_x
        ))
        y_axis = np.fft.fftshift(np.fft.fftfreq(
            self['simulation', 'kspace', 'shape_full', 1],
            d_y
        ))
        return x_axis, y_axis

    def get_reciprocal_space_axes_mrad(self) -> tuple[np.ndarray, np.ndarray]:
        axes = self.get_reciprocal_space_axes_invA()
        out = []
        for axis in axes:
            out.append( axis / self.get_wavenumber_invA() * 1e3 )
        return tuple(out)

    def get_reciprocal_space_grids_invA(self, ROI=False):
        # invsupercell_dims_invA = self.get_reciprocal_supercell_dims_invA()
        # grids = np.array(np.meshgrid(
        #     np.linspace(-invsupercell_dims_invA[0] / 2, invsupercell_dims_invA[0] / 2,
        #             self['simulation', 'kspace', 'shape_full', 0]),
        # np.linspace(-invsupercell_dims_invA[1] / 2, invsupercell_dims_invA[1] / 2,
        #             self['simulation', 'kspace', 'shape_full', 1]),
        # ))
        grids = np.array(np.meshgrid(*self.get_reciprocal_space_axes_invA(), indexing='ij'))
        if ROI:
            grids = self.crop_arr_qspace_2ROI(grids)
        return grids

    def get_reciprocal_space_grids_mrad(self, ROI=False):
        grids = self.get_reciprocal_space_grids_invA(ROI=ROI) / self.get_wavenumber_invA() * 1e3
        return grids




    def get_frequency_axis_THz(self, ROI=False):
        """Returns axis of frequencies in THz

        Parameters
        ----------
        ROI : bool, default False
            if True returns only the ROI part of full frequency axis,
            it then corresponds directly to the results' energy axis

        Returns
        -------
        np.ndarray
        """
        freqs = np.fft.fftfreq(self['trajectory','chunks','size'], self['trajectory','timestep_effective_fs'] / 1e3)
        freqs = np.fft.fftshift(freqs)

        if ROI:
            freqs = self.cropROI_arr_freq(freqs)

        return freqs

    def get_energy_axis_meV(self, ROI=False):
        """Returns axis of energies in meV

        Parameters
        ----------
        ROI : bool, default False
            if True returns only the ROI part of full energy axis,
            it then corresponds directly to the results' energy axis

        Returns
        -------
        np.ndarray
        """
        freqs = self.get_frequency_axis_THz(ROI=ROI)
        energies = units.convert_THz2meV(freqs)
        return energies

    def get_wavenumber_invA(self):
        voltage = self['beam','energy_keV'] * 1e3
        lwn = 1 / c.h * (2 * c.m_e * c.e * voltage *
                            (1 + (c.e * voltage) / (2 * c.m_e * c.c ** 2))
                            ) ** 0.5
        return lwn * 1e-10  # convert to inv Angstroms


    def crop_arr_qspace_2ROI(self, arr, from_shape:str='full'):
        """Crops input array and returns array cropped to ROI in qspace according to config
        crops last two axes
        """
        start = copy.copy(self['simulation', 'kspace', 'ROI_min_indices'])
        stop = copy.copy(self['simulation', 'kspace', 'ROI_max_indices'])
        if from_shape == 'full':
            pass
        elif from_shape in ['bandwidth_limited', 'bwl', 'limited']:
            for i in range(2):
                start[i] = start[i] - np.ceil((self['simulation', 'kspace', "shape_full",i] - self['simulation', 'kspace', "shape_bandwidth_limited",i])/2).astype(int)
                start[i] = start[i] if start[i] > 0 else 0
                stop[i] = stop[i] - np.ceil((self['simulation', 'kspace', "shape_full",i] - self['simulation', 'kspace', "shape_bandwidth_limited",i])/2).astype(int)
                stop[i] = stop[i] if stop[i] < self['simulation', 'kspace', "shape_bandwidth_limited",i] else self['simulation', 'kspace', "shape_bandwidth_limited",i]

        return arr[..., start[0]:stop[0], start[1]:stop[1]]

        # if from_shape == 'full':
        #     shape_tot_x, shape_tot_y = self['simulation', 'kspace', 'shape_full']
        # elif from_shape in ['bandwidth_limited', 'bwl', 'limited']:
        #     shape_tot_x, shape_tot_y = self['simulation', 'kspace', 'shape_bandwidth_limited']
        # else:
        #     raise ValueError(f'from_shape "{from_shape}" not recognized')
        #
        # shape_ROI_x, shape_ROI_y = self['simulation', 'kspace', 'ROI_shape']
        #
        # if self['simulation', 'kspace', 'ROI_mode'] in [None,"center"]:
        #     return arr[...,
        #            (shape_tot_x - shape_ROI_x) // 2:(shape_tot_x + shape_ROI_x) // 2,
        #            (shape_tot_y - shape_ROI_y) // 2:(shape_tot_y + shape_ROI_y) // 2
        #            ]


    def cropROI_arr_freq(self, arr):
        """Crops input array and returns array cropped to ROI in freq according to config

        arr: (self['chunks','size'])
        """
        start, stop = self['simulation','frequency_THz','ROI_indices']
        return arr[start:stop,...]

    def print_progress(self):
        status_list = self['computation_batches', 'status_list']
        total = len(status_list)
        finished = status_list.count(2)
        in_progress = status_list.count(1)
        untouched = status_list.count(0)

        message = textwrap.dedent(f"""
        CURRENT STATUS OF {self['datafolder']} :
        out of total {total} batches, there is
        finished    : {finished:4} ~ {finished/total*100:3.0f} %
        in progress : {in_progress:4} ~ {in_progress/total*100:3.0f} %
        untouched   : {untouched:4} ~ {untouched/total*100:3.0f} %
        """)

        print(message)

    @staticmethod
    def convert_config_file_for_different_machine(
            config_file:str,
            local_project_directory: str,
            remote_project_directory: str,
    ):
        """Opens the confing_file and replaces all occurrences
        of local_project_directory by remote_project_directory

        DISCLAIMER: potentially buggy?

        Parameters
        ----------
        config_file: str
            configuration file to be converted
        local_project_directory : str
            directory string to be replaced by
        remote_project_directory : str
        """
        with open(config_file) as f:
            text: str = f.read()

        text = text.replace(local_project_directory, remote_project_directory)

        with open(config_file, 'w') as f:
            f.write(text)

        print(
            f'Converted : {config_file}',
            f'     from : {local_project_directory}',
            f'       to : {remote_project_directory}',
            sep='\n'
            )


class Calculator:
    """Calculator object used to perform calculation of one computation batch

    Parameters
    ----------
    config: Config or str
        Connfig object to be used OR configuration file to be converted as str
    batch_id: int
        id of batch to be computed
    logger: logging.Logger, optional
        used for logging
    device: str | torch.device, optional
        overwrites device set up by config
        if not provided, defaults to device from config



    Attributes
    ----------
    batch_id: int
        id of batch to be computed
    batch_params: dict
        direct acces to batch-specifig config params
    logger: looging.Logger | tools.NullLogger
    zarr_store: zarr.Store
    zarr_array: zarr.Array

    Methods
    -------
    run()
        Perform calculation for give batch

    Examples
    --------
    import torched_tacaw as tt

    config = './config.yaml'
    calculator = tt.Calculator(config, 0)

    """
    def __init__(
            self,
            config,
            batch_id,
            logger=None,
            device=None,
    ):
        self.logger = tools.logger_or_null(logger)

        if isinstance(config, Config):
            self.config = config
        else:
            self.config = Config.load_from_yaml(config, logger=logger)

        self.batch_id  = batch_id
        self.batch_params = self.config['computation_batches','param_list',self.batch_id]

        self.zarr_store = LocalStore(self.config['storage']['intensities_zarray'])
        self.zarr_array = zarr.open(
            self.zarr_store, mode='a',
            # synchronizer=zarr.sync.ProcessSynchronizer(self.zarr_store.path + '.sync')
        )

        if device is not None:
            self.device = device
            if self.logger is not None:
                self.logger.info(f'calculator initialized with device: {self.device}')
        else:
            self.device = self.config['simulation']['device']
            if self.logger is not None:
                self.logger.info(f'calculator initialized with default device: {self.device}')


        self.simplelog(f'calculator will compute batch {self.batch_id:03}: {self.config["computation_batches","param_list",self.batch_id]}')


    def work(self) -> None:
        """Perform calculation for give batch"""
        self.make_flat_init_wavefunctions()
        self.allocate_final_wavefunctions()
        self.load_trajectory()
        self.perform_multislice()
        self.perform_tacaw()
        self.update_zarr_array()


    def simplelog(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.info(*args, **kwargs)
        else:
            pass

    def simplelog_debug(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.debug(*args, **kwargs)
        else:
            pass

    # NOT NEEDED?
    def get_scanning_grids_A(self):
        """returns an array in shape (2, nx, ny) of meshgrids"""
        # return self.config.get_scanning_grids(self.batch_params['scanning_batch_coordinates'])
        return self.config.get_scanning_grids_coordinates_cartesian(self.batch_params['scanning_batch_coordinates'])

    def make_flat_init_wavefunctions(self):
        scanning_grids_A = self.get_scanning_grids_A()
        self.simplelog(f'scanning_grids_A.shape: {scanning_grids_A.shape}')

        # ----------------
        # supercell reciprocal dimensions and grids for later use
        # DONE: this section shall be replaced by corresponding getters from config
        # supercell_dims_A = np.array(self.config['sample','unitcell']) * np.array(self.config['sample','supercell_int'])
        # realpixel_dims_A = supercell_dims_A[:2] / np.array(self.config['simulation','kspace','shape_full'])
        # invsupercell_dims_invA = 1 / realpixel_dims_A
        # reciprocal_space_grids_invA = np.array(np.meshgrid(
        #     np.linspace(-invsupercell_dims_invA[0] / 2, invsupercell_dims_invA[0] / 2, self.config['simulation','kspace','shape_full',0]),
        #     np.linspace(-invsupercell_dims_invA[1] / 2, invsupercell_dims_invA[1] / 2, self.config['simulation','kspace','shape_full',1]),
        # ))
        # ----------------

        supercell_dims_A = self.config.get_supercell_dims_A()
        reciprocal_space_grids_invA = self.config.get_reciprocal_space_grids_invA()

        fourier_shifters = np.exp(-2j * np.pi * np.einsum('amn,akl->mnkl',
                                                          scanning_grids_A,
                                                          reciprocal_space_grids_invA
                                                          )
                                  )
        old_shape = fourier_shifters.shape
        fourier_shifters_flat = np.reshape(
            fourier_shifters,
            [old_shape[0]*old_shape[1],old_shape[2],old_shape[3]]
        )

        if self.logger is not None:
            self.logger.info(f'preparing initial wavefunctions')
        base_init_wave = pyms.focused_probe(
            gridshape   = self.config['simulation','kspace','shape_full'],
            rsize       = supercell_dims_A[:2],
            eV          = self.config['beam', 'energy_keV']*1e3,
            app         = self.config['beam', 'conv_ang_mrad'],
            # beam_tilt=[0, 0],
            # aperture_shift=[0, 0],
            # tilt_units='mrad',
            # df=0,
            # aberrations=[],
            # q=None,
            # app_units='mrad',
            qspace=True,  # returns wf in qspace to be fftshifted
        )
        base_init_wave = np.fft.fftshift(base_init_wave, axes=(0, 1))

        self.init_waves_flat = np.einsum(
            'skl,kl->skl',
            fourier_shifters_flat,
            base_init_wave
        )

        self.init_waves_flat = np.fft.ifftshift(self.init_waves_flat, axes=(-2, -1))

        # init_waves_orig_shape = init_waves.shape

        # self.init_waves_flat = np.reshape(
        #     init_waves,
        #     [init_waves_orig_shape[0] * init_waves_orig_shape[1],
        #      init_waves_orig_shape[2],
        #      init_waves_orig_shape[3]]
        # )

        if self.logger:
            self.logger.info(f'flat initial waves created: init_waves_flat.shape = {self.init_waves_flat.shape}')


        # # DEBUG - dump final wavefunctions to disk
        # debugfile = self.config['datafolder'] + f'debug/init_waves_flat{self.batch_id:03}.npy'
        # tools.ensure_valid_path_file(debugfile)
        # np.save(debugfile, self.init_waves_flat)
        # self.simplelog(f'{debugfile} saved')


    def load_trajectory(self):
        """Loads trajectory from trajectory file into self.trajectory
        as a subscriptable object returning atoms when accessed by index.

        If file format is not provided, it will guess the file format from the file extension

        Examples
        --------
        self.trajectory[0] -> ase.Atoms
        """
        # trajectory_file = self.config['trajectory', 'file']
        #
        # if 'file_type' in self.config['trajectory']:
        #     trajectory_file_type = self.config['trajectory']['file_type']
        # else:
        #     # guess file_type based on the file extension
        #     # trajectory_file_type = 'traj'
        #     trajectory_file_type = os.path.splitext(trajectory_file)[-1]
        #
        # if 'file_kwargs' in self.config['trajectory']:
        #     trajectory_file_kwargs = self.config['trajectory']['kwargs']
        # else:
        #     trajectory_file_kwargs = {}
        #
        #
        # if trajectory_file_type in ['traj', '.traj', 'ase']: # ASE .traj file
        #     self.trajectory = Trajectory(trajectory_file)
        #
        # elif trajectory_file_type in ['lammps', '.lammps']:
        #     self.trajectory = io.LammpsTrajectoryReader(trajectory_file, **trajectory_file_kwargs)

        trajectory_file = self.config['trajectory', 'file']

        if 'file_type' in self.config['trajectory']:
            trajectory_file_type = self.config['trajectory']['file_type']
        else:
            trajectory_file_type = None

        if 'file_kwargs' in self.config['trajectory']:
            trajectory_file_kwargs = self.config['trajectory']['kwargs']
        else:
            trajectory_file_kwargs = {}

        self.trajectory = io.TrajctoryReader(trajectory_file, trajectory_file_type, **trajectory_file_kwargs)


    def allocate_final_wavefunctions(self):
        shape = [
            1,  # dummy axis to be used maybe in furture
            self.config['trajectory', 'chunks', 'size'],
            self.init_waves_flat.shape[0],
            #self.config['beam', 'scanning_batch_shape',0] * self.config['beam', 'scanning_batch_shape',1], # flattened
            *self.config['simulation', 'kspace', 'ROI_shape'],
        ]

        self.final_wavefunctions = torch.empty(
            shape,
            dtype = torch.complex128,
            device = self.device
        )

        if self.logger is not None:
            logger = self.logger.getChild('allocate_final_wavefunctions()')
            logger.info(f'final wavefunctions allocated in shape {shape}')


    def perform_multislice(self):
        logger = self.logger.getChild('multislice') if self.logger is not None else None
        if logger is not None:
            logger.info('preparing to perform multislice...')

        # --------- Initialize crystal structure ---------- #

        base_structure = self.trajectory[0]
        center_atoms_in_cell = self.config['simulation','center_atoms_in_cell']

        natoms = len(base_structure)
        # atomlist = np.concatenate(
        #     [base_structure.cell.scaled_positions(base_structure.positions),
        #      base_structure.numbers.reshape(natoms, 1)],
        #     axis=1
        # )

        # crystal = pyms.structure(
        #     atoms.cell.diagonal(),
        #     atomlist,
        #     np.zeros(natoms),
        #     np.ones(natoms)
        # )

        # center atoms in the cell
        if center_atoms_in_cell:
            if logger is not None:
                logger.info('centering atoms in cell ...')
            base_structure_centered = base_structure.copy()
            base_structure_centered.center()
            shift_vector = (base_structure_centered.get_center_of_mass()
                            - base_structure.get_center_of_mass()
                                )
            del base_structure_centered

        chunk_start = self.config['trajectory', 'chunks', 'starts', self.batch_params['trajectory_chunk_id']]

        # del atomlist


        # --------- perform multislice for every snapshot in chunk ---------- #

        subslices = np.linspace(1.0 / self.config['simulation','n_slices'], 1.0, self.config['simulation','n_slices'])
        self.simplelog_debug(f'subslices: {subslices}')

        if logger is not None:
            logger.info('performing multislice on each snapshot in chunk...')
        for i in range(self.config['trajectory', 'chunks', 'size']):
            snapshot_index = chunk_start+i*self.config['trajectory','chunks','step']
            if logger is not None:
                logger.info(f'├─ begining multislice of snapshot {i} ({snapshot_index} in .trj)')
            atoms = self.trajectory[snapshot_index]

            if center_atoms_in_cell:
                atoms.translate(shift_vector)
                self.simplelog_debug('│   ├─ atoms shifted to center')

            atomlist = np.concatenate(
                [atoms.cell.scaled_positions(atoms.positions),
                 atoms.numbers.reshape(natoms, 1)],
                axis=1
            )
            crystal = pyms.structure(
                atoms.cell.diagonal(),
                atomlist,
                np.zeros(natoms),
                np.ones(natoms)
            )

            # multislice precursors: transmission fctns & propagator
            P, T = pyms.multislice_precursor(
                crystal,
                self.config['simulation', 'kspace', 'shape_full'],
                self.config['beam','energy_keV'] * 1e3,
                # subslices   = self.config['simulation','subslices'],
                subslices   = subslices ,
                nT          = 1,
                device      = self.device,
                showProgress=False,
                displacements=False,
                fractional_occupancy=False,
                band_width_limiting=self.config['simulation', 'kspace', 'bandwidth_limiting']
            )

            if logger is not None:
                logger.debug(f'│   ├─ ms precursor finished')

            fin_wave_temporary = pyms.multislice(
                self.init_waves_flat,
                self.config['simulation','n_slices'],
                P, T,
                device_type=self.device,
                return_numpy=False,
                qspace_in=True,
                qspace_out=True,
                subslicing=True
            )
            fin_wave_temporary = torch.fft.fftshift(fin_wave_temporary, dim = (-2, -1))

            if logger is not None:
                logger.debug(f'│   ├─ multislice finished --> cutting kspace and reshaping...')

            # TODO: add "crop_mode" to config in format "qe" | None
            # slice out ROI in kspace
            # currently this ROI is in the center of kspace implicitly
            _, shape_tot_x, shape_tot_y = fin_wave_temporary.shape
            _, _, _, shape_ROI_x, shape_ROI_y = self.final_wavefunctions.shape
            fin_wave_temporary = self.config.crop_arr_qspace_2ROI(fin_wave_temporary, 'bwl')
            # fin_wave_temporary = fin_wave_temporary[
            #                       :,
            #                       (shape_tot_x - shape_ROI_x) // 2:(shape_tot_x + shape_ROI_x) // 2,
            #                       (shape_tot_y - shape_ROI_y) // 2:(shape_tot_y + shape_ROI_y) // 2,
            #                      ]

            self.simplelog_debug( f'│   └─ part of shape {fin_wave_temporary.shape} was cropped')

            self.final_wavefunctions[0,i, :, :, :] = fin_wave_temporary

            # # DEBUG - dump final wavefunctions to disk
            # debugfile = self.config['datafolder'] + f'debug/snapshot_fin_wavefun_{self.batch_id}_{i}.npy'
            # tools.ensure_valid_path_file(debugfile)
            # np.save(debugfile, fin_wave_temporary.cpu())
            # self.simplelog(f'{debugfile} saved')

            del atoms, atomlist, crystal, P, T, fin_wave_temporary



        del chunk_start, base_structure, natoms, subslices
        if center_atoms_in_cell:
            del shift_vector

        # # DEBUG - dump final wavefunctions to disk
        # debugfile = self.config['datafolder'] + f'debug/final_wavefunctions{self.batch_id}.npy'
        # tools.ensure_valid_path_file(debugfile)
        # np.save(debugfile, self.final_wavefunctions.cpu())
        # self.simplelog(f'{debugfile} saved')


    def perform_tacaw(self):
        logger = self.logger

        def get_window():
            # TODO: move this to Config
            window_config = self.config['simulation', 'window']
            if isinstance(window_config, str):
                window_config = dict(type=window_config)

            if isinstance(window_config, dict): # backwards compatibility with "name" instead of type
                if 'name' in window_config:
                    window_config['type'] = window_config['name']

            match window_config['type']:
                case 'hann':
                    window = scipy.signal.windows.tukey(self.config['trajectory','chunks','size'],1, sym=True)
                    self.logger.info('using hann window')
                case 'tukey':
                    try:
                        alpha = window_config['alpha']
                    except KeyError:
                        raise Exception('parameter alpha is missing for tukey window')
                    window = scipy.signal.windows.tukey(self.config['trajectory', 'chunks', 'size'], alpha, sym=True)
                    self.logger.info(f'using tukey window with alpha={alpha}')
                case _:
                    raise Exception(f"Unknown window type: {window_config['type']}")

            return window

        if logger is not None:
            logger.info('performing tacaw (windowing, FFT, ROI, modsquared)...')
        # windowing
        self.final_wavefunctions = torch.einsum(
            'ae...,e->ae...',
            self.final_wavefunctions,
            torch.tensor(get_window(), dtype=torch.complex128, device=self.device)
        )
        # FFT, FFTshift
        self.final_wavefunctions = torch.fft.fftshift(
            torch.fft.fft(
                self.final_wavefunctions,
                dim=1
            ),
            dim =(1,)
        )

        # reduce energy axis to ROI
        start, stop = self.config['simulation','frequency_THz','ROI_indices']
        self.final_wavefunctions = self.final_wavefunctions[:, start:stop, :, :, :]
        del start, stop

        # squared modulus
        self.intensity_freq = self.final_wavefunctions
        del self.final_wavefunctions
        self.intensity_freq = torch.abs(self.intensity_freq)**2

        # normalize ( sum I = trajlen * trajlen * Nx_red * Ny_red )
        if logger is not None:
            logger.info(f'renormalizing intensities:')
            logger.info(f'  from sum:          {torch.sum(self.intensity_freq)}')
            logger.info(f'  with shape:        {self.intensity_freq.shape}')
        self.intensity_freq = self.intensity_freq / (
                    self.config['trajectory','chunks','size'] ** 2
                    * self.config['simulation','kspace','shape_full',0]
                    * self.config['simulation','kspace','shape_full',1]
        ) * 2 * np.pi
        if logger:
            logger.info(f'  --> fin sum:       {torch.sum(self.intensity_freq)}')
            logger.info(f'  nof scanpoints in batch: {self.config["beam","scanning","batch_shape",0] * self.config["beam","scanning","batch_shape",1]}')
            logger.info(f'  totprob beam (<1): {torch.sum(self.intensity_freq) / (self.config["beam","scanning","batch_shape",0] * self.config["beam","scanning","batch_shape",1])}')


        # multiply by prefactor
        # TODO: change this to use config.get_energy_axis_ROI
        full_energy_axis_THz = torch.linspace(
            *self.config['simulation','frequency_THz', 'full'],
            self.config['trajectory','chunks','size'],
            device=self.device
        )
        ROI_energy_axis_THz = full_energy_axis_THz[self.config['simulation','frequency_THz', 'ROI_indices',0]:self.config['simulation','frequency_THz', 'ROI_indices',1]]
        del full_energy_axis_THz
        ROI_energy_axis_meV = units.convert_THz2meV(ROI_energy_axis_THz)
        del ROI_energy_axis_THz


        # boltzmann_term = np.exp( - energy_axis_meV * c.milli * c.eV  / ( c.Boltzmann * temperature_K ) )
        betaE = ROI_energy_axis_meV * c.milli * c.eV / (c.Boltzmann * self.config['sample', 'temperature_K'])

        # prefactor = betaE / (1 - torch.exp(-betaE))
        def calculate_prefactor(x):
            # Use a small threshold to check for values close to zero
            # if not problem with divergence around zero
            eps = 1e-6
            near_zero = torch.abs(x) < eps
            regular = x / (1 - torch.exp(-x))
            return torch.where(
                near_zero,
                torch.tensor(1.0, dtype=x.dtype, device=x.device),
                regular
            )
        prefactor = calculate_prefactor(betaE)

        self.tacaw = torch.einsum('e,de...->de...', prefactor, self.intensity_freq)
        del self.intensity_freq

        # if logger:
        #     logger.info(f'betaE sanity-check:')
        #     logger.info(f'                sum: {torch.sum(betaE) }')
        #     logger.info(f'                min: {torch.min(betaE)}')
        #     logger.info(f'             minabs: {torch.min(torch.abs(betaE))}')
        #     logger.info(f'                max: {torch.max(betaE)}')
        #
        # if logger:
        #     logger.info(f'prefactor sanity-check:')
        #     logger.info(f'                sum: {torch.sum(prefactor) }')
        #     logger.info(f'                min: {torch.min(prefactor)}')
        #     logger.info(f'                max: {torch.max(prefactor)}')

        if logger:
            logger.info(f'tacaw normalization sanity-check:')
            logger.info(f'       sum of tacaw: {torch.sum(self.tacaw) }')
            logger.info(f'  totprob beam (<1): {torch.sum(self.tacaw) / self.tacaw.shape[-3]}')  # TODO: REFORM


        if logger is not None:
            logger.info('tacaw calculation done.')


    def update_zarr_array(self):
        if self.logger is not None:
            self.logger.info(f'updating zarr array...')

        # transfer tacaw to cpu
        addend = self.tacaw.cpu().numpy()

        current_scanning_shape = self.get_scanning_grids_A().shape[1:]

        newshape = [1,
            self.config['simulation','frequency_THz','ROI_len'],
            *current_scanning_shape,
            # self.config['beam', 'scanning_batch_shape', 0],
            # self.config['beam', 'scanning_batch_shape', 1],
            self.config['simulation','kspace','ROI_shape',0],
            self.config['simulation','kspace','ROI_shape',1]
            ]

        self.simplelog(f'batch of shape {addend.shape} to reshaped to {newshape}')

        # unflatten tacaw in scanning axes
        addend = addend.reshape(newshape)
        # divide tacaw by number of chunks for averaging
        addend = addend / self.config['trajectory','chunks', 'nof']
        self.simplelog(f'batch of shape {addend.shape} will be dumped to the disk')

        # # DEBUG - dump addend to disk
        # debugfile = self.config['datafolder']+'debug/addend.npy'
        # tools.ensure_valid_path_file(debugfile)
        # np.save(debugfile, addend)
        # self.simplelog(f'{debugfile} saved')


        # Lock the chunk, read, update, and write back
        lock = FileLock(self.config['storage', 'intensities_zarray']
                        + f".{self.batch_params['scanning_batch_coordinates'][0]}-{self.batch_params['scanning_batch_coordinates'][1]}.lock",
                        timeout=15*60)
        with lock:
            # print(f'gammacama, {self.config['beam', 'scanning_numbers']}, {self.config['beam', 'scanning_batch_shape']}, {self.batch_params['scanning_batch_coordinates']}')
            scanning_subspace_start, scanning_subspace_stop = indices_of_cutout_from_array(
                self.config['beam', 'scanning','shape'],
                self.config['beam', 'scanning','batch_shape'],
                self.batch_params['scanning_batch_coordinates']
            )

            # print(scanning_subspace_start, scanning_subspace_stop)

            self.simplelog(f'part of scanning points between {scanning_subspace_start} and {scanning_subspace_stop} will be updated')
            chunk = self.zarr_array[
                :,  # redundant
                :,  # energy
                scanning_subspace_start[0]:scanning_subspace_stop[0],
                scanning_subspace_start[1]:scanning_subspace_stop[1],
                :,
                :
            ]
            if chunk is not None:
                if self.logger is not None:
                    self.logger.info(f'chunk.shape = {chunk.shape}')
                    self.logger.info(f'tacaw.shape = {self.tacaw.shape}')
                chunk += addend
            else:
                chunk = addend

            self.zarr_array[
                :,  # redundant
                :,  # energy
                scanning_subspace_start[0]:scanning_subspace_stop[0],
                scanning_subspace_start[1]:scanning_subspace_stop[1],
                :,
                :
            ] = chunk
        if self.logger:
            self.logger.info(f'Updated Zarr array at {self.batch_params}')


class Master:
    def __init__(
            self,
            config_file,
            logger=None,
            device=None,
    ):
        self.config_file        = config_file
        self.logger             = logger

        if device is not None:
            self.device = device
            if self.logger is not None:
                self.logger.info(f'calculator initialized with device: {self.device}')
        else:
            self.device = self.config['simulation']['device']
            if self.logger is not None:
                self.logger.info(f'calculator initialized with default device: {self.device}')

        self.batches_left = True

    def simplelog(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.info(*args, **kwargs)
        else:
            pass

    def initialize_batch(self):
        if self.logger is not None:
            self.logger.info(f'='*60)
            self.logger.info(f'Initializing new batch:')
        # Lock the chunk, read, update, and write back
        lock = FileLock(self.config_file + '.lock', timeout=15*60)
        with lock:
            config = Config.load_from_yaml(self.config_file)

            status_list = config['computation_batches','status_list']
            try:
                self.batch_id = status_list.index(0)
                if self.logger is not None:
                    self.logger.info(f'batch_id: {self.batch_id} initialized')
            except ValueError:
                if self.logger is not None:
                    self.logger.info(f'no unstarted batch found')
                return False        # returns false if there is no unstarted batch to be found

            config['computation_batches']['status_list'][self.batch_id] = 1

            config.dump_to_yaml(self.config_file)

            return True  # returns True if there is unstarted batch

    def process_batch(self):
        calculator = Calculator(
            self.config_file,
            self.batch_id,
            logger=self.logger,
            device=self.device
        )

        calculator.work()

    def finish_batch(self):
        lock = FileLock(self.config_file + '.lock', timeout=15 * 60)
        with lock:
            config = Config.load_from_yaml(self.config_file)

            config['computation_batches']['status_list'][self.batch_id] = 2  # label as finished

            config.dump_to_yaml(self.config_file)



    def run(self):
        if self.logger is not None:
            self.logger.info(f' ●   M A S T E R   S T A R T E D   ● ')

        while self.batches_left:
            self.batches_left = self.initialize_batch()
            if not self.batches_left:
                break

            self.process_batch()
            self.finish_batch()

        if self.logger is not None:
            self.logger.info(f' ●   M A S T E R   F I N I S H E D   ● ')


    def restart_unfinished(self):
        "restarts unfinished calculation"

        if self.logger is not None:
            self.logger.info(f'RESTART: rewriting {self.config_file}')
        # Lock the chunk, read, update, and write back
        lock = FileLock(self.config_file + '.lock', timeout=3*60)
        with lock:
            config = Config.load_from_yaml(self.config_file)

            status_list = config['computation_batches','status_list']
            for i, status in enumerate(status_list):
                if status == 1:
                    if self.logger is not None:
                        self.logger.info(f'batch {i} will be repeated')
                    config['computation_batches']['status_list'][i] = 0

            config.dump_to_yaml(self.config_file)


        self.run()