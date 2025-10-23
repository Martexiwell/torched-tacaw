"""
Input / Output routines

"""
import os
from collections import deque

import ase
from ase.io.lammpsrun import read_lammps_dump
import ase.io

from ase.parallel import paropen

class LammpsTrajectoryReader:
    """ Returns an object that reads the trajectory from Lammps trajectory file when indexed

    Parameters
    ----------
    filename : str
        filename where trajectory is stored


    Examples
    --------
    trajectory = LammpsTrajectoryReader('trajectory.lammps', specorder=['Ba', 'Fe', 'O'])

    atoms = trajectory[35]


    # for i in range(999):
    #     snapshot = read_lammps_dump('atom_pos.lammps', index=i, specorder=['Ba', 'Fe', 'O'])

    """
    def __init__(self, filename:str, **kwargs):
        self.filename = filename
        self.kwargs = kwargs

    def __getitem__(self, index: int) -> ase.Atoms:
        snapshot = read_lammps_dump(self.filename, index=index, **self.kwargs)
        return snapshot

    def __len__(self) -> int:
        """
        Returns the number of snapshots in the trajectory

        this method is constructed based on the excerpts from ase

        Returns
        -------

        """
        """Process cleartext lammps dumpfiles

        :param fileobj: filestream providing the trajectory data
        :param index: integer or slice object (default: get the last timestep)
        :returns: list of Atoms objects
        :rtype: list
        """

        if isinstance(self.filename, str):
            fileobj = paropen(self.filename)

        # Load all dumped timesteps into memory simultaneously
        lines = deque(fileobj.readlines())

        len_timstep_block = 0
        n_lines_total = len(lines)

        _atoms_lines_found = 0
        while _atoms_lines_found < 2:
            line = lines.popleft()
            if 'ITEM: ATOMS' in line:
                _atoms_lines_found += 1
            len_timstep_block += 1

        n_timesteps = n_lines_total // len_timstep_block

        return n_timesteps



class TrajctoryReader:
    """
    Returns an object that reads the trajectory from trajectory file
    returns ase.Atoms object when indexed

    Parameters
    ----------
    filename : str
        filename where trajectory is stored
    filetype : str, optional
        if not provided, filetype is guessed based on extension
        readable filetypes are currently .traj and .lammps

    Attributes
    ----------
    filename : str
    filetype : str | None
    kwargs : dict
        other parameters passed to the readers
    trajectory
        indexable object returning ase.Atoms upon indexing
    """
    def __init__(self, filename:str, filetype:str=None, **kwargs):
        self.filename = filename
        self.filetype:str = filetype
        self.kwargs = kwargs

        # guess filetpe if not provided:
        if self.filetype is None:
            self.filetype = os.path.splitext(self.filename)[-1]


        if self.filetype in ['traj', '.traj', 'ase']: # ASE .traj file
            self.trajectory = ase.io.Trajectory(self.filename)

        elif self.filetype in ['lammps', '.lammps']:
            self.trajectory = LammpsTrajectoryReader(self.filename, **self.kwargs)


    def __getitem__(self, index: int) -> ase.Atoms:
        return self.trajectory[index]

    def __len__(self) -> int:
        return self.trajectory.__len__()
