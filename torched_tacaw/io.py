"""
Input / Output routines

"""

import ase
from ase.io.lammpsrun import read_lammps_dump
import ase.io

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



class TrajctoryReader:

    def __init__(self, filename:str, **kwargs):
        self.filename = filename
        filetype = kwargs.pop('filetype', None)

