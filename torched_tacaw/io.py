"""
Input / Output routines

"""
import os
from collections import deque

import numpy as np


import ase
from ase.io.lammpsrun import read_lammps_dump
import ase.io.lammpsrun as lammpsrun

import ase.io
from ase.io import Trajectory

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





def convert_lammps_dump_text_to_traj(
        lammps_file:str,
        traj_file:str,
        **kwargs
):
    """Processes cleartext lammps dumpfiles and dumps them to ase .traj file format

    Parameters
    ----------
    lammps_file : str
    traj_file : str
    kwargs : dict
        other parameters passed to the readers


    Notes
    -----
    This function is mostly alteration of ase.io.lammpsrun.read_lammps_dump().

    """

    def _parse_pbc(tilt_items: list[str]) -> list[bool]:
        """Handle pbc conditions."""
        pbc_items = tilt_items[-3:] if len(tilt_items) >= 3 else ['f', 'f', 'f']
        return ['p' in d.lower() for d in pbc_items]

    def _parse_box_bound(line: str, lines: deque) -> tuple:
        # save labels behind "ITEM: BOX BOUNDS" in triclinic case
        # (>=lammps-7Jul09)
        tilt_items = line.split()[3:]
        cell_lines = [lines.popleft() for _ in range(3)]
        cell_data = np.loadtxt(cell_lines)

        # general triclinic boxes (>=patch_17Apr2024)
        if tilt_items[0] == 'abc':
            cell = cell_data[:, :3]
            celldisp = cell_data[:, 3]
            pbc = _parse_pbc(tilt_items)
            return cell, celldisp, pbc

        diagdisp = cell_data[:, :2].flatten()

        # determine cell tilt (triclinic case!)
        if len(cell_data[0]) > 2:
            # for >=lammps-7Jul09 use labels behind "ITEM: BOX BOUNDS"
            # to assign tilt (vector) elements ...
            offdiag = cell_data[:, 2]
            # ... otherwise assume default order in 3rd column
            # (if the latter was present)
            if len(tilt_items) >= 3:
                sort_index = [tilt_items.index(i) for i in ['xy', 'xz', 'yz']]
                offdiag = offdiag[sort_index]
        else:
            offdiag = np.zeros(3)

        cell, celldisp = lammpsrun.construct_cell(diagdisp, offdiag)

        pbc = _parse_pbc(tilt_items)

        return cell, celldisp, pbc



    trajectory_writer = Trajectory(traj_file, mode='w')

    fileobj = paropen(lammps_file)

    # Load all dumped timesteps into memory simultaneously
    lines = deque(fileobj.readlines())
    # index_end = get_max_index(index)

    n_atoms = 0
    n_snapshots = 0

    # avoid references before assignment in case of incorrect file structure
    cell, celldisp, pbc, info = None, None, False, {}

    while len(lines) > n_atoms:
        line = lines.popleft()

        if 'ITEM: TIMESTEP' in line:
            line = lines.popleft()
            # !TODO: pyflakes complains about this line -> do something
            ntimestep = int(line.split()[0])  # NOQA
            info['timestep'] = ntimestep

        if 'ITEM: NUMBER OF ATOMS' in line:
            line = lines.popleft()
            n_atoms = int(line.split()[0])

        if 'ITEM: BOX BOUNDS' in line:
            cell, celldisp, pbc = _parse_box_bound(line, lines)

        if 'ITEM: ATOMS' in line:
            colnames = line.split()[2:]
            datarows = [lines.popleft() for _ in range(n_atoms)]
            data = np.loadtxt(datarows, dtype=str, ndmin=2)
            out_atoms = lammpsrun.lammps_data_to_ase_atoms(
                data=data,
                colnames=colnames,
                cell=cell,
                celldisp=celldisp,
                atomsobj=lammpsrun.Atoms,
                pbc=pbc,
                **kwargs,
            )
            out_atoms.info.update(info)

            trajectory_writer.write(out_atoms)

            n_snapshots += 1
            print(f"snapshot {n_snapshots:5} converted", end='\r')