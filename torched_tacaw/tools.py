import numpy as np
import pathlib
import logging


def rotmat2(theta):
    """Returns 2d rotational matrix

        Parameters
        ----------
        theta : float
            angle in radians

        Returns
        -------
        np.ndarray
            rotation matrix
        """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def rotmat3(theta, axis:int|str=None):
    """Returns 3d rotational matrix

    Parameters
    ----------
    theta : float
        angle in radians
    axis : str or int, default='z'
        axis of rotation

    Returns
    -------
    np.ndarray
        rotation matrix
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    if axis == 'z' or axis == 2 or axis == None:
        pass
    elif axis == 'x' or axis == 0:
        R = np.roll(R, (1, 1), (0,1))
    elif axis == 'y' or axis == 1:
        R = np.roll(R, (2, 2), (0,1))
    else:
        raise ValueError(f'invalid axis {axis} of type {type(axis)} - must be "z" or "x" or "y", or 2 or 0 or 1')

    return R


def ensure_valid_path_file(filename):
    """TODO:
    Should ensure that the path is valid
    by creating all the necessary folders
    """
    foldername = '/'.join(filename.split('/')[:-1])+'/'
    folderpath = pathlib.Path(foldername)

    if not folderpath.exists():
        print(f'creating folder {foldername}')
        folderpath.mkdir(parents=True, exist_ok=True)
        return 1
    else:
        return 0



def indices_of_cutout_from_array(array_shape, cutout_shape, cutout_coordinates ):

    # print('I: ', array_shape, cutout_coordinates, cutout_shape)

    start   = [int(i*j) for i,j in zip(cutout_coordinates, cutout_shape) ]
    stop    = np.array([
        start[i] + cutout_shape[i] if start[i] + cutout_shape[i] < array_shape[i] else array_shape[i]
        for i in range(len(array_shape))
    ])

    # print(f'XXX {start} {stop}')

    return start, stop

def cutout_from_array(array, start, stop):
    slices = tuple(slice(start[i], stop[i]) for i in range(len(start)))

    # Use the slices to cut out the desired portion of the array
    cutout = array[slices]
    return cutout

def get_nth_cutout_from_array(array, cutout_shape, cutout_coordinates):
    start, stop = indices_of_cutout_from_array(array.shape, cutout_shape, cutout_coordinates)
    return cutout_from_array(array, start, stop)



# Logging handler
# ===============
def _do_nothing(self, *args, **kwargs):
    return None

_silent_methods = {k:_do_nothing for k,v in logging.Logger.__dict__.items() if callable(v)}
# "fake" logger class that can be used as the logger object, just does nothing
NullLogger = type("NullLogger", (logging.Logger,), _silent_methods)

def logger_or_null(logger) -> logging.Logger | NullLogger:
    """Returns logger or NullLogger if input is None, raises ValueError if input is invalid

    Parameters
    ----------
    logger : logging.Logger | str | bool | None
        : logging.Logger
            -> logger          # input is used as logger
        : str
            -> logging.getLogger(logger)
        True
            -> logging.getLogger(__name__)
        None or False
            -> NullLogger()    # no logging
    """
    if logger is None or logger is False:
        return NullLogger()
    elif isinstance(logger, logging.Logger):
        return logger
    elif isinstance(logger, str):
        return logging.getLogger(logger)
    elif logger is True:
        return logging.getLogger(__name__)
    else:
        raise Exception(f'invalid logger object provided: {logger}')


def add_shot_noise(pdf, n_counts=None, beam_current_A=None, dwell_time_s=None):
    """
    Add shot noise to a normalized PMF.

    Parameters
    ----------
    pdf : np.ndarray
        Normalized probability array (sums to 1.0)
    n_counts : int or float, optional
        Total number of electrons (counts) to simulate.
    beam_current_A : float, optional
        Beam current in Amperes.
    dwell_time_s : float, optional
        Per-voxel dwell time in seconds.

    Returns
    -------
    np.ndarray
        Noisy pdf, renormalized to sum to 1.0

    Notes
    -----
    Provide either `n_counts` OR both `beam_current_A` and `dwell_time_s`.
    """
    e = 1.602176634e-19  # elementary charge, Coulombs

    if n_counts is not None:
        total_counts = n_counts
    elif beam_current_A is not None and dwell_time_s is not None:
        total_counts = (beam_current_A * dwell_time_s) / e
    else:
        raise ValueError(
            "Provide either `n_counts` or both `beam_current_A` and `dwell_time_s`."
        )

    counts = np.random.poisson(pdf * total_counts)

    total = counts.sum()

    return counts / total_counts


def debugger():
    ensure_valid_path_file('testfolder0/testfolder1/test.txt')

if __name__ == "__main__":
    debugger()