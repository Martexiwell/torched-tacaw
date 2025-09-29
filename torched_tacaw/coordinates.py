"""
This submodule is used to work with coordinate systems, usually 2D coordinates
as used in other files.
By convention, we are using 'ij' indexing - first is x-axis followed by y-axis

"""
import numpy as np

def scanning_grids_indices(shape: list | tuple | np.ndarray) -> np.ndarray:
    """retrurns grids of scannning indices

    Examples
    --------
    >>> scanning_grids_indices([2,3])
    np.array([[[0, 0, 0],
               [1, 1, 1]],
              [[0, 1, 2],
               [0, 1, 2]]])
    """
    # shape = self['beam','scanning','shape']

    x_axis = np.arange(shape[0])
    y_axis = np.arange(shape[1])

    grids = np.array(np.meshgrid(x_axis, y_axis, indexing='ij'))

    return grids


def scanning_grids_coordinates_fractional(shape: list | tuple | np.ndarray) -> np.ndarray:
    """Returns grids of fractional coordinates from 1 by 1 square given by shape

    Examples
    --------
    >>> scanning_grids_coordinates_fractional([2,3])
    np.array([[[0.        , 0.        , 0.        ],
               [0.5       , 0.5       , 0.5       ]],
              [[0.        , 0.33333333, 0.66666667],
               [0.        , 0.33333333, 0.66666667]]])
    """

    indices_grids = scanning_grids_indices(shape)

    fractional_grids = np.einsum('a,akl->akl', 1 / np.array(shape), indices_grids)

    return fractional_grids


def scanning_grids_coordinates_cartesian(
    shape: list | tuple | np.ndarray,
    basis_vectors: list | tuple | np.ndarray = None,
    origin: list | tuple | np.ndarray = None,
) -> np.ndarray:
    """Returns grids of cartesian coordinates corresponding to grid-points
    (given by shape) in parallelogram defined by origin and two vectors

    Parameters
    ----------
    shape : list or similar
        shape of scanning grid as [m,n]
    basis_vectors : nested list or similar, default = [[1,0], [0,1]]
        basis vectors provided "as a list of row vectors"
        e.g. [[1,0],[3,2]] is a basis with canonical
        a = [1,0] but b = [3,2]
    origin : list or similar, default = [0,0]
        origin of the parallelogram

    Returns
    -------
    np.ndarray
        grids of cartesian coordinates in the shape (2,m,n)
    """
    basis_vectors = basis_vectors if basis_vectors is not None else [[1.,0.],[0.,1.]]
    origin = origin if origin is not None else [0.,0.]

    basis_matrix = np.array(basis_vectors).T  # basis vectors as columns in matrix from assumed two vectors in row
    origin = np.array(origin)

    # fractional_coordinates = self.get_scanning_grids_coordinates_fractional()
    fractional_coordinates = scanning_grids_coordinates_fractional(shape)

    cartesian_coordinates = np.einsum(
        'ij,jkl->ikl',
        basis_matrix,
        fractional_coordinates
    ) + origin[:, np.newaxis, np.newaxis]

    return cartesian_coordinates


def extend_coordinate_grids(grids: np.ndarray, tiles: list | tuple | np.ndarray):
    """Extends coordinate grids by tiling the provided grids
    with added bias so that the returned grid is extension of the provided one
    (!) assumes that the grid is equidistant (!)
    
    Examples
    --------
    >>> grids = np.array([[[0.  0. ]
    ...                    [1.  1. ]
    ...                    [2.  2. ]]
    ...
    ...                   [[0.  0.2]
    ...                    [0.  0.2]
    ...                    [0.  0.2]]])
    ... tiles = [2,3]
    ...
    ... tile_coordinate_grid(grids, tiles)
    array([[[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
            [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ],
            [2. , 2. , 2. , 2. , 2. , 2. , 2. , 2. , 2. ],
            [3. , 3. , 3. , 3. , 3. , 3. , 3. , 3. , 3. ]],

           [[0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6],
            [0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6],
            [0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6],
            [0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6]]])

    """
    dx = grids[0,1,0] - grids[0,0,0]
    dy = grids[1,0,1] - grids[1,0,0]
    X = grids[0]
    Y = grids[1]
    
    X = np.concatenate([np.concatenate([ 
                X
                + ix * (np.zeros(X.shape) + dx + X[-1,0]-X[0,0])
        for iy in range(tiles[1])], axis = 1) 
        for ix in range(tiles[0])], axis = 0)
    Y = np.concatenate([np.concatenate([ 
                Y
                + iy * (np.zeros(Y.shape) + dy + Y[0,-1]-Y[0,0])
        for iy in range(tiles[1])], axis = 1) 
        for ix in range(tiles[0])], axis = 0)

    return np.array([X,Y])


def tile_array(array: np.ndarray, tiles: list | tuple | np.ndarray) -> np.ndarray:
    """Tiles the input array by repeating it along respective axes

    Parameters
    ----------
    array : np.ndarray
        shape e.g. (x,y)
    tiles : list or similar
        e.g. [m,n]

    Returns
    -------
    np.ndarray
        with shape e.g. (m*x, n*y)
    """
    for axis, tiling in enumerate(tiles):
        array = np.concatenate([array]*tiling, axis=axis)
    return array

def debug():
    shape = [5, 8]

    basis = [[2, -1],
             [1, 3]]

    origin = [-10, 10]

    indices_grids = scanning_grids_indices(shape)
    print(indices_grids.shape)
    print(indices_grids)
    assert indices_grids.shape == (2, *shape)

    fractional_grids = scanning_grids_coordinates_fractional(shape)
    print(fractional_grids.shape)
    print(fractional_grids)
    assert fractional_grids.shape == (2, *shape)

    cartesian_grids = scanning_grids_coordinates_cartesian(shape, basis, origin)
    print(cartesian_grids.shape)
    print(cartesian_grids)
    assert cartesian_grids.shape == (2, *shape)





if __name__ == '__main__':
    debug()