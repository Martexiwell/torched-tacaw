"""
N-D Bresenham line algo
"""
import numpy as np
# def _bresenhamline_nslope(slope):
#     """
#     Normalize slope for Bresenham's line algorithm.
#
#     >>> s = np.array([[-2, -2, -2, 0]])
#     >>> _bresenhamline_nslope(s)
#     array([[-1., -1., -1.,  0.]])
#
#     >>> s = np.array([[0, 0, 0, 0]])
#     >>> _bresenhamline_nslope(s)
#     array([[ 0.,  0.,  0.,  0.]])
#
#     >>> s = np.array([[0, 0, 9, 0]])
#     >>> _bresenhamline_nslope(s)
#     array([[ 0.,  0.,  1.,  0.]])
#     """
#     scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
#     zeroslope = (scale == 0).all(1)
#     scale[zeroslope] = np.ones(1)
#     normalizedslope = np.array(slope, dtype=np.double) / scale
#     normalizedslope[zeroslope] = np.zeros(slope[0].shape)
#     return normalizedslope
#
# def _bresenhamlines(start, end, max_iter):
#     """
#     Returns npts lines of length max_iter each. (npts x max_iter x dimension)
#
#     >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
#     >>> _bresenhamlines(s, np.zeros(s.shape[1]), max_iter=-1)
#     array([[[ 3,  1,  8,  0],
#             [ 2,  1,  7,  0],
#             [ 2,  1,  6,  0],
#             [ 2,  1,  5,  0],
#             [ 1,  0,  4,  0],
#             [ 1,  0,  3,  0],
#             [ 1,  0,  2,  0],
#             [ 0,  0,  1,  0],
#             [ 0,  0,  0,  0]],
#     <BLANKLINE>
#            [[ 0,  0,  2,  0],
#             [ 0,  0,  1,  0],
#             [ 0,  0,  0,  0],
#             [ 0,  0, -1,  0],
#             [ 0,  0, -2,  0],
#             [ 0,  0, -3,  0],
#             [ 0,  0, -4,  0],
#             [ 0,  0, -5,  0],
#             [ 0,  0, -6,  0]]])
#     """
#     if max_iter == -1:
#         max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
#     npts, dim = start.shape
#     nslope = _bresenhamline_nslope(end - start)
#
#     # steps to iterate on
#     stepseq = np.arange(1, max_iter + 1)
#     stepmat = np.tile(stepseq, (dim, 1)).T
#
#     # some hacks for broadcasting properly
#     bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat
#
#     # Approximate to nearest int
#     return np.array(np.rint(bline), dtype=start.dtype)
#
# def bresenhamline(start, end, max_iter=5):
#     """
#     Returns a list of points from (start, end] by ray tracing a line b/w the
#     points.
#     Parameters:
#         start: An array of start points (number of points x dimension)
#         end:   An end points (1 x dimension)
#             or An array of end point corresponding to each start point
#                 (number of points x dimension)
#         max_iter: Max points to traverse. if -1, maximum number of required
#                   points are traversed
#
#     Returns:
#         linevox (n x dimension) A cumulative array of all points traversed by
#         all the lines so far.
#
#     >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
#     >>> bresenhamline(s, np.zeros(s.shape[1]), max_iter=-1)
#     array([[ 3,  1,  8,  0],
#            [ 2,  1,  7,  0],
#            [ 2,  1,  6,  0],
#            [ 2,  1,  5,  0],
#            [ 1,  0,  4,  0],
#            [ 1,  0,  3,  0],
#            [ 1,  0,  2,  0],
#            [ 0,  0,  1,  0],
#            [ 0,  0,  0,  0],
#            [ 0,  0,  2,  0],
#            [ 0,  0,  1,  0],
#            [ 0,  0,  0,  0],
#            [ 0,  0, -1,  0],
#            [ 0,  0, -2,  0],
#            [ 0,  0, -3,  0],
#            [ 0,  0, -4,  0],
#            [ 0,  0, -5,  0],
#            [ 0,  0, -6,  0]])
#     """
#     # Return the points as a single array
#     return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])

def bresenham_line(start, end):
    """

    Parameters
    ----------
    start
    end

    Returns
    -------
    np.ndarray

    Examples
    --------
    >>> bresenham_line([2,10], [13,5])
    array([[ 2 10]
           [ 3 10]
           [ 4  9]
           [ 5  9]
           [ 6  8]
           [ 7  8]
           [ 8  7]
           [ 9  7]
           [10  6]
           [11  6]
           [12  5]])

    """
    start = np.array(start, dtype=int)
    end = np.array(end, dtype=int)

    diff = end - start

    nof_points = np.max(np.abs(diff))
    long_axis = np.arange(nof_points)

    line = start[np.newaxis,:] + long_axis[:,np.newaxis] * diff[np.newaxis, :] / nof_points
    line = np.round(line).astype(int)

    return line


def test():
    def print_grid(grid):
        for line in grid:
            print(*line, sep='')

    testgrid = np.zeros([20,20], int)


    print_grid(testgrid)

    line = bresenham_line([2,10], [13,5])
    print('---')
    print(line)
    print('---')
    for point in line:
        testgrid[point[0]][point[1]] += 1

    print_grid(testgrid)




if __name__ == "__main__":
    # import doctest
    # doctest.testmod()
    test()
