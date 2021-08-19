import numpy as np


def calc_mixing_length(z, h, alpha = 0.4/3):
    """
    Calculates the mixing length for each height in z
    Based on Poggi et al 2004
    Zero-plane displacement height is taken as (2/3)*h, appropriate for dense canopies (Katul et al 2004)
    Default value for alpha = 0.4/3
    TODO: Update documentation with references
    TODO: alpha = 0.4/3, based on Katul...  what do we use here?

    Inputs:
    -------
    z : vector of heights [m]
    h : canopy height [m]
    alpha : unitless parameter

    Outputs:
    --------
    mixing_length : mixing length [m] at each height in z
    """

    dz = z[1] - z[0]  # Calculate the vertical discretization interval
    d = 0.67 * h  # zero-plane displacement height [m]
    subcanopy_height = (alpha * h - 2 * dz) / 0.2

    mixing_length = np.piecewise(z, [z < subcanopy_height, (z >= subcanopy_height) & (z < d), z >= d],
                                 [lambda z: 0.2 * z + 2 * dz, alpha * h, lambda z: 0.4 * (z - d) + alpha * h])
    return mixing_length

# TODO: Solve Km and Uz
