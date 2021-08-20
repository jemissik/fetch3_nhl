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

    dz = z[1] - z[0]  # Vertical discretization interval
    d = 0.67 * h  # zero-plane displacement height [m]
    subcanopy_height = (alpha * h - 2 * dz) / 0.2

    mixing_length = np.piecewise(z, [z < subcanopy_height, (z >= subcanopy_height) & (z < d), z >= d],
                                 [lambda z: 0.2 * z + 2 * dz, alpha * h, lambda z: 0.4 * (z - d) + alpha * h])
    return mixing_length

# TODO: Solve Km and Uz

def calc_uz(Uz, ustar):
    """
    Calculates the wind speed at canopy height z adjusted by the friction velocity
    Eqn A.5 from Mirfenderesgi et al 2016

    Inputs:
    Uz : wind speed at height z [m s-1]
    ustar : friction velocity [m s-1]

    Outputs:
     uz : wind speed [m s-1] at canopy height z adjusted by the friction velocity
    """
    uz = Uz*ustar
    return uz

def thomas_tridiagonal (aa, bb, cc, dd):
    """
    Thomas algorithm for solving tridiagonal matrix
    TODO Add documentation
    Inputs:

    Outputs:
    """

    #initialize arrays
    n = len(bb)
    bet = np.zeros(n)
    gam = np.zeros(n)
    q = np.zeros(n)

    bet[0] = bb[0]
    gam[0] = dd[0]/bb[0]

    for i in range(1, n):
        bet[i] = bb[i] - (aa[i] * cc[i - 1] / bet[i - 1])
        gam[i] = (dd[i] - aa[i] * gam[i - 1]) / bet[i]

    q[n-1] = gam[n-1]

    for i in range(n-2, -1, -1):
        q[i] = gam[i]-(cc[i]*q[i+1]/bet[i])

    return q



