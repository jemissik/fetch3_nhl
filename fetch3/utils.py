import numpy as np
#############################################
# Helper functions
#################################################

# Function to do to 2d interpolation
def interpolate_2d(x, zdim):
    """
    Interpolates input to 2d for canopy-distributed values

    Parameters
    ----------
    x : [type]
        input
    zdim : [type]
        length of z dimension
    """
    x_2d = np.zeros(shape=(zdim, len(x)))
    for i in np.arange(0,len(x),1):
        x_2d[:,i]=x[i]
    return x_2d

def neg2zero(x):
    return np.where(x < 0, 0, x)