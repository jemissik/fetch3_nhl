import numpy as np

def calc_Kg(Ta):
    """
    Calculate the temperature-dependent conductance coefficient
    From Ewers et al 2007
    Equation A.2 from Mirfenderesgi et al 2016
    Inputs:
    Ta : air temperature [deg C]

    Outputs:
    Kg: temperature-dependent conductance coefficient [kPa m3 kg-1]
    """

    Kg = 115.8 * 0.4236 * Ta
    return Kg

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

    mixing_length = np.piecewise(z.astype(float), [z < subcanopy_height, (z >= subcanopy_height) & (z < d), z >= d],
                                 [lambda z: 0.2 * z + 2 * dz, alpha * h, lambda z: 0.4 * (z - d) + alpha * h])
    return mixing_length

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

    q[-1] = gam[-1]

    for i in range(n-2, -1, -1):
        q[i] = gam[i]-(cc[i]*q[i+1]/bet[i])

    return q

def solve_Uz(z, mixing_length,Cd ,a_s, U_top):
    """
    Solves the momentum equation to calculate the vertical wind profile.
    Applies no-slip boundary condition: wind speed  =  0 at surface (z = 0).
    Model for turbulent diffusivity of momentum is from Poggi 2004, eqn 6 (TODO doesn't match Mirfenderesgi)

    Inputs:
    _______
    z : vector of heights [m]
    mixing_length : mixing length [d] at each height in z
    Cd : drag coefficient [unitless], assumed to be 0.2 (Katul et al 2004)
    TODO a_s: leaf surface area [m2]
    U_top : Measured wind speed at top of canopy [m s-1]

    Outputs:
    ________
    Km : turbulent diffusivity of momentum at each height in z [m s-1]
    U : wind speed at each height in z [m s-1]
    """

    dz = z[1] - z[0]  # Vertical discretization interval
    n = len(z)
    U_bottom = 0  # no-slip boundary
    U = np.linspace(U_bottom, U_top, n)  # Vertical wind speed profile, beginning iteration with linear profile

    # model for diffusivity, from Poggi et al 2004, eqn 6
    def calc_Km(mixing_length, dU):
        Km = (mixing_length ** 2) * np.abs(dU)
        return Km

    # start iterative solution
    err = 10**9

    while err > 0.0001:
        # dU/dz
        dU = np.zeros(n)
        dU[1:] = np.diff(U)/dz
        dU[0] = dU[1]

        # calculate Km
        Km = calc_Km(mixing_length, dU)

        # Set up coefficients for ODE
        a1 = -Km
        dKm = np.concatenate(([Km[1]-Km[0]], np.diff(Km)))  # Use Km[1]-Km[0] for first 2 elements of dKm
        a2 = -dKm/dz
        a3 = Cd * a_s * np.abs(U)  # TODO should it be abs(u)? it isn't in MATLAB version

        # Set the elements of the tridiagonal matrix
        upd = (a1 / (dz * dz) + a2 / (2 * dz))
        dia = (-a1 * 2 / (dz * dz) + a3)
        lod = (a1 / (dz * dz) - a2 / (2 * dz))
        co = np.zeros(n)
        co[0] = U_bottom
        co[-1] = U_top
        lod[0] = 0
        lod[-1] = 0
        upd[0] = 0
        upd[-1] = 0
        dia[0] = 1
        dia[-1] = 1

        # Solve tridiagonal matrix using Thomas algorithm
        Un = thomas_tridiagonal(lod, dia, upd, co)
        err = np.max(np.abs(Un - U))

        # Use successive relaxations in iterations
        eps1 = 0.5
        U = eps1 * Un + (1 - eps1) * U

    return U, Km

def calc_gb(uz, d):
    """
    Calculates the leaf boundary layer conductance

    Inputs:
    ________
    uz: wind speed at canopy height z [m s-1]
    d: characteristic leaf length [m]

    Outputs:
    ________________
    gb: leaf boundary layer conductance [mol m-2 s-1]
    TODO figure out where the units for gb come from

    """
    gb = 0.147 * (uz/d)**0.5
    return gb

def calc_geff(gb, gs):
    """
    Calculates the effective leaf conductance
    Eqn A.3 of Mirfenderesgi

    Inputs:
    ----------
    gb : boundary layer conductance [mol m-2 s-1]
    gs : stomatal conductance [mol m-2 s-1]

    Outputs:
    -------
    geff : effective leaf conductance [mol m-2 s-1]

    """
    geff = (gb * gs) / (gb + gs)
    return geff

def calc_zenith_angle(doy, lat, long, standard_meridian, time_of_day):
    """
    Calculates the solar zenith angle, based on Campbell & Norman, 1998

    Inputs:
    ----------
    doy : Day of year (Ordinal day, e.g. 1 = Jan 1)
    lat : Latitude
    long : Longitude (Needs to be negative for deg W, positive for deg E)
    standard_meridian : Standard meridian [deg] for the given location (Needs to be negative for deg W, positive for deg E)
    time_of_day : Time of day (hours) in local standard time

    Outputs:
    -------
    zenith_angle_deg : zenith angle of the sun [degrees]

    """
    # Calculate the solar declination angle, Eqn 11.2, Campbell & Norman
    declination_angle_rad = np.arcsin(0.39785 * np.sin(np.deg2rad(278.97 + 0.9856 * doy + 1.9165 * np.sin(np.deg2rad(356.6 + 0.9856 * doy)))))

    # Calculate the equation of time, Eqn 11.4, Campbell & Norman
    f = np.deg2rad(279.575 + 0.98565) # in radians
    ET = (-104.7 * np.sin(f) + 596.2 * np.sin(2 * f) + 4.3 * np.sin (3 * f) - 12.7 * np.sin(4 * f) - 429.3 * np.cos (f) - 2.0 * np.cos(2 * f) + 19.3 * np.cos(3 * f))/3600

    # Calculate the longitude correction
    # + 1/15 or an hour for each degree east of standard meridian
    # - 1/15 or an hour for each degree west of standard meridian
    long_correction = (long - standard_meridian) * 1/15

    # Calculate the time of solar noon (t0), Eqn 11.3, Campbell & Norman
    t0 = 12 - long_correction - ET

    # Calculate the zenith angle, Eqn 11.1, Campbell & Norman
    lat_rad = np.deg2rad(lat)
    zenith_angle_rad = np.arccos(np.sin(lat_rad) * np.sin(declination_angle_rad)
                                 + np.cos(lat_rad) * np.cos(declination_angle_rad) * np.cos(np.pi/12 * (time_of_day - t0)))
    zenith_angle_deg = np.rad2deg(zenith_angle_rad)

    return zenith_angle_deg

def calc_rad_attenuation(zenith_angle, LAI, Cf = 0.85, x = 1):
    """
    Calculates the vertical attenuation of radiation through the canopy

    Inputs:
    ----------
    zenith_angle
    LAI : Normalized leaf area index at each height in z
    Cf : Clumping fraction [unitless], assumed to be 0.85 (Forseth & Norman 1993) unless otherwise specified
    x : Ratio of horizontal to vertical projections of leaves (leaf angle distribution), assumed spherical (x=1)

    Outputs:
    -------
    P0 : attenuation fraction of PAR penetrating the canopy at each level z [unitless]
    # TODO MATLAB version has both LAI and LAD in parameters, only LAD is used. What are the correct units? 
    # TODO MATLAB version flips the arrays... why? needs to be done here? 
    """
    # Calculate the light extinction coefficient (unitless)
    k = (((x**2 + np.tan(np.deg2rad(zenith_angle)) ** 2) ** 0.5) * np.cos(np.deg2rad(zenith_angle))) / (x + 1.744 * (x + 1.182) ** 0.773)

    # Calculate P0
    P0 = np.exp(k * LAI * Cf)