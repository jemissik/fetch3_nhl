"""
#################
NHL transpiration
#################
This module calculates non-hydrodynamically limited transpiration.
This code was ported from FETCH2's MATLAB code (see Mirfenderesgi et al 2016)
"""
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d

def calc_esat(Tair):
    """
    Calculates the saturation vapor pressure using the Clausius-Clapeyron equation.

    Parameters
    ----------
    Tair : array_like
        Air temperature [degrees C]

    Returns
    -------
    es : array_like
        The saturation vapor pressure [kPa] corresponding to each element in Tair
    """

    #constants
    e0 = 0.611 #kPa
    T0 = 273 #K
    Rv = 461 #J K-1 kg -1, gas constant for water vapor
    Lv = 2.5 * 10**6 #J kg-1

    Tair = Tair + 273.15 #convert temperature to Kelvin
    es = e0 * np.exp((Lv/Rv)*(1/T0 - 1/Tair))
    return es

def calc_vpd_kPa(RH, Tair):
    """
    Calculates vapor pressure deficit from air temperature and relative humidity.

    Parameters
    ----------
    RH : array_like
        Relative humidity [%]
    Tair : array_like
        Air temperature [deg C]

    Returns
    -------
    VPD : array_like
        Vapor pressure deficit [kPa]
    """

    es = calc_esat(Tair)
    eactual = RH*es/100

    return (es - eactual)

def calc_Kg(Tair):
    """
    Calculates the temperature-dependent conductance coefficient.
    Formulation from Ewers et al 2007
    Equation A.2 from Mirfenderesgi et al 2016

    Parameters
    ----------
    Tair : array_like
        Air temperature [deg C]

    Returns
    -------
    Kg : array_like
        Temperature-dependent conductance coefficient [kPa m3 kg-1]
    """
    Kg = 115.8 + 0.4226 * Tair
    return Kg

def calc_mixing_length(z, h, alpha_ml = 0.1):
    """
    Calculates the mixing length for each height in z.
    Based on Poggi et al 2004
    Zero-plane displacement height is taken as (2/3)*h, appropriate for dense canopies (Katul et al 2004).

    Parameters
    ----------
    z : array
        vector of heights [m]
    h : float
        canopy height [m]
    alpha_ml : float
        unitless parameter

    Returns
    -------
    mixing_length : array
        mixing length [m] at each height in z
    """

    dz = z[1] - z[0]  # Vertical discretization interval
    d = 0.67 * h  # zero-plane displacement height [m]
    subcanopy_height = (alpha_ml * h - 2 * dz) / 0.2

    mixing_length = np.piecewise(z.astype(float), [z < subcanopy_height, (z >= subcanopy_height) & (z < d), z >= d],
                                 [lambda z: 0.2 * z + 2 * dz, alpha_ml * h, lambda z: 0.4 * (z - d) + alpha_ml * h])
    return mixing_length


def thomas_tridiagonal (aa, bb, cc, dd):
    """
    Thomas algorithm for solving tridiagonal matrix
    #TODO update docstring
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

def solve_Uz(z, dz, Cd ,a_s, U_top, **kwargs):
    """
    Solves the momentum equation to calculate the vertical wind profile.
    Applies no-slip boundary condition: wind speed=0 at surface (z = 0).
    Model for turbulent diffusivity of momentum is from Poggi 2004, eqn 6

    Parameters
    ----------
    z : array
        vector of heights [m]
    dz : float
        Vertical discretization interval [m]
    Cd : float
        drag coefficient [unitless]
    a_s: float
        leaf surface area [m2]
    U_top : float
        Measured wind speed at top of canopy [m s-1]
    **kwargs to be passed to calc_mixing_length

    Returns
    -------
    Km : array
        turbulent diffusivity of momentum at each height in z [m s-1]
    U : array
        wind speed at each height in z [m s-1]
    """

    n = len(z)
    U_bottom = 0  # no-slip boundary
    U = np.linspace(U_bottom, U_top, n)  # Vertical wind speed profile, beginning iteration with linear profile

    mixing_length = calc_mixing_length(z, **kwargs)

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
        a3 = Cd * a_s * np.abs(U)

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

def calc_gb(uz, d = 0.0015):
    """
    Calculates the leaf boundary layer conductance and resistance, assuming laminar boundary layer.

    Parameters
    ----------
    uz : array-like
        wind speed at canopy height z [m s-1]
    d : float
        characteristic leaf length [m]. By default 0.0015.

    Returns
    -------
    gb : array-like
        leaf boundary layer conductance
    rb : array_like
        leaf boundary layer resistance

    References
    ----------
    Monteith J, Unsworth M. Principles of environmental physics: plants, animals, and the atmosphere.
    Academic Press; 2013 Jul 26.

    """
    rb = (395 * 29 / 1150) * (d / (np.sqrt(uz ** 2) + 0.001)) ** 0.5
    gb = 1/rb
    return gb, rb

def calc_geff(gb, gs):
    """
    Calculates the effective leaf conductance.
    Eqn A.3 of Mirfenderesgi

    Parameters
    ----------
    gb : array_like
        boundary layer conductance [mol m-2 s-1]
    gs : array_like
        stomatal conductance [mol m-2 s-1]

    Returns
    -------
    geff : array_like
        effective leaf conductance [mol m-2 s-1]

    """
    geff = (gb * gs) / (gb + gs)
    return geff

def calc_zenith_angle(doy, lat, long, time_offset, time_of_day, zenith_method='CN'):
    """
    Calculates the solar zenith angle, based on Campbell & Norman, 1998.

    Also has options for 2 alternate methods: use a constant zenith angle of 0 (i.e.
    pretend the sun is always directly overhead), or replicate the results of the
    FETCH2 NHL code (which is missing an adjustment for longitude).

    Parameters
    ----------
    doy : int
        Day of year (Ordinal day, e.g. 1 = Jan 1)
    lat : float
        Latitude
    long : float
        Longitude (Needs to be negative for deg W, positive for deg E)
    time_offset : int
        Time offset [in hours] for local standard time zone, e.g, for
        Pacific Standard Time, time_offset = -8
    time_of_day : float
        Time of day (hours) in local standard time
    zenith_method : str
        Method to use for the calculations. Options are:
            * "CN" - Campbell & Norman 1998
            * "constant" - constant zenith angle of 0
            * "fetch2" - replicate the results of the FETCH2 NHL code

    Note: Be sure that time of day and time offset are in local standard time, not daylight savings

    Returns
    -------
    zenith_angle_deg : zenith angle of the sun [degrees]

    """
    #Calculation if using the Campbell & Norman method
    if zenith_method == "CN":
        # Calculate the standard meridian (in degrees) from the time zone offset
        standard_meridian = time_offset * 15

        # Calculate the solar declination angle, Eqn 11.2, Campbell & Norman
        declination_angle_rad = np.arcsin(0.39785 * np.sin(np.deg2rad(278.97 + 0.9856 * doy + 1.9165 * np.sin(np.deg2rad(356.6 + 0.9856 * doy)))))

        # Calculate the equation of time, Eqn 11.4, Campbell & Norman
        f = np.deg2rad(279.575 + 0.98565 * doy) # in radians. NOTE: typo in my version of Campbell & Norman book
        ET = (-104.7 * np.sin(f) + 596.2 * np.sin(2 * f) + 4.3 * np.sin (3 * f) - 12.7 * np.sin(4 * f) - 429.3 * np.cos (f) - 2.0 * np.cos(2 * f) + 19.3 * np.cos(3 * f))/3600

        # Calculate the longitude correction
        # + 1/15 of an hour for each degree east of standard meridian
        # - 1/15 of an hour for each degree west of standard meridian
        long_correction = (long - standard_meridian) * 1/15

        # Calculate the time of solar noon (t0), Eqn 11.3, Campbell & Norman
        t0 = 12 - long_correction - ET

        # Calculate the zenith angle, Eqn 11.1, Campbell & Norman
        lat_rad = np.deg2rad(lat)
        zenith_angle_rad = np.arccos(np.sin(lat_rad) * np.sin(declination_angle_rad)
                                    + np.cos(lat_rad) * np.cos(declination_angle_rad) * np.cos(np.deg2rad(15 * (time_of_day - t0))))
        zenith_angle_deg = np.rad2deg(zenith_angle_rad)
    elif zenith_method == "constant":
        zenith_angle_deg = 0;
    elif zenith_method == "fetch2":
        # This code was ported directly from the FETCH2 NHL module
        #compute Solar declination angle
        adjusted_time_of_day = time_of_day + 5 #accounts for the adjustment in Time used in the FETCH2 input
        CF = np.pi/180
        LAT = lat*CF
        xx = 278.97 + 0.9856 * doy + 1.9165 * np.sin((356.6 + 0.9856 * doy) * CF)
        dd = np.arcsin(0.39785 * np.sin(xx * CF))
        #compute Zenith angle
        f = (279.575 + 0.9856 * doy) * CF
        ET = (-104.7*np.sin(f) + 596.2*np.sin(2*f) + 4.3*np.sin(3*f)
              -12.7*np.sin(4*f) -429.3*np.cos(f) - 2*np.cos(2*f)
              +19.3*np.cos(3*f))/3600

        TF = 0 * (long/15) + ET
        aa = np.sin(LAT)*np.sin(dd)+(np.cos(LAT))*np.cos(dd)*np.cos(15*(adjusted_time_of_day-12-TF)*CF)
        ZEN=np.arccos(aa)
        zenith_angle_deg = np.rad2deg(ZEN)

    return zenith_angle_deg

def calc_rad_attenuation(PAR, LAD, dz, Cf = 0.85, x = 1, **kwargs):
    """
    Calculates the vertical attenuation of radiation through the canopy

    Parameters
    ----------
    PAR : float
        photosynthetically active radiation at canopy top [umol m-2 s-1]
    LAD : array
        Leaf area density [m2leaf m-2crown m-1stem] at each height in z
    dz : float
        Vertical discretization interval [m]
    Cf : float
        Clumping fraction [unitless]. By default assumed to be 0.85 (Forseth & Norman 1993)
    x : float
        Ratio of horizontal to vertical projections of leaves (leaf angle distribution).
        By default assumed to be spherical (x=1).
    **kwargs to be passed to calc_zenith_angle

    Returns
    -------
    P0 : array
        attenuation fraction of PAR penetrating the canopy at each level z [unitless]
    Qp : array
        absorbed photosynthetically active radiation at each level within the canopy [umol m-2 s-1]
    """
    zenith_angle = calc_zenith_angle(**kwargs)
    # Calculate the light extinction coefficient (unitless)
    xn1=np.sqrt(x * x + (np.cos(np.deg2rad(zenith_angle))) ** 2)
    xd1=(x + 1.774 * np.cos(np.deg2rad(zenith_angle)) * (x + 1.182) **(-0.733))
    k = xn1/xd1

    LAI_cumulative = (LAD*dz)[::-1].cumsum()[::-1] # Cumulative sum from top of canopy
    # Calculate P0 and Qp
    P0 = np.exp(-k * LAI_cumulative * Cf)
    Qp = P0 * PAR

    return P0, Qp, zenith_angle

def calc_gs_Leuning(g0, m, A, c_s, gamma_star, VPD, D0 = 3):
    """
    Calculates gs according to Leuning 1995

    Parameters
    ----------
    g0 : float
        cuticular conductance [mol m-2 s-1], residual stomatal conductance at the
        light compensation point (empirically fitted parameter)
    m : float
        empirically fitted parameter [unitless]
    A : float
        net CO2 assimilation rate [umol CO2 m-2 s-1]
    c_s : float
        atmospheric CO2 concentration [umol mol-1]
    gamma_star : float
        CO2 compensation point [umol mol-1]
    VPD : float
        VPD [kPa]
    D0 : float
        reference vapor pressure [kPa], by default assumed to be 3.0 kPa

    Returns
    -------
    gs : float
        stomatal conductance [mol H2O m-2 s-1]
    """

    gs = g0 + m * abs(A)/((c_s - gamma_star) * (1 + VPD/D0))
    return gs

def solve_leaf_physiology(Tair, Qp, Ca, Vcmax25, alpha_p, VPD, **kwargs):
    """
    Calculates photosynthesis and stomatal conductance
    Uses Leuning model for stomatal conductance

    Parameters
    ----------
    Tair : float
        Air temperature [deg C]
    Qp : float
        absorbed photosynthetically active radiation at each level within the canopy [umol m-2 s-1]
    Ca : float
        CO2 concentration [umol/mol]
    Vcmax25 : float
        Farquhar model parameter
    alpha_p : float
        Farquhar model parameter
    VPD : float
        Vapor pressure deficit [kPa]
    **kwargs for calc_gb

    Returns
    -------
    A : float
        photosynthesis [umol m-2 s-1]
    gs : float
        stomatal conductance [mol m-2 s-1]
    Ci : float
        intracellular CO2 concentration [umol mol-1]
    Cs : float
        CO2 concentration at leaf surface [umol mol-1]
    gb : float
        boundary layer conductance [mol m-2 s-1]
    geff : float
        effective leaf conductance [mol m-2 s-1]

    """
    # Parameters
    #Farquhar model
    Kc25 = 300 # [umol mol-1] Michaelis-Menten constant for CO2, at 25 deg C
    Ko25 = 300 # [mmol mol-1] Michaelis-Menten constant for O2, at 25 deg C
    e_m = 0.08 # [mol mol-1]
    o = 210 #[mmol mol-1]
    #Leuning model
    g0 = 0.01 #[mol m-2 s-1]
    m = 4.0  #unitless

    # Adjust the Farquhar model parameters for temperature
    Vcmax = Vcmax25 * np.exp( 0.088 * (Tair - 25)) / (1 + np.exp(0.29 * (Tair - 41)))
    Kc = Kc25 * np.exp(0.074 * (Tair -25))
    Ko = Ko25 * np.exp(0.018 * (Tair - 25))

    #Calculate gamma_star and Rd
    Rd = 0.015 * Vcmax  # Dark respiration [umol m-2 s-1]
    gamma_star = (3.69 + 0.188 * (Tair - 25) + 0.0036 * (Tair -25 ) ** 2) * 10

    # equation for RuBP saturated rate of CO2 assimilation
    def calc_Ac(Vcmax, Ci, gamma_star, Kc, o, Ko, Rd):
        return Vcmax * (Ci - gamma_star)/(Ci + Kc * (1 + o / Ko)) - Rd

    # equation for RuBP limited rate of CO2 assimilation
    def calc_Aj(alpha_p, e_m, Qp, Ci, gamma_star, Rd):
        return alpha_p * e_m * Qp * (Ci - gamma_star) / (Ci + 2 * gamma_star) - Rd

    # Solve for An, gs, and Ci
    Ci = 0.99 * Ca
    Cs = Ca  # CO2 concentration at the surface
    err = 10000
    count = 0
    while (err > 0.01) & (count < 200):

        #Calculate photosynthesis
        Aj = calc_Aj(alpha_p, e_m, Qp, Ci, gamma_star, Rd)
        Ac = np.full(len(Aj),calc_Ac(Vcmax, Ci, gamma_star, Kc, o, Ko, Rd))

        A = np.minimum(Ac, Aj)

        # Calculate stomatal conductance
        gs = calc_gs_Leuning(g0, m, A, Cs, gamma_star, VPD)

        # Calculate leaf boundary layer resistance
        gb, rb = calc_gb(**kwargs)
        Cs = np.maximum(Ca - A * rb, np.full(len(A), 0.1 * Ca))
        Ci2 = Cs - A / gs
        err = max(np.abs(Ci - Ci2))
        Ci = Ci2
        count += 1

    geff = calc_geff(gb, gs)

    A[0] = A[1]
    Ci[0] = Ci[1]
    Cs[0]=Cs[1]
    gs[0]=gs[1]
    gb[0]=gb[1]
    geff[0]=geff[1]

    return A, gs, Ci, Cs, gb, geff

def calc_transpiration_leaf(VPD, Tair, geff, Press):
    """
    Calculates the water vapor source from the leaf.

    Parameters
    ----------
    VPD : float
        vapor pressure deficit [kPa]
    Tair : float
        air temperature [deg C]
    geff : float
        effective leaf conductance [mol m-2_leaf s-1]
    Press : float
        air pressure [kPa]

    Returns
    -------
    transpiration_leaf : float
        water vapor source per unit leaf area [kg s-1 m-2_leaf]
    """
    Kg = calc_Kg(Tair)  #kPa m3 kg-1
    rhov = 44.6 * Press / 101.3 * 273.15 / (Tair + 273.15)  # water vapor density, mol m-3
    transpiration_leaf = 0.4 * (geff * VPD) / (Kg * rhov)  # kg s-1 m-2_leaf

    return transpiration_leaf

def calc_respiration(Tair):
    """
    Calculates respiration.
    Based on Q10 model.

    Parameters
    ----------
    Tair : float
        Air temperature [deg C]

    Returns
    -------
    Re : float
        Respiration [umol CO2 m-2 s-1]
    """
    Tr = 10
    RE10 = 2.6
    Q10 = 2.25
    Re = RE10 * Q10 **((Tair - Tr)/Tr)
    return Re

def solve_C_closure(z, Kc, Ca, S_initial, Re, a_s, Tair, Qp, Vcmax25, alpha_p, VPD,**kwargs):

    CF = 1.15 * 1000 / 29
    Re = Re / CF
    S = S_initial / CF

    dz = z[1] - z[0]
    C = Ca

    #start iterative solution
    err = 10 ** 9
    while err > 0.0001:
        # set up coefficients for ODE
        a1 = Kc
        dKc = np.concatenate(([Kc[1]-Kc[0]], np.diff(Kc)))  # Use Kc[1]-Kc[0] for first 2 elements of dKc
        a2 = dKc / dz
        a3 = 0 * z
        a4 = S

        upd = (a1 / (dz * dz) + a2 / (2 * dz))
        dia = (-a1 * 2 / (dz * dz) + a3)
        lod = (a1 / (dz * dz) - a2 / (2 * dz))
        co = a4

        lod[0] = 0
        dia[0] = 1
        upd[0] = -1
        co[0] = Re * dz / (Kc[0] + 0.00001)
        lod[-1] = 0
        dia[-1] = 1
        upd[-1] = 0
        co[-1] = Ca[-1]

        # Use Thomas algorithm to solve
        Cn = thomas_tridiagonal(lod, dia, upd, co)
        err = np.max(np.abs(Cn - C))

        #use successive relaxations in iterations
        eps1 = 0.1
        C = (eps1 * Cn + (1 - eps1) * C)
        Ca = C
        A, gs, Ci, Cs, gb, geff = solve_leaf_physiology(Tair, Qp, Ca, Vcmax25, alpha_p, VPD, **kwargs)
        S = -A * a_s / CF

    # Fluxes are computed in umol/m2/s; Sources are computed in umol/m3/s
    Fc = -np.concatenate(Re, np.diff(Kc)) / dz
    Fc = CF * Fc
    S = S * CF

    return C, Fc, S

def calc_LAI_vertical(LADnorm, z_h_LADnorm, tot_LAI_crown, dz, h):
    """
    Creates vertical leaf area distribution.

    Parameters
    ----------
    LADnorm : array
        Vertical gradient of normalized LAD [unitless]
    z_h_LADnorm : array
        z/h for LAD [unitless: m/m]
    tot_LAI_crown : float
        total leaf area per crown area [m2_leaf m-2_crown]
    dz : float
        Vertical discretization interval [m]
    h : float
        Canopy height [m]

    Returns
    -------
    LAD: array
        Leaf area density on new vertical grid [m2leaf m-2crown m-1stem]

    """
    z_LAD = z_h_LADnorm * h  # Heights for LAD points
    dz_LAD = z_LAD[1] - z_LAD[0]
    zmin = 0
    z = np.arange(zmin, h, dz)  # New array for vertical resolution

    #Calculate LAD
    LAD = LADnorm * tot_LAI_crown / dz_LAD  #[m2leaf m-2crown m-1stem]

    # Interpolate LAD to new vertical resolution
    f = interp1d(z_LAD, LAD, bounds_error = False, fill_value='extrapolate')
    LAD_z = f(z)

    # scale so new integrated LAD matches the original total LAI per crown (corrects for interpolation error)
    LAD_z = LAD_z * tot_LAI_crown / sum(LAD_z*dz)

    return LAD_z

def calc_NHL(dz, h, Cd, U_top, ustar, PAR, Ca, Vcmax25, alpha_gs, alpha_p, total_LAI_sp, plot_area, total_crown_area_sp, mean_crown_area_sp, LADnorm, z_h_LADnorm, RH, Tair, Press, Cf=0.85, x=1, **kwargs):
    """
    Calculate NHL transpiration
    #TODO make docstring

    Parameters
    ----------
    dz : float
        Vertical discretization interval [m]
    h : float
        Canopy height [m]
    Cd : float
        drag coefficient [unitless]
    U_top : float
        Measured wind speed at top of canopy [m s-1]
    ustar : float
        friction velocity [m s-1]
    PAR : float
        photosynthetically active radiation at canopy top [umol m-2 s-1]
    Ca : float
        CO2 concentration [umol/mol]
    Vcmax25 : float
        Maximum carboxylation capacity of Rubisco at 25 deg C
    alpha_gs : float
        fitting parameter of stomatal conductance model
    total_LAI_sp : _type_
        _description_
    plot_area : _type_
        _description_
    total_crown_area_sp : _type_
        _description_
    mean_crown_area_sp : _type_
        _description_
    LADnorm : _type_
        _description_
    z_h_LADnorm : _type_
        _description_
    RH : _type_
        _description_
    Tair : _type_
        _description_
    Press : _type_
        _description_
    Cf : float, optional
        _description_, by default 0.85
    x : int, optional
        _description_, by default 1

    Returns
    -------
    _type_
        _description_
    """    """"""

    # Calculate VPD
    VPD = calc_vpd_kPa(RH, Tair = Tair)

    #Set up vertical grid
    zmin = 0
    z = np.arange(zmin, h, dz)  # [m]

    # Calculate leaf area for each vertical layer (for one tree)
    tot_LAI_crown = total_LAI_sp * plot_area / total_crown_area_sp  # LAI per crown area [m2_leaf m-2_crown]

    # Distrubute leaves vertically, and assign leaf area to stem
    LAD = calc_LAI_vertical(LADnorm, z_h_LADnorm, total_LAI_sp, dz, h) #[m2leaf m-2crown m-1stem]

    # Calculate wind speed at each layer
    U, Km = solve_Uz(z, dz, Cd , LAD , U_top, h = h)

    # Adjust the diffusivity and velocity by Ustar
    # Eqn A.5 from Mirfenderesgi et al 2016

    U = U * ustar
    Km = Km * ustar

    # Calculate radiation at each layer
    P0, Qp, zenith_angle = calc_rad_attenuation(PAR, LAD, dz, Cf = Cf, x = x, **kwargs)

    # Solve conductances
    A, gs, Ci, Cs, gb, geff = solve_leaf_physiology(Tair, Qp, Ca, Vcmax25, alpha_p, VPD = VPD, uz = U)

    # Calculate the transpiration per m-1 [ kg H2O s-1 m-1_stem]
    NHL_trans_leaf = calc_transpiration_leaf(VPD, Tair, geff, Press)  #[kg H2O m-2leaf s-1]
    NHL_trans_sp_stem = NHL_trans_leaf * LAD * mean_crown_area_sp / dz # [kg H2O s-1 m-1stem m-2ground]

    #Add data to dataset
    ds = xr.Dataset(data_vars=dict(
        U = (["z"], U),
        Km = (["z"],Km),
        P0 = (["z"], P0),
        Qp = (["z"], Qp),
        A = (["z"], A),
        gs = (["z"], gs),
        Ci = (["z"], Ci),
        Cs = (["z"], Cs),
        gb = (["z"], gb),
        geff = (["z"], geff),

        NHL_trans_leaf=(["z"], NHL_trans_leaf),
        NHL_trans_sp_stem = (["z"], NHL_trans_sp_stem),
        ),
        coords=dict(z=(["z"], z)),
        attrs=dict(description="Model output")
        )

    return ds, LAD, zenith_angle

def calc_NHL_timesteps(dz, h, Cd, met_data, Vcmax25, alpha_gs, alpha_p,
            total_LAI_spn, plot_area, total_crown_area_spn, mean_crown_area_spn, LAD_norm, z_h_LADnorm,
            lat, long, time_offset = -5, **kwargs):
    """
    Calls NHL for each timestep in the met data
    #TODO docstring

    Parameters
    ----------
    dz : _type_
        _description_
    h : _type_
        _description_
    Cd : _type_
        _description_
    met_data : _type_
        _description_
    Vcmax25 : _type_
        _description_
    alpha_gs : _type_
        _description_
    alpha_p : _type_
        _description_
    total_LAI_spn : _type_
        _description_
    plot_area : _type_
        _description_
    total_crown_area_spn : _type_
        _description_
    mean_crown_area_spn : _type_
        _description_
    LAD_norm : _type_
        _description_
    z_h_LADnorm : _type_
        _description_
    lat : _type_
        _description_
    long : _type_
        _description_
    time_offset : int, optional
        _description_, by default -5

    Returns
    -------
    _type_
        _description_
    """

    zmin = 0
    z = np.arange(zmin, h, dz)  # [m]

    NHL_tot_trans_sp_tree_all = np.empty((len(met_data)))
    zenith_angle_all = np.empty((len(met_data)))


    datasets = []
    for i in range(0,len(met_data)):
        ds, LAD, zenith_angle = calc_NHL(
            dz, h, Cd, met_data.WS_F.iloc[i], met_data.USTAR.iloc[i], met_data.PPFD_IN.iloc[i], met_data.CO2_F.iloc[i], Vcmax25, alpha_gs, alpha_p,
            total_LAI_spn, plot_area, total_crown_area_spn, mean_crown_area_spn, LAD_norm, z_h_LADnorm,
            met_data.RH.iloc[i], met_data.TA_F.iloc[i], met_data.PA_F.iloc[i], doy = met_data.Timestamp.iloc[i].dayofyear, lat = lat,
            long= long, time_offset = time_offset, time_of_day = met_data.Timestamp[i].hour + met_data.Timestamp[i].minute/60, **kwargs)

        zenith_angle_all[i] = zenith_angle
        datasets.append(ds)
    d2 = xr.concat(datasets, pd.Index(met_data.Timestamp, name="time"))
    return d2, LAD, zenith_angle_all

def calc_stem_wp_response(stem_wp, wp_s50, c3):
    """
    Calculates the restriction for NHL transpiration
    due to the hydrodynamic effects of xylem water potential.
    From eqn 2 of Mirfenderesgi et al 2016

    Parameters
    ----------
    stem_wp : float
        Stem water potential (function of z and time) [Pa]
    wp_s50 : float
        Empirical shape parameter describing the inflection point
        of the leaf stem water potential response curve [Pa]
    c3 : float
        Shape parameter for stomatal response

    Returns
    -------
    wp_response: float
        restriction for NHL transpiration
    """
    wp_response = np.exp(-((stem_wp)/wp_s50)**c3)
    return wp_response

def calc_transpiration_nhl(nhl_transpiration, stem_wp_fn, LAD):
    """Calculates transpiration for FETCH3"""
    return nhl_transpiration * stem_wp_fn * LAD

def write_outputs(output_vars, dir):
    """
    Writes NHL outputs to csv files

    Parameters
    ----------
    output_vars : _type_
        _description_
    """

    #Writes model outputs to csv files

    for var in output_vars:
        pd.DataFrame(output_vars[var]).to_csv(dir / ('nhl_' + var + '.csv'), index = False, header=False)

def write_outputs_netcdf(dir, ds):
    """
    Writes model output to netcdf file

    Parameters
    ----------
    ds : [xarray dataset]
    """

    #save dataset
    ds.to_netcdf(dir / 'nhl_out.nc')
