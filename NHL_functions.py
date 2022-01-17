from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d

def calc_esat(Tair):
    '''
    Calculates the saturation vapor pressure using the Clausius-Clapeyron equation

    Inputs:
    Tair: Air temperature [degrees C]

    Outputs:
    The saturation vapor pressure [kPa]
    '''
    #constants
    e0 = 0.611 #kPa
    T0 = 273 #K
    Rv = 461 #J K-1 kg -1, gas constant for water vapor
    Lv = 2.5 * 10**6 #J kg-1

    Tair = Tair + 273.15 #convert temperature to Kelvin
    es = e0 * np.exp((Lv/Rv)*(1/T0 - 1/Tair))
    return es

def calc_vpd_kPa(RH, Tair):
    '''
    Calculates vapor pressure deficit from air temperature and relative humidity.

    Inputs:
    RH: relative humidity [%]
    Tair: air temperature [deg C]

    Outputs:
    Vapor pressure deficit [kPa]
    '''

    #calculate esat
    es = calc_esat(Tair)
    eactual = RH*es/100

    return (es - eactual)

def calc_Kg(Tair):
    """
    Calculate the temperature-dependent conductance coefficient
    From Ewers et al 2007
    Equation A.2 from Mirfenderesgi et al 2016

    Inputs:
    Tair : air temperature [deg C]

    Outputs:
    Kg: temperature-dependent conductance coefficient [kPa m3 kg-1]
    """
    Kg = 115.8 + 0.4226 * Tair
    return Kg

def calc_mixing_length(z, h, alpha = 0.1):
    """
    Calculates the mixing length for each height in z
    Based on Poggi et al 2004
    Zero-plane displacement height is taken as (2/3)*h, appropriate for dense canopies (Katul et al 2004)

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
    Applies no-slip boundary condition: wind speed  =  0 at surface (z = 0).
    Model for turbulent diffusivity of momentum is from Poggi 2004, eqn 6

    Inputs:
    _______
    z : vector of heights [m]
    mixing_length : mixing length [d] at each height in z
    Cd : drag coefficient [unitless], assumed to be 0.2 (Katul et al 2004)
    a_s: leaf surface area [m2]
    U_top : Measured wind speed at top of canopy [m s-1]
    **kwargs to be passed to calc_mixing_length

    Outputs:
    ________
    Km : turbulent diffusivity of momentum at each height in z [m s-1]
    U : wind speed at each height in z [m s-1]
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
    Calculates the leaf boundary layer conductance and resistance, assuming laminar boundary layer

    Inputs:
    ________
    uz: wind speed at canopy height z [m s-1]
    d: characteristic leaf length [m]

    Outputs:
    ________________
    gb: leaf boundary layer conductance
    rb: leaf boundary layer resistance

    References
    ----------
    Monteith J, Unsworth M. Principles of environmental physics: plants, animals, and the atmosphere. Academic Press; 2013 Jul 26.

    """
    rb = (395 * 29 / 1150) * (d / (np.sqrt(uz ** 2) + 0.001)) ** 0.5
    gb = 1/rb
    return gb, rb

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

def calc_zenith_angle(doy, lat, long, time_offset, time_of_day):
    """
    Calculates the solar zenith angle, based on Campbell & Norman, 1998

    Inputs:
    ----------
    doy : Day of year (Ordinal day, e.g. 1 = Jan 1)
    lat : Latitude
    long : Longitude (Needs to be negative for deg W, positive for deg E)
    time_offset : Time offset [in hours] for local standard time zone, e.g, for Pacific Standard Time, time_offset = -8
    time_of_day : Time of day (hours) in local standard time

    Note: Be sure that time of day and time offset are in local standard time, not daylight savings

    Outputs:
    -------
    zenith_angle_deg : zenith angle of the sun [degrees]

    """
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

    return zenith_angle_deg

def calc_rad_attenuation(PAR, LAD, dz, alpha, Cf = 0.85, x = 1, **kwargs):
    """
    Calculates the vertical attenuation of radiation through the canopy

    Inputs:
    ----------
    PAR : photosynthetically active radiation at canopy top [umol m-2 s-1]
    LAI : Normalized leaf area index at each height in z
    Cf : Clumping fraction [unitless], assumed to be 0.85 (Forseth & Norman 1993) unless otherwise specified
    x : Ratio of horizontal to vertical projections of leaves (leaf angle distribution), assumed spherical (x=1)
    **kwargs for calc_zenith_angle

    Outputs:
    -------
    P0 : attenuation fraction of PAR penetrating the canopy at each level z [unitless]
    Qp : absorbed photosynthetically active radiation at each level within the canopy
    """
    zenith_angle = calc_zenith_angle(**kwargs)
    # Calculate the light extinction coefficient (unitless)
    xn1=np.sqrt(alpha * alpha + (np.cos(np.deg2rad(zenith_angle))) ** 2)
    xd1=(alpha + 1.774 * np.cos(np.deg2rad(zenith_angle)) * (alpha + 1.182) **(-0.733))
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
    g0 : [mol m-2 s-1]
        [cuticular conductance, residual stomatal conductance at the light compensation point (empirically fitted parameter)]
    m : [unitless]
        [empirically fitted parameter]
    A : [umol CO2 m-2 s-1]
        [net CO2 assimilation rate]
    c_s : [umol mol-1]
        [atmospheric CO2 concentration]
    gamma_star : [umol mol-1]
        [CO2 compensation point]
    VPD : [kPa]
        [VPD]
    D0 : [kPa]
        [reference vapor pressure, assumed to be 3.0 kPa]

    Returns
    -------
    gs [mol H2O m-2 s-1]
        [stomatal conductance]
    """

    gs = g0 + m * abs(A)/((c_s - gamma_star) * (1 + VPD/D0))
    return gs

def solve_leaf_physiology(Tair, Qp, Ca, Vcmax25, alpha_p, VPD, **kwargs):
    """
    Calculates photosynthesis and stomatal conductance
    Uses Leuning model for stomatal conductance

    Parameters
    ----------
    Tair : [deg C]
        [Air temperature]
    VPD : [Kpa]
        Vapor pressure deficit
    Qp : [type]
        [description]
    Ca : [type]
        CO2 concentration
    U : Wind speed of
        [description]
    Vcmax25 : [type]
        Farquhar model parameter
    alpha_p : [type]
        Farquhar model parameter
    d : [m]
        Leaf length scale for aerodynamic resistance
    D0 : [kPa]
        [reference vapor pressure, assumed to be 3.0 kPa]
    **kwargs for calc_gb

    Outputs
    -------
    A : photosynthesis [umol m-2 s-1]
    gs : stomatal conductance [mol m-2 s-1]
    Ci : intracellular CO2 concentration [umol mol-1]
    Cs : CO2 concentration at leaf surface [umol mol-1]
    gb : boundary layer conductance [mol m-2 s-1]
    geff : effective leaf conductance [mol m-2 s-1]

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
    Calculates the water vapor source from the leaf

    Parameters
    ----------
    VPD : kPa
        vapor pressure deficit
    Tair : deg C
        air :
    geff : mol m-2_leaf s-1
        effective leaf conductance
    Press : kPa
        air pressure

    Returns
    -------
    [kg s-1 m-2_leaf]
        water vapor source per unit leaf area
    """
    Kg = calc_Kg(Tair)  #kPa m3 kg-1
    rhov = 44.6 * Press / 101.3 * 273.15 / (Tair + 273.15)  # water vapor density, mol m-3
    transpiration_leaf = 0.4 * (geff * VPD) / (Kg * rhov)  # kg s-1 m-2_leaf

    return transpiration_leaf

def calc_respiration(Tair):
    """
    Calculates respiration
    Based on Q10 model

    Parameters
    ----------
    Tair : [deg C]
        Air temperature

    Returns
    -------
    [umol CO2 m-2 s-1]
        Respiration
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
    Creates vertical leaf area distribution

    Parameters
    ----------
    LAD : [type]
        Vertical gradient of normalized LAD
    z_h_LAD : [unitless: m/m]
        z/h for LAD
    dz : [m]
        Vertical discretization interval
    h : [m]
        Canopy height

    Returns
    -------
    LAD on new vertical grid [m2leaf m-2crown m-1stem]


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

    Parameters
    ----------
    z : m
        [Height vector]
    total_LAI_sp : [m2_leaf m-2_ground]
        [total LAI for each species]
    plot_area : [m2]
        [Total plot area]
    total_crown_area_sp : [m2]
    mean_crown_area_sp : [m2]
    LAD : [m2_leaf m-1]
    VPD : [kPa]
    Tair : [deg C]
    geff : mol m-2_leaf s-1
        effective leaf conductance
    Press : [kPa]
        air pressure

    Returns
    -------
    NHL transpiration
    """

    # Calculate VPD
    VPD = calc_vpd_kPa(RH, Tair = Tair)

    #Set up vertical grid
    zmin = 0
    z = np.arange(zmin, h, dz)  # [m]

    # Calculate leaf area for each vertical layer (for one tree)
    tot_LAI_crown = total_LAI_sp * plot_area / total_crown_area_sp  # LAI per crown area [m2_leaf m-2_crown]

    # Distrubute leaves vertically, and assign leaf area to stem
    LAD = calc_LAI_vertical(LADnorm, z_h_LADnorm, tot_LAI_crown, dz, h) #[m2leaf m-2crown m-1stem]

    # Calculate wind speed at each layer
    U, Km = solve_Uz(z, dz, Cd , LAD , U_top, h = h)

    # Adjust the diffusivity and velocity by Ustar
    U = U * ustar
    Km = Km * ustar

    # Calculate radiation at each layer
    P0, Qp, zenith_angle = calc_rad_attenuation(PAR, LAD, dz, alpha_gs, Cf, x, **kwargs)

    # Solve conductances
    A, gs, Ci, Cs, gb, geff = solve_leaf_physiology(Tair, Qp, Ca, Vcmax25, alpha_p, VPD = VPD, uz = U)

    # Calculate the transpiration per m-1 [ kg H2O s-1 m-1_stem]
    NHL_trans_leaf = calc_transpiration_leaf(VPD, Tair, geff, Press)  #[kg H2O m-2leaf s-1]
    NHL_trans_sp_stem = NHL_trans_leaf * LAD  # [kg H2O s-1 m-1stem m-2ground]

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
            lat, long, time_offset = -5):

    zmin = 0
    z = np.arange(zmin, h, dz)  # [m]

    NHL_tot_trans_sp_tree_all = np.empty((len(met_data)))
    zenith_angle_all = np.empty((len(met_data)))


    datasets = []
    for i in range(0,len(met_data)):
        if i%50==0:
            print('Calculating step ' + str(i))
        ds, LAD, zenith_angle = calc_NHL(
            dz, h, Cd, met_data.WS_F.iloc[i], met_data.USTAR.iloc[i], met_data.PPFD_IN.iloc[i], met_data.CO2_F.iloc[i], Vcmax25, alpha_gs, alpha_p,
            total_LAI_spn, plot_area, total_crown_area_spn, mean_crown_area_spn, LAD_norm, z_h_LADnorm,
            met_data.RH.iloc[i], met_data.TA_F.iloc[i], met_data.PA_F.iloc[i], doy = met_data.Timestamp.iloc[i].dayofyear, lat = lat,
            long= long, time_offset = time_offset, time_of_day = met_data.Timestamp[i].hour + met_data.Timestamp[i].minute/60)

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
    stem_wp : [type]
        Stem water potential (function of z and time) [Pa]
    wp_s50 : [type]
        Empirical shape parameter describing the inflection point
        of the leaf stem water potential response curve [Pa]
    c3 : [type]
        Shape parameter for stomatal response

    Returns
    -------
    [type]
        [description]
    """
    wp_response = np.exp(-((stem_wp)/wp_s50)**c3)
    return wp_response

def calc_transpiration_nhl(nhl_transpiration, stem_wp_fn, LAD):
    return nhl_transpiration * stem_wp_fn * LAD

def write_outputs(output_vars):

    #Writes model outputs to csv files

    working_dir = Path.cwd()

    # make output directory if one doesn't exist
    (working_dir /'output').mkdir(exist_ok=True)

    for var in output_vars:
        pd.DataFrame(output_vars[var]).to_csv(working_dir / 'output' / ('nhl_' + var + '.csv'), index = False, header=False)

def write_outputs_netcdf(ds):
    """
    Writes model output to netcdf file

    Parameters
    ----------
    ds : [xarray dataset]
    """



    working_dir = Path.cwd()

    # make output directory if one doesn't exist
    (working_dir /'output').mkdir(exist_ok=True)

    #save dataset
    ds.to_netcdf(working_dir / 'output' /  'nhl_out.nc')
