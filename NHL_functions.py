import numpy as np

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


	return e0 * np.exp((Lv/Rv)*(1/T0 - 1/Tair))

def calc_vpd_kPa(Tair, RH):
	'''
    Calculates vapor pressure deficit from air temperature and relative humidity.

	Inputs:
	Tair: Air temperature [degrees C]
	RH: relative humidity [%]

	Outputs:
	Vapor pressure deficit [kPa]
	'''

	#calculate esat
	es = calc_esat(Tair)
	eactual = RH*es/100

	return (es - eactual)

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

def calc_gb(uz, d = 0.0015):
    """
    Calculates the leaf boundary layer conductance and resistance, assuming laminar boundary layer 
    TODO See Monteith & Unsworth 2013? or another source? 
    TODO Need to make separate functions for water vapor and CO2 

    Inputs:
    ________
    uz: wind speed at canopy height z [m s-1]
    d: characteristic leaf length [m]

    Outputs:
    ________________
    gb: leaf boundary layer conductance [TODO units??]
    rb: leaf boundary layer resistance [TODO units??]
    
    References
    ----------
    Monteith J, Unsworth M. Principles of environmental physics: plants, animals, and the atmosphere. Academic Press; 2013 Jul 26.

    """
    rb = (395 * 29 / 1150) * (d / (np.sqrt(uz ** 2) + 0.001)) ** 0.5  #TODO change sqrt (u^2) part 
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

    Outputs:
    -------
    zenith_angle_deg : zenith angle of the sun [degrees]

    """
    # Calculate the standard meridian (in degrees) from the time zone offset 
    standard_meridian = time_offset * 15 
    
    # Calculate the solar declination angle, Eqn 11.2, Campbell & Norman
    declination_angle_rad = np.arcsin(0.39785 * np.sin(np.deg2rad(278.97 + 0.9856 * doy + 1.9165 * np.sin(np.deg2rad(356.6 + 0.9856 * doy)))))

    # Calculate the equation of time, Eqn 11.4, Campbell & Norman
    f = np.deg2rad(279.575 + 0.98565) # in radians
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
                                 + np.cos(lat_rad) * np.cos(declination_angle_rad) * np.cos(np.pi/12 * (time_of_day - t0)))
    zenith_angle_deg = np.rad2deg(zenith_angle_rad)

    return zenith_angle_deg

def calc_rad_attenuation(PAR, zenith_angle, LAI, Cf = 0.85, x = 1):
    """
    Calculates the vertical attenuation of radiation through the canopy

    Inputs:
    ----------
    PAR : photosynthetically active radiation at canopy top [umol m-2 s-1]
    zenith_angle
    LAI : Normalized leaf area index at each height in z
    Cf : Clumping fraction [unitless], assumed to be 0.85 (Forseth & Norman 1993) unless otherwise specified
    x : Ratio of horizontal to vertical projections of leaves (leaf angle distribution), assumed spherical (x=1)

    Outputs:
    -------
    P0 : attenuation fraction of PAR penetrating the canopy at each level z [unitless]
    Qp : absorbed photosynthetically active radiation at each level within the canopy
    # TODO MATLAB version has both LAI and LAD in parameters, only LAD is used. What are the correct units? 
    # TODO MATLAB version flips the arrays... why? needs to be done here? 
    """
    # Calculate the light extinction coefficient (unitless)
    k = (((x**2 + np.tan(np.deg2rad(zenith_angle)) ** 2) ** 0.5) * np.cos(np.deg2rad(zenith_angle))) / (x + 1.744 * (x + 1.182) ** 0.773)

    # Calculate P0 and Qp
    P0 = np.exp(k * LAI * Cf)
    Qp = P0 * PAR
    
    return P0, Qp

def calc_gs_Leuning(g0, m, A, c_s, gamma_star, VPD, D0 = 3):
    """
    [Calculates gs according to Leuning 1995 
    TODO check units for gs of CO2 vs H2O.. is there some multiplier?]

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
        [VPD TODO Should be the humidity deficit at the leaf surface]
    D0 : [kPa]
        [reference vapor pressure, assumed to be 3.0 kPa]

    Returns
    -------
    gs [mol H2O m-2 s-1]
        [stomatal conductance]
    """
    
    gs = g0 + m * A/((c_s - gamma_star)(1 - VPD/D0))
    return gs

def solve_leaf_physiology(Tair, VPD, Qp, Ca, U, Vcmax25, alpha_p, d = 0.0015, D0 = 3):
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
    while (max(err) > 0.01) & count < 200: 
        
        #Calculate photosynthesis
        Ac = calc_Ac(Vcmax, Ci, gamma_star, Kc, o, Ko, Rd)
        Aj = calc_Aj(alpha_p, e_m, Qp, Ci, gamma_star, Rd)
        A = min(Ac, Aj)
        
        # Calculate stomatal conductance 
        gs = calc_gs_Leuning(g0, m, A, Cs, gamma_star, VPD, D0 = D0)
        
        # Calculate leaf boundary layer resistance 
        gb, rb = calc_gb(U, d)
        Cs = max(Ca - A * rb, 0.1 * Ca)
        Ci2 = Cs - A / gs
        err = np.abs(Ci - Ci2)
        Ci = Ci2
        count += 1
    
    geff = calc_geff(gb, gs)  # TODO The same conductance is used for gb,co2 and gb,h2o. need to fix. 
    
    A[0] = A[1]
    Ci[0] = Ci[1]
    Cs[0]=Cs[1]
    gs[0]=gs[1]
    gb[0]=gb[1]
    geff[0]=geff[1]

    return A, gs, Ci, Cs, gb, geff