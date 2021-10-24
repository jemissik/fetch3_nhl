# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:10:31 2019

@author: mdef0001
"""
import time
start = time.time()
#importing libraries
import numpy as np
from numpy.core.numeric import True_
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
import scipy
from numpy.linalg import multi_dot

from FETCH2_loading_LAD import *
from met_data import *
from jarvis import *
from canopy import *

'''
#importing variables
from FETCH2_loading_LAD import params, working_dir, dt0, dt, tmax, dz, nz, nz_r, nz_s, z, z_soil, z_upper, \
    nz_clay, nz_sand, q_rain, step_time, Head_bottom_H, H_initial, SW_in,  \
    f_Ta_2d, f_s_2d, f_d_2d, VPD_2d, NET_2d, delta_2d, LAD
'''
##### these definitions are temporary until other parts of code are restructured
hx50 = params['hx50']
ga = params['ga']
gama = params['gama']
lamb = params['lamb']
Cp = params['Cp']
gb = params['gb']
gsmax = params['gsmax']
nl = params['nl']
Emax = params['Emax']
UpperBC = params['UpperBC']
BottomBC = params['BottomBC']
theta_1_clay = params['theta_1_clay']
theta_2_clay = params['theta_2_clay']
theta_1_sand = params['theta_1_sand']
theta_2_sand = params['theta_2_sand']
#####

############## inital condition #######################

#setting profile for initial condition
if BottomBC==0:
    H_initial[0]=Head_bottom_H[0]


##############Temporal discritization according to MODEL resolution
t_num = np.arange(0,tmax+dt0,dt0)         #[s]
nt = len(t_num)  #number of time steps
########################################

#function for stem xylem: K and C
def Porous_media_xylem(arg,params,i):

    #arg= potential [Pa]
    cavitation_xylem=np.zeros(shape=(len(arg)))

    for i in np.arange(0,len(cavitation_xylem),1):
        if arg[i]>0:
            cavitation_xylem[i]=1
        else:
            cavitation_xylem[i]=(1-1/(1+np.exp(params['ap']*(arg[i]-params['bp']))))

    #Index Ax/As - area of xylem per area of soil
    #kmax = m/s
    K=params['kmax']*params['Aind_x']*cavitation_xylem

    #CAPACITANCE FUNCTION AS IN BOHRER ET AL 2005
    C=np.zeros(shape=len(z[nz_r:nz]))

    C=((params['Aind_x']*params['p']*params['sat_xylem'])/(params['Phi_0']))*((params['Phi_0']-arg)/params['Phi_0'])**(-(params['p']+1))

    return C,K, cavitation_xylem
########################################################################################

#function for root xylem: K and C
def Porous_media_root(arg,params,dz,theta):
     #arg= potential (Pa)
    stress_kr=np.zeros(shape=(len(arg)))

    for i in np.arange(0,len(stress_kr),1):
        if arg[i]>0:
            stress_kr[i]=1
        else:
            stress_kr[i]=(1-1/(1+np.exp(params['ap']*(arg[i]-params['bp']))))  #CAVITATION CURVE FOR THE ROOT XYLEM


    #Index Ar/As - area of root xylem per area of soil
    #considered 1 following VERMA ET AL 2014 {for this case}

    #Keax = effective root axial conductivity
    K=params['Ksax']*params['Aind_r']*stress_kr #[m2/s Pa]

    #KEEPING CAPACITANCE CONSTANT - using value according to VERMA ET AL., 2014
    C=np.zeros(shape=nz_r-nz_s)
    #C[:]=1.1*10**(-11)  #(1/Pa)

    #CAPACITANCE FUNCTION AS IN BOHRER ET AL 2005

    #considering axial area rollowing basal area [cylinder]
    C=((params['Aind_r']*params['p']*params['sat_xylem'])/(params['Phi_0']))*((params['Phi_0']-arg)/params['Phi_0'])**(-(params['p']+1))

    return C, K, stress_kr

###############################################################################

#vanGenuchten for soil K and C
def vanGenuchten(arg,params,z):

    #arg = potential from Pascal to meters
    theta=np.zeros(shape=len(arg))
    arg=((arg)/(params['g']*params['Rho']))   #m

    Se=np.zeros(shape=len(arg))
    K=np.zeros(shape=len(arg))
    C=np.zeros(shape=len(arg))
    #considering l = 0.5

    for i in np.arange(0,len(arg),1):
        if z[i]<=params['clay_d'] : #clay_d=4.2m for verma

            if arg[i]<0:
            #Compute the volumetric moisture content
                theta[i] = (params['theta_S1'] - params['theta_R1'])/((1 + (params['alpha_1']*abs(arg[i]))**params['n_1'])**params['m_1']) + params['theta_R1']  #m3/m3
            #Compute the effective saturation
                Se[i] = ((theta[i] - params['theta_R1'])/(params['theta_S1'] - params['theta_R1'])) ## Unitless factor
            #Compute the hydraulic conductivity
                K[i]=params['Ksat_1']*Se[i]**(1/2)*(1 - (1 - Se[i]**(1/params['m_1']))**params['m_1'])**2   # van genuchten Eq.8 (m/s) #
            if arg[i]>=0:
                theta[i]=params['theta_S1']
                K[i]=params['Ksat_1']

            C[i]=((-params['alpha_1']*np.sign(arg[i])*params['m_1']*(params['theta_S1']-params['theta_R1']))/(1-params['m_1']))*Se[i]**(1/params['m_1'])*(1-Se[i]**(1/params['m_1']))**params['m_1']

        if z[i]>params['clay_d']: #sand

            if arg[i]<0:
             #Compute the volumetric moisture content
                theta[i] = (params['theta_S2'] - params['theta_R2'])/((1 + (params['alpha_2']*abs(arg[i]))**params['n_2'])**params['m_2']) + params['theta_R2']  #m3/m3
            #Compute the effective saturation
                Se[i] = ((theta[i] - params['theta_R2'])/(params['theta_S2'] - params['theta_R2'])) ## Unitless factor
            #Compute the hydraulic conductivity
                K[i]=params['Ksat_2']*Se[i]**(1/2)*(1 - (1 - Se[i]**(1/params['m_2']))**params['m_2'])**2   # van genuchten Eq.8 (m/s) #
            if arg[i]>=0:
                theta[i]=params['theta_S2']
                K[i]=params['Ksat_2']

            C[i]=((-params['alpha_2']*np.sign(arg[i])*params['m_2']*(params['theta_S2']-params['theta_R2']))/(1-params['m_2']))*Se[i]**(1/params['m_2'])*(1-Se[i]**(1/params['m_2']))**params['m_2']




    K=(K/(params['Rho']*params['g'])) # since H is in Pa
    C=(C/(params['Rho']*params['g'])) # since H is in Pa


    return C, K,theta, Se

###############################################################################

###############################################################################

def Picard(H_initial):
    #picard iteration solver, as described in the supplementary material
    #solution following Celia et al., 1990

    # Stem water potential [Pa]

    ######################################################################


    # Define matrices that we’ll need in solution(similar to celia et al.[1990])
    x=np.ones(((nz-1),1))
    DeltaPlus  = np.diagflat(-np.ones((nz,1))) + np.diagflat(x,1)  # delta (i+1) -delta(i)

    y=-np.ones(((nz-1,1)))
    DeltaMinus = np.diagflat(np.ones((nz,1))) + np.diagflat(y,-1)    #delta (i-1) - delta(i)

    p=np.ones(((nz-1,1)))
    MPlus  = np.diagflat(np.ones((nz,1))) + np.diagflat(p,1)

    w=np.ones((nz-1,1))
    MMinus = np.diagflat(np.ones((nz,1))) + np.diagflat(w,-1)

    ############################Initializing the pressure heads/variables ###################
    #only saving variables EVERY HALF HOUR
    dim=np.mod(t_num,1800)==0
    dim=sum(bool(x) for x in dim)

    H = np.zeros(shape=(nz,dim)) #Stem water potential [Pa]
    trans_2d=np.zeros(shape=(len(z_upper),dim))
    K=np.zeros(shape=(nz,dim))
    Capac=np.zeros(shape=(nz,dim))
    S_kx=np.zeros(shape=(nz-nz_r,dim))
    S_kr=np.zeros(shape=(nz_r-nz_s,dim))
    S_sink=np.zeros(shape=(nz_r-nz_s,dim))
    Kr_sink=np.zeros(shape=(nz_r-nz_s,dim))
    THETA=np.zeros(shape=(nz_s,dim))
    EVsink_ts=np.zeros(shape=((nz_r-nz_s),dim))
    infiltration=np.zeros(shape=dim)


    f_leaf_2d=np.zeros(shape=(len(z_upper),nt))
    Pt_2d=np.zeros(shape=(len(z_upper),nt))
    gs_2d=np.zeros(shape=(len(z_upper),nt))
    gc_2d=np.zeros(shape=(len(z_upper),nt))

    S_stomata=np.zeros(shape=(len(z[nz_r:nz]),nt))
    S_S=np.zeros(shape=(nz,nt))
    theta=np.zeros(shape=(nz_s))
    Se=np.zeros(shape=(nz_s,nt))
    Kr=np.zeros(shape=(nz_r-nz_s))

    #H_initial = inital water potential [Pa]
    H[:,0] = H_initial[:]

#################################### ROOT MASS DISTRIBUTION FORMULATION ############################################

    #root mass distribution following VERMA ET AL 2O14

    z_dist=np.arange(0,params['Root_depth']+dz,dz)
    z_dist=np.flipud(z_dist)

    r_dist=(np.exp(params['qz']-((params['qz']*z_dist)/params['Root_depth']))*params['qz']**2*(params['Root_depth']-z_dist))/(params['Root_depth']**2*(1+np.exp(params['qz'])*(-1+params['qz'])))


####################################################################################################################################



   #INITIALIZING THESE VARIABLES FOR ITERATIONS
    cnp1m=np.zeros(shape=(nz))
    knp1m=np.zeros(shape=(nz))
    stress_kx=np.zeros(shape=(nz-nz_r))
    stress_kr=np.zeros(shape=(nz_r-nz_s))
    stress_roots=np.zeros(shape=(nz_r-nz_s))
    deltam=np.zeros(shape=(nz))


    #vector for adding potentials in B matrix
    TS=np.zeros(shape=(nz_r))

    niter = 0
    sav=0


    for i in np.arange(1,nt,1):
        #use nt for entire period

        # Initialize the Picard iteration solver - saving variables every half-hour
        if i==1:
            hn=H[:,0] #condition for initial conditions
        else:
            hn=hnp1mp1 #condition for remaining time steps

        hnp1m = hn


        # Define a dummy stopping variable
        stop_flag = 0

        # Define an iteration counter


        while(stop_flag==0):
        #=========================== above-ground xylem ========================
             # Get C,K,for soil, roots, stem

            #VanGenuchten relationships applied for the soil nodes
            cnp1m[0:nz_s], knp1m[0:nz_s],theta[:], Se[:,i]=vanGenuchten(hnp1m[0:nz_s],params,z_soil)

            #Equations for C, K for the root nodes
            cnp1m[nz_s:nz_r],knp1m[nz_s:nz_r],stress_kr[:] = Porous_media_root(hnp1m[nz_s:nz_r],params,dz,theta[nz_s-(nz_r-nz_s):nz_s])

            #Equations for C, K for stem nodes
            cnp1m[nz_r:nz],knp1m[nz_r:nz], stress_kx[:] = Porous_media_xylem(hnp1m[nz_r:nz],params,i)


            #% Compute the individual elements of the A matrix for LHS


            C=np.diagflat(cnp1m)

            #interlayer hydraulic conductivity - transition between roots and stem
            #calculated as a simple average
            knp1m[nz_r]=(knp1m[nz_r-1]+knp1m[nz_r])/2

            #interlayer between clay and sand
            knp1m[nz_clay]=(knp1m[nz_clay]+knp1m[nz_clay+1])/2

            #equation S.17
            kbarplus = (1/2)*np.matmul(MPlus,knp1m)  #1/2 (K_{i} + K_{i+1})

            kbarplus[nz-1]=0    #boundary condition at the top of the tree : no-flux
            kbarplus[nz_s-1]=0  #boundary condition at the top of the soil

            Kbarplus =np.diagflat(kbarplus)

            #equation S.16
            kbarminus = (1/2)*np.matmul(MMinus,knp1m)  #1/2 (K_{i-1} - K_{i})

            kbarminus[0]=0    #boundary contition at the bottom of the soil
            kbarminus[nz_s]=0 #boundary contition at the bottom of the roots : no-flux

            Kbarminus = np.diagflat(kbarminus)

            ##########ROOT WATER UPTAKE TERM ############################
            stress_roots=np.zeros(shape=(len(z[nz_s-(nz_r-nz_s):nz_s])))

            #FEDDES root water uptake stress function
            #parameters from VERMA ET AL 2014: Equations S.73, 74 and 75 supplementary material

            #clay
            for k,j in zip(np.arange(nz_s-(nz_r-nz_s),nz_clay+1,1),np.arange(0,((len(stress_roots-1))-(nz_sand-nz_clay)),1)): #clay
                if theta[k]<=theta_1_clay:
                    stress_roots[j]=0
                if theta_1_clay < theta[k] and theta[k]<= theta_2_clay:
                    stress_roots[j]=(theta[k]-theta_1_clay)/(theta_2_clay-theta_1_clay)
                if theta[k] > theta_2_clay:
                    stress_roots[j]=1
            #sand
            for k,j in zip(np.arange(nz_clay+1,nz_s,1),np.arange(len(stress_roots)-(nz_sand-nz_clay),len(stress_roots),1)): #sand
               if theta[k]<=theta_1_sand:
                    stress_roots[j]=0
               if theta_1_sand < theta[k] and theta[k] <=theta_2_sand:
                    stress_roots[j]=(theta[k]-theta_1_sand)/(theta_2_sand-theta_1_sand)
               if theta[k] > theta_2_sand:
                    stress_roots[j]=1


            #specific radial conductivity under saturated soil conditions
            Ksrad=stress_roots*params['Kr'] #stress function is unitless

            #effective root radial conductivity
            Kerad=Ksrad*r_dist  #[1/sPa] #Kr is already divided by Rho*g

            #effective root radial conductivity
            Kr=Kerad


#######################################################################
            #tridiagonal matrix
            A = (1/dt0)*C - (1/(dz**2))*(np.dot(Kbarplus,DeltaPlus) - np.dot(Kbarminus,DeltaMinus))



            #Infiltration calculation - only infitrates if top soil layer is not saturated
            #equation S.53
            if UpperBC==0:
                q_inf=min(q_rain[i],
                                ((params['theta_S2']-theta[-1])*(dz/dt0))) #m/s


################################## SINK/SOURCE TERM ON THE SAME TIMESTEP #####################################
              #equation S.22 suplementary material
            if params['Root_depth']==params['Soil_depth']:
                #diagonals
                for k,e in zip(np.arange(0,nz_s,1),np.arange(0,(nz_r-nz_s),1)):
                    A[k,k]=A[k,k]-Kr[e] #soil ---  from 0:soil top
                for j, w in zip(np.arange(nz_s,nz_r,1),np.arange(0,(nz_r-nz_s),1)):
                    A[j,j]=A[j,j]-Kr[w] #root ---- from soil bottom :root top

                #terms outside diagonals
                for k , j,e in zip(np.arange((nz_r-nz_s),nz_r,1),np.arange((0),nz_s,1),np.arange(0,(nz_r-nz_s),1)):
                    A[j,k]=+Kr[e] #root

                for k, j,e in zip(np.arange(nz_s,nz_r,1),np.arange(0,nz_s,1),np.arange(0,(nz_r-nz_s),1)):
                    A[k,j]=+Kr[e] #soil


                 #residual for vector Right hand side vector
                TS[0:nz_s]=-Kr*(hnp1m[0:nz_s]-hnp1m[nz_s:nz_r]) #soil
                TS[(nz_s):nz_r]=+Kr*(hnp1m[0:nz_s]-hnp1m[nz_s:nz_r]) #root


            else:
                 #diagonals
                for k,e in zip(np.arange(nz_s-(nz_r-nz_s),nz_s,1),np.arange(0,(nz_r-nz_s),1)):
                    A[k,k]=A[k,k]-Kr[e] #soil --- (soil-roots):soil
                for j, w in zip(np.arange(nz_s,nz_r,1),np.arange(0,(nz_r-nz_s),1)):
                    A[j,j]=A[j,j]-Kr[w] #root ---- soil:root

                #terms outside diagonals
                for k , j,e in zip(np.arange(nz_s,nz_r,1),np.arange(nz_s-(nz_r-nz_s),nz_s,1),np.arange(0,(nz_r-nz_s),1)):
                    A[j,k]=+Kr[e] #root

                for k, j,e in zip(np.arange(nz_s,nz_r,1),np.arange(nz_s-(nz_r-nz_s),nz_s,1),np.arange(0,(nz_r-nz_s),1)):
                    A[k,j]=+Kr[e] #soil


                #residual for vector Right hand side vector
                TS[nz_s-(nz_r-nz_s):nz_s]=-Kr*(hnp1m[nz_s-(nz_r-nz_s):nz_s]-hnp1m[nz_s:nz_r]) #soil
                TS[(nz_s):nz_r]=+Kr*(hnp1m[nz_s-(nz_r-nz_s):nz_s]-hnp1m[nz_s:nz_r]) #root


########################################################################################################

            ##########TRANSPIRATION FORMULATION #################
            Pt_2d[:,i] = calc_transpiration(SW_in[i], NET_2d[:,i], delta_2d[i], Cp, VPD_2d[:,i], lamb,
                                            gb, ga, f_Ta_2d[:,i], f_s_2d[:,i], f_d_2d[:,i], jarvis_fleaf(hn[nz_r:nz]))

            #SINK/SOURCE ARRAY : concatenating all sinks and sources in a vector
            S_S[:,i]=np.concatenate((TS,-Pt_2d[:,i])) #vector with sink and sources



            #dummy variable to help breaking the multiplication into parts
            matrix2=multi_dot([Kbarplus,DeltaPlus,hnp1m]) - multi_dot([Kbarminus,DeltaMinus,hnp1m])

            #% Compute the residual of MPFD (right hand side)

            R_MPFD = (1/(dz**2))*(matrix2) + (1/dz)*params['Rho']*params['g']*(kbarplus - kbarminus) - (1/dt0)*np.dot((hnp1m - hn),C)+(S_S[:,i])

            #bottom boundary condition - known potential - \delta\Phi=0
            if BottomBC==0:
                A[1,0]=0
                A[0,1]=0
                A[0,0]=1
                R_MPFD[0]=0



            if UpperBC==0:  #adding the infiltration on the most superficial soil layer [1/s]
                R_MPFD[nz_s-1]=R_MPFD[nz_s-1]+(q_inf)/dz

            if BottomBC==2: #free drainage condition: F1-1/2 = K at the bottom of the soil
                R_MPFD[0]=R_MPFD[0]-(kbarplus[0]*params['Rho']*params['g'])/dz


            #Compute deltam for iteration level m+1 : equations S.25 to S.41 (matrix)
            deltam = np.dot(linalg.pinv2(A),R_MPFD)


            if  np.max(np.abs(deltam[:])) < params['stop_tol']:  #equation S.42
                stop_flag = 1
                hnp1mp1 = hnp1m + deltam

                #Bottom boundary condition at bottom of the soil
                #setting for the next time step value for next cycle
                if BottomBC==0:
                    hnp1mp1[0] = Head_bottom_H[i]

                #saving output variables only every 30min
                if np.mod(t_num[i],1800)==0:
                    sav=sav+1

                    H[:,sav] = hnp1mp1 #saving potential
                    trans_2d[:,sav]=Pt_2d[:,i] #1/s
                    hsoil=hnp1mp1[nz_s-(nz_r-nz_s):nz_s]
                    hroot=hnp1mp1[(nz_s):(nz_r)]
                    EVsink_ts[:,sav]=-Kr[:]*(hsoil-hroot)  #sink term soil #saving

                    #saving output variables
                    K[:,sav]=knp1m
                    THETA[:,sav]=theta
                    Capac[:,sav]=cnp1m
                    S_kx[:,sav]=stress_kx
                    S_kr[:,sav]=stress_kr
                    S_sink[:,sav]=stress_roots
                    Kr_sink[:,sav]=Kr

                    if UpperBC==0 and q_rain[i]>0:
                        infiltration[sav]=q_inf
                niter=niter+1

                if params['print_run_progress']:
                    if (niter % 50) == 0:
                        print("calculated time steps",niter)

            else:
                hnp1mp1 =hnp1m + deltam
                hnp1m = hnp1mp1




    return H*(10**(-6)), K,S_stomata,theta, S_kx, S_kr,C,Kr_sink, Capac, S_sink,EVsink_ts,THETA, infiltration,trans_2d

H,K,S_stomata,theta, S_kx, S_kr,C,Kr_sink, Capac, S_sink,EVsink_ts, THETA, infiltration,trans_2d = Picard(H_initial)  #calling the function



######################################################################################

####################### Water balance ###################################

theta_i=sum(THETA[:,1]*dz)
theta_t=sum(THETA[:,-1]*dz)
theta_tot=theta_i-theta_t  #(m)
theta_tot=theta_tot*1000  #(mm)

infilt_tot=sum(infiltration)*dt*1000 #mm
if UpperBC==0:
    theta_tot=(theta_tot)+infilt_tot
############################

EVsink_total=np.zeros(shape=(len(EVsink_ts[0])))
for i in np.arange(1,len(EVsink_ts[0]),1):
    EVsink_total[i]=sum(-EVsink_ts[:,i]*dz)  #(1/s) over the simulation times dz [m]= m

root_water=sum(EVsink_total)*1000*dt #mm
#############################

transpiration_tot=sum(sum(trans_2d))*1000*dt*dz ##mm

df_waterbal = pd.DataFrame(data={'theta_i':theta_i,
               'theta_t':theta_t, 'theta_tot':theta_tot, 'infilt_tot':infilt_tot,
                'root_water':root_water, 'transpiration_tot':transpiration_tot}, index = [0])

#summing during all time steps and multiplying by 1000 = mm  #
#the dt factor is accounting for the time step - to the TOTAl and not the rate


############################################################################
df_time = pd.DataFrame(data=step_time.index.values,index=step_time)

#########################################################


d = {'trans':(sum(trans_2d[:,:]*dz)*1000)} #mm/s
df_EP = pd.DataFrame(data=d,index=step_time[:])

trans_h=dt*df_EP['trans'].resample('60T').sum() # hourly accumulated simulated transpiration


#output
output_vars = {'H':H, 'K': K, 'S_stomata':S_stomata, 'theta':theta, 'S_kx':S_kx, 'S_kr':S_kr, 'C':C,
               'Kr_sink':Kr_sink, 'Capac':Capac, 'EVsink_ts':EVsink_ts,
               'THETA':THETA, 'infiltration':infiltration, 'trans_2d':trans_2d,'EVsink_total':EVsink_total}

for var in output_vars:
    pd.DataFrame(output_vars[var]).to_csv(working_dir / 'output' / (var + '.csv'), index = False, header=False)

df_waterbal.to_csv(working_dir / 'output' / ('df_waterbal' + '.csv'), index=False, header=True)
df_EP.to_csv(working_dir / 'output' / ('df_EP' + '.csv'), index=False, header=True)


print(f"run time: {time.time() - start} s")