from FETCH2_loading_LAD import *
#######################################################################
#LEAF AREA DENSITY FORMULATION (LAD) [1/m]
#######################################################################
#Simple LAD formulation to illustrate model capability
#following Lalic et al 2014
####################
def calc_LAD(z_Above):

    z_LAD=z_Above[1:]
    LAD=np.zeros(shape=(int(params['Hspec']/dz)))  #[1/m]

    #LAD function according to Lalic et al 2014
    for i in np.arange(0,len(z_LAD),1):
        if  0.1<=z_LAD[i]<params['z_m']:
            LAD[i]=params['L_m']*(((params['Hspec']-params['z_m'])/(params['Hspec']-z_LAD[i]))**6)*np.exp(6*(1-((params['Hspec']-params['z_m'])/(params['Hspec']-z_LAD[i]))))
        if  params['z_m']<=z_LAD[i]<params['Hspec']:
            LAD[i]=params['L_m']*(((params['Hspec']-params['z_m'])/(params['Hspec']-z_LAD[i]))**0.5)*np.exp(0.5*(1-((params['Hspec']-params['z_m'])/(params['Hspec']-z_LAD[i]))))
        if z_LAD[i]==params['Hspec']:
            LAD[i]=0
        return LAD
LAD = calc_LAD(z_Above)