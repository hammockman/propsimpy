"""
An attempt to recreate Theodor Schmidt's PropSim program.

It compares OK against the two published test cases. Not exact, but not far off.


TODO:
* generalize to arbitrary number of elements (DONE) at arbitrary radii
* compare results against the two examples in the papers
* refactor
* get CD from XFOIL data


Notes:
* it appears the BASIC compiler only looked at the first 2 chars of any vaiariable name so SW==SWPTAREA etc

REF:
Schmidt, T. A Simple Program for Propeller-Performance Prediction. Human Power Vol. 7 No. 2, (1988).
Schmidt, T. Propeller simulation with PropSim. Human Power 47:3--7 (1999)

jh, Mar 2022
"""

import numpy as np


#### TEST1

Diameter = 0.5 # m
Ptch = 0.66 # m
U = 2 # m/s
R1 = 100 # rpm
R2 = 300 # rpm
Incr = 10 # rpm
Blades = 2
Range = "N" # limit top rpm to low blade loadings
CH = np.array([0.01, 0.075, 0.085, 0.09, 0.09, 0.085, 0.076, 0.067, 0.054]) # m, CH[0] near hub
HubDiaRatio = 0.1
nelem = 9



#### TEST2

Diameter = 0.4
Ptch = 0.46
U = 0.5
R1 = 100 # rpm
R2 = 300 # rpm
Incr = 10 # rpm
Blades = 2
RCH = np.array([0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25])
CH = np.array([0.075, 0.085, 0.09, 0.09, 0.085, 0.076, 0.067, 0.054, 0]) # m, CH[0] near hub
HubDiaRatio = 0.1
nelem = 9



#### ICIELA

Diameter = 15*25.4/1000.
Ptch = 12*25.4/1000.
#U = 3.1 # m/s = 6 knots
U = 4.0
#U = 0.01
R1 = 100 # rpm
R2 = 1500 # rpm
Incr = 100 # rpm
Blades = 3
HubDiaRatio = 0.13
# chord data estimated from imagery see ipynb
#nelem = 9
#CH = np.array([0.11830418, 0.09045039, 0.09747757, 0.11830382, 0.13603253, 0.14741299, 0.14789966, 0.13580973, 0.09026126])
nelem = 19
CH = np.array([
    0.13441003, 0.1054195 , 0.09263512, 0.09083029, 0.09256338,
    0.09970571, 0.10973234, 0.11933387, 0.12808184, 0.13603253,
    0.14144855, 0.14681655, 0.14840396, 0.14799135, 0.14757875,
    0.13942701, 0.12551326, 0.10463595, 0.06081509])


def main(Diameter=Diameter,
         Ptch=Ptch,
         HubDiaRatio=HubDiaRatio,
         CH=CH,
         U=U,
         R1=R1,
         R2=R2,
         Incr=Incr,
         Blades=Blades,
         Range=False,
         Header=False,
         nelem = nelem):

    # sor is my innovation to help convergence - even though non-convergence was caused by bugs...
    #sor = 0.99 # factor controlling how slowly slip gets updated (0 = original replacement, 1 = will never update)
    sor = 0


    from scipy.interpolate import interp1d, interp2d
    FF = interp2d(
        np.linspace(0, 1.5, 16),
        (45e3, 50e3, 60e3, 70e3, 80e3, 90e3, 100e3, 125e3, 150e3, 175e3, 200e3, 2e6, 4e6, 6e6),
        np.array([ # 14 x 16
            [ 25  , 23  , 27  , 50  , 70  , 80  , 80  , 80  , 75  , 70  , 60  , 50, 50  , 200, 1000, 1000], # Re<45,000
            [ 24  , 22  , 25  , 40  , 50  , 55  , 57  , 58  , 55  , 50  , 45  , 42, 46  , 180, 1000, 1000], # Re~50,000
            [ 23  , 21  , 23  , 35  , 40  , 46  , 46  , 46  , 46  , 46  , 42  , 38, 45  , 150,  800, 1000], # Re~60,000
            [ 22  , 21  , 22  , 31  , 35  , 39  , 39  , 39  , 39  , 39  , 37  , 35, 41  , 140,  700, 1000], # Re~70,000
            [ 22  , 21  , 21  , 27  , 30  , 33  , 33  , 33  , 33  , 33  , 32  , 31, 37  , 130,  700, 1000], # Re~80,000
            [ 21  , 21  , 21  , 23  , 25  , 27  , 27  , 27  , 27  , 27  , 27  , 27, 33  , 120,  600, 1000], # Re~90,000
            [ 20  , 20  , 20  , 20  , 19.9, 19.5, 19  , 19  , 19.1, 19.5, 20.2, 23, 30  , 100,  600, 1000], # Re~100,000
            [ 19  , 16.8, 18.6, 18.3, 18  , 17.5, 17.2, 17.3, 17.8, 18.3, 19.2, 21, 28  ,  84,  600, 1000], # Re~125,000
            [ 18  , 17.7, 17.2, 16.5, 15.6, 15  , 14.6, 14.8, 15.3, 16.2, 17.3, 20, 25  ,  80,  500, 1000], # Re~150,000
            [ 17.5, 16.3, 15  , 13.6, 12.7, 12  , 12  , 12.4, 13.6, 14.6, 16  , 18, 24  ,  60,  400, 1000], # Re~175,000
            [ 17  , 14.5, 12  , 10.2,  9.5,  9.3,  9.4,  9.8, 10.8, 12  , 13.8, 16, 21  ,  42,   84,  168], # Re~200,000
            [ 12  , 10.3,  9.3,  8.6,  8.3,  8  ,  8  ,  8.3,  9  , 10  , 11.1, 13, 15.5,  20,   40,   80], # Re<2e6
            [  7  ,  7  ,  7  , 7  ,   7  ,  7  ,  7  ,  7.1,  7.5,  8  ,  8.8, 10, 12  ,  15,   20,   30], # Re<4e6
            [  6.2,  6.2,  6.2, 6.2,   6.2,  6.3,  6.4,  6.6,  6.9,  7.3,  8  ,  9, 10.1,  12,   14,   17]  # Re<4e6
        ]),
        kind='linear',
        bounds_error=False,
        fill_value=None # extrapolate based on nearest neighbour
        )

    halfrho = 512 # fresh water = 500, salt water = 512, air = 0.625
    visc = 1e6 # water = 1e6, air = 7e5

    # todo: generalize to irregular radii
    iel = np.arange(1,nelem+1)
    R = Diameter/(2*(nelem+1)) * iel
    dR = iel*0.+Diameter/(2*(nelem+1))
    #CH = interp1d(RCH, CH, kind='linear', bounds_error=False, fill_value=(RCH[0], 0))(R)

    ii = (2*R)>(Diameter*HubDiaRatio)
    R = R[ii]
    CH = CH[ii]
    dR = dR[ii]
    nelem = ii.sum()

    Circum = np.pi*Diameter # m
    TD = np.arctan(Ptch/Circum)*180./np.pi # tip angle (deg)
    SwptArea = np.pi*(Diameter/2)**2
    A = CH*dR*Blades
    #BETA = np.arctan(Ptch/(Circum*I/10.))
    BETA = np.arctan(Ptch/2/np.pi/R)
    BETA[R<=(HubDiaRatio*Diameter/2.)] = 0.
    Atot = A.sum() # + A[-1]/4.# total blade area
    BAR = Atot/SwptArea # blade area ratio
    AR = (Diameter/2)**2/Atot*Blades # blade aspect ratio
    print(f"DIAMETER: {Diameter:0.2f}")
    print(f"PITCH: {Ptch:0.2f}")
    print(f"SWEPT AREA: {SwptArea:0.4f} m^2")
    print(f"BLADE AREA RATIO: {BAR:0.4f}")
    print(f"BLADE ASPECT RATIO: {AR:0.2f}")
    print(f"TIP ANGLE: {TD:0.2f} deg")
    print(f"\nSTATION\tRADIUS\tdR\tCHORD\tANGLE")
    for i in range(nelem):
        print(f"{i+1}\t{R[i]:0.3f}\t{dR[i]:0.3f}\t{CH[i]:0.3f}\t{BETA[i]*180./np.pi:0.2f}")

    # establish valid rpm range
    RPMs = np.arange(R1, R2+Incr, Incr)
    VR = Circum*RPMs/60.
    DELTA = np.arctan(U/(np.outer(VR, R)))
    ALPHA = BETA - DELTA # angle of incidence at blade element
    R3 = R1
    RPMbad = RPMs[np.where(ALPHA<-0.01)[0]] # rpms where any element has -ve lift
    #if len(RPMbad)>0:
    #    R3 = RPMbad.max() # limit program to +ve lift
    R4 = R2
    RRbad = RPMs[np.where(ALPHA>0.26)[0]]  # rpms where any blade element is overloaded
    if (len(RPMbad)>0) and (Range):
       R4 = RPMs.min() # restrict blade loading

    #print(R1, R2, R3, R4)

    # loop over rotational speeds
    q = 0
    PP = np.zeros(nelem) # power in
    PW = np.zeros(nelem) # power out
    T  = np.zeros(nelem) # thrust
    CL = np.zeros(nelem)
    CD = np.zeros(nelem)
    UR = np.zeros(nelem)
    VR = np.zeros(nelem)
    W = np.zeros(nelem)
    Q = np.zeros(nelem)
    ALPHA = np.zeros(nelem)
    DELTA = np.zeros(nelem)
    Re = np.zeros(nelem)
    ID = np.zeros(nelem)
    D = np.zeros(nelem)
    L = np.zeros(nelem)
    F = np.zeros(nelem)
    ETA = np.zeros(nelem)
    EF = np.zeros(nelem)
    CT = np.zeros(nelem)
    C2 = np.zeros(nelem)
    Torque = np.zeros(nelem)

    print(f"\nrpm\tPin\tPout\tETA\tETA_F\tT\tQ\tCL[5]\tSLIP")
    for rpm in np.arange(R3, R4+Incr, Incr):
        #vr = Circum*rpm/60.
        #ur = U*(1+q) # speed through disc, q is slip factor # NOT USED FURTHER
        for i in range(nelem):
            k = 0
            while True:
                k += 1
                UR[i] = U*(1+Q[i])
                #VR[i] = vr*((i+1.)/10.)
                VR[i] = 2*np.pi*R[i]*rpm/60
                W[i] = np.sqrt(UR[i]**2+VR[i]**2) # resultant speed at blade segment
                DELTA[i] = np.arctan(UR[i]/VR[i])
                ALPHA[i] = BETA[i]-DELTA[i] # angle of attack
                CL[i] = ALPHA[i]*5.75/(1.+2./AR)+0.35 # coefficient of lift
                if CL[i]<0:
                    next_rpm = True
                else:
                    next_rpm = False
                Re[i] = visc*W[i]*CH[i] # Reynolds number

                #X = max(min(int(np.round(CL[i]*10.+0.5)), 15)-1, 0)
                #if Re[i]<45000:
                #    Y = 0
                #elif Re[i]<105000:
                #    Y = int(Re[i]/1e4 - 1.5)
                #elif Re[i]<212500:
                #    Y = int(Re[i]/2.5e4 + 2.5)
                #elif Re[i]<2e6:
                #    Y = 11
                #elif Re[i]<4e6:
                #    Y = 12
                #else:
                #    Y = 13
                #print(Re[i], Y, CL[i], X, FF[Y,X])
                #CD[i] = FF[Y,X]/1000. # profile coefficient of drag
                CD[i] = FF(CL[i], Re[i])/1000.

                if CL[i]>1.2:
                    CL[i] = 1.2 # pathetic attempt to simulate approaching stall
                ID[i] = CL[i]**2/(np.pi*AR) # induced drag
                L[i] = halfrho*A[i]*CL[i]*W[i]**2 # lift
                D[i] = halfrho*A[i]*(CD[i] + ID[i])*W[i]**2 # drag
                T[i] = L[i]*np.cos(DELTA[i]) - D[i]*np.sin(DELTA[i]) # thrust ## DE==DELTA
                F[i] = L[i]*np.sin(DELTA[i]) + D[i]*np.cos(DELTA[i]) # lateral force
                Torque[i] = F[i]*R[i]
                PP[i] = F[i]*VR[i] # power in
                PW[i] = T[i]*U # power out
                ETA[i] = PW[i]/PP[i] # efficiency
                #CT[i] = T[i]/(halfrho*Diameter*((i+1.)/10.)*np.pi*U**2*Diameter/20.) # coefficient of thrust ## DI=Diameter
                CT[i] = T[i]/(halfrho*R[i]*2*np.pi*U**2*dR[i]) # coefficient of thrust ## DI=Diameter
                C2[i] = Q[i]*4*(1. + Q[i]) # also coefficient of thrust
                EF[i] = 2./(1. + np.sqrt(1. + CT[i])) # Froude efficiency
                #print(Q[i], (1./EF[i] - 1.), (1 - sor)*(1./EF[i] - 1.) + sor*Q[i])
                Q[i] = (1 - sor)*(1./EF[i] - 1.) + sor*Q[i]
                #print(rpm, k, Q[i], VR[i], W[i], BETA[i], DELTA[i], ALPHA[i], CL[i], CT[i], C2[i], abs(C2[i]/CT[i] - 1.), next_rpm)
                if (abs(C2[i]/CT[i] - 1.)<0.05):
                    break
        t = T.sum()
        torque = Torque.sum()
        pp = PP.sum()
        pw = PW.sum()
        eta = pw/pp # "eta + PW/PP" ????????????????
        ct = t/(halfrho*SwptArea*U**2) # SW==SWPTAREA
        ef = 2./(1. + np.sqrt(1. + ct))
        q = 1./ef - 1.
        c2 = 4*q*(1. + q)
        uj = U*(1. + 2.*q)

        print(f"{rpm:0.0f}\t{pp:0.0f}\t{pw:0.0f}\t{eta:0.2f}\t{ef:0.2f}\t{t:0.0f}\t{torque:0.0f}\t{CL[nelem//2]:0.2f}\t{q:0.3f}")
        #print(Q)

        yield {
            'D': Diameter,
            'P': Ptch,
            'n': rpm,
            'Pin': pp,
            'Pout': pw,
            'T': t,
            'Q': torque,
            'eta': eta,
            'eta_F': ef,
            'slip': q,
            }




if __name__=="__main__":
    main()
