import numpy as np
from org.orekit.orbits import KeplerianOrbit, PositionAngle
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants
from src import satellite


class Walker:
    def __init__(self, name, a, i, n, p, f, epochDate, payload):
        self.name = name
        self.a = a # semimajor axis
        self.i = i # inclination
        self.n = n # number of satellites
        self.p = p # number of equally spaced planes
        self.f = f # relative spacing between planes
        self.epochDate = epochDate
        self.payload = payload

    def createConstellation(self):
        s = int(self.n/self.p) # number of satellites per plane
        pu = 2*np.pi/self.n # pattern unit
        delAnom = pu*self.p # in plane spacing between satellites
        delRaan = pu*s #node spacing
        phasing = pu*self.f
        inertialFrame = FramesFactory.getEME2000()
        sats = []
        for planeNum in range(self.p):
            for satNum in range(s):
                anom = (satNum * delAnom + phasing * planeNum) % (2*np.pi)
                # orbit = KeplerianOrbit(self.a, 0.0001, self.i, 0.0, planeNum*delRaan, anom, PositionAngle.TRUE,
                #                   inertialFrame, self.epochDate, Constants.WGS84_EARTH_MU)
                orbit = KeplerianOrbit(float(self.a), 0.0001, float(self.i), 0.0, float(planeNum*delRaan), float(anom),
                                              PositionAngle.TRUE,
                                              inertialFrame, self.epochDate, Constants.WGS84_EARTH_MU)
                sat = satellite.Satellite("sat_walker_%d", orbit, self.payload)
                sats.append(sat)
        self.sats = sats
