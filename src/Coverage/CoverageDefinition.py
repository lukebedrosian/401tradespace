import numpy as np
from org.orekit.bodies import GeodeticPoint, OneAxisEllipsoid, BodyShape
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants, IERSConventions


class CoverageDefinition:
    def __init__(self, name, granularity, constellation):
        self.name = name
        self.constellation = constellation  # hashset of constellations
        self.planet = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                                       Constants.WGS84_EARTH_FLATTENING,
                                       FramesFactory.getITRF(IERSConventions.IERS_2003, True))
        self.minLatitude = -np.pi/2
        self.maxLatitude = np.pi/2
        self.minLongitude = -np.pi
        self.maxLongitude = np.pi
        grid = []
        nlong = int((self.maxLongitude - self.maxLongitude)/granularity)
        nlat = int((self.maxLatitude - self.maxLatitude)/granularity)
        longitudes = np.arange(self.minLongitude, self.maxLongitude, granularity)
        latitudes = np.arange(self.minLatitude, self.maxLatitude, granularity)
        numPoints = 0
        for long in longitudes:
            for lat in latitudes:
                point = GeodeticPoint(float(lat), float(long), 0.0)
                coveragepoint = CoveragePoint(point, str(numPoints))
                grid.append(coveragepoint)
                numPoints += 1
        self.grid = grid
        self.numPoints = numPoints

class CoveragePoint:

    def __init__(self, point, name):
        self.point = point
        self.name = name
        self.inView = 0
