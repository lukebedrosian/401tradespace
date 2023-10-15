import numpy as np
from org.orekit.bodies import GeodeticPoint, OneAxisEllipsoid, BodyShape
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants, IERSConventions
import coveragePoint


class CoverageDefinition:
    def __init__(self, name, granularity, style):
        self.name = name
        self.grid  # hashset of coverage points
        self.constellations  # hashset of constellations
        self.planet = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                                       Constants.WGS84_EARTH_FLATTENING,
                                       FramesFactory.getITRF(IERSConventions.IERS_2003, True))
        self.minLatitude = np.radians(-90)
        self.maxLatitude = np.radians(90)
        self.minLongitude = np.radians(-180)
        self.maxLongitude = np.radians(180)
        grid = set()

        numPoints = 0
        for long in range(self.minLongitude, self.maxLongitude, granularity):
            for lat in range(self.minLatitude, self.maxLatitude, granularity):
                point = GeodeticPoint(lat, long, 0.0)
                coveragepoint = coveragePoint(point, str(numPoints))
                grid.add(coveragepoint)
                numPoints += 1

