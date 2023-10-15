from math import radians
import pandas as pd
import numpy as np
import orekit

from orekit.pyhelpers import setup_orekit_curdir
from org.hipparchus.geometry.euclidean.threed import RotationOrder, Vector3D, RotationConvention
from org.orekit.attitudes import NadirPointing, YawSteering, LofOffset, FieldAttitude;
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid;
from org.orekit.frames import FramesFactory, LOFType, StaticTransform;
from org.orekit.orbits import KeplerianOrbit, PositionAngle;
from org.orekit.propagation.analytical import EcksteinHechlerPropagator;
from org.orekit.propagation import PropagatorsParallelizer
from org.orekit.propagation.events import EclipseDetector, EventsLogger
from org.orekit.propagation.events.handlers import ContinueOnEvent
from org.orekit.time import AbsoluteDate;
from org.orekit.time import TimeScalesFactory;
from org.orekit.utils import Constants, IERSConventions
from org.orekit.utils import PVCoordinatesProvider
from org.orekit.propagation.events import FieldOfViewDetector
from org.orekit.propagation.events import GroundFieldOfViewDetector
from org.orekit.propagation.events import AbstractDetector
from org.orekit.propagation.sampling import MultiSatStepHandler

def orbit_period(a):
    # seconds
    return 2 * np.pi * np.sqrt(a**3 / 3.986e14)
def propogate_constellation(walker):
    earthFrame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    Earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                             Constants.WGS84_EARTH_FLATTENING,
                             earthFrame)
    attitudeLaw = LofOffset(FramesFactory.getEME2000(),
                            LOFType.VVLH,
                            RotationOrder.XYZ,
                            0.0, 0.0, 0.0)
    propogators = []
    sats = walker.sats
    epochDate = walker.epochDate
    for i in sats:
        propagator = EcksteinHechlerPropagator(i.orbit, attitudeLaw,
                                               Constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS,
                                               Constants.EIGEN5C_EARTH_MU, Constants.EIGEN5C_EARTH_C20,
                                               Constants.EIGEN5C_EARTH_C30, Constants.EIGEN5C_EARTH_C40,
                                               Constants.EIGEN5C_EARTH_C50, Constants.EIGEN5C_EARTH_C60)
        propogators.append(propagator)
    handler = MultiSatStepHandler(propogators, walker.epochDate)
    parallelProp = PropagatorsParallelizer(propogators, handler)
    propogationDuration = 10*orbit_period(walker.a)
    timeStep = 60.0 #seconds
    discrete_times = np.arange(timeStep, propogationDuration + timeStep, timeStep)
    fovEvent = FieldOfViewDetector()
    for i in discrete_times:
        state = propagator.propagate(epochDate, epochDate.shiftedBy(float(i)))

