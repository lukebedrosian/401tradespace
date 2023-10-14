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



def propogate_orbit_data(h, i, raan, epoch, pointing_off_nadir):
    alt = h
    h = float(h)*1E3 + 6378137.0
    vm = orekit.initVM()
    setup_orekit_curdir()
    utc = TimeScalesFactory.getUTC()
    i = radians(i)
    raan = radians(raan)  # right ascension of ascending node
    omega = radians(0.0)  # argument of perigee
    lv = radians(0.0)  # true anomaly
    e = 0.0
    epochDate = AbsoluteDate(epoch[0], epoch[1], epoch[2], epoch[3], epoch[4], epoch[5], utc) # AbsoluteDate(int year, int month, int day, int hour, int minute, double second, TimeScale timeScale)
    initialDate = epochDate
    inertialFrame = FramesFactory.getEME2000()
    initialOrbit = KeplerianOrbit(h, e, i, omega, raan, lv,
                                  PositionAngle.TRUE,
                                  inertialFrame, epochDate, Constants.WGS84_EARTH_MU)
    earthFrame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    sun = CelestialBodyFactory.getSun()
    sun_pv = PVCoordinatesProvider.cast_(sun)  # But we want the PVCoord interface
    Earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                             Constants.WGS84_EARTH_FLATTENING,
                             earthFrame)
    # nadirLaw = NadirPointing(FramesFactory.getEME2000(), Earth)
    attitudeLaw = LofOffset(FramesFactory.getEME2000(),
                             LOFType.VVLH,
                             RotationOrder.XYZ,
                             0.0, 0.0, 0.0)
    # yawSteeringLaw = YawSteering(FramesFactory.getEME2000(), nadirLaw, sun, Vector3D.MINUS_I)
    eclipse_detector = EclipseDetector(sun, Constants.SUN_RADIUS, Earth).withUmbra().withHandler(ContinueOnEvent())
    logger = EventsLogger()
    logged_detector = logger.monitorDetector(eclipse_detector)
    propagator = EcksteinHechlerPropagator(initialOrbit, attitudeLaw,
                                           Constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS,
                                           Constants.EIGEN5C_EARTH_MU, Constants.EIGEN5C_EARTH_C20,
                                           Constants.EIGEN5C_EARTH_C30, Constants.EIGEN5C_EARTH_C40,
                                           Constants.EIGEN5C_EARTH_C50, Constants.EIGEN5C_EARTH_C60)
    propagator.addEventDetector(logged_detector)
    state = propagator.propagate(initialDate, initialDate.shiftedBy(5.0))
    discrete_times = np.arange(dt, propogation_duration + dt, dt)
    # Propagate from the initial date for the fixed duration
    propogation_duration = orbit_period(alt)
    print(propogation_duration)
    dt = 30.0 # time step - 60 seconds
    propogation_duration = np.floor((propogation_duration + dt) / dt) * dt
    discrete_times = np.arange(dt, propogation_duration + dt, dt)
    n = int(propogation_duration / dt)
    dates = []
    eclipses = []
    satX = []
    satY = []
    satZ = []
    sunX = []
    sunY = []
    sunZ = []
    satpos = []
    sunpos = []
    for i in discrete_times:
        state = propagator.propagate(initialDate, initialDate.shiftedBy(float(i)))
        satPos = state.getPVCoordinates().getPosition()
        sunPos = sun_pv.getPVCoordinates(state.getDate(), state.getFrame()).getPosition()
        inertToSpacecraft = StaticTransform.cast_(state.toTransform())
        sunInert = sun_pv.getPVCoordinates(state.getDate(), state.getFrame()).getPosition()
        sunSat = np.array(inertToSpacecraft.transformPosition(sunInert).normalize().toArray())
        earthSat = np.array(inertToSpacecraft.transformPosition(Vector3D.ZERO).normalize().toArray())
        # earthSat *= -1.0
        dates.append(str(state.getAttitude().getDate()))
        eclipses.append(eclipse_detector.g(state) <= 0)
        satX.append(earthSat[0])
        satY.append(earthSat[1])
        satZ.append(earthSat[2])
        sunX.append(sunSat[0])
        sunY.append(sunSat[1])
        sunZ.append(sunSat[2])
        satpos.append(satPos)
        sunpos.append(sunPos)
    d = {'Date': dates, 'Eclipse': eclipses, 'Sat X': satX, 'Sat Y': satY, 'Sat Z': satZ, 'Sun X': sunX, 'Sun Y': sunY, 'Sun Z': sunZ, 'Sat Pos': satpos, 'Sun Pos': sunpos}
    df = pd.DataFrame(data=d)
    return df
