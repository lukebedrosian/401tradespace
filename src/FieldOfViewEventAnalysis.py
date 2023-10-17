import numpy as np
from org.hipparchus.geometry.euclidean.threed import RotationOrder, Vector3D
from org.orekit.attitudes import LofOffset
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.frames import FramesFactory, LOFType, TopocentricFrame, StaticTransform
from org.orekit.geometry.fov import CircularFieldOfView
from org.orekit.propagation import PropagatorsParallelizer
from org.orekit.propagation.analytical import EcksteinHechlerPropagator
from org.orekit.propagation.events import FieldOfViewDetector, ElevationDetector, BooleanDetector
from org.orekit.propagation.events.handlers import PythonEventHandler, EventHandler
from org.orekit.propagation.sampling import MultiSatStepHandler, PythonMultiSatStepHandler
from org.orekit.utils import Constants, IERSConventions
from java.util import Arrays

class SimpleStepHandler(PythonMultiSatStepHandler):

    def init(self, s, t):
        pass

    def handleStep(self, interpolators):
        pass

    def finish(self, finalStates):
        pass


class FieldOfViewEventAnalysis:
    def __init__(self, covDefs, halfangle):
        self.covDefs = covDefs
        self.halfangle = halfangle

    def call(self):
        attitudeLaw = LofOffset(FramesFactory.getEME2000(),
                                LOFType.VVLH,
                                RotationOrder.XYZ,
                                0.0, 0.0, 0.0)
        fov = CircularFieldOfView(Vector3D.PLUS_K, float(np.radians(self.halfangle)), 0.0)
        propagators = []
        propagatorStates = []
        detectors = []
        sats = []
        points = []
        inertialFrame = FramesFactory.getEME2000()
        earthFrame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        earth = OneAxisEllipsoid(
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
            Constants.WGS84_EARTH_FLATTENING,
            earthFrame)

        for cdef in self.covDefs:
            point_to_satDetector = []
            for i in range(len(cdef.grid)):
                point_to_satDetector.append([])
            for s in range(len(cdef.constellation.sats)):
                detectorsInSat = []
                propagator = EcksteinHechlerPropagator(cdef.constellation.sats[s].orbit, attitudeLaw,
                                                       Constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS,
                                                       Constants.EIGEN5C_EARTH_MU, Constants.EIGEN5C_EARTH_C20,
                                                       Constants.EIGEN5C_EARTH_C30, Constants.EIGEN5C_EARTH_C40,
                                                       Constants.EIGEN5C_EARTH_C50, Constants.EIGEN5C_EARTH_C60)
                fovSizeStep = cdef.constellation.sats[s].orbit.getKeplerianPeriod() / 100.0
                threshold = 1e-3
                for p in range(len(cdef.grid)):
                    topo = TopocentricFrame(earth, cdef.grid[p].point, 'target')
                    fd = FieldOfViewDetector(topo, fov)
                    ed = ElevationDetector(60.0, threshold, topo).withConstantElevation(30.0)
                    detector = BooleanDetector.andCombine(ed).notCombine(fd)
                    detectors.append(detector)
                    sats.append(cdef.constellation.sats[s])
                    propagator.addEventDetector(detector)
                    points.append(cdef.grid[p])
                    point_to_satDetector[p].append(detector)
                propagatorStates.append(propagator.getInitialState())
                propagators.append(propagator)
        handler = SimpleStepHandler()
        propogationDuration = 2 * cdef.constellation.sats[0].orbit.getKeplerianPeriod()
        parallelProp = PropagatorsParallelizer(Arrays.asList(propagators), handler)
        timeStep = 60.0  # seconds
        discrete_times = np.arange(timeStep, propogationDuration + timeStep, timeStep)
        epochDate = self.covDefs[0].constellation.epochDate
        grid = self.covDefs[0].grid
        cdef = self.covDefs[0]
        print(discrete_times)
        for t in discrete_times:
            print(t)
            inView = [0] * len(grid)
            state = parallelProp.propagate(epochDate, epochDate.shiftedBy(float(t)))
            for s in range(len(cdef.constellation.sats)):
                for p in range(len(grid)):
                    dets = point_to_satDetector[p]
                    for d in range(len(dets)):
                        detector = dets[d]
                        if detector.g(state.get(s)) > 0.0:
                            inView[p] = 1
                            break
            print(inView)
            percentCoverage = float(sum(inView)) / float(len(inView))
            print("Coverage Percent of Globe: " + str(percentCoverage * 100) + "%")
