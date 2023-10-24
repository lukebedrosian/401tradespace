import numpy as np
from org.hipparchus.geometry.euclidean.threed import RotationOrder, Vector3D
from org.orekit.attitudes import LofOffset
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.frames import FramesFactory, LOFType, TopocentricFrame, StaticTransform
from org.orekit.geometry.fov import CircularFieldOfView
from org.orekit.propagation import PropagatorsParallelizer
from org.orekit.propagation.analytical import EcksteinHechlerPropagator
from org.orekit.propagation.events import FieldOfViewDetector, ElevationDetector, BooleanDetector, DateDetector
from org.orekit.propagation.events.handlers import PythonEventHandler, EventHandler
from org.orekit.propagation.sampling import MultiSatStepHandler, PythonMultiSatStepHandler, OrekitStepInterpolator, \
    OrekitStepNormalizer, OrekitStepHandler, PythonOrekitStepHandler
from org.orekit.utils import Constants, IERSConventions
from operator import xor
from java.util import Arrays
import logging
import threading
import time


class MultiSatHandler(PythonMultiSatStepHandler):
    def init(self):
        pass
    def handleStep(self, s0, t):
        pass
    def finish(self, finalStates):
        pass

class MultiSatStepNormalizer(PythonMultiSatStepHandler):

    def __init__(self, h, grid, sats, detectors, hashmap, epochDate, coverages):
        self.h = np.abs(h)
        self.forward = True
        self.lastState = None
        self.grid = grid
        self.sats = sats
        self.detectors = detectors
        self.hashmap = hashmap
        self.startDate = epochDate
        self.coverages = coverages
        super(MultiSatStepNormalizer, self).__init__()

    def init(self, s0, t):
        self.lastState = None
        self.forward = True
        pass

    def handleStep(self, interpolators):

        step = self.h
        interpolator1 = OrekitStepInterpolator.cast_(interpolators.get(0)) # interpolators should be interpolated together
        print(interpolator1.getCurrentState().getDate())
        if self.startDate.compareTo(interpolator1.getCurrentState().getDate()) >= 0:
            return

        self.forward = interpolator1.isForward()
        if not self.forward:
            step = -self.h
        self.lastState = interpolator1.getPreviousState()
        states = []
        for interpolator in interpolators:
            state = OrekitStepInterpolator.cast_(interpolator).getCurrentState()
            states.append(state)
        nextTime = self.startDate

        currentDate = interpolator1.getCurrentState().getDate()
        nextInStep = xor(self.forward, (nextTime.compareTo(currentDate) > 0))

        while nextTime.compareTo(currentDate) < 0:
            self.lastState = OrekitStepInterpolator.cast_(interpolators.get(0)).getInterpolatedState(nextTime)
            states = []
            for interpolator in interpolators:
                state = OrekitStepInterpolator.cast_(interpolator).getInterpolatedState(nextTime)
                states.append(state)
            nextTime = nextTime.shiftedBy(float(step))

            nextInStep = xor(self.forward, nextTime.compareTo(states[0].getDate()) > 0)
            inView = [0] * len(self.grid)
            for s in range(len(self.sats)):
                state = states[s]
                for p in range(len(self.grid)):
                    dets = self.hashmap[p]
                    for d in range(len(dets)):
                        detector = dets[d]
                        if detector.g(state) > 0.0:
                            inView[p] = 1
                            break
            percentCoverage = float(sum(inView)) / float(len(inView))
            self.coverages.append(percentCoverage)
        # print(self.coverages)
        self.startDate = interpolator1.getCurrentState().getDate()


    def finish(self, finalStates):
        pass

    def getResult(self):
        return self.coverages


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
        epochDate = self.covDefs[0].constellation.epochDate
        propogationDuration = self.covDefs[0].constellation.sats[0].orbit.getKeplerianPeriod()
        timeStep = 120.0  # seconds
        discrete_times = np.arange(timeStep, propogationDuration + timeStep, timeStep)
        dates = []
        for shift in discrete_times:
            dates.append(epochDate.shiftedBy(float(shift)))
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

        grid = self.covDefs[0].grid
        cdef = self.covDefs[0]
        val = 10
        coverages = []
        normalized_handler = MultiSatStepNormalizer(float(60.0*5), grid, cdef.constellation.sats, detectors, point_to_satDetector, epochDate, coverages)
        propagationDuration = 2 * cdef.constellation.sats[0].orbit.getKeplerianPeriod()
        parallelProp = PropagatorsParallelizer(Arrays.asList(propagators), normalized_handler)
        print(normalized_handler.coverages)
        state = parallelProp.propagate(epochDate, epochDate.shiftedBy(float(propagationDuration)))
        # while (normalized_handler.coverages == []):
        #     pass
        # print('coverages:', normalized_handler.coverages)
        return normalized_handler.coverages
        print(epochDate, state.get(0).getDate())
        # for t in discrete_times:
        #     print('t:', t)
        #     inView = [0] * len(grid)
        #     print('starting propogation')
        #
        #     print('state propogated')
        #     for s in range(len(cdef.constellation.sats)):
        #         for p in range(len(grid)):
        #             dets = point_to_satDetector[p]
        #             for d in range(len(dets)):
        #                 detector = dets[d]
        #                 if detector.g(state.get(s)) > 0.0:
        #                     inView[p] = 1
        #                     break
        #     print(inView)
        #     percentCoverage = float(sum(inView)) / float(len(inView))
        #     print("Coverage Percent of Globe: " + str(percentCoverage * 100) + "%")

