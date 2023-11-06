from src.Coverage import walker
from src.Coverage import FieldOfViewEventAnalysis
from src.Coverage import CoverageDefinition


class Design:
    def __init__(self, name, parameters, variables):
        self.name = name
        self.halfangle = parameters[0]
        self.granularity = parameters[1]
        self.epoch = parameters[2]
        self.semimajoraxis = variables[0]
        self.inclination = variables[1]
        self.numsats = variables[2]
        self.numplanes = variables[3]
        self.f = variables[4]

        self.constellation = walker.Walker(self.name, self.semimajoraxis, self.inclination, self.numsats, self.numplanes, self.f, self.epoch, "delta")
        self.constellation.createConstellation()

        self.coveragedefinition = CoverageDefinition.CoverageDefinition(self.name, self.granularity, self.constellation)

        self.coverageanalysis = FieldOfViewEventAnalysis.FieldOfViewEventAnalysis([self.coveragedefinition], self.halfangle)

        self.coverage = self.coverageanalysis.call()

    def getCoverage(self):
        coverage = self.coverageanalysis.call()
        return coverage

    def getCoveragePercent(self):
        coverage = self.coverageanalysis.call()
        toReturn = float(sum(coverage))/float(len(coverage))
        return toReturn
