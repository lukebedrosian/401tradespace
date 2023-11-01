import src.Design

class Evaluator:
    def __init__(self, design):
        self.design = design
        print(design.constellation)

    def evaluate(self):
        #performance scores
        coveragePct = self.design.getCoveragePercent()
        toReturn =  [coveragePct, self.design.numplanes/8.0]
        return toReturn

