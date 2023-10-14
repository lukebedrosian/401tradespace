import orekit
import numpy as np
from src import walker
from org.orekit.time import TimeScalesFactory, AbsoluteDate
from orekit.pyhelpers import setup_orekit_curdir

vm = orekit.initVM()
setup_orekit_curdir()
epochDate = AbsoluteDate(2020, 1, 1, 0, 0, 00.000, TimeScalesFactory.getUTC()) # AbsoluteDate(int year, int month, int day, int hour, int minute, double second, TimeScale timeScale)
payload = "hi"
altitude = 9000 #km
i = 28.5 #degrees
semiMajorAxis = (altitude+6378) * 1000
w = walker.Walker("One", semiMajorAxis, np.radians(i), 20, 3, 1, epochDate, payload)
w.createConstellation()

for i in w.sats:
    print(i.orbit)
