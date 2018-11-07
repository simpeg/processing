import DCIPtools as DCIP
import matplotlib.pyplot as plt
import numpy as np
from SimPEG import DC

fname = "/Users/juan/PyDev/processing/testDataIP/Seabridge_Final.DAT"
patch = DCIP.loadDias(fname)
survey_dc = patch.createDcSurvey("IP", ip_type="decay")
G = DC.Utils.geometric_factor(survey_dc)
d_G = patch.getGeometricFactor()
rho = patch.getApparentResistivity()
# plt.plot(survey_dc.dobs, '.')
# plt.plot(rho, '.', ms=1)
# plt.show()
print(survey_dc.dobs.shape)