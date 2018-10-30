import matplotlib.pyplot as plt
from SimPEG import (
    Mesh,  Problem,  Survey,  Maps,  Utils,
    EM,  DataMisfit,  Regularization,  Optimization,
    InvProblem,  Directives,  Inversion
)
from SimPEG.EM.Static import DC, Utils as DCUtils
from SimPEG.EM.Static.Utils import plot_pseudoSection
# from pymatsolver import Pardiso as Solver

import numpy as np
import JDataObject as Jdata

################################################################
# define the file required for import
fileName = "/Users/juan/Documents/testData/FieldSchool_2017new.DAT"
fileName2 = "/Users/juan/Documents/testData/FieldSchool_Mesh.txt"
fileName3 = "/Users/juan/Documents/testData/FS2017_a1-2.obs"

# ln conductivity
ln_sigback = -5.
ln_sigc = -3.
ln_sigr = -6.

patch = Jdata.loadDias(fileName)
# survey = patch.createDcSurvey()
# survey = Jdata.loadDiasToDcSurvey(fileName)
# plt.plot(np.asarray(xpp), np.asarray(ypp), 'o')
survey = DCUtils.StaticUtils.readUBC_DC3Dobs(fileName3)
# print len(xpp)
# 3D Mesh
#############################################################

# Create mesh and center it
mesh = Mesh.TensorMesh._readUBC_3DMesh(fileName2)  # Read in/create the mesgh
survey = survey['dc_survey']
survey.getABMN_locations()
uniq = Utils.uniqueRows(np.vstack((survey.a_locations, survey.b_locations, survey.m_locations, survey.n_locations)))
electrode_locations = uniq[0]
actinds = Utils.surface2ind_topo(mesh, electrode_locations, method='cubic')
survey.drapeTopo(mesh, actinds)

# Begin Formulating Problem
############################################################
# print minE, maxE, minN, maxN

# Setup Problem with exponential mapping and Active cells only in the core mesh
expmap = Maps.ExpMap(mesh)
mapactive = Maps.InjectActiveCells(mesh=mesh, indActive=actinds,
                                   valInactive=np.log(1e-8))
mapping = expmap * mapactive
problem = DC.Problem3D_N(mesh, sigmaMap=mapping)
problem.pair(survey)
problem.Solver = Solver

# %pylab inline
# Assign Uncertainty
out = plt.hist(np.log10(abs(survey.dobs) + 1e-5), bins=100)
# print out
plt.show()

survey.std = 0.05
survey.eps = 1e-2

# Tikhonov Inversion#
############################################################

# Initial Model
sig_half = 1. / 90
m0 = np.log(sig_half) * np.ones(mapping.nP)
dmis = DataMisfit.l2_DataMisfit(survey)           # Data Misfit

# Regularization
regT = Regularization.Simple(mesh, indActive=actinds, alpha_s=1e-6,
                             alpha_x=1., alpha_y=1., alpha_z=1.)

# Optimization Scheme
opt = Optimization.InexactGaussNewton(maxIter=10)

# Form the problem
opt.remember('xc')
invProb = InvProblem.BaseInvProblem(dmis, regT, opt)

# Directives for Inversions
beta = Directives.BetaEstimate_ByEig(beta0_ratio=1e+1)
Target = Directives.TargetMisfit()
betaSched = Directives.BetaSchedule(coolingFactor=5., coolingRate=2)

inv = Inversion.BaseInversion(invProb, directiveList=[beta, Target,
                              betaSched])

# Run Inversion
minv = inv.run(m0)

# Plotting ---------------------------------------------
# plt.plot([minE, maxE], [minN, maxN], 'o')
# plt.show()
