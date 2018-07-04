"""
3D DC inversion of Dipole Dipole array
======================================

This is an example for 3D DC Inversion. The model consists of 2 spheres,
one conductive, the other one resistive compared to the background.

We restrain the inversion to the Core Mesh through the use an Active Cells
mapping that we combine with an exponetial mapping to invert
in log conductivity space. Here mapping,  :math:`\\mathcal{M}`,
indicates transformation of our model to a different space:

.. math::
    \\sigma = \\mathcal{M}(\\mathbf{m})

Following example will show you how user can implement a 3D DC inversion.
"""

from SimPEG import (
    Mesh, Maps, Utils,
    DataMisfit, Regularization, Optimization,
    InvProblem, Directives, Inversion
)
from SimPEG.EM.Static import DC, Utils as DCUtils
from SimPEG.EM.Static import IP as IPUtils
import numpy as np
import matplotlib.pyplot as plt
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

np.random.seed(12345)

# =============================
# methods


def getRxData():
    xyz = open("/Users/juan/Documents/testData/IdealizedStations_Rx.csv")
    x = []
    y = []
    z = []
    for line in xyz:
        x_, y_, z_ = line.split(',')
        x.append(float(x_))
        y.append(float(y_))
        z.append(float(z_))
    xyz.close()
    x1 = np.asarray(x)
    y1 = np.asarray(y)
    z1 = np.asarray(z)
    rx_electrodes = np.c_[x1, y1, z1]
    return rx_electrodes


def getTxData():
    xyz = open("/Users/juan/Documents/testData/IdealizedStations_Tx.csv")
    x = []
    y = []
    z = []
    for line in xyz:
        x_, y_, z_ = line.split(',')
        x.append(float(x_))
        y.append(float(y_))
        z.append(float(z_))
    xyz.close()
    x1 = np.asarray(x)
    y1 = np.asarray(y)
    z1 = np.asarray(z)
    tx_electrodes = np.c_[x1, y1, z1]
    return tx_electrodes


def generateSurvey(rx, tx, min_dipole_size, max_dipole_size):
    """
     Generates a survey to through into a forward model
     INPUT:
          rx_dx = array of Rx x spacings
          rx_dy = array of Rx y spacings
          Tx_dx = array of Tx x spacings
          Tx_dy = array of Tx y spacings
    """
    SrcList = []
    rx_length = rx.shape[0]
    for idk in range(tx.shape[0]):
        rx1 = []
        rx2 = []
        for idx in range(rx_length):
            node1 = rx[idx, :]
            for idj in range(idx, rx_length):
                node2 = rx[idj, :]
                dist = np.sqrt(np.sum((node1 - node2)**2))
                distE = np.abs(node1[0] - tx[idk, 0])
                if distE < 80:
                    if (min_dipole_size) < dist < (max_dipole_size):
                        rx1.append(node1)
                        rx2.append(node2)
                    # print(dist)
        rx1 = np.asarray(rx1)
        rx2 = np.asarray(rx2)
        rxClass = DC.Rx.Dipole(rx1, rx2)
        srcClass = DC.Src.Pole([rxClass], tx[idk, :])
        SrcList.append(srcClass)

    survey = DC.Survey(SrcList)

    return survey


fileName1 = "/Users/juan/Documents/testData/fmdata.obs"
fileName2 = "/Users/juan/Documents/testData/forwardmodel.msh"
mesh = Mesh.TensorMesh._readUBC_3DMesh(fileName2)  # Read in/create the mesgh
# print(mesh.gridCC[:, 0], mesh.gridCC[:, 1], mesh.gridCC[:, 2])
x0 = 374700.
x1 = 375000.
y0 = 6275880.
y0_1 = 1000. * np.sin(45 * np.pi / 180) + y0
y1 = 6275900.
y1_1 = 1000. * np.sin(45 * np.pi / 180) + y1
z0 = 1800.
z1 = z0 - (1000. * np.cos((45 * np.pi / 180)))
m1 = (z0 - z1) / (y0_1 - y0)
lims = np.arange(0, 200, 25) + y0
lims1 = np.arange(0, 200, 25) + y1
z_lim = z0 - m1 * (lims - y0)
z_lim1 = z0 - m1 * (lims1 - y1)
y_lim = y0 + (1. / m1) * (lims - y0)
y_lim1 = y1 + (1. / m1) * (lims1 - y1)
print(y_lim)
print(y_lim1)
x0 = (np.max(mesh.gridCC[:, 0]) + np.min(mesh.gridCC[:, 0])) / 2. + 50
y0 = (np.max(mesh.gridCC[:, 1]) + np.min(mesh.gridCC[:, 1])) / 2. - 50
z0 = 2350

(np.max(mesh.gridCC[:, 2]) + np.min(mesh.gridCC[:, 2])) / 2.
r0 = 500
print(x0, y0, z0)
csph = (np.sqrt((mesh.gridCC[:, 0] - x0)**2. + (mesh.gridCC[:, 1] - y0)**2. +
                (mesh.gridCC[:, 2] - z0)**2.)) < r0
print(csph.size)
# Define model Background
# ln_sigback = -5.
rx = getRxData()
tx = getTxData()
# plt.plot(rx[:, 0], rx[:, 1], 'o')
# plt.show()

survey = generateSurvey(rx, tx, 45, 65)
survey.getABMN_locations()
uniq = Utils.uniqueRows(np.vstack((survey.a_locations,
                                   survey.b_locations,
                                   survey.m_locations,
                                   survey.n_locations)))
electrode_locations = uniq[0]
actinds = Utils.surface2ind_topo(mesh, electrode_locations, method='cubic')
# # print(actinds.size)
# survey.drapeTopo(mesh, actinds)
sigma = np.ones(mesh.nC) * 1. / 100.
sigma[csph] = (1. / 5000.) * np.ones_like(sigma[csph])
sigma[~actinds] = 1. / 1e8
rho = 1. / sigma
# Setup Problem with exponential mapping and Active cells only in the core mesh
# Use Exponential Map: m = log(rho)
actmap = Maps.InjectActiveCells(
    mesh, indActive=actinds, valInactive=np.log(1e8)
)
mapping = Maps.ExpMap(mesh) * actmap
# Generate mtrue
ncy = mesh.nCy
ncz = mesh.nCz
ncx = mesh.nCx
mtrue = rho
print(mtrue.min(), mtrue.max())
clim = [10, 10000.]
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax = Utils.mkvc(ax)
# ax[0].title('Ground Truth, Vertical')
# ax[0].set_aspect('equal')
# ax[1].title('Ground Truth, Vertical')
# ax[1].set_aspect('equal')
dat = mesh.plotSlice(((mtrue)), ax=ax[0], normal='Z', clim=clim,
                     ind=int((ncz / 2) - 4))
ax[0].plot(rx[:, 0], rx[:, 1], 'or')
mesh.plotSlice(((mtrue)), ax=ax[1], normal='Y', clim=clim,
               ind=int(ncy / 2))
mesh.plotSlice(((mtrue)), ax=ax[2], normal='X', clim=clim,
               ind=int(ncx / 2))
cbar_ax = fig.add_axes([0.82, 0.15, 0.05, 0.7])
cb = plt.colorbar(dat[0], ax=cbar_ax)
fig.subplots_adjust(right=0.85)
cb.set_label('rho')
cbar_ax.axis('off')
plt.show()

# problem = DC.Problem3D_CC(mesh, sigmaMap=mapping)
# problem.pair(survey)
# problem.Solver = Solver
# print("dpred time")
# survey.dpred(mtrue[actinds])
# print("makeSyntheticData")
# survey.makeSyntheticData(mtrue, std=0.05, force=True)
# DCUtils.writeUBC_DCobs(fileName1,
#                        survey,
#                        3,
#                        'GENERAL',
#                        survey_type='pole-dipole')
# plt.plot(rx[:, 0], rx[:, 1], 'o')
# plt.plot(tx[:, 0], tx[:, 1], 'dr')
# for i in range(len(dipoles)):
#     plto = np.asarray(dipoles[i])
#     plt.plot(plto[:, 0], plto[:, 1], '-')
# plt.title("Rx Dipole Density")
# plt.xlabel("Easting (m)")
# plt.ylabel("Northing (m)")
# plt.show()
# 3D Mesh
#########

# # Cell sizes
# csx, csy, csz = 1., 1., 0.5
# # Number of core cells in each direction
# ncx, ncy, ncz = 41, 31, 21
# # Number of padding cells to add in each direction
# npad = 7
# # Vectors of cell lengths in each direction with padding
# hx = [(csx, npad, -1.5), (csx, ncx), (csx, npad, 1.5)]
# hy = [(csy, npad, -1.5), (csy, ncy), (csy, npad, 1.5)]
# hz = [(csz, npad, -1.5), (csz, ncz)]
# # Create mesh and center it
# mesh = Mesh.TensorMesh([hx, hy, hz], x0="CCN")

# # 2-spheres Model Creation
# ##########################

# # Spheres parameters
# x0, y0, z0, r0 = -6., 0., -3.5, 3.
# x1, y1, z1, r1 = 6., 0., -3.5, 3.

# # ln conductivity
# ln_sigback = -5.
# ln_sigc = -3.
# ln_sigr = -6.

# # Define model
# # Background
# mtrue = ln_sigback * np.ones(mesh.nC)

# # Conductive sphere
# csph = (np.sqrt((mesh.gridCC[:, 0] - x0)**2. + (mesh.gridCC[:, 1] - y0)**2. +
#                 (mesh.gridCC[:, 2] - z0)**2.)) < r0
# mtrue[csph] = ln_sigc * np.ones_like(mtrue[csph])

# # Resistive Sphere
# rsph = (np.sqrt((mesh.gridCC[:, 0] - x1)**2. + (mesh.gridCC[:, 1] - y1)**2. +
#                 (mesh.gridCC[:, 2] - z1)**2.)) < r1
# mtrue[rsph] = ln_sigr * np.ones_like(mtrue[rsph])

# # Extract Core Mesh
# xmin, xmax = -20., 20.
# ymin, ymax = -15., 15.
# zmin, zmax = -10., 0.
# xyzlim = np.r_[[[xmin, xmax], [ymin, ymax], [zmin, zmax]]]
# actind, meshCore = Utils.meshutils.ExtractCoreMesh(xyzlim, mesh)


# # Function to plot cylinder border
# def getCylinderPoints(xc, zc, r):
#     xLocOrig1 = np.arange(-r, r + r / 10., r / 10.)
#     xLocOrig2 = np.arange(r, -r - r / 10., -r / 10.)
#     # Top half of cylinder
#     zLoc1 = np.sqrt(-xLocOrig1**2. + r**2.) + zc
#     # Bottom half of cylinder
#     zLoc2 = -np.sqrt(-xLocOrig2**2. + r**2.) + zc
#     # Shift from x = 0 to xc
#     xLoc1 = xLocOrig1 + xc * np.ones_like(xLocOrig1)
#     xLoc2 = xLocOrig2 + xc * np.ones_like(xLocOrig2)

#     topHalf = np.vstack([xLoc1, zLoc1]).T
#     topHalf = topHalf[0:-1, :]
#     bottomHalf = np.vstack([xLoc2, zLoc2]).T
#     bottomHalf = bottomHalf[0:-1, :]

#     cylinderPoints = np.vstack([topHalf, bottomHalf])
#     cylinderPoints = np.vstack([cylinderPoints, topHalf[0, :]])
#     return cylinderPoints


# # Setup a synthetic Dipole-Dipole Survey
# # Line 1
# xmin, xmax = -15., 15.
# ymin, ymax = 0., 0.
# zmin, zmax = 0, 0
# endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
# survey1 = DCUtils.gen_DCIPsurvey(endl, "dipole-dipole", dim=mesh.dim,
#                                  a=3, b=3, n=8)

# # Line 2
# xmin, xmax = -15., 15.
# ymin, ymax = 5., 5.
# zmin, zmax = 0, 0
# endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
# survey2 = DCUtils.gen_DCIPsurvey(endl, "dipole-dipole", dim=mesh.dim,
#                                  a=3, b=3, n=8)

# # Line 3
# xmin, xmax = -15., 15.
# ymin, ymax = -5., -5.
# zmin, zmax = 0, 0
# endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
# survey3 = DCUtils.gen_DCIPsurvey(endl, "dipole-dipole", dim=mesh.dim,
#                                  a=3, b=3, n=8)

# # Concatenate lines
# survey = DC.Survey(survey1.srcList + survey2.srcList + survey3.srcList)

# # Setup Problem with exponential mapping and Active cells only in the core mesh
# expmap = Maps.ExpMap(mesh)
# mapactive = Maps.InjectActiveCells(mesh=mesh, indActive=actind,
#                                    valInactive=-5.)
# mapping = expmap * mapactive
# problem = DC.Problem3D_CC(mesh, sigmaMap=mapping)
# problem.pair(survey)
# problem.Solver = Solver

# survey.dpred(mtrue[actind])
# survey.makeSyntheticData(mtrue[actind], std=0.05, force=True)


# # Tikhonov Inversion
# ####################

# # Initial Model
# m0 = np.median(ln_sigback) * np.ones(mapping.nP)
# # Data Misfit
# dmis = DataMisfit.l2_DataMisfit(survey)
# # Regularization
# regT = Regularization.Simple(mesh, indActive=actind, alpha_s=1e-6,
#                              alpha_x=1., alpha_y=1., alpha_z=1.)

# # Optimization Scheme
# opt = Optimization.InexactGaussNewton(maxIter=10)

# # Form the problem
# opt.remember('xc')
# invProb = InvProblem.BaseInvProblem(dmis, regT, opt)

# # Directives for Inversions
# beta = Directives.BetaEstimate_ByEig(beta0_ratio=1e+1)
# Target = Directives.TargetMisfit()
# betaSched = Directives.BetaSchedule(coolingFactor=5., coolingRate=2)

# inv = Inversion.BaseInversion(invProb, directiveList=[beta, Target,
#                                                       betaSched])
# # Run Inversion
# minv = inv.run(m0)

# # Final Plot
# ############

# fig, ax = plt.subplots(2, 2, figsize=(12, 6))
# ax = Utils.mkvc(ax)

# cyl0v = getCylinderPoints(x0, z0, r0)
# cyl1v = getCylinderPoints(x1, z1, r1)

# cyl0h = getCylinderPoints(x0, y0, r0)
# cyl1h = getCylinderPoints(x1, y1, r1)

# clim = [(mtrue[actind]).min(), (mtrue[actind]).max()]

# dat = meshCore.plotSlice(((mtrue[actind])), ax=ax[0], normal='Y', clim=clim,
#                          ind=int(ncy / 2))
# ax[0].set_title('Ground Truth, Vertical')
# ax[0].set_aspect('equal')

# meshCore.plotSlice((minv), ax=ax[1], normal='Y', clim=clim, ind=int(ncy / 2))
# ax[1].set_aspect('equal')
# ax[1].set_title('Inverted Model, Vertical')

# meshCore.plotSlice(((mtrue[actind])), ax=ax[2], normal='Z', clim=clim,
#                    ind=int(ncz / 2))
# ax[2].set_title('Ground Truth, Horizontal')
# ax[2].set_aspect('equal')

# meshCore.plotSlice((minv), ax=ax[3], normal='Z', clim=clim, ind=int(ncz / 2))
# ax[3].set_title('Inverted Model, Horizontal')
# ax[3].set_aspect('equal')

# for i in range(2):
#     ax[i].plot(cyl0v[:, 0], cyl0v[:, 1], 'k--')
#     ax[i].plot(cyl1v[:, 0], cyl1v[:, 1], 'k--')
# for i in range(2, 4):
#     ax[i].plot(cyl1h[:, 0], cyl1h[:, 1], 'k--')
#     ax[i].plot(cyl0h[:, 0], cyl0h[:, 1], 'k--')

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# cb = plt.colorbar(dat[0], ax=cbar_ax)
# cb.set_label('ln conductivity')

# cbar_ax.axis('off')

# plt.show()
