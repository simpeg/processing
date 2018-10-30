import matplotlib.pyplot as plt
import numpy as np
import JDataObject as Jdata
import pylab as plt
from scipy.interpolate import griddata

################################################################
# define the file required for import
fileName = "/Users/juan/Documents/testData/DAT_B.DAT"

unitType = "appResistivity"
# unitType = "appChareability"
xp = []
yp = []
val = []
val2 = []
Nlevel = ["N = 1", "N = 2", "N = 3", "N = 4", "N = 5", "N = 6", "N = 7"]
z_n = [-100, -200, -300, -400, -500, -600, -700]
vmin_rho, vmax_rho = 10, 4000
vmin_mx, vmax_mx = 0, 18

# =================================================================
# Code Start
patch = Jdata.loadDias(fileName)               # loads data
# calculated mid-pt data points
for src in range(len(patch.readings)):
    for rx in range(len(patch.readings[src].Vdp)):
        xp.append(
            patch.readings[src].Vdp[rx].getXplotpoint(patch.readings[src].Idp))
        yp.append(
            patch.readings[src].Vdp[rx].getZplotpoint(patch.readings[src].Idp))
        val.append(
            np.abs(patch.readings[src].Vdp[rx].Rho))
        val2.append(
            np.abs(patch.readings[src].Vdp[rx].Mx))
# convert to numpy
midx = np.asarray(xp)
midz = np.asarray(yp)
rho = np.asarray(val)
mx = np.asarray(val2)
xNLevel = np.min(midx) - 200
x_n = np.zeros(len(z_n))
x_n = x_n + xNLevel
# Grid points
grid_x, grid_z = np.mgrid[np.min(midx):np.max(midx),
                          np.min(midz):np.max(midz)]
# create an axes to plot on
ax = plt.subplot(2, 1, 1, aspect='equal')
# create an axes to plot on
ax2 = plt.subplot(2, 1, 2, aspect='equal')
# check which data to plot
# if unitType == "appResistivity":
vmin = vmin_rho
vmax = vmax_rho
ax.axes.set_title("Apparent Resistivity (ohm-m)", y=1.14)
grid_rho = griddata(np.c_[midx, midz], rho.T, (grid_x, grid_z),
                    method='cubic')
grid_rho = np.ma.masked_where(np.isnan(grid_rho), grid_rho)
name = 'custom_div_cmap'
pcolorOpts = {}
from matplotlib.colors import LinearSegmentedColormap
custom_map = LinearSegmentedColormap.from_list(name=name, colors=['aqua', [0, 0.85, 1, 1], [0.1, 1, 0.1, 1], 'yellow', [1, 0.7, 0, 1], [1, 0.2, 0.2, 1], [0.95, 0.9, 1, 1]], N=200)
CS = ax.contour(grid_x[:, 0], grid_z[0, :], grid_rho.T, 15, linewidths=0.5, colors='k')
ph = ax.pcolormesh(grid_x[:, 0], grid_z[0, :], grid_rho.T, cmap=custom_map, clim=(vmin, vmax), vmin=vmin, vmax=vmax, **pcolorOpts)
# , vmin=vmin, vmax=vmax, {})
cbar = plt.colorbar(ph, ax=ax, format="%.0f", fraction=0.04, orientation="vertical")
cbar.set_label("App.Res.", size=12)
for i, txt in enumerate(val):
    ax.annotate(int(rho[i]), (midx[i], midz[i]), size=8)
# else:
vmin = vmin_mx
vmax = vmax_mx
ax2.axes.set_title("Apparent Chargeability (mV/V)", y=1.11)
grid_mx = griddata(np.c_[midx, midz], mx.T, (grid_x, grid_z),
                   method='cubic')
grid_mx = np.ma.masked_where(np.isnan(grid_mx), grid_mx)
name = 'custom_div_cmap1'
pcolorOpts = {}
# from matplotlib.colors import LinearSegmentedColormap
# custom_map = LinearSegmentedColormap.from_list(name=name, colors=['aqua', [0, 0.85, 1, 1], [0.1, 1, 0.1, 1], 'yellow', [1, 0.7, 0, 1], [1, 0.2, 0.2, 1], [0.95, 0.9, 1, 1]], N=200)
CS2 = ax2.contour(grid_x[:, 0], grid_z[0, :], grid_mx.T, 15, linewidths=0.5, colors='k')
ph2 = ax2.pcolormesh(grid_x[:, 0], grid_z[0, :], grid_mx.T, cmap=custom_map, clim=(vmin, vmax), vmin=vmin, vmax=vmax, **pcolorOpts)
# , vmin=vmin, vmax=vmax, {})
cbar2 = plt.colorbar(ph2, ax=ax2, format="%.0f", fraction=0.04, orientation="vertical")
cbar2.set_label("App.Mx.", size=12)
for i, txt in enumerate(val):
    ax2.annotate(int(mx[i]), (midx[i], midz[i]), size=8)

for i, txt in enumerate(Nlevel):
    ax.annotate(Nlevel[i], (x_n[i], z_n[i]), size=9)
    ax2.annotate(Nlevel[i], (x_n[i], z_n[i]), size=9)

ax.axes.get_yaxis().set_visible(False)
ax.axes.set_ylim(np.min(midz) - 50, np.max(midz) + 50)
ax.axes.set_xlim(np.min(midx) - 250, np.max(midx) + 100)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.tick_params(labelsize=8)
ax2.axes.get_yaxis().set_visible(False)
ax2.axes.set_ylim(np.min(midz) - 50, np.max(midz) + 50)
ax2.axes.set_xlim(np.min(midx) - 250, np.max(midx) + 100)
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position('top')
ax2.tick_params(labelsize=8)
plt.show()
