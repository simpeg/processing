import matplotlib.pyplot as plt
import numpy as np
import DCIPtools as DCIP


def getTime():
    timeFrom = [2040, 2060, 2080, 2120, 2160, 2200,
                2240, 2320, 2400,
                2480, 2560, 2640,
                2720, 2800, 2960,
                3120, 3280, 3440,
                3600, 3760]
    timeTo = [2060, 2080, 2120, 2160, 2200, 2240,
              2320, 2400, 2480, 2560, 2640, 2720,
              2800, 2960, 3120, 3280, 3440,
              3600, 3760, 3920]
    return timeFrom, timeTo


fileName = "/Users/juan/Documents/testData/L1.DAT"
patch = DCIP.loadDias(fileName)
timeFrom, timeTo = getTime()
timeTo = np.asarray(timeTo)
timeFrom = np.asarray(timeFrom)
timeCenter = (timeTo + timeFrom) / 2.
decay = (patch.readings[0].Vdp[25].Vs /
         patch.readings[0].Vdp[25].Vp)
w_ = np.ones(timeCenter.size)             # cole-cole weights
w_[:3] = 0.3
c = 0.65
tau = 0.35
r = 4.0
stored_error = 0
min_iter = 3
for iter in range(12):
    c, tau, M, error, vs = DCIP.getColeCole(decay,
                                            c,
                                            tau,
                                            r,
                                            timeCenter,
                                            w_)
    delta_error = abs(stored_error - error) / ((stored_error + error) / 2.)
    print("iter: %i | c: %f | tau: %f | M: %f | error: %f | delta: %f" %
          (iter, c, tau, M, error, delta_error))
    stored_error = error
    # r = r / 2.
    if delta_error > 0.002 or iter < min_iter:
        r = r / 2.
    elif delta_error < 0.002:
        print("convergence accomplished! DONE")
        break

print(c, tau, M, error)
# print(decay)

plt.plot(timeCenter, decay, 'o-')
plt.plot(timeCenter, vs, 'o-r')
# for i in range(1):
#     for j in range(len(patch.readings[i].Vdp)):
#         decay = patch.readings[i].Vdp[j].Vs
#         plt.plot(timeCenter, decay, 'o-')
plt.show()
percent_diff = (
    np.sum(np.abs((decay - vs)) /
           ((vs + decay) / 2.)))
print(percent_diff)
print(np.sum((decay - vs)**2 * w_) / 20.)
