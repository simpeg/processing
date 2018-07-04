import DCIPtools as DCIP
import matplotlib.pyplot as plt
import numpy as np


def getData():
    xyz = open("/Users/juan/Documents/testData/SR-PG/L100_R48_SR_PG.xyz")
    xt = []
    for line in xyz:
        x = line
        xt.append(float(x))

    xt = np.asarray(xt)
    return xt


xt = getData()
amp = DCIP.getFrequnceyResponse(xt)
freqs = np.arange(0, amp.size) * (75. / amp.size)
print(amp.size, freqs.size)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(xt)
ax2.semilogy(freqs, amp)
plt.show()
