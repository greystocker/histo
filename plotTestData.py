import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

hmap = np.load('testDataExVivo.npy')

proj = np.sum(hmap, axis=1)

maxlvl = np.max(proj[:])
minlvl = np.min(proj[:])

fig = plt.figure()

gs = GridSpec(1, 2, figure=fig)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax1.set_xlim([-10, 1100])
ax1.set_ylim([0, 5])
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Max TEAPAM Projection Signal')
ax1.set_title('Max TEAPAM Projection Amplitude')
ax0.set_xlabel('Z (mm)')
ax0.set_ylabel('X (mm)')
ax0.set_title('TEAPAM')
plt.pause(5)

amp = np.zeros(proj.shape[0])

for locN in range(proj.shape[0]):
    if locN:
        ax0.clear()

    ax0.imshow(proj[locN, :, :], vmin=minlvl, vmax=maxlvl)
    ax0.set_title(str(locN*2)+' ms')

    amp[locN] = np.max(proj[locN, :, :]) - np.median(proj[locN, :, :])

    ax1.plot(2*locN, amp[locN]/1e5, 'r.',)
    plt.pause(.001)


# plt.figure()
# plt.plot(np.linspace(0, 1000, 500), amp)
