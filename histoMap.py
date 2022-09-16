import numpy as np
import os
import matplotlib.pyplot as plt
import numpy.linalg as la
import multiprocess as mp
import itertools
import tqdm
from matplotlib.gridspec import GridSpec


class map:
    """Class for cavitation mapping"""

    def __init__(self, wfs, fs=-1, c=1500, xpos=np.arange(-5e-3, 5e-3, 0.5e-3), ypos=np.arange(-5e-3, 5e-3, 0.5e-3), zpos=np.arange(-5e-3, 5e-3, 0.5e-3)):
        """
            wfs: 
            xpos: numpy array of x positions for mapping grid [m]
            xpos: numpy array of x positions for mapping grid [m]
            xpos: numpy array of x positions for mapping grid [m]
            xdcr: numpy array of transducer element coordinates [m]
            fs: sampling frequency of transducer [Hz]"""

        self.wfs = wfs
        if len(self.wfs.shape) == 2:
            print('Reformatting wfs...')
            self.wfs = self.wfs[np.newaxis, :, :]

        if fs == -1:
            print('Enter the sampling frequency being used [Hz]: ')
            self.fs = input()
            print('Using fs = ', self.fs)
        else:
            self.fs = fs

        self.xpos = xpos
        self.ypos = ypos
        self.zpos = zpos

        self.xdcr = np.load('arrayCoords.npy')[:260, :]
        self.c = c

        if len(self.wfs.shape) == 3:
            self.npulses = self.wfs.shape[0]
        else:
            self.npulses = 1

        self.t0 = 0
        self.dur = self.wfs.shape[-1]/self.fs
        self.totchan = len(self.xdcr)
        print('totchan:', self.totchan)

        self.delayIdx = []
        # self.delayIdx = self.calcPhaseShiftIdxs()

    def readBinary(fname, nchan=8):
        """Used to read binary data and convert to numpy array
        inputs:
            fname: file name
            nchan: number of channels per board (default of 8)"""

        reclen = int(round(self.fs*self.dur))
        reclen_extra = reclen+2
        data = np.zeros((self.npulses, self.totchan, reclen))

        for i in np.arange(1, int(self.totchan/nchan)):

            filename = fname+str(i-1)
            f = open(filename, 'rb')
            A = np.fromfile(f, dtype=np.int16)
            B = np.reshape(A, (self.npulses, nchan, reclen_extra))
            C = B[:, :, 2:]
            data[:, (i-1)*8+0:i*8, :] = C

        return data

    def calcPhaseShiftIdxs(self,):
        """Function for calculating phase delay (units of index) for TEAPAM and other mapping techniques."""

        print('Calculating Phase Delays with c = ', self.c)

        delayIdx = np.zeros((len(self.xpos), len(self.ypos),
                            len(self.zpos), self.xdcr.shape[0]), dtype=int)

        for xi in range(len(self.xpos)):
            x = self.xpos[xi]
            for yi in range(len(self.ypos)):
                y = self.ypos[yi]
                for zi in range(len(self.zpos)):
                    z = self.zpos[zi]
                    pos = [x, y, z]
                    deltaDDir = self.xdcr - pos
                    # deltaD = (deltaDDir[:, 0]**2 + deltaDDir[:, 1]** 2 + deltaDDir[:, 2]**2)**0.5
                    deltaD = la.norm(deltaDDir, axis=1)**0.5
                    delayIdx[xi, yi, zi, :] = np.rint(
                        (deltaD - min(deltaD))/self.c*self.fs)

        return delayIdx

    def teapam(self, loc):
        """Function to perform TEAPAM for a single point (xi,yi,zi) [index] in previously defined grid,
            using previously defined phase delays"""

        if type(loc) != list:
            loc = [loc]
        result = np.zeros((len(loc), self.wfs.shape[0]))

        for lIdx, (xi, yi, zi) in enumerate(loc):
            rowsE, column_indicesE = np.ogrid[:self.wfs.shape[1],
                                              :self.wfs.shape[2]]
            r = self.delayIdx[xi, yi, zi, :]

            column_indicesE = column_indicesE - r[:, np.newaxis]

            for i in range(self.wfs.shape[0]):
                # resultE = self.wfs[i, rowsE, column_indicesE.astype('int')]
                resultE = self.wfs[i, rowsE, column_indicesE]
                # result[lIdx, i] = np.sum(np.square(np.sum(resultE, axis=0)))
                result[lIdx, i] = la.norm(np.sum(resultE, axis=0))

        return loc, result

    def init_worker(self,):
        """Initialization function for parallel computing"""
        pass

    def parallelTEAPAM(self, numProcesses=4):
        """Function to run parallel TEAPAM calculations for map object
        inputs:
            numProcesses (Optional): number of parallel processes to run (default: 4)
                Optimal number will be different for different machines"""

        self.delayIdx = self.calcPhaseShiftIdxs()

        xind = np.arange(len(self.xpos))
        yind = np.arange(len(self.ypos))
        zind = np.arange(len(self.zpos))

        locations = list(itertools.product(*[xind, yind, zind]))
        hmapO = np.zeros((np.shape(self.wfs)[0], len(
            self.xpos), len(self.ypos), len(self.zpos)))

        with mp.Pool(processes=numProcesses, initializer=self.init_worker, initargs=()) as p:
            for (loc, res) in tqdm.tqdm(p.imap(self.teapam, locations), total=len(list(locations))):
                for lIdx, (xi, yi, zi) in enumerate(loc):
                    hmapO[:, xi, yi, zi] = res[lIdx]

        return hmapO


if __name__ == "__main__":

    # wfs = np.load('C:\\Users\labuser\\Dropbox (University of Michigan)\\Rapid Ablation Study\\jan6\\20VCalib.npy')[
    #     :, :260, 700:1000]
    # fs = 12.5e6

    # print(wfs.shape)

    nchan = 8
    npulses = 501
    dur = 100e-6
    fs = 5e6
    reclen = int(round(fs*dur))
    reclen_extra = reclen+2

    data = np.zeros((npulses, 260, reclen))
    for i in np.arange(1, 33):
        #        folder = '/home/liverarray/Documents/Greyson/jan12_2_foci_water_1Hz_3y/'
        #        filename = folder+'2_foci_water_1Hz_3y'+str(i-1)
        # folder = 'C:\\Users\\labuser\\Dropbox (University of Michigan)\\Bubble Dissolution Study\\August Pig\\augPigDiss14\\augPigDiss14_'
        folder = 'C:\\Users\\labuser\\Dropbox (University of Michigan)\\Bubble Dissolution Study\\ExVivoDegassedDiss\\9_9_22_DissMeasureSamp1_1\\9_9_22_DissMeasureSamp1_1_'
        filename = folder+str(i-1)
        # print(filename)
        f = open(filename, 'rb')
        A = np.fromfile(f, dtype=np.int16)
        B = np.reshape(A, (npulses, nchan, reclen_extra))
        C = B[:, :, 2:]
        data[:, (i-1)*8+0:i*8, :] = C
    wfs = data[1:, :, 50:200]
    # plt.figure()
    # plt.imshow(wfs)
    # plt.colorbar()
    # plt.show()

    mapI = map(wfs, fs=fs)

    mapI.xpos = np.arange(-3e-3, 3e-3, 1e-3)
    mapI.ypos = np.arange(-3e-3, 3e-3, 1e-3)
    mapI.zpos = np.arange(-2e-3, 8e-3, 1e-3)

    hmap = mapI.parallelTEAPAM()
    np.save('testDataExVivo', hmap)
    # proj = np.sum(hmap, axis=1)

    # maxlvl = np.max(proj[:])
    # minlvl = np.min(proj[:])

    # fig = plt.figure()

    # gs = GridSpec(1, 2, figure=fig)
    # ax0 = fig.add_subplot(gs[0, 0])
    # ax1 = fig.add_subplot(gs[0, 1])

    # for locN in range(proj.shape[0]):
    #     if locN:
    #         ax0.clear()

    #     ax0.imshow(proj[locN, :, :], vmin=minlvl, vmax=maxlvl)
    #     ax0.set_title(locN*2)

    #     ax1.plot(locN, np.max(proj[locN, :, :]), '.')
    #     plt.pause(.1)

    # print(hmap.shape)
