""" Treatment planning package for histotripsy """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class treatment:

    def __init__(self, steering_width, point_spacing, shape='sphere'):
        """steering_width: diameter of volume
        point_spacing: spacing between points (mm) (default = 1 mm)
        shape: sphere or cube (default = cube)"""

        self.xdcr = np.load('arrayCoords.npy')
        self.steering_width = steering_width
        self.point_spacing = point_spacing
        self.shape = shape
        self.corner = [90.6, 77.2, -73.9]
        self.pts = np.round(self.hcp_maker(), decimals=2)
        self.numPts = len(self.pts)
        # self.treatmentDF = self.makeShellDF()
        self.treatmentDF = self.makeLundtLatDF()
        # self.treatmentDF = self.makeRasterDF()
        self.pts = [self.treatmentDF['x'].values,self.treatmentDF['y'].values,self.treatmentDF['z'].values]
        
    def makeRasterDF(self, ):
        """Function to make DF for raster scan treatment"""
        return self.initDF()

    def makeLundtLatDF(self, minLatSpacing=5, minAxSpacing=5):
        """Function for grouping points based on a minimum separation
        inputs:
            df: initialized treatment dataframe
            minLatSpacing: minimum lateral spacing [mm]
            minAxSpacing: minimum axial spacing [mm]"""
        
        df = self.initDF()
        groupsX = np.arange(np.min(self.pts[:, 0]), np.max(
            self.pts[:, 0]), minLatSpacing)
        groupsY = np.arange(np.min(self.pts[:, 1]), np.max(
            self.pts[:, 1]), minLatSpacing)
        groupsZ = np.arange(np.min(self.pts[:, 2]), np.max(
            self.pts[:, 2]), minAxSpacing)

        for i in range(len(groupsX)):
            df.loc[df['x'] >= groupsX[i], 'xGroup'] = i

        for i in range(len(groupsY)):
            df.loc[df['y'] >= groupsY[i], 'yGroup'] = i

        for i in range(len(groupsZ)):
            df.loc[df['z'] >= groupsZ[i], 'zGroup'] = i

        subGroup = 0

        for xi in range(len(groupsX)):
            for yi in range(len(groupsY)):
                for zi in range(len(groupsZ)):
                    df.loc[(df['xGroup'] == xi) & (df['yGroup'] == yi)
                           & (df['zGroup'] == zi), 'subGroup'] = subGroup
                    if np.isin(subGroup, df['subGroup'].values):
                        subGroup = subGroup + 1

        numPts = len(df)  # Get length of df

        # Add column for relative X coor within group
        df['xRelInd'] = [None]*numPts
        # df['xRelCoor'] = df['x'] - groupsX[df['xGroup']]
        # Add column for relative Y coor within group
        df['yRelInd'] = [None]*numPts
        # Add column for relative Z coor within group
        df['zRelInd'] = [None]*numPts

        for group in range(df['subGroup'].max()+1):
            xcoors = np.sort(
                np.unique(df[df['subGroup'] == group]['x'].values))
            for xi in range(len(xcoors)):
                df.loc[(df['subGroup'] == group) & (df['x']
                                                    == xcoors[xi]), 'xRelInd'] = xi

            ycoors = np.sort(
                np.unique(df[df['subGroup'] == group]['y'].values))
            for yi in range(len(ycoors)):
                df.loc[(df['subGroup'] == group) & (df['y']
                                                    == ycoors[yi]), 'yRelInd'] = yi

            zcoors = np.sort(
                np.unique(df[df['subGroup'] == group]['z'].values))
            for zi in range(len(zcoors)):
                df.loc[(df['subGroup'] == group) & (df['z']
                                                    == zcoors[zi]), 'zRelInd'] = zi

        newDF = self.getSubGroupOrder(df)

        oI = 0
        zRel = np.sort(np.unique(newDF['zRelInd'].values))[::-1]
        yRel = np.unique(newDF['yRelInd'].values)
        np.random.shuffle(yRel)
        xRel = np.unique(newDF['xRelInd'].values)
        np.random.shuffle(xRel)
        for zi in zRel:
            for yi in yRel:
                for xi in xRel:
                    groups = newDF[(newDF['xRelInd'] == xi) & (newDF['yRelInd'] == yi) & (
                        newDF['zRelInd'] == zi)]['subGroupOrder'].values
                    for group in groups:
                        newDF.loc[(newDF['xRelInd'] == xi) & (newDF['yRelInd'] == yi) & (
                            newDF['zRelInd'] == zi) & (newDF['subGroupOrder'] == group), 'order'] = oI
                        # if np.isin(oI, newDF['order'].values):
                        oI = oI + 1
        # newDF['order'] = (newDF['order']-newDF['order'].max()).abs()
        newDF.sort_values(by=['order'], inplace=True)
        newDF.set_index('order', inplace=True)

        return newDF

    def getSubGroupOrder(self, df):

        df['subGroupOrder'] = [None]*len(df)
        orderI = df['subGroup'].max()
        for i in range(int(df['zGroup'].max())+1):
            groups = np.unique(df[df['zGroup'] == i]['subGroup'].values)
            for group in groups:
                df.loc[df['subGroup'] == group, 'subGroupOrder'] = orderI
                orderI = orderI-1

        return df

    def initDF(self,):
        """Function to initialize treatment dataframe after treatment points
        have been defined - note that this function does not populate subgroups, etc"""
        numPts = len(self.pts[:, 0])        # get number of points in volume
        # get all unique x & y values in set of points

        # generate dataframe with coors
        df = pd.DataFrame(self.pts, columns=['x', 'y', 'z'])

        # Add some columns to dataframe
        df['xGroup'] = [None]*numPts  # which x-plane the point is located in
        df['yGroup'] = [None]*numPts  # which y-plane the point is located in
        # which column the point is located in - actual value is meaningless
        df['colNum'] = [None]*numPts
        # just used to identify columns between eachother
        # which ablation sub-group the point is part of
        df['subGroup'] = [None]*numPts
        # final order which the transducer will fire at
        df['order'] = [None]*numPts

        return df

    def makeShellDF(self,):
        """Function to create shell-based anti-blocking treatment dataframe"""
        df = self.initDF()
        df = self.columnize(df)
        df = self.seedSubGroup(df)
        df = self.windowGroup(df)
        df = self.shellOrder(df)
        print(df)
        # df.sort_values(by=['order'], inplace=True)
        # df.set_index('order', inplace=True)

        # Need to add final ordering

        return df

    def shellOrder(self, df):

        df.sort_values(by=['subGroup'], inplace = True)

        numSGs = df['subGroup'].max()

        oi = 0

        for i in range(numSGs):
            sdf = df[df['subGroup']==i]
            order = np.arange(oi,oi+len(sdf))
            
            locInd = sdf.index[0]

            df.loc[df['subGroup']==i,'order'] = order
            oi = np.max(order)+1
        
        newDF = df.sort_values(by=['order'])
        newDF.set_index('order', inplace=True)
        return newDF

    def hcp_maker(self,):
        """Used to generate a treatment grid of hexagonally packed points
        steering_width: diameter of volume
        point_spacing: spacing between points (mm) (default = 1 mm)
        shape: sphere or cube (default = cube)"""

        self.point_spacing *= 0.5

        nz = np.ceil((self.steering_width/self.point_spacing) /
                     (2*np.sqrt(6)/3)).astype(int)
        ny = np.ceil((self.steering_width/self.point_spacing) /
                     np.sqrt(3)+nz % 2).astype(int)
        nx = np.ceil((self.steering_width/self.point_spacing) /
                     2+(nz+ny) % 2).astype(int)

        nPoints = nx*ny*nz

        pts = np.zeros((nPoints, 3))
        m = 0
        i, j, k = 0, 0, 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    pts[m, 0] = (2*i+((j+k) % 2))*self.point_spacing
                    pts[m, 1] = (np.sqrt(3)*(j+1/3*(k % 2)))*self.point_spacing
                    pts[m, 2] = ((2*np.sqrt(6)/3)*k)*self.point_spacing
                    m += 1

        pts -= np.mean(pts, axis=0)

        if self.shape == 'sphere':
            final_idx = 0
            pts_final = np.zeros(pts.shape)
            for i in range(len(pts)):
                rad = (pts[i, 0]**2 + pts[i, 1]**2 + pts[i, 2]**2)**(1/2)
                if rad < self.steering_width/2:
                    pts_final[final_idx, :] = pts[i, :]
                    final_idx = final_idx + 1
            pts = pts_final[0:final_idx, :]
        pts[:,2] = pts[:,2]*-1
        return(pts)

    def plotTreatment(self, groups=False):
        "Used to quickly and simply make 3D scatter plot of points"
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if groups == False:
            ax.scatter(self.pts[:, 0], self.pts[:, 1],
                       self.pts[:, 2], marker='o', color='b')

        if groups == True:
            for i in range(self.treatmentDF['subGroup'].max()+1):
                sdf = self.treatmentDF[self.treatmentDF['subGroup'] == i]

                ax.scatter(sdf['x'], sdf['y'],
                           sdf['z'], marker='o')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    def plotTransducer(self, ):
        "Used to quickly and simply make 3D scatter plot of the transducer"
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.xdcr[:, 0], self.xdcr[:, 1],
                   self.xdcr[:, 2], marker='o', color='r')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    def blockCheck(self, EFS, block):
        '''Used to check if shield point blocks EFSloc based on transducer aperture, returns True if blocked'''
        '''corner: 1x3 array of positive x,y corner of array, focused at [0, 0, 0], assumed to be symmetric'''
        '''Array assumed to be in -z space, pointed at origin'''
        '''EFS: 1x3 array of location of point you are steering to'''
        '''block: 1x3 array of location of point which may block aperture'''

        tCorner = [self.corner[0], self.corner[1], -1*self.corner[2]]
        EFSloc = [EFS[0], EFS[1], -1*EFS[2]]
        shield = [block[0], block[1], -1*block[2]]

        assert all(
            i > 0 for i in tCorner), "x,y coordinates need to be > 0, z coordinate needs to be < 0"
        if shield[2] < EFSloc[2]:
            return False

        def subcheck(tCorner, EFSloc, shield):

            l = tCorner[0] - EFSloc[0]
            m = tCorner[1] - EFSloc[1]
            n = tCorner[2] - EFSloc[2]

            N = (shield[2] - EFSloc[2])/n

            if tCorner[1] > 0:
                ycheck = N*m + EFSloc[1] > shield[1]
            else:
                ycheck = N*m + EFSloc[1] < shield[1]
            if tCorner[0] > 0:
                xcheck = N*l + EFSloc[0] > shield[0]
            else:
                xcheck = N*l + EFSloc[0] < shield[0]

            return (ycheck & xcheck)

        check1 = subcheck(tCorner, EFSloc, shield)
        tCorner[0] = tCorner[0] * -1
        check2 = subcheck(tCorner, EFSloc, shield)
        tCorner[1] = tCorner[1] * -1
        check3 = subcheck(tCorner, EFSloc, shield)
        tCorner[0] = tCorner[0] * -1
        check4 = subcheck(tCorner, EFSloc, shield)
        tCorner[1] = tCorner[1] * -1
        return (check1 & check2 & check3 & check4)

    def columnize(self, df):
        numPts = len(self.pts[:, 0])        # get number of points in volume
        # get all unique x & y values in set of points
        uXVals = np.unique(self.pts[:, 0])
        uYVals = np.unique(self.pts[:, 1])

        xi = 0
        for xcoor in uXVals:    # Determine which xGroup point is part of
            df.loc[df.x == xcoor, 'xGroup'] = xi
            xi = xi + 1

        yi = 0
        for ycoor in uYVals:         # Determine which yGroup point is part of
            df.loc[df.y == ycoor, 'yGroup'] = yi
            yi = yi + 1

        # grouping into different column numbers - not all values are used for arbitrary volumes
        df['colNum'] = df['xGroup']*len(uYVals) + df['yGroup']

        return df

    def seedSubGroup(self, df):
        ''' Not to generally be used outside of other functions '''
        ''' df: dataframe from columnize(pts) function '''
        ''' outputs dataframe with subGroup seeded NOT FINAL!!!! '''
        numPts = len(df)
        columns = np.unique(df['colNum'])  # get all unique column numbers

        # generate seed values for determining subGroups
        # this block of code basically guesses which subGroup the point will be in to
        # speed things up later.  All it does is takes the point furthest from the transducer
        # in each column, sets that to subGroup = 0, the next furthest to be subGroup = 1, etc.
        groupInd = 0
        while df.isnull().sum(axis=0)['subGroup']:
            print((numPts-df.isnull().sum(axis=0)
                  ['subGroup'])/numPts*100, '% done.')
            nulldf = df[df.isnull()['subGroup']]
            # print(nulldf.head())
            for col in columns:
                subFrame = nulldf[(nulldf['colNum'] == col)]
                # print(subFrame)
                if not subFrame.empty:
                    subLoc = subFrame[subFrame['z'] == max(subFrame['z'])]
                    locInd = subLoc.index[0]
                    df.iloc[locInd, df.columns.get_loc('subGroup')] = groupInd

            groupInd = groupInd+1
        print('Finished!')
        return df

    def windowGroup(self, df):
        tCorner = self.corner
        numPts = len(df)
        df['bCheck'] = [None]*numPts
        sGNum = 0
        while ((len(df[df['subGroup'] == sGNum]) > 0) or (sGNum == 0)):
            # % done line isn't totally accurate, will print out 100% done multiple times...
            # but it gives some idea of how progress is going
            print(((numPts-df.isnull().sum(axis=0)
                  ['bCheck'])/numPts)*100, '% done')
            checkDF = df[df['subGroup'] == sGNum]
            prevDF = df[df['subGroup'] <= sGNum]

            for index, row in checkDF.iterrows():
                blockDF = prevDF[prevDF['z'] < row['z']]
                for sIndex, sRow in blockDF.iterrows():
                    # tCorner = [cx, cy, cz]
                    EFSloc = [row['x'], row['y'], row['z']]
                    shield = [sRow['x'], sRow['y'], sRow['z']]
                    ind = self.blockCheck(EFSloc, shield)
                    # print(ind)
                    if ind == True:
                        # print('Blocked!')
                        df.iloc[sIndex, df.columns.get_loc(
                            'subGroup')] = df.iloc[sIndex, df.columns.get_loc('subGroup')] + 1
                        # drop this loc from prevDF
                        prevDF = df[df['subGroup'] <= sGNum]
                df.iloc[index, df.columns.get_loc('bCheck')] = 1
            sGNum = sGNum + 1
        print('Finished!')
        return df


    def animateTreatment(self, ):
        """Function for animating treatment pattern
        Scatter-plots treatment points in order, but does not account for how subgroups
        may be repeated before continuing in the treatment order (i.e. it plots all points one time)"""

        xlims = (self.treatmentDF['x'].min()*1.1,self.treatmentDF['x'].max()*1.1)
        ylims = (self.treatmentDF['y'].min()*1.1,self.treatmentDF['y'].max()*1.1)
        zlims = (self.treatmentDF['z'].min()*1.1,self.treatmentDF['z'].max()*1.1)


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_zlim(zlims)

        for index,row in self.treatmentDF.iterrows():
            ax.clear()
            ax.scatter(row['x'], row['y'], row['z'])
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.set_zlim(zlims)
            plt.pause(0.001)
            fig.show()


if __name__ == "__main__":

    test = treatment(10, 1, shape='sphere')
    test.animateTreatment()
    # test.plotTreatment(groups=True)
    