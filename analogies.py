import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex

def makePyramid(image, levels=4, filename=None):
    if filename:
        image = cv2.imread(filename)
    G = image.copy()
    gpImage = [G]
    for i in xrange(levels):
        G = cv2.pyrDown(G)
        gpImage.append(G)
    return gpImage

#numpy slicing/stacking is faster than python bytecode looping
def extractFeatures(features, neighborhoodSize, pixelI, pixelJ):
    height, width, num = features.shape
    radius = neighborhoodSize / 2
    startI, endI = pixelI - radius, pixelI + radius
    startJ, endJ = pixelJ - radius, pixelJ + radius
    mirrorTop, mirrorBottom, mirrorLeft, mirrorRight = [False]*4
    if startI < 0:
        startI = 0
        mirrorTop = True
    if endI > height - 1:
        endI = height - 1
        mirrorBottom = True
    if startJ < 0:
        startJ = 0
        mirrorLeft = True
    if endJ > width - 1:
        endJ = width - 1
        mirrorRight = True
    neighborhood = features[startI:endI + 1, startJ:endJ + 1]
    if mirrorTop:
        nH, nW, _ = neighborhood.shape
        toStack = neighborhood[0:neighborhoodSize - nH, :]
        neighborhood = np.vstack((np.flipud(toStack), neighborhood))
    if mirrorBottom:
        nH, nW, _ = neighborhood.shape
        toStack = neighborhood[nH*2 - neighborhoodSize:nH, :]
        neighborhood = np.vstack((neighborhood, np.flipud(toStack)))
    if mirrorLeft:
        nH, nW, _ = neighborhood.shape
        toStack = neighborhood[:, 0:neighborhoodSize - nW]
        neighborhood = np.hstack((np.fliplr(toStack), neighborhood))
    if mirrorRight:
        nH, nW, _ = neighborhood.shape
        toStack = neighborhood[:, nW*2 - neighborhoodSize:nW]
        neighborhood = np.hstack((neighborhood, np.fliplr(toStack)))
    neighborhood = np.array(neighborhood).reshape(neighborhoodSize*neighborhoodSize, num)
    neighborhood = neighborhood.reshape(neighborhood.shape[0]*neighborhood.shape[1])
    return neighborhood

def extractAllFeatures(features, neighborhoodSize):
    height, width, num = features.shape
    allF = []
    for i in range(height):
        for j in range(width):
            allF.append(extractFeatures(features, neighborhoodSize, i, j))
    return allF

def createImageAnalogy(A, A1, B):
    #Intialize B1
    height, width, channels = B.shape
    B1 = np.zeros((height,width,channels), np.uint8)

    #Compute Gaussian Pyramids
    levels = 4
    APyramid = makePyramid(A, levels)
    A1Pyramid = makePyramid(A1, levels)
    BPyramid = makePyramid(B, levels)
    B1Pyramid = makePyramid(B1, levels)

    #Features
    #features are BGR...for now
    AFeatures = A.copy()
    BFeatures = B.copy()

    #Search Structures
    #...

    #For level in coarsest to finest
    for l in reversed(xrange(levels)):
        print "level is: " + str(l)
        B1_l = B1Pyramid[l]
        B1_lh, B1_lw, _ = B1_l.shape
        print "level shape is: " + str(B1_l.shape)
        neigh = generateNN(APyramid, A1Pyramid, l, levels)
        #For pixel in B1_l in linescan order
        for i in range(0, B1_lh):
            for j in range(0, B1_lw):
                # print "Curr iter: " + str((i,j))
                p_x, p_y = bestMatch(neigh, BPyramid, B1Pyramid, l, (i, j), levels)
                B1_l[i][j] = A1[p_y][p_x]
        print "Done with level"
    return B1Pyramid

def generateNN(APyramid, A1Pyramid, l, levels=4):
    neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean')
    A_l = APyramid[l]
    #the paper treats l - 1 as the coarser layer which is l+1 in our indexing
    A_l_1 = APyramid[l+1]
    A1_l = A1Pyramid[l]
    A1_l_1 = A1Pyramid[l+1]

    searchHeight, searchWidth, _ = A_l.shape
    completeSearchSpace = []
    for i in range(searchHeight):
        for j in range(searchWidth):
            #at coarsest level consider only this level
            if l >= levels - 1: 
                A_lFeatures = extractFeatures(A_l, 5, i, j) #treat only layer as fine 5x5
                A1_lFeatures = extractFeatures(A1_l, 5, i, j) #treat only layer as fine 5x5
                Fp_l = np.hstack((A_lFeatures, A1_lFeatures))
            else:
                A_lFeatures = extractFeatures(A_l, 5, i, j) #fine 5x5 divide coordinates 2x?
                A_l_1Features = extractFeatures(A_l_1, 3, i/2, j/2) #coarse 3x3

                A1_lFeatures = extractFeatures(A1_l, 5, i, j) 
                A1_l_1Features = extractFeatures(A1_l_1, 3, i/2, j/2)

                Fp_l = np.hstack((A_lFeatures, A_l_1Features, A1_lFeatures, A1_l_1Features))

            completeSearchSpace.append(Fp_l)
    completeSearchSpace = np.array(completeSearchSpace)
    neigh = ANN(completeSearchSpace)
    #neigh.fit(completeSearchSpace) old full KNN
    return neigh

def ANN(searchSpace):
    dimension = searchSpace[0].shape[0]
    t = AnnoyIndex(dimension, metric='euclidean')
    for i in range(len(searchSpace)):
        t.add_item(i, searchSpace[i])
    t.build(10)
    return t

def bestMatch(neigh, BPyramid, B1Pyramid, l, q, levels=4):
    B_l = BPyramid[l]
    #the paper treats l - 1 as the coarser layer which is l+1 in our indexing
    B_l_1 = BPyramid[l+1]
    B1_l = B1Pyramid[l]
    B1_l_1 = B1Pyramid[l+1]

    if l >= levels - 1:
        B_lFeatures = extractFeatures(B_l, 5, q[0], q[1]) #treat only layer as fine 5x5
        B1_lFeatures = extractFeatures(B1_l, 5, q[0], q[1]) #treat only layer as fine 5x5
        Fq_l = np.hstack((B_lFeatures, B1_lFeatures))
    else:
        B_lFeatures = extractFeatures(B_l, 5, q[0], q[1])
        B_l_1Features = extractFeatures(B_l_1, 3, q[0]/2, q[1]/2)

        B1_lFeatures = extractFeatures(B1_l, 5, q[0], q[1])
        B1_l_1Features = extractFeatures(B1_l_1, 3, q[0]/2, q[1]/2) #fine 5x5 multiply coordinates 2x?

        Fq_l = np.hstack((B_lFeatures, B_l_1Features, B1_lFeatures, B1_l_1Features))

    #Find
    indices = neigh.get_nns_by_vector(Fq_l,1)
    i = indices[0]
    x = i % B_l.shape[1]
    y = i / B_l.shape[1]
    return x, y

def main():
    A = cv2.imread('images/blurA.bmp')
    Ap = cv2.imread('images/blurAp.bmp')
    B = cv2.imread('images/blurB.bmp')
    # im = cv2.pyrDown(im)
    resultPyramid = createImageAnalogy(A, Ap, B)
    cv2.imshow('result', resultPyramid[0])
    cv2.imshow('result1', resultPyramid[1])
    cv2.imshow('result2', resultPyramid[2])
    cv2.imshow('result3', resultPyramid[3])


if __name__ == "__main__":
    main()