import cv2

def makePyramid(image, filename=None):
    if filename:
        image = cv2.imread(filename)
    G = image.copy()
    gpImage = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpImage.append(G)
    return gpImage
