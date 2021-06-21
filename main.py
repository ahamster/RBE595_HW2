# Courtland Hui
# RBE595
# HW2 - Harris Corner Detector

import cv2
import numpy as np


# Credits: borrowed code from cianconway on github
def match(desc1,desc2,threshold=0.5): #Match method, for each corner in image 1, selects it's match in image 2 using normalized cross corelation
    
    n = len(desc1[0])
    
    # pair-wise distances
    d = np.ones((len(desc1),len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            ncc_value = np.sum(d1 * d2) / (n-1)
            if ncc_value > threshold:
                d[i,j] = ncc_value
            
    ndx = np.argsort(d)
    matchscores = ndx[:,0]
    
    return matchscores

# Credits: borrowed code from cianconway on github
def drawMatches(img1, kp1, img2, kp2, matches, inliers = None):
    # Create a new output image that concatenates the two images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns, y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        inlier = False

        if inliers is not None:
            for i in inliers:
                if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
                    inlier = True

        # Draw a small circle at both co-ordinates
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points, draw inliers if we have them
        if inliers is not None and inlier:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
        elif inliers is not None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

        if inliers is None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    return out

# Credits: inspired by code from stheakanath for descriptors
def get_descriptors(image, coordinates, wid=5):
    # Get features from points
    desc = []
    img_height, img_width = image.shape[1], image.shape[0]
    # print("height, width:", image.shape[1], image.shape[0])
    for y, x in zip(*coordinates):
        if x-wid < 0 or y-wid < 0 or y+wid > img_height or x+wid > img_width:
            # print("x,y:",x,y, "minx, miny:",x-wid, y-wid, "maxx,maxy:", x+wid, y+wid)
            patch_val = 0
        else:
                      
            patch = image[(x-wid, y-wid), (x+wid, y+wid)]
            patch_val = (patch - patch.mean()) / patch.std()
        desc.append(patch_val)
    
    return desc

def harris_corner(img):
    window_size_ratio = .05 # as a fraction of image height
    window_type = 'gaussian' # 'gaussian' or 'square' window
    sigma = 0.025 # variance for gaussian window
    alpha = 0.04 # 0.04–0.15
    threshold_ratio = 0.3 # threshold ratio of max value 0.2 is pretty good

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Start Harris Corner Detector Algorithm
    # 1. Compute Gaussian derivatives at each pixel
    """ # Partial derivative kernel with respect to x
    Gx = np.array([[0, 0, 0], [2, 0, -2], [0, 0, 0]])
    #Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    Gx = Gx/(np.sum(Gx) if np.sum(Gx)!=0 else 1)
    print(Gx)
    Ix = cv2.filter2D(img_gray,-1,Gx)

    # Partial derivative kernel with respect to y
    Gy = np.array([[-0, -2, 0], [0, 0, 0], [0, 2, 0]])
    #Gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    Gy = Gy/(np.sum(Gy) if np.sum(Gy)!=0 else 1)
    Iy = cv2.filter2D(img_gray,-1,Gy)
    print(Gy) """

    #Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    #Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    Iy, Ix = np.gradient(img_gray)

    """ cv2.imshow('Image Window Ix', Ix)
    cv2.imshow('Image Window Iy', Iy)
    # add wait key. window waits until user presses a key
    cv2.waitKey(0)
    # and finally destroy/close all open windows
    cv2.destroyAllWindows() """

    # 2. Compute second moment matrix M in a Gaussian window around each pixel
    # Compute products of partial derivates for M
    Ixx = Ix**2
    Ixy = Ix*Iy
    Iyy = Iy**2

    # Window size
    height, width = img_gray.shape
    window_size = round(window_size_ratio*height)

    # Window
    if window_type == 'gaussian':
        x = np.linspace(-1, 1, window_size)
        y = np.linspace(-1, 1, window_size)
        x, y = np.meshgrid(x, y)
        G_sigma = np.exp(-(x**2+y**2)/(2*sigma**2))/(2*np.pi*sigma**2)
        #print(G_sigma)
    elif window_type == 'square':
        G_sigma = np.ones([window_size, window_size])
    else:
        print("Window type incorrect, choose either 'gaussian' or 'square'.")
        quit()

    # Computing S (Derivative squared image convoluted with gaussian window)
    Sxx = cv2.filter2D(Ixx,-1, G_sigma)
    Sxy = cv2.filter2D(Ixy,-1, G_sigma)
    Syy = cv2.filter2D(Iyy,-1, G_sigma)

    # 3. Compute corner response function R
    R = Sxx*Syy-Sxy**2-alpha*(Sxx+Syy)**2

    # 4. Ouputs R positions after thresholding
    R_positions = R > threshold_ratio * R.max()
    return R_positions.nonzero()

# main code

position = (10,50) # pos for text overlay

# Open and read images
path1 = 'image1.jpg'
path2 = 'image2.jpg'

img1 = cv2.imread(path1)
img2 = cv2.imread(path2)

if img1 is None or img2 is None:
    print("Image not found")
    quit()

harris_points_img1 = harris_corner(img1)
harris_points_img2 = harris_corner(img2)

img1_circled = img1.copy()
for y, x in zip(*harris_points_img1):
    cv2.rectangle(img1_circled, (x-5, y-5), (x+5, y+5), (0, 0, 255), 1)
cv2.putText(img1_circled, "Harris Corners Detected: " + str(len(harris_points_img1[0])), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
cv2.imshow('Image1 Window of R', img1_circled)
img2_circled = img2.copy()
for y, x in zip(*harris_points_img2):
    cv2.rectangle(img2_circled, (x-5, y-5), (x+5, y+5), (0, 0, 255), 1)
cv2.putText(img2_circled, "Harris Corners Detected: " + str(len(harris_points_img2[0])), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
cv2.imshow('Image2 Window of R', img2_circled)
cv2.waitKey(0)
cv2.destroyAllWindows()

desc1 = get_descriptors(img1, harris_points_img1) # get patch of image for each point and compute 
desc2 = get_descriptors(img2, harris_points_img2)

kp1 = harris_points_img1
kp2 = harris_points_img2

""" # unable to get matching to work :( #
matches = match(desc1, desc2)
matchImg = drawMatches(img1,kp1,img2,kp2,matches)
cv2.imshow('InlierMatches.png', matchImg) """

#########################
# Video portion of code #
#########################

# create VideoCapture object
cap = cv2.VideoCapture('hw_video.mp4')

if (cap.isOpened() == False):
    print('Error while trying to open video. Plese check again...')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# define codec and create VideoWriter object
out = cv2.VideoWriter('hw_video_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))

while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        # add harris corner detections and count to each frame

        frame_with_harris_points = harris_corner(frame)

        for y, x in zip(*frame_with_harris_points):
            cv2.rectangle(frame, (x-5, y-5), (x+5, y+5), (0, 0, 255), 1)
    
        cv2.putText(frame, "Harris Corners Detected: " + str(len(frame_with_harris_points[0])), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)

        # save video frame
        out.write(frame)
        # display frame
        cv2.imshow('Video', frame)

        # press `q` to exit
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break

    # if no frame found
    else:
        break

# release VideoCapture()
cap.release()

# close all frames and video windows
cv2.destroyAllWindows()