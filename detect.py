import os
from tkinter import filedialog
import numpy as np
from collections import defaultdict
import cv2
from math import sqrt
from pprint import pprint
from matplotlib import pyplot as plt
from IPython.display import Image, display
from skimage.segmentation import quickshift, felzenszwalb, slic, watershed
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from sklearn.feature_extraction import image
from sklearn.cluster import SpectralClustering, spectral_clustering
from sklearn.cluster import KMeans


from skimage import (
    data, restoration, util
)

import imutils


from tkinter import Tk
from tkinter.filedialog import askopenfilename

class CircleDetectorBuilder(object):

    #---------------------------------------------------------
    # TOP PRIORITY
    # Noise Reduction
    
    
    # Blob Descriptor for texture recongnition
    # Local Binary Pattern for texture classification https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html
    # Gabor Filter https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_gabor.html
    # Try out template Matching (pyramid)

    
    def __init__(self, filename: str, showFlag: bool, C: float):
        self.img = None
        self.filename = filename
        self.originalImage = None
        self.filteredImage = None
        self.images = [self.originalImage]
        self.showFlag = showFlag
        self.keypoints = None
        self.C = C
        self.circles = None

    def with_read_image_unchanged(self):
        self.img = cv2.imread(self.filename, cv2.IMREAD_UNCHANGED)
        self.originalImage = self.img.copy()
        return self
    
    def with_read_image(self):
        self.img = cv2.imread(self.filename)
        self.originalImage = self.img.copy()
        return self

    def with_resize_absolute(self, toX=800.0, toY=640.0):
        self.img = cv2.resize(self.img, (toX, toY))
        self.originalImage = self.img.copy()
        return self
    
    def with_resize_relative(self, factor):
        self.originalImage = self.img.copy()
        return NotImplemented
    
    def with_histogram_equal(self):
        self.img = cv2.equalizeHist(self.img)
        self.push_image()
        return self
    
    def with_hue_shift(self, amount=30):
        dimensions = self.img[0,0]
        if len(dimensions) == 3:
            b_channel, g_channel, r_channel = cv2.split(self.img)
            alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50
            self.img = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        if len(dimensions) == 4:
            alpha = self.img[:,:,3]


        bgr = self.img[:,:,0:3]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)

        hnew = np.mod(h + amount, 180).astype(np.uint8)
        hsv_new = cv2.merge([hnew,s,v])

        desaturated_image = hsv_new.copy()
        desaturated_image[:, :, 1] = desaturated_image[:, :, 1] * 0.0  # You can adjust the factor to control desaturation


        bgr_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
        desaturated_image_new = cv2.cvtColor(desaturated_image, cv2.COLOR_HSV2BGR)

        B, G, R = cv2.split(desaturated_image_new)
        # bgra = cv2.cvtColor(bgr_new, cv2.COLOR_BGR2BGRA)
        # bgra[:,:,3] = alpha

        self.img = G
        self.push_image()

        # cv2.imshow("BGR before Hue shift.png", bgr)
        # cv2.imshow("BGR after Hue shift.png", bgr_new)
        # cv2.imshow("BGR after Hue desat shift.png",desaturated_image_new)

        return self
    
    def with_split_B_G_R(self, choose="R"):
        (B, G, R) = cv2.split(self.img)

        match choose:
            case "B":
                self.img = B
            case "G":
                self.img = G
            case "R":
                self.img = R
            case _:
                self.img = R

        # cv2.imshow("B Channel.png", B)
        # cv2.imshow("G Channel.png", G)
        # cv2.imshow("R Channel.png", R)

        return self
    
    def with_grayscale(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.push_image()
        return self
    
    def with_clahe(self, clipLimit=2.0, tileGridSize=(8,8)):
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        self.img = clahe.apply(self.img)
        self.push_image()
        return self
    
    def with_global_histogram(self):
        return NotImplemented
    
    def with_adaptive_threshold(self, blockSize: int, _C: float, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, maxValue=255):
        self.img = cv2.adaptiveThreshold(self.img, maxValue, adaptiveMethod, thresholdType, blockSize, self.C)
        self.push_image()
        return self
    
    def with_threshold(self, thresh=0.0, maxVal=255.0, threshHoldType=cv2.THRESH_OTSU):
        _, self.img = cv2.threshold(self.img, thresh, maxVal, type= threshHoldType | cv2.THRESH_BINARY)
        self.push_image()
        return self
    
    
    def with_pyr_mean_shift_filter(self, sp=2, sr=12, maxLevel=2):
        self.img = cv2.pyrMeanShiftFiltering(self.img, sp, sr, maxLevel=2)
        #cv2.imshow("Mean shift filterd", self.img) 
        self.filteredImage = self.img.copy()
        return self
    
    def with_gaussian_blur(self, sigmaX, sigmaY, kernelSize=(5,5), borderType=0):
        self.img = cv2.GaussianBlur(self.img, kernelSize, borderType, sigmaX, sigmaY)
        #cv2.imshow("Gauss", self.img.copy())
        self.filteredImage = self.img.copy()
        #self.push_image()
        return self
    
    def with_bilateral_blur(self, d=15):
        self.img = cv2.bilateralFilter(self.img, 3, 64, 64)
        #cv2.imshow("bilatral blur", self.img.copy())
        return self
    

    def with_invert_image(self):
        self.img = cv2.bitwise_not(self.img)
        self.push_image()
        return self
    
    
    def with_median_blur(self, kernelSize=3):
        self.img = cv2.medianBlur(self.img, kernelSize)
        self.push_image()
        return self
    
    def with_blur(self, kernelSize=3):
        self.img = cv2.blur(self.img, (kernelSize, kernelSize))
        self.push_image()
        return self
    
    def with_egde_preserving_filter(self, k=3, threshhold=127):
        self.img = cv2.ximgproc.edgePreservingFilter(self.img, k, threshhold)
        self.filteredImage = self.img
        #cv2.imshow("egde_perserving_filter", self.img)
        return self
    
    def with_erosion(self, kernelX=5, kernelY=5, iterations=1, borderType=cv2.BORDER_CONSTANT):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelX, kernelY))
        self.img = cv2.erode(self.img, kernel=kernel, iterations=iterations, borderType=borderType)
        self.push_image()
        return self
    
    def with_dilation(self, kernelX=5, kernelY=5, iterations=1, borderType=cv2.BORDER_CONSTANT):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelX, kernelY))
        self.img = cv2.dilate(self.img ,kernel=kernel, iterations=iterations, borderType=borderType)
        self.push_image()
        return self
    
    def with_morphology(self, operation=cv2.MORPH_OPEN, kernelX=3, kernelY=3, iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelX, kernelY))
        self.img = cv2.morphologyEx(self.img, operation, kernel, iterations)
        self.push_image()
        return self
    
    def with_divide(self):
        self.img = cv2.divide(self.img, cv2.GaussianBlur(self.img, (5,5), 33, 33), scale=255)
        self.push_image()
        return self
    
    def with_watershed(self):
        # sure background area
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        sure_bg = self.img
        
        #cv2.imshow('Before Transform.jpg', self.img)
        # Distance transform
        
        dist = cv2.distanceTransform(self.img, cv2.DIST_L2, 5)
        #cv2.imwrite('Distance Transform.jpg', dist)
        
        
        dist2 = cv2.normalize(dist, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #cv2.imwrite('Distance Transform Normalize.jpg', dist2)
        
        #foreground area
        dist2 = dist2.astype(np.uint8)
        #self.with_adaptive_threshold(31, self.C, maxValue=0.1 * dist.max())
        
        ret, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, cv2.THRESH_BINARY)
        
        #ret, sure_fg = cv2.threshold(dist2, 0, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)
        
        sure_fg = self.img.astype(np.uint8)
        #cv2.imshow('Sure Foreground', sure_fg)
        
        # unknown area
        unknown = cv2.subtract(sure_bg, sure_fg)
        #cv2.imshow('Unknown', unknown)
        
        # Marker labelling
        # sure foreground 
        ret, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that background is not 0, but 1
        markers += 1
        # mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # fig, ax = plt.subplots(figsize=(6, 6))
        # ax.imshow(markers, cmap="gray")
        # ax.axis('off')
        # plt.show()
        
        # watershed Algorithm
        markers = cv2.watershed(self.filteredImage, markers)
        
        
        # fig, ax = plt.subplots(figsize=(5, 5))
        # ax.imshow(markers, cmap="tab20b")
        # ax.axis('off')
        # plt.show()
        
        labels = np.unique(markers)
        
        tree = []
        for label in labels[:]:  
        
        # Create a binary image in which only the area of the label is in the foreground 
        #and the rest of the image is in the background   
            target = np.where(markers == label, 255, 0).astype(np.uint8)
            
        # Perform contour extraction on the created binary image
            contours, hierarchy = cv2.findContours(
                target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            tree.append(contours[0])
        
        # Draw the outline
        
        #img = cv2.drawContours(self.originalImage, tree, -1, color=(255, 255, 255), thickness=1)
        self.img = cv2.drawContours(self.originalImage.copy(), tree, -1, color=(255, 255, 255), thickness=cv2.FILLED)
        #cv2.imshow("Contours",self.img)
                
        diff = cv2.subtract(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(self.originalImage, cv2.COLOR_BGR2GRAY))
        #cv2.imshow("Difference",diff)
        self.img = diff
        
        return self

    def with_canny_edge(self, thresHold1=100.0, thresHold2=200.0, apertureSize=3, L2gradient=False):
        self.img = cv2.Canny(self.img, 100 ,200)
        self.push_image()
        return self
    
    def with_detect_blobs_MSER(self):
        # Throws segmentation fault
        # Set our filtering parameters
        # Initialize parameter setting using cv2.SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()

        
        # Set Area filtering parameters
        params.filterByArea = False        
        params.minArea = 10
        
        # Set Circularity filtering parameters
        params.filterByCircularity = True 
        params.minCircularity = 0.5
        
        # Set Convexity filtering parameters
        params.filterByConvexity = False
        params.minConvexity = 0.5
            
        # Set inertia filtering parameters
        params.filterByInertia = False
        params.minInertiaRatio = 0.4
        
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        #cv2.imshow("before detection", self.img)
        
        # Detect blobs
        keypoints = detector.detect(self.img)
        
        self.keypoints = keypoints
        
        #Use for Debug
        #pprint(vars(self.keypoints))

        # Draw blobs on our image as red circles
        blank = np.zeros((1, 1)) 
        blobs = cv2.drawKeypoints(self.originalImage, keypoints, blank, (209, 0, 255),
                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        #cv2.imshow("Keypoints", blobs)
        return self

    def with_detect_circles(self, method=cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=200, param2=100):
        self.circles = cv2.HoughCircles(image=self.img,
                            method=method,
                            dp=dp,
                            minDist=minDist,
                            param1=param1,
                            param2=param2)
        
        return self

    def show(self, offSetX=0, offSetY=0):
        if self.circles is None:
            print("No circles were detected or order of build steps is wrong")
            self.show_images_with_offset_wrapper(offSetX, offSetY)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return self
    
        self.circles = np.uint16(np.around(self.circles))
        for i in self.circles[0,:]:
            cv2.circle(self.originalImage, (i[0],i[1]),i[2],(255,0,0),2)
            cv2.circle(self.originalImage, (i[0],i[1]),2,(255,0,0),2)

        self.show_images_with_offset_wrapper(offSetX, offSetY)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self

    def push_image(self):
        if self.showFlag:
            self.images.append(self.img.copy())

    def concat_images_and_display(self, x, y):
        if self.showFlag:
            cv2.namedWindow('Circle Detection')
            cv2.moveWindow('Circle Detection', x, y)
            cv2.imshow('Circle Detection', np.concatenate((self.images[1::]), 1))

    def show_images_with_offset_wrapper(self, x, y):
        self.concat_images_and_display(x, y)
        cv2.namedWindow('Final Image')
        cv2.moveWindow('Final Image', x, y+1200)
        cv2.imshow('Final Image', self.originalImage)
        

def k_means_segmentation(filename, K):
    img = cv2.imread(filename)
    img = cv2.resize(img, (640, 480))
    img_or = img.copy()
    #cv2.imshow('original', img.copy())
    
    img = cv2.GaussianBlur(img, (5,5), sigmaX=11, sigmaY=11)
    img = cv2.pyrMeanShiftFiltering(img, 5, 10, maxLevel=1)
    #cv2.imshow('Mean Shift Filter',img.copy())
    
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 300)
    ret,label,center=cv2.kmeans(Z, K, None, criteria, 50, cv2.KMEANS_PP_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    #cv2.imshow('res2',res2)
    
    return res2, img_or
     
def meanshift(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image, (420, 320))
    image = cv2.pyrMeanShiftFiltering(image, 10, 10, maxLevel=1)
    # Convert the image to RGB format (scikit-image expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    
    #image_rgb_resized = cv2.GaussianBlur(image_rgb_resized, (5,5), 11, 11)
    image_lab = cv2.pyrMeanShiftFiltering(image_lab, 20, 20, maxLevel=1)
    
    cv2.imshow("lab", image)
    
    # Apply quick Mean Shift-based segmentation
    segments = quickshift(image_lab, ratio=3.0, max_dist=50.0)
    
    print(len(segments))
    
    # Display the segmented grayscale image
    plt.figure(figsize=(8, 6))
    plt.subplot(121)
    plt.imshow(segments, cmap='magma')
    plt.title('Segmented Grayscale Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.show()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def felzenszwalb_segmentation(filename):
    image = cv2.imread(filename)

    # Convert the image to RGB format (scikit-image expects RGB)
    image_rgb = cv2.resize(image, (480, 320))
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    # Apply Felzenszwalb segmentation
    # The parameters to adjust are scale and min_size
    # - `scale` controls the trade-off between color similarity and spatial proximity. Smaller values lead to more segments.
    # - `min_size` sets the minimum component size. Smaller values may result in smaller segments.
    segments = felzenszwalb(image_rgb, scale=100)

    # Display the segmented grayscale image
    plt.figure(figsize=(8, 6))
    plt.subplot(121)
    plt.imshow(segments, cmap='gray')
    plt.title('Segmented Grayscale Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.show()

def slic_segmentation(filename):
    # Load the image using OpenCV
    image = cv2.imread(filename)
    image_rgb = cv2.resize(image, (480, 320))
    # Convert the image to RGB format (scikit-image expects RGB)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

    # Apply SLIC segmentation
    # The parameters to adjust are n_segments and compactness
    # - `n_segments` controls the approximate number of superpixels to generate.
    # - `compactness` controls the trade-off between color similarity and spatial proximity.
    segments = slic(image_rgb, n_segments=2, compactness=10, sigma=1)

    # Display the segmented grayscale image
    plt.figure(figsize=(8, 6))
    plt.subplot(121)
    plt.imshow(segments, cmap='gray')
    plt.title('Segmented Grayscale Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.show()
    
def watershed_segmentation(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image_gray = cv2.resize(image, (480, 320))
    return NotImplemented
    return NotImplemented

def chooseFile() -> str:
    root = Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
    initialdir="./perpendicular", title="Choose an image",
    filetypes=[(
        "Images Files", ["*.png", "*.jpg", "*.jpeg", "*.bmp"])])
    print(filename)
    return filename
       
def float_range(start: float, stop: float, increment: float) -> float:
    while start < stop:
        yield start
        start += increment
        
def compare_k_means(filename, K, C, blocksize):
    K = K
    start=2
    
    C= C
    blockSize = blocksize
    
    fig, axs = plt.subplots(K-1, 5, figsize=(15, 15))
    
    for i in range(start, K+1):
        print("K={}".format(i))
        
        img, original = k_means_segmentation(filename, i)
        org = original
                        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        _, img_binary = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
        
        _, img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        
        img_adaptive = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)
        
        fig.suptitle('K-means K={}'.format(K))
        axs[i-start, 0].imshow(img.copy())
        axs[i-start, 0].set_title('K-means K={}'.format(i))
        
        axs[i-start, 1].imshow(img_binary.copy())
        axs[i-start, 1].set_title('K-means with Binary K={}'.format(i))
        
        axs[i-start, 2].imshow(img_otsu.copy())
        axs[i-start, 2].set_title('K-means with Otsu K={}'.format(i))
        
        axs[i-start, 3].imshow(img_adaptive.copy())
        axs[i-start, 3].set_title('K-means Adaptive K={}, Blocksize={}, C={}'.format(i, blockSize, C))
        
        axs[i-start, 4].hist(img.ravel(),256,[0,256])
        axs[i-start, 4].set_title('K-means Histogram K={}'.format(i))
        
    plt.show()
    
def get_all_filenames(dir: str) -> list:
    # Specify the directory path where you want to list filenames
    directory_path = dir

    # Initialize an empty list to store filenames
    file_list = []

    # Use the os.listdir() function to get a list of filenames in the directory
    try:
        for filename in os.listdir(directory_path):
            # Append the filename to the list
            file_list.append(filename)
    except OSError as e:
        print(f"Error: {e}")
    return file_list

def elbow_method(data: list, filename: str):
    inertias = []

    for i in range(1,10):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    #plt.figure() # uncomment to 
    plt.plot(range(1,10), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig("Elbow method_{}.png".format(filename.split('.')[0]))
    
def plot_blobs(blobs, title):
    coord, label = zip(*blobs)  # Unzip the list of (x, y) coordinates
    x, y = zip(*coord)
    plt.scatter(x, y, marker='o', label=title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(title)
    plt.legend()

if __name__ == "__main__":


    # BINARIZATION_NIBLACK: int
    # BINARIZATION_SAUVOLA: int
    # BINARIZATION_WOLF: int
    # BINARIZATION_NICK: int

    # DTF_NC: int
    # DTF_IC: int
    # DTF_RF: int
    # GUIDED_FILTER: int
    # AM_FILTER: int
    # EdgeAwareFiltersList = int
    # """One of [DTF_NC, DTF_IC, DTF_RF, GUIDED_FILTER, AM_FILTER]"""
    
    #filename = chooseFile()
    
    # img = cv2.imread(filename)
    # img = cv2.resize(img, (480, 320))
    # img_filt = cv2.ximgproc.edgePreservingFilter(img, 5, 127)
    # print("debug")
    
    
    # img_ori = img.copy()
    # img = cv2.GaussianBlur(img, (9,9), sigmaX=33, sigmaY=33)
    # img = cv2.pyrMeanShiftFiltering(img, 10, 10)
    
    # #cv2.imshow("Original",img.copy())
    
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # t_sauvola = cv2.ximgproc.niBlackThreshold(img, 255, cv2.THRESH_BINARY, 57, 0.2, binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)
    
    # for i in float_range(-0.5, 0.5, 0.01):
    #     i = round(i, 3)
    #     print(i)
    #     t_sauvola = cv2.ximgproc.niBlackThreshold(img, 255, cv2.THRESH_BINARY, 11, i, binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)
    #     cv2.imwrite("/home/whoami/projects/python/hough-detect/threshold/threshold_sauvola{}.jpg".format(i), t_sauvola)
    
    # cv2.imwrite("/home/whoami/projects/python/hough-detect/threshold/Original.jpg", img_ori)
    
    #t_sauvola = cv2.ximgproc.niBlackThreshold(img, 255, cv2.THRESH_BINARY_INV, 67, 0.1, binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)

    # # Canny Edge Detection
    # edges = cv2.Canny(image=img, threshold1=5, threshold2=15, L2gradient=True) # Canny Edge Detection
    # #Display Canny Edge Detection Image
    # cv2.imshow('Canny Edge Detection', edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # t_sauvola = cv2.ximgproc.niBlackThreshold(img, 255, cv2.THRESH_BINARY_INV, 67, 0.1, binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)
    
    # cv2.imshow("Thres", t_sauvola)
    # cv2.waitKey(0)
    # # Initializing parameter setting using cv2.SimpleBlobDetector function
    # params = cv2.SimpleBlobDetector_Params()
    
    # params.filterByColor = True
    # params.blobColor = 255
    
    # # Filter by area (value for area here defines the pixel value)
    # params.filterByArea = False
    # params.minArea = 5
    
    # # Filter by circularity
    # params.filterByCircularity = False
    # params.minCircularity = 0.5
    
    # # Filter by convexity
    # params.filterByConvexity = False
    # params.minConvexity = 0.1
        
    # # Filter by inertia ratio
    # params.filterByInertia = False
    # params.minInertiaRatio = 0.01
    
    # # Creating a blob detector using the defined parameters
    # detector = cv2.SimpleBlobDetector_create(params)
        
    # # Detecting the blobs in the image
    # keypoints = detector.detect(t_sauvola)
    
    # # Drawing the blobs that have been filtered with green on the image
    # blank = np.zeros((1, 1))
    # blobs = cv2.drawKeypoints(img_ori, keypoints, blank, (255, 255, 0),
    #                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    
    # # Setting the grid size
    # plt.figure(figsize=(20,20))
    
    # # Displaying the image
    # plt.subplot(121)
    # plt.title('Original')
    # plt.imshow(img_ori, cmap='gray')
    
    # plt.subplot(122)
    # plt.title('Blobs')
    # plt.imshow(blobs)
    
    # plt.show()
    # cv2.destroyAllWindows()
    
    # #NiBlack
    # t_black = cv2.ximgproc.niBlackThreshold(img, 255, cv2.THRESH_BINARY_INV, 31, 0.5, binarizationMethod=cv2.ximgproc.BINARIZATION_NIBLACK )
    # #WOLF
    # t_wolf = cv2.ximgproc.niBlackThreshold(img, 255, cv2.THRESH_BINARY_INV, 31, 0.5, cv2.ximgproc.BINARIZATION_WOLF)
    # #NICK
    # t_nick = cv2.ximgproc.niBlackThreshold(img, 255, cv2.THRESH_BINARY_INV, 31, 0.5, cv2.ximgproc.BINARIZATION_NICK)
    
    # t_sauvola_inv = cv2.bitwise_not(t_sauvola)
    
    # t_sauvola_cross = cv2.morphologyEx(t_sauvola, cv2.MORPH_DILATE, (5, 5), iterations=2)
    
    # cv2.imshow("Thres sauvola", t_sauvola)
    
    # cv2.imshow("Thres sauvola inv", t_sauvola_inv)
    
    # #noise removal
    # kernel = np.ones((3,3),np.uint8)
    # opening = cv2.morphologyEx(t_sauvola,cv2.MORPH_OPEN,kernel)
    
    # # sure background area
    # sure_bg = cv2.dilate(opening,kernel)
    
    # # Finding sure foreground area
    # dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2, 5)
    
    # dist_transform = cv2.normalize(dist_transform, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # ret, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255,0)
    
    # cv2.imwrite('Distance Transform Normalize.jpg', dist_transform)
    
    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg,sure_fg)

    # cv2.imshow("Sure Foreground", sure_fg)
    # cv2.imshow("Sure Background", sure_bg)
    # cv2.imshow("Unknown", unknown)
    
    # # Marker labelling
    # ret, markers = cv2.connectedComponents(sure_fg)
    # # Add one to all labels so that sure background is not 0, but 1
    # markers = markers+1
    # # Now, mark the region of unknown with zero
    # markers[unknown==255] = 0
    
    # # plt.figure()
    # # plt.imshow(markers, cmap="jet")
    # # plt.show()
    
    # markers = cv2.watershed(img_ori.copy(), markers)
    
    # labels = np.unique(markers)
    
    # tree = []
    # for label in labels[:]:  
    
    # # Create a binary image in which only the area of the label is in the foreground 
    # #and the rest of the image is in the background   
    #     target = np.where(markers == label, 255, 0).astype(np.uint8)
        
    # # Perform contour extraction on the created binary image
    #     contours, hierarchy = cv2.findContours(
    #         target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    #     )
    #     tree.append(contours[0])
        
    # contours = cv2.drawContours(img_ori.copy(), tree, -1, color=(255, 255, 255), thickness=cv2.FILLED)
    # cv2.imshow("Contours", contours)
    
    # diff = cv2.subtract(cv2.cvtColor(contours, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY))
    
    # diff[diff != 0] = 255
    
    # cv2.imshow("Difference",diff)
    
    # cv2.imshow("Thres black",t_black)
    # cv2.imshow("Thres wolf",t_wolf)
    # cv2.imshow("Thres nick",t_nick)
    
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    #compare_k_means(filename, 6, 15, 51)
    
    # image = cv2.imread(filename)
    # image = cv2.resize(image, (480, 320))
    # image = cv2.GaussianBlur(image, (5,5), sigmaX=33, sigmaY=33)
    # image = cv2.pyrMeanShiftFiltering(image, 10, 10)
    # cv2.imshow("PYRMEAN",image)
    
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # #image_gray = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 15)
    # #cv2.imshow("threshold",image_gray)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # #blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)

    # # Compute radii in the 3rd column.
    # #blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    # #blobs_dog = blob_dog(image_gray, threshold=0.2)
    # #blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    # blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

    # blobs_list = [blobs_doh]
    # color = ['yellow']
    # title = ['Laplacian of Gaussian']
    # sequence = zip(blobs_list, color, title)

    # fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    # ax = axes.ravel()

    # for idx, (blobs, color, title) in enumerate(sequence):
    #     ax[idx].set_title(title)
    #     ax[idx].imshow(image)
    #     for blob in blobs:
    #         y, x, r = blob
    #         c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
    #         ax[idx].add_patch(c)
    #     ax[idx].set_axis_off()

    # plt.tight_layout()
    # plt.show()

    
    # --------------------------- Segmentation methods
        
    #meanshift(filename)
    #felzenszwalb_segmentation(filename)
    #slic_segmentation(filename)
    
    # Try out different threshold methods
    #.with_adaptive_threshold(51,15) C >= 0 when not much to none background C < 0 when Background in Image 15, -15 solid values
    # cb = CircleDetectorBuilder(filename, True) \
    # .with_read_image_unchanged() \
    # .with_resize_absolute(800, 640) \
    # .with_hue_shift() \
    # .with_gaussian_blur(kernelSize=(9,9)) \
    # .with_adaptive_threshold(51,-15) \
    # .with_morphology(operation=cv2.MORPH_CLOSE) \
    # .with_gaussian_blur(kernelSize=(15, 15)) \
    # .with_detect_circles(method=cv2.HOUGH_GRADIENT_ALT, param1=300, param2=0.7 ) \
    # .show()


    # Detect without Background
    # cb = CircleDetectorBuilder(filename, True, 15) \
    # .with_read_image() \
    # .with_resize_absolute(480, 360) \
    # .with_pyr_mean_shift_filter() \
    # .with_hue_shift() \
    # .with_gaussian_blur(kernelSize=(5,5))\
    # .with_adaptive_threshold(67, 0) \
    # .with_morphology(operation=cv2.MORPH_OPEN, iterations=1) \
    # .with_watershed() \
    # .show()


    #Detect with Background
    # cb = CircleDetectorBuilder(filename, True, -15) \
    # .with_read_image() \
    # .with_resize_absolute(480, 360) \
    # .with_bilateral_blur() \
    # .with_pyr_mean_shift_filter() \
    # .with_hue_shift() \
    # .with_adaptive_threshold(67, 0) \
    # .with_morphology(operation=cv2.MORPH_OPEN, iterations=4) \
    # .with_watershed() \
    # .show()


    #Important
    # cb = CircleDetectorBuilder(filename, True, -15) \
    # .with_read_image() \
    # .with_resize_absolute(480, 320) \
    # .with_gaussian_blur(33, 33, kernelSize=(5,5)) \
    # .with_pyr_mean_shift_filter(20,20, maxLevel=1) \
    # .with_hue_shift() \
    # .with_adaptive_threshold(67, 0) \
    # .with_watershed() \
    # .with_gaussian_blur(11, 11) \
    # .with_detect_blobs_MSER() \
    # .show()
    
    #Important fast filtering
    
    path_to_dir = "/home/whoami/projects/python/hough-detect/perpendicular"
    
    files: list[str] = get_all_filenames(path_to_dir)
    i = 0
    for filename in files:
        absolute_file_path = path_to_dir + '/' + filename
        
        cb = CircleDetectorBuilder(absolute_file_path, True, -15) \
        .with_read_image() \
        .with_resize_absolute(480, 320) \
        .with_egde_preserving_filter(k=5) \
        .with_pyr_mean_shift_filter(5,10, maxLevel=1) \
        .with_grayscale() \
        .with_adaptive_threshold(67, 0) \
        .with_watershed() \
        .with_gaussian_blur(11, 11) \
        .with_detect_blobs_MSER() \
        
        # Post processing
        points = cb.keypoints
        pointSize = []
        
        for point in points:
            pointSize.append((point.size, point.pt[0], point.pt[1]))
        
        mean = np.mean(np.asarray(pointSize[0]))
        sigma = np.std(np.asarray(pointSize[0]))
        
        minimun = mean - sigma
        maximum = mean + sigma
        
        print("Mean = {}".format(mean))
        print("Standard Deviation = {}".format(sigma))
        
        for point in pointSize:
            if point[0] < minimun or point[0] > maximum:
                pointSize.remove(point)
        
        data = []
        
        for point in pointSize:
            data.append((point[1], point[2]))

        kmeans = KMeans(n_clusters=2)
        kmeans.fit(data)
        cluster_labels = kmeans.labels_
        
        # clustering = SpectralClustering(n_clusters=2,
        # assign_labels='kmeans',
        # random_state=0).fit(np.array(data))
        
        # cluster_labels = clustering.labels_
        
        labeled_data = zip(data, cluster_labels)
        
        labeled_data = list(labeled_data)
        # Assuming you know some properties of the blobs in each cluster
        cluster_0_blobs = [blob for i, blob in enumerate(labeled_data) if blob[1] == 0]
        cluster_1_blobs = [blob for i, blob in enumerate(labeled_data) if blob[1] == 1]
        
        cv2.imshow("Image", cb.originalImage)
        plot_blobs(cluster_0_blobs, "Cluster 0 Blobs")
        plot_blobs(cluster_1_blobs, "Cluster 1 Blobs")
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    
    # Prove assumtation that most Tree trunk size is normal distributed
    
    # draw the left over blobs

    #Important
    # cb = CircleDetectorBuilder(filename, True, 15) \
    # .with_read_image() \
    # .with_resize_absolute(420, 320) \
    # .with_gaussian_blur(33, 33, (3,3)) \
    # .with_pyr_mean_shift_filter(20,20, maxLevel=1) \
    # .with_hue_shift() \
    # .with_adaptive_threshold(51, 0) \
    # .with_watershed() \
    # .with_gaussian_blur(11, 11) \
    # .with_detect_blobs_MSER() \
    # .show()

    # Background
    # cb = CircleDetectorBuilder(filename, True, -15) \
    # .with_read_image() \
    # .with_resize_absolute(420, 320) \
    # .with_gaussian_blur(33, 33, kernelSize=(5,5)) \
    # .with_pyr_mean_shift_filter(10,20, maxLevel=2) \
    # .with_hue_shift() \
    # .with_adaptive_threshold(67, 0) \
    # .with_watershed()\
    # .with_gaussian_blur(33, 33, kernelSize=(5,5)) \
    # .with_detect_blobs_MSER() \
    # .show()