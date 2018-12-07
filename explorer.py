import matplotlib.image as mpimg
import numpy as np 
import glob
import cv2 
from skimage.feature import hog

car_paths = ['./data/vehicles/GTI_Far/image*.png',
             './data/vehicles/GTI_Left/image*.png',
             './data/vehicles/GTI_MiddleClose/image*.png',
             './data/vehicles/GTI_Right/image*.png',
             './data/vehicles/KITTI_extracted/*.png']
not_car_paths = ['./data/non-vehicles/Extras/extra*.png',
                 './data/non-vehicles/GTI/image*.png']     

class Parameter:
    cspace = 'GRAY' 
    orient = 8  
    pix_per_cell = 16 
    cell_per_block = 1 
    spatial_size = (16, 16) 
    hist_bins = 16 

    def getParams(self):
        return(self.cspace,self.orient,self.pix_per_cell,self.cell_per_block,self.spatial_size,self.hist_bins)

#Load Image as RGB, 64x64
def loadCarImages(paths, max_load = 1000000):
    cars = []
    for path in paths:
        count = 0
        images = glob.glob(path)
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            img = BGR2RGB(img)
            img = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)
            cars.append(img)
            h_flip = cv2.flip(img,0)
            cars.append(h_flip)
            count += 1 
            if(count == max_load):
                break
        print("Loaded image count is {}, augmented to {} in path [{}]".format(count,count*2,path))                        
    return(cars)

def BGR2RGB(img):
    r_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return(r_img)

def RGB2BGR(img):
    r_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return(r_img)

def RGB2GRAY(img):
    r_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return(r_img)

# Define a function to compute color histogram features  
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)             
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel() 
    # Return the feature vector
    return features

# Input is RGB
# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    # return rhist, ghist, bhist, bin_centers, hist_features
    return rhist, ghist, bhist, bin_centers, hist_features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features
    
# Define a function to return HOG features and visualization
def hog_feature(img, orient=8, pix_per_cell=16, cell_per_block=1, vis=False, feature_vec=True):
    if vis == True:
        feature, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return feature, hog_image
    else:      
        feature = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return feature


def extract_feature_param(image, pm):
    feature = extract_feature(image,orient=pm.orient,pix_per_cell=pm.pix_per_cell,
                cell_per_block=pm.cell_per_block,
              cspace=pm.cspace, spatial_size=pm.spatial_size,
              hist_bins=pm.hist_bins, hist_range=pm.hist_range)
    return(feature)              


# Extract features from a 64x64 RGB image 
def extract_feature(image, orient=8, pix_per_cell=16, cell_per_block=1, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):

    # apply color conversion if other than 'RGB'
    # if cspace != 'RGB':
    #     if cspace == 'HSV':
    #         feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #     elif cspace == 'LUV':
    #         feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    #     elif cspace == 'HLS':
    #         feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    #     elif cspace == 'YUV':
    #         feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    # else: 
    feature_image = np.copy(image)      

    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    # Apply color_hist() also with a color space option now
    rh, gh, bh, bincen, hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
    # Apply HOG() also with a color space option now
    hog_features = np.ravel(hog_feature(RGB2GRAY(feature_image),orient=orient,
                   pix_per_cell=pix_per_cell,cell_per_block=cell_per_block, vis=False,feature_vec=True))  
#   print("spatial f.dtype={}, hist f.dtype={}, hog f.dtype={}".format(spatial_features[0].dtype,hist_features[0].dtype,hog_features[0].dtype))
    feature = np.concatenate((spatial_features,hist_features,hog_features))        
    return feature

def extract_features(images, param):
    features = []
    for image in images:
        feature = extract_feature_param(image,param)
        features.append(feature)
    return(features)


# Do normalize from the feature list input - X 
# output is Gaussian normalized feature list. 
def normalize(X):
    X_scaler = StandardScaler().fit(X)       
    scaled_X = X_scaler.transform(X)
    return(scaled_X)






