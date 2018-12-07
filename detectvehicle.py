import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import explorer as ex
import detector as dt
import glob

from scipy.ndimage.measurements import label
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

INPUT_FILE = 'project_video.mp4'
OUTPUT_FILE = 'output.mp4'
TEST_IMAGES = './test_images/screenshot##.jpg'
savecount = 0
cap = cv2.VideoCapture(INPUT_FILE)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_FILE,fourcc,20.0,(1280,720))
SAMPLE_FILENAME = './test/sample1.jpg'
base_img = None
mtx = None
dmtx = None
svc = None 
X_scaler = None 
print("Current working CV2 version is {}.".format(cv2.__version__))

param = ex.Parameter()


def initialize(maxload=1000000):
    global svc,X_scaler,param

    #Initialize parameter 
    param.color_space = 'GRAY' 
    param.orient = 8  
    param.pix_per_cell = 16 
    param.cell_per_block = 1 
    param.spatial_size = (16, 16) 
    param.hist_bins = 16 
    param.hist_range = (0,256) 

    #Limit load 100 for test.
    cars = ex.loadCarImages(ex.car_paths,maxload)
    notcars = ex.loadCarImages(ex.not_car_paths,maxload)
    car_features = ex.extract_features(cars,param)
    notcar_features = ex.extract_features(notcars,param)
    print("Car feature sample shape {}".format(car_features[0].shape))
    print("Not Car feature sample shape {}".format(notcar_features[0].shape))
    print("Car features count is {}, Not Car feature counts is {}.".format(len(car_features),len(notcar_features)))

    #prepare Scaler 
    X = np.vstack((car_features, notcar_features))
    X = X.astype(np.float64) 
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    rand_state = np.random.randint(0, 100)

    #Spilt Data 
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
    print('Using spatial binning of:',param.spatial_size,'and', param.hist_bins,'histogram bins.',' Feature vector length:', len(X_train[0]))
    # Use a linear SVC as a Classifier
    svc = SVC()
    t=time.time()
    # Train the Support Vector Machine 
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')    
    return(cars,notcars,svc,X_scaler) 

def doFindCar(image):
    global svc, X_scaler,param
    # w1 = dt.slide_window(image, x_start_stop=[640, 1180], y_start_stop=[400, 600], xy_window=(72, 72), xy_overlap=(0.7, 0.7))                       
    # w2 = dt.slide_window(image, x_start_stop=[640, 1280], y_start_stop=[420, 650], xy_window=(96, 96), xy_overlap=(0.5, 0.5))      
    # w3= dt.slide_window(image, x_start_stop=[760, 1280], y_start_stop=[420, 720], xy_window=(128, 128), xy_overlap=(0.5, 0.5))                     
    #w4= dt.slide_window(image, x_start_stop=[0, 520], y_start_stop=[420, 720], xy_window=(128, 128), xy_overlap=(0.5, 0.5))                     

    # w1 = dt.slide_window(image, x_start_stop=[300, 1080], y_start_stop=[400, 600], xy_window=(72, 72), xy_overlap=(0.6, 0.6))                       
    # w2 = dt.slide_window(image, x_start_stop=[0, 1280], y_start_stop=[420, 650], xy_window=(96, 96), xy_overlap=(0.5, 0.5))                       
    # w3= dt.slide_window(image, x_start_stop=[760, 1280], y_start_stop=[420, 720], xy_window=(128, 128), xy_overlap=(0.4, 0.4))                     
    # w4= dt.slide_window(image, x_start_stop=[0, 520], y_start_stop=[420, 720], xy_window=(128, 128), xy_overlap=(0.4, 0.4))  

    w1 = dt.slide_window(image, x_start_stop=[500, 1080], y_start_stop=[400, 520], xy_window=(72, 72), xy_overlap=(0.6, 0.6))                       
    w2 = dt.slide_window(image, x_start_stop=[350, 1280], y_start_stop=[400, 680], xy_window=(96, 96), xy_overlap=(0.5, 0.5))                       
    w3= dt.slide_window(image, x_start_stop=[760, 1280], y_start_stop=[450, 720], xy_window=(128, 128), xy_overlap=(0.5, 0.5))                     
    w4= dt.slide_window(image, x_start_stop=[300, 520], y_start_stop=[450, 720], xy_window=(128, 128), xy_overlap=(0.5, 0.5))    
    windows = np.concatenate((w1,w2,w3))
    window_img = dt.draw_boxes(image, windows, color=(0, 0, 255), thick=3)
    on_windows = dt.search_windows(image, windows, svc, X_scaler,param)
    on_window_img = dt.draw_boxes(image, on_windows, color=(255,0, 0), thick=3)
    # print("Slide windows count : {}. search window count : {}".format(len(windows),len(on_windows)))
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = dt.add_heat(heat,on_windows)
    heat = dt.apply_threshold(heat,1)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = dt.draw_labeled_bboxes(np.copy(image), labels) 
    return(draw_img)

def doDetectVehicle():                 
    global savecount,TEST_IMAGES,svc
    count = 0
    while(cap.isOpened()):
        ret,frame = cap.read()        
        if(ret == True):      
            frame = ex.BGR2RGB(frame)  
            if(count % 1 == 0):                
                frame = doFindCar(frame)    
                frame = ex.RGB2BGR(frame)      
                cv2.imshow('Vehicle Detection',frame)
                out.write(frame)
            count+=1
            if(cv2.waitKey(1) & 0xFF == ord('q')):
                break            
            else:
                if(cv2.waitKey(1) & 0xFF == ord('s')):
                    filename=TEST_IMAGES.replace("##",str(savecount))
                    cv2.imwrite(filename,frame)                    
                    print("Image file [{}] saved.".format(filename))
                    savecount=savecount+1
        else:
            break        
    cap.release()
    out.release()
    print("Vehicel Detection complete.")

initialize()
doDetectVehicle()
cv2.waitKey(0)        
cv2.destroyAllWindows()