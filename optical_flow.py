'''
    This code captures video from a webcam, calculates optical flow between frames
    (either sparse using Lucas-Kanade or dense using Farneback's method), and visualizes the motion of points across the video frames.
'''

#%% OpenCV based real-time optical flow estimation and tracking
# Ref: https://github.com/daisukelab/cv_opt_flow/tree/master
import numpy as np
import cv2
'''
    numpy (np): A library used for numerical operations, mainly to handle arrays and matrices.

    cv2 (OpenCV): A computer vision library that provides functions for image and video processing, including operations like optical flow.
'''

#%% Generic Parameters
color = np.random.randint(0,255,(100,3)) # Create some random colors
'''
    color: This generates 100 random colors, each represented by a 3-element array (RGB color). These colors will be used to visualize optical flow paths.
'''

#%% Parameters for Lucas Kanade optical flow approach [Ref: https://cseweb.ucsd.edu//classes/sp02/cse252/lucaskanade81.pdf]
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
'''
    feature_params: Defines parameters for Shi-Tomasi corner detection, which is used to detect good features (corner points) in the image.

        maxCorners: The maximum number of corners to return.

        qualityLevel: Minimum accepted quality of corners.

        minDistance: Minimum distance between detected corners.

        blockSize: Size of the local window to calculate corner quality.

    Changing the Shi-Tomasi Corner Detection (feature_params)

        Increasing maxCorners will detect more feature points to track. Reducing it will track fewer points.

        A lower value of qualityLevel (e.g., 0.1) will detect more features (including weaker ones), while a higher value (e.g., 0.9) will detect only the strongest features.

        Reducing minDistance will allow the detection of feature points that are closer together. Increasing it will space the points further apart.

        A larger value of blockSize can help in detecting corners in larger blocks, potentially improving detection in more complex scenes.
'''

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                       maxLevel = 2,
                       criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
'''
    lk_params: Defines parameters for the Lucas-Kanade optical flow calculation.

        winSize: Size of the window used for optical flow calculation.

        maxLevel: The number of pyramid levels for optical flow.

        criteria: Termination criteria. The algorithm stops when either the number of iterations or the accuracy is reached.

    
    Changing the Lucas-Kanade Optical Flow Parameters (lk_params)
        winSize controls the size of the window used to compute the flow.
        A smaller value will make the flow estimation more sensitive to small movements, but might be less stable.
        A larger value will smoothen the flow estimation, but may miss finer details.

        Increasing maxLevel will allow the algorithm to consider more pyramid levels, which helps detect larger motions, but may reduce precision for small movements.

        Criteria dictates how the algorithm terminates.
        You can change the 0.03 threshold to a higher value (e.g., 0.1) to allow for more movement before terminating,
        or reduce it to stop the calculation sooner for more accurate tracking.
'''

#%% Flow estimation is always with respect to previous frame and the below code is required to be done for the first time as called from main
def set1stFrame(frame):
    
    # Converting to gray scale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params) # Corner detection using https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame)
    
    return frame_gray,mask,p0
'''
    set1stFrame: This function processes the first frame from the video.

        frame_gray: Converts the captured frame to grayscale, which simplifies the processing.

        p0: Detects good feature points (corners) to track in the subsequent frames using Shi-Tomasi.

        mask: Creates a blank image of the same size as the input frame, which will be used to draw the flow paths.

        Returns: The grayscale frame, the mask, and the list of corner points p0.
'''

#%% Lucas Kanade optical flow approach [Ref: https://cseweb.ucsd.edu//classes/sp02/cse252/lucaskanade81.pdf]
def LucasKanadeOpticalFlow (frame,old_gray,mask,p0):
    
    # Converting to gray scale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # calculate optical flow
    if (p0 is None or len(p0) ==0):
        p0 = np.array([[50, 50], [100, 100]], dtype=np.float32).reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray,
                                           p0, None, **lk_params)
    
    if p1 is not None:    
    
        # Select good points (skip no points to avoid errors)
        good_new = p1[st==1]
        good_old = p0[st==1]
    
        # draw the tracks
        for i, (new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a),int(b)), (int(c),int(d)), color[i].tolist(), 2)
            frame_gray = cv2.circle(frame_gray, (int(a),int(b)), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
    
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    
    return img,old_gray,p0
'''
    LucasKanadeOpticalFlow: This function implements the Lucas-Kanade optical flow estimation.

        Converts the current frame to grayscale.

        Checks if there are no points to track (p0 is None or empty) and initializes some points if needed.

        cv2.calcOpticalFlowPyrLK: Calculates optical flow (movement) of the points from the previous frame to the current frame.

        If good flow points (p1) are detected, the function draws the movement vectors (lines) and updates the mask to visualize it.

        The previous frame (old_gray) and the feature points (p0) are updated for the next iteration.
'''


#%% Computes a dense optical flow using the Gunnar Farneback's algorithm.
step = 16 

def DenseOpticalFlowByLines(frame, old_gray):
    
    # Converting to gray scale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
    
    h, w = frame_gray.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2,-1)
    
    flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)  # https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af 
    '''
        0.5 (pyr_scale): This parameter controls the image scale used in pyramid calculations.
        Increasing it may give a better estimate of global motion but might make it less sensitive to small local movements.

        15 (win_size): This parameter defines the window size for each level of the pyramid.
        Larger values will provide smoother flow estimation at the cost of finer details.

        3 (iterations): This controls how many iterations to run at each pyramid level.
        Increasing it allows for better refinement but might slow down the process.

        5 (poly_n): The size of the neighborhood used for polynomial expansion.
        Larger values will allow the model to capture more local variations in motion, but might make the flow estimation noisy.

        1.2 (poly_sigma): This affects the standard deviation of the Gaussian filter used to smooth the image.
        Larger values will make the flow estimation more stable but blur finer details.
    '''


    fx, fy = flow[y,x].T
    
    # Plot the streamlines
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)    
    cv2.polylines(frame, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(frame, (x1, y1), 1, (0, 255, 0), -1)
    return frame
'''
    DenseOpticalFlowByLines: This function estimates dense optical flow using Gunnar Farneback's method.

        Converts the current frame to grayscale.

        Creates a grid of points where flow is to be calculated (mgrid).

        cv2.calcOpticalFlowFarneback: Computes dense optical flow (motion) between the previous and current frames.

        Visualizes the flow using green lines and circles on the image.
'''

#%% Open CV Video Capture and frame analysis
cap = cv2.VideoCapture(0)
'''
    cv2.VideoCapture(0): Opens the webcam (0 is typically the default camera).

    cap.isOpened(): Checks if the camera was successfully opened. If not, it raises an error.
'''

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

firstframeflag = 1
'''
    firstframeflag: A flag to ensure the first frame is processed separately for initialization.
'''

# The loop will break on pressing the 'q' key
while True:
    try:
        
        if (firstframeflag):
            # Capture one frame
            ret, frame = cap.read() 
            
            old_gray,mask,p0 = set1stFrame(frame)            
          
            firstframeflag = 0
        
        # Capture one frame
        ret, frame = cap.read()  
        
        # Comment/uncomment respective lines (line 227/228) to activate the desired results. 
        img = DenseOpticalFlowByLines(frame, old_gray)
        #img,old_gray,p0 = LucasKanadeOpticalFlow(frame,old_gray,mask,p0)
        
        cv2.imshow("Optical Flow", img)
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   
    except KeyboardInterrupt:
        break

'''
    The loop continuously captures frames from the webcam:

        For the first frame, it initializes the optical flow and feature points using set1stFrame.

        For subsequent frames, it computes the dense optical flow using DenseOpticalFlowByLines.

        The flow is visualized on the image and displayed with cv2.imshow.

        The loop can be stopped by pressing the 'q' key (cv2.waitKey(1) checks for this).

        The loop also handles KeyboardInterrupt to break out of the loop cleanly.
'''

cap.release()
cv2.destroyAllWindows()
'''
    cap.release(): Releases the webcam for cleanup.

    cv2.destroyAllWindows(): Closes any OpenCV windows that were opened.
'''