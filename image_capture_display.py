'''
    This script captures video from a webcam, processes each frame in real time, and applies color segmentation to detect red, green, and blue colors.
    The segmented images are displayed alongside the original frame, allowing the user to see the detected colors.
    The program exits when the 'q' key is pressed or a manual interruption occurs.
'''

# Reference: https://pyimagesearch.com/2014/08/04/opencv-python-color-detection/
import cv2
import numpy as np
'''
    cv2: Imports OpenCV, which is used for computer vision tasks such as image processing and color detection.

    numpy: Imports NumPy, which provides powerful numerical operations and array handling.
'''

#%% Defining a list of boundaries in the RGB color space 
# (or rather, BGR, since OpenCV represents images as NumPy arrays in reverse order) 
# Refer to https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
boundaries = [
	([17, 15, 100], [50, 56, 200]), # For Red
	([86, 31, 4], [220, 88, 50]), # For Blue
	([25, 90, 4], [62, 200, 50])] # For Green 
'''
    Defines a list of color boundaries in the BGR format (since OpenCV uses BGR instead of RGB).

    Each boundary consists of:

        A lower bound (minimum BGR values).

        An upper bound (maximum BGR values).

    These ranges allow segmentation of red, blue, and green colors in an image.

    When increase the range of a color (e.g. from ([17, 15, 100], [50, 56, 200]) to ([0, 0, 100], [100, 100, 255]),
    the red detection will be more inclusive and might detect more shades of red.)

    When limiting the range of a color (e.g. from ([17, 15, 100], [50, 56, 200]) to ([0, 0, 100], [50, 50, 255]),
    only very bright red pixels will be detected, filtering out darker reds.
'''

#%% Normalize the Image for display (Optional)
def normalizeImg (Img):
    Img= np.float64(Img) # Converting to float to avoid errors due to division
    norm_img = (Img - np.min(Img))/(np.max(Img) - np.min(Img)) # Normalizing pixel values to range [0,1]
    norm_img = np.uint8(norm_img * 255.0) # Converting back to uint8 for display
    return norm_img
'''
    Converts the image to a floating-point format to prevent integer division errors.

    Normalizes pixel values so they lie between 0 and 1.

    Scales them back to 0-255 and converts them back to uint8 format for proper image display.
'''

#%% Open CV Video Capture and frame analysis
cap = cv2.VideoCapture(0)
'''
    Opens the default webcam (0 refers to the primary webcam).

    This allows capturing live video frames for color detection.
'''

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
'''
    Ensures that the webcam was successfully accessed.

    If it fails, the program raises an error and stops execution.
'''

# The loop will break on pressing the 'q' key
# Starts an infinite loop for real-time video processing.
# Encapsulated in a try-except block to handle unexpected errors (e.g., manual interruption).
while True:
    try:

        # Capture one frame
        ret, frame = cap.read()
        '''
            cap.read() captures a frame from the webcam.

            ret is a boolean indicating success (True) or failure (False).

            frame is the captured image (a NumPy array representing pixel values).
        '''
        
        output=[]
        '''
            Creates an empty list to store the segmented images for red, green, and blue colors.
        '''
        
        # loop over the boundaries
        # Iterates over the color boundaries defined earlier.
        for (lower, upper) in boundaries:

            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")
            '''
                Converts the lower and upper boundary lists into NumPy arrays.

                The data type is set to uint8 (unsigned 8-bit integer) since image pixels are typically stored in this format.
            '''
            
            # find the colors within the specified boundaries and apply the mask (basically segmenting for colours)
            mask = cv2.inRange(frame, lower, upper)
            '''
                Applies thresholding to keep only pixels within the specified color range.

                Pixels inside the range are set to 255 (white).

                Pixels outside the range are set to 0 (black).

                This results in a binary mask where only the desired color appears.
            '''
            
            output.append(cv2.bitwise_and(frame, frame, mask = mask)) #Segmented frames are appended
            '''
                Uses bitwise AND operation to retain only the pixels that fall within the mask.

                This results in a new image where only the selected color is visible, while everything else is black.

                The segmented image is appended to the output list.
            '''
        
        # Output is appeneded to be of size Pixels X 3 (for R, G, B)
        red_img = normalizeImg(output[0])
        green_img = normalizeImg(output[1])
        blue_img = normalizeImg(output[2])
        '''
            Calls normalizeImg() on each segmented image.

            This ensures pixel values are scaled appropriately for proper visualization.
        '''
       
        # horizontal Concatination for displaying the images and colour segmentations
        catImg = cv2.hconcat([frame,red_img,green_img,blue_img])
        '''
            Combines the original frame and the three segmented images into a single display.

            cv2.hconcat() horizontally stacks the images.
        '''

        cv2.imshow("Images with Colours",catImg)
        '''
            Displays the concatenated image window with:

                The original frame.

                Red-segmented image.

                Green-segmented image.

                Blue-segmented image.
        '''
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
        '''
            Waits for 1 millisecond for a key press.

            If the 'q' key is pressed, the loop breaks, and the program exits.
        '''
   
    except KeyboardInterrupt:
        break
    '''
        Handles a manual interruption (Ctrl+C).

        If the user forcibly stops the program, it gracefully exits the loop.
    '''

cap.release()
cv2.destroyAllWindows()
'''
    Releases the webcam so other programs can access it.

    Closes all OpenCV windows to free up system resources.
'''
