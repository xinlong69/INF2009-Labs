import cv2
import numpy as np

# For HoG (Histogram of Oriented Gradients) feature extraction
from skimage import feature # Refer to https://scikit-image.org/ for full capabalities of scikit-image library
from skimage import exposure # For image contrast adjustment
'''
    cv2 (OpenCV): Used for video capture, image processing, and display.

    numpy: Provides support for numerical operations on arrays.

    skimage.feature: Provides the hog() function for extracting Histogram of Oriented Gradients (HoG) features.

    skimage.exposure: Used to rescale the intensity of images for better visualization.
'''

#%% Open CV Video Capture and frame analysis
cap = cv2.VideoCapture(0)
'''
    Opens the default webcam (0).

    Alternative: If using an external camera, change 0 to 1 or the respective camera index.
'''

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
'''
    Ensures the webcam is successfully opened.

    If it fails, an I/O error is raised.
'''

# The loop will break on pressing the 'q' key
# A loop to continuously capture and process frames.
# The loop will run until ‘q’ is pressed or an exception occurs.
while True:
    try:
        # Capture one frame
        ret, frame = cap.read()
        '''
            ret is True if the frame was read successfully, False otherwise.

            frame contains the current image from the webcam.
        '''
        
        # resizing for faster detection
        # Resizing reduces processing time, making real-time detection faster.
        frame = cv2.resize(frame, (256, 256)) #Uncomment and see the speed up
        
        # Converting to gray scal as HOG feature extraction in scikit-image works only on gray scale image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        '''
            OpenCV captures frames in BGR format, but HoG in scikit-image only works on grayscale images.

            Converts BGR → Grayscale for HoG feature extraction.
        '''
        
        # Extact the HoG featues from the image
        (H, hogImage) = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
    	cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
    	visualize=True)
        '''
            feature.hog() extracts HoG features.

            HoG Parameters Explained:

                orientations=9: Uses 9 orientation bins for gradient computation.

                pixels_per_cell=(8, 8): Divides the image into 8×8 pixel cells.

                cells_per_block=(2, 2): Normalizes features over 2×2 block of cells.

                transform_sqrt=True: Applies square-root transformation for normalization.

                block_norm="L1": Uses L1 normalization to scale feature vectors.

                visualize=True: Returns the HoG image representation (hogImage).

                
            Lower orientations (e.g., from orientation=9 to 6) → Less detail, smoother features
            Higher orientations (e.g., from orientation=9 to 18) → More detailed features, but noisier image

            Larger pixels_per_cell (e.g., from pixels_per_cell=(8, 8) to (16, 16)) → Larger blocks, less fine details
            Smaller pixels_per_cell (e.g., from pixels_per_cell=(8, 8) to (4, 4)) → Finer details but more computation

            Larger blocks (e.g., from pixels_per_cell=(2, 2) to (3,3)) → More generalized features, smoother detection
            Smaller blocks (e.g., from pixels_per_cell=(2, 2) to (1,1)) → Very local features, might introduce noise

            Change block_norm="L1"

                Options:
                    "L1" : L1 normalization
                    "L2-Hys" : More robust against lighting changes
                    "L2" : Normalization with better contrast preservation

                🔎 Effect:
                    "L2-Hys" usually gives better feature representation for object detection.
                    "L1" might make subtle features more visible.

            Change transform_sqrt=False
                Options:
                    True: Helps with illumination changes (better contrast).
                    False: Can result in darker feature maps.

        '''
        
        # Rescale intensity to be within 0-255 (contrast stretching)
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))        
        hogImage = hogImage.astype("uint8")
        '''
            Rescales HoG feature intensities to the range 0-255 for better visualization.

            Converts the float64 image to uint8 for OpenCV compatibility.
        '''
        
        # Converting gray to RGB
        hogImg = cv2.cvtColor(hogImage,cv2.COLOR_GRAY2RGB)
        '''
            Converts grayscale HoG image to RGB format for better visualization.
        '''
        
        # Horizontal concatenation to show both input image and its HoG features.
        catImg = cv2.hconcat([frame,hogImg])        
        cv2.imshow("HOG Image", catImg)
        '''
            cv2.hconcat([frame, hogImg]): Concatenates original and HoG images horizontally.

            cv2.imshow("HOG Image", catImg): Displays both images together.
        '''
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
        '''
            Breaks the loop when 'q' is pressed.
        '''
        
    except KeyboardInterrupt:
        break
    '''
        Stops execution if Ctrl+C is pressed.
    '''

cap.release()
cv2.destroyAllWindows()
'''
    cap.release(): Releases the webcam.

    cv2.destroyAllWindows(): Closes all OpenCV windows.
'''

# %%
