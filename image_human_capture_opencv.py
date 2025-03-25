'''
    This script uses OpenCV’s Haar Cascade Classifier to detect faces in a webcam feed. 
'''

# For the code to work the Open source Haar Cascade model has to be downloaded and kept in the same folder. 
# Please download the .xml from https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml

import cv2
'''
    cv2 (OpenCV) is a computer vision library that provides image processing functions.
'''

# Initiate the Face Detection Cascade Classifier
haarcascade = "haarcascade_frontalface_alt2.xml"
detector = cv2.CascadeClassifier(haarcascade)
'''
    Haar Cascade Classifier is a pre-trained XML model used for face detection.

    The CascadeClassifier loads the model from haarcascade_frontalface_alt2.xml (must be in the same folder).

    This classifier detects frontal human faces.
'''

#%% Open CV Video Capture and frame analysis
cap = cv2.VideoCapture(0)
'''
    Opens the webcam (0 represents the default webcam).

    If using an external camera, change 0 to 1 or the correct index.
'''

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

'''
    Ensures the camera is available before proceeding.

    If not accessible, it raises an error.
'''

# The loop will break on pressing the 'q' key
# Continuously captures frames from the webcam.
while True:
    try:
        # Capture one frame
        ret, frame = cap.read()
        '''
            Captures one frame from the webcam feed.

            ret is True if successful, otherwise False.

            frame is the captured image.
        '''
        
        # resizing for faster detection
        frame = cv2.resize(frame, (256, 256)) #Comment and see the speed up
        '''
            Speeds up face detection by reducing image size to 256x256 pixels.

            Comment this line to see the performance difference.
        '''
        
        # Converting to gray scale as feature extraction works only on gray scale image
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        '''
            Haar Cascade works best on grayscale images.

            Converts the color image (BGR) to grayscale.
        '''
        
        # Detect faces using the haarcascade classifier on the "grayscale image"
        faces = detector.detectMultiScale(image_gray)
        '''
            detectMultiScale() finds faces in the grayscale image.

            Returns a list of bounding boxes around detected faces.
        '''
       
        # returns the bounding boxes for the detected objects and display them using rectangles 
        for face in faces:
            (x,y,w,d) = face
            # Draw a white coloured rectangle around each face using the face's coordinates
            # on the "image_template" with the thickness of 2 
            cv2.rectangle(frame,(x,y),(x+w, y+d),(255, 255, 255), 2)
        '''
            Loops through all detected faces.

            Extracts x, y, w, d (top-left corner and width/height).

            Draws a white rectangle around each detected face:

                (x, y) → Top-left corner.

                (x+w, y+d) → Bottom-right corner.

                (255, 255, 255) → White color.

                2 → Line thickness.
        '''
       
        # resizes the video so its easier to see on the screen
        frame = cv2.resize(frame,(720,720))
        '''
            Increases the displayed frame size to 720x720 pixels for better visibility.
        '''

        # Display the resulting frame
        cv2.imshow("frame",frame)
        '''
            Opens a window to display the frame.

            Detected faces are shown with rectangles.
        '''
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

        '''
            Checks if the 'q' key is pressed.

            If true, exits the loop.
        '''
        
    except KeyboardInterrupt:
        break
    '''
        Allows manual termination of the script using CTRL+C.
    '''

cap.release()
cv2.destroyAllWindows()
'''
    Stops the webcam.

    Closes all OpenCV windows.
'''