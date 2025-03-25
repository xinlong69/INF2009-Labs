'''
    This script captures live video from the webcam and detects people in real-time using Histogram of Oriented Gradients (HOG)
    and a pre-trained Support Vector Machine (SVM) model.
    The detected person closest to the center of the screen is highlighted, and directional instructions (left, right, or center)
    are printed based on their position.
'''

import cv2
import numpy as np
'''
    cv2: OpenCV library for computer vision tasks.

    numpy: For handling arrays and mathematical operations.
'''

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
'''
    cv2.HOGDescriptor(): Initializes the HOG feature extractor, which extracts gradient-based features from images.

    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()):

    Loads a pre-trained SVM-based person detector from OpenCV’s default model.
'''

#sets how many pixels away from the center a person needs to be before the head stops
center_tolerance = 5; 
'''
    If the detected person is within ±5 pixels of the center, they are considered centered.


    Smaller value (5): More sensitive to small movements.
    Larger value (20): Only significant movements trigger direction change.
'''

#%% Open CV Video Capture and frame analysis
cap = cv2.VideoCapture(0)
'''
    Opens the default webcam (0 refers to the primary camera).

    If using an external camera, change to 1 or 2 (depending on your setup).
'''

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
'''
    Ensures the camera is accessible, otherwise raises an error.
'''

# The loop will break on pressing the 'q' key
while True:
    try:
        # Capture one frame
        ret, frame = cap.read()
        '''
            Reads a single frame from the webcam.

            ret: Boolean indicating success (True) or failure (False).
            frame: The captured image.

            Lower Resolution (e.g., 256x256): Faster but less detailed.
            Higher Resolution (e.g., 512x512): More accurate but slower.
        '''
        
        # resizing for faster detection
        frame = cv2.resize(frame, (256, 256)) #Uncomment and see the speed up
        '''
            A smaller image speeds up detection but reduces accuracy.
        '''

       # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = hog.detectMultiScale(frame, winStride=(1,1), scale = 1.05)
        '''
            detectMultiScale() detects objects (people in this case) and returns:

                boxes: The bounding boxes around detected persons.
                weights: Confidence scores for each detection.

            🔹 Parameter Tuning:

                winStride=(1,1): Smaller values detect more people but increase processing time.
                scale=1.05: Determines the scaling factor between image pyramids (higher values detect larger objects).
            
                
            Lower scale (e.g., 1.05): Detects smaller objects but increases processing time.
            Higher scale (e.g., 1.2): Faster but may miss smaller people.

            Smaller winStride (e.g., (2,2)): More precise but slower.
            Larger winStride (e.g., (8,8)): Faster but may miss detections.
        '''

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        '''
            Converts (x, y, width, height) format to (x1, y1, x2, y2), which OpenCV uses for drawing rectangles.
        '''

        centers = []
        for box in boxes:
            #get the distance from the center of each box's center x cord to the center of the screen and ad them to a list
            center_x = ((box[2]-box[0])/2)+box[0]
            x_pos_rel_center = (center_x-70)
            dist_to_center_x = abs(x_pos_rel_center)
            centers.append({'box': box, 'x_pos_rel_center': x_pos_rel_center, 'dist_to_center_x':dist_to_center_x})    
        
        '''
            Computes the horizontal center of each detected person (center_x).

            Measures how far they are from a reference center (70 pixels from the left).

            Stores data in a list sorted by distance from the center.
        '''
        
        if len(centers) > 0:
            # Sorts the list by distance_to_center
            sorted_boxes = sorted(centers, key=lambda i: i['dist_to_center_x'])
            '''
                If at least one person is detected, sort the list by proximity to the center.
            '''

            # Draws the box
            center_box = sorted_boxes[0]['box']

            for box in range(len(sorted_boxes)):

                # display the detected boxes in the colour picture
                if box == 0:
                    cv2.rectangle(frame, (sorted_boxes[box]['box'][0],sorted_boxes[box]['box'][1]), (sorted_boxes[box]['box'][2],sorted_boxes[box]['box'][3]), (0,255, 0), 2)
                else:
                    cv2.rectangle(frame, (sorted_boxes[box]['box'][0],sorted_boxes[box]['box'][1]), (sorted_boxes[box]['box'][2],sorted_boxes[box]['box'][3]),(0,0,255),2)
            
            '''
                Draws a green box (0, 255, 0) around the center-most person.

                Draws red boxes (0, 0, 255) around other detected people.
            '''
            
            # Retrieves the distance from center from the list and determins if the head should turn left, right, or stay put and turn lights on
            Center_box_pos_x = sorted_boxes[0]['x_pos_rel_center']

            if -center_tolerance <= Center_box_pos_x <= center_tolerance:
                # Turn on eye light
                print("center") # Person is centered
            elif Center_box_pos_x >= center_tolerance:
                # Turn head to the right
                print("right") # Person is on the right, turn head right
            elif Center_box_pos_x <= -center_tolerance:
                # Turn head to the left
                print("left") # Person is on the left, turn head left

            print(str(Center_box_pos_x))
            '''
                Compares the center position with center_tolerance:

                    Within ±5 pixels → "center"

                    Beyond +5 pixels → "right"

                    Below -5 pixels → "left"
            '''

        else:
            # prints out that no person has been detected            
            print("nothing detected")
            '''
                If no one is detected, it prints "nothing detected".
            '''

        #resizes the video so its easier to see on the screen
        frame = cv2.resize(frame,(720,720))
        '''
            Enlarges the frame for better visibility.

            
            Lower Resolution (e.g., 256x256): Faster but less detailed.
            Higher Resolution (e.g., 512x512): More accurate but slower.
        '''

        # Display the resulting frame
        cv2.imshow("frame",frame)
        '''
            Shows the real-time detection with bounding boxes.
        '''
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
        '''
            Press 'q' to exit the loop and close the webcam.
        '''

    except KeyboardInterrupt:
        break

cap.release()
cv2.destroyAllWindows()

'''
    Releases the webcam and closes OpenCV windows when the script exits.
'''